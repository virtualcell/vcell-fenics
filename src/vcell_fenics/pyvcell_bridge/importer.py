"""Translate a VCell ``MathDescription`` (pyvcell's lowered math model) into a
formalism :class:`~vcell_fenics.formalism.schema.MathDescription` (doc §2.6).

This is the **problem-generation front door** of the template-surface architecture
(`docs/modeling/validation-and-diagnostics.md` §1.1): VCell already guarantees that
its math is built from well-posed templates, so a VCell-imported model lands on the
*template* surface (T1/T2/T4), not the weak-form escape hatch — well-posed by
construction. The translator implements the §2.6.1 direct maps:

- ``CompartmentSubDomain`` → ``Subdomain(kind="volume")``; ``MembraneSubDomain`` →
  ``Subdomain(kind="surface")`` (both with ``motion: none`` — this pyvcell model
  carries no membrane velocity yet).
- ``PdeEquation`` → ``bulk_radv_diff`` (compartment) / ``surface_pde_with_dilution``
  (membrane), with ``diffusion`` / ``source`` (the reaction ``rate``) /
  ``relative_advection`` (the ``velocity``) slots and ``initial_condition``;
  ``steady`` picks the temporality.
- ``OdeEquation`` → ``lumped_ode`` (``rate`` slot).
- numeric ``Constant`` → ``ParameterConstant``; symbolic ``Constant`` and **pure**
  ``MathFunction`` (no variable reference) → ``ParameterExpression``.

**Functions and observables.** A VCell ``MathFunction`` that references state variables
(e.g. ``I = gL*(V - VL)``) cannot be a parameter (the formalism forbids variable refs in
parameters). Such functions are **inlined** into the equation expressions that use them
(so each equation is self-contained for the compiler) and **also** surfaced as
:class:`Observable`\\ s — derived outputs kept on the side, never inside the math (§2.6.3).
See :mod:`~vcell_fenics.pyvcell_bridge.inlining`.

Expression strings are run through :func:`translate_expression` (coordinate/time
namespacing, ``^``→``**``). :func:`import_model` returns the math + observables; the
convenience :func:`import_math_description` returns just the math.

The translator reads pyvcell's typed pydantic model directly; the pyvcell types are
imported under ``TYPE_CHECKING`` (annotations only), so the module type-checks against the
real ``pyvcell.vcml.models_math`` classes without a runtime import of pyvcell.

**Boundary conditions.** Per-face box BCs (§2.6.2) map ``Flux``→Neumann / ``Value``→Dirichlet
(with VCell's default-Dirichlet-from-IC rule), needing the geometry ``dim`` to tell which of the
six faces are real. Membrane ``JumpCondition``s map to Neumann BCs on the bulk species at the
membrane (the composable bulk-surface pattern §1.6.5), with cross-membrane volume-species
references wrapped in ``trace(·)``.

**Loud rejection.** Constructs that would change the math if dropped are never dropped silently:
stochastic / particle dynamics (§2.6.3) raise :class:`VcellImportError`; a boundary-bearing PDE
without ``dim``, a non-Flux/Value face type (periodic), and a jump-condition species living in both
compartments raise :class:`NotImplementedError` pointing at the follow-up.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCNeumann,
    BoundaryCondition,
    Equation,
    MathDescription,
    MotionNone,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.pyvcell_bridge.expression import translate_expression
from vcell_fenics.pyvcell_bridge.inlining import _IDENT_RE, FunctionResolution, resolve_functions

if TYPE_CHECKING:
    from pyvcell.vcml.models_math import (
        CompartmentSubDomain,
        MembraneSubDomain,
        PdeEquation,
        Velocity,
    )
    from pyvcell.vcml.models_math import (
        MathDescription as VcmlMathDescription,
    )

    # A volume or surface subdomain of the VCell math model.
    VcmlSubDomain = CompartmentSubDomain | MembraneSubDomain


class VcellImportError(Exception):
    """A VCell ``MathDescription`` uses a construct that cannot be imported into the
    formalism (an out-of-scope §2.6.3 construct, or a malformed equation)."""


@dataclass(frozen=True)
class Observable:
    """A derived output quantity (a VCell ``MathFunction`` that references state variables),
    kept *outside* the math description (§2.6.3). ``expression`` is in formalism syntax and
    may reference variables, parameters, ``geom.*`` and ``sim.t``; ``subdomain`` is the
    function's home subdomain, if VCell recorded one."""

    name: str
    expression: str
    subdomain: str | None = None


@dataclass(frozen=True)
class ImportResult:
    """The result of importing a VCell model: the solvable formalism ``MathDescription`` plus
    the observables sidecar (derived quantities, not part of the math/solve)."""

    math: MathDescription
    observables: tuple[Observable, ...] = field(default=())


def import_model(vcml: VcmlMathDescription, *, geometry: str | None = None, dim: int | None = None) -> ImportResult:
    """Translate a pyvcell ``MathDescription`` into a formalism ``MathDescription`` plus its
    observables. ``geometry`` defaults to the VCell math description's ``name`` (it is
    cross-checked against a real Geometry at solve time, not here).

    ``dim`` is the spatial dimension of the geometry (0–3). It is required to import per-face
    boundary conditions (§2.6.2): VCell's `boundary_types` are model-wide and list all six box
    faces (`Xm…Zp`) even for a 2D model, so the dimension says which faces are real (`x_minus` /
    `x_plus` for 2D, plus `z_*` for 3D). Without ``dim`` a PDE that carries any non-default
    boundary is rejected rather than mis-imported."""

    variable_names = _collect_variable_names(vcml)
    resolution = resolve_functions(list(vcml.functions), variable_names)
    # Bulk (volume) species → the compartment(s) each lives in: the species set drives trace-wrapping
    # of cross-membrane flux references (§1.6.5); the compartment map detects a species on both sides
    # of a membrane (which our per-variable BC cannot yet disambiguate).
    species_compartments: dict[str, set[str]] = {}
    for compartment in vcml.compartment_subdomains:
        for pde in compartment.pde_equations:
            species_compartments.setdefault(pde.name, set()).add(compartment.name)
    bulk_species = set(species_compartments)

    subdomains: list[Subdomain] = []
    variables: list[Variable] = []
    equations: list[Equation] = []
    boundary_conditions: list[BoundaryCondition] = []

    for compartment in vcml.compartment_subdomains:
        _reject_stochastic(compartment)
        subdomains.append(Subdomain(name=compartment.name, kind="volume", motion=MotionNone()))
        _translate_subdomain_equations(
            compartment, "volume", variables, equations, boundary_conditions, resolution, dim
        )

    for membrane in vcml.membrane_subdomains:
        _reject_stochastic(membrane)
        subdomains.append(Subdomain(name=membrane.name, kind="surface", motion=MotionNone()))
        _translate_subdomain_equations(membrane, "surface", variables, equations, boundary_conditions, resolution, dim)
        boundary_conditions.extend(_translate_jump_conditions(membrane, bulk_species, species_compartments, resolution))

    parameters = _translate_parameters(vcml, resolution)
    observables = _build_observables(vcml, resolution)

    # Drop zero-flux Neumann BCs: a zero D∇u·n is the natural (no-flux) default, so imposing it is
    # redundant. It also avoids a spurious constraint where it doesn't belong — VCell emits all-faces
    # no-flux defaults for *every* compartment, so an interior compartment (not touching the box) gets
    # box-face BCs it has no boundary for; those would otherwise be rejected at solve.
    numeric = {c.name: value for c in vcml.constants if (value := _as_float(c.exp)) is not None}
    boundary_conditions = [bc for bc in boundary_conditions if not _is_zero_neumann(bc, numeric)]

    math = MathDescription(
        geometry=geometry if geometry is not None else vcml.name,
        subdomains=subdomains,
        variables=variables,
        equations=equations,
        parameters=parameters,
        boundary_conditions=boundary_conditions,
    )
    return ImportResult(math=math, observables=tuple(observables))


def _is_zero_neumann(bc: BoundaryCondition, numeric: dict[str, float]) -> bool:
    """Whether ``bc`` is a Neumann flux that resolves to zero — a literal ``0`` or a single numeric
    constant equal to 0 (VCell's per-face no-flux default, e.g. ``u_boundaryXm = 0``)."""

    if not isinstance(bc, BCNeumann):
        return False
    expression = bc.expression.strip()
    literal = _as_float(expression)
    return (literal == 0.0) if literal is not None else (numeric.get(expression) == 0.0)


def import_math_description(
    vcml: VcmlMathDescription, *, geometry: str | None = None, dim: int | None = None
) -> MathDescription:
    """Convenience wrapper returning just the formalism ``MathDescription`` (the observables
    sidecar is dropped). See :func:`import_model`."""

    return import_model(vcml, geometry=geometry, dim=dim).math


def _collect_variable_names(vcml: VcmlMathDescription) -> set[str]:
    """The state-variable names — the equation-governed names (plus any declared
    ``MathVariable``s). Used to decide which functions reference variables."""

    names: set[str] = {v.name for v in vcml.variables}
    for compartment in vcml.compartment_subdomains:
        names.update(p.name for p in compartment.pde_equations)
        names.update(o.name for o in compartment.ode_equations)
    for membrane in vcml.membrane_subdomains:
        names.update(p.name for p in membrane.pde_equations)
        names.update(o.name for o in membrane.ode_equations)
    return names


def _translate_subdomain_equations(
    subdomain: VcmlSubDomain,
    kind: str,
    variables: list[Variable],
    equations: list[Equation],
    boundary_conditions: list[BoundaryCondition],
    res: FunctionResolution,
    dim: int | None,
) -> None:
    """Append the variables + template equations for one subdomain's PDEs and ODEs (inlining
    variable-referencing functions into each expression), plus the per-face boundary conditions."""

    pde_template = "bulk_radv_diff" if kind == "volume" else "surface_pde_with_dilution"

    def expr(raw: str | None) -> str | None:
        if raw is None:
            return None
        return translate_expression(res.inline(raw) or "")

    for pde in subdomain.pde_equations:
        boundary_conditions.extend(_translate_boundaries(pde, subdomain, kind, expr, dim))
        terms: dict[str, str] = {}
        if pde.diffusion is not None:
            terms["diffusion"] = expr(pde.diffusion)  # type: ignore[assignment]
        if pde.rate is not None:
            terms["source"] = expr(pde.rate)  # type: ignore[assignment]
        advection = _velocity_vector(pde.velocity, res)
        if advection is not None:
            terms["relative_advection"] = advection
        variables.append(Variable(name=pde.name, subdomain=subdomain.name, type="scalar"))
        equations.append(
            TemplateEquation(
                template=pde_template,
                variable=pde.name,
                subdomain=subdomain.name,
                temporality="steady_state" if pde.steady else "time_dependent",
                terms=terms,
                initial_condition=expr(pde.initial),
            )
        )

    for ode in subdomain.ode_equations:
        if ode.rate is None:
            raise VcellImportError(f"ODE for {ode.name!r} on {subdomain.name!r} has no rate expression")
        variables.append(Variable(name=ode.name, subdomain=subdomain.name, type="scalar"))
        equations.append(
            TemplateEquation(
                template="lumped_ode",
                variable=ode.name,
                subdomain=subdomain.name,
                temporality="time_dependent",
                terms={"rate": expr(ode.rate)},  # type: ignore[dict-item]
                initial_condition=expr(ode.initial),
            )
        )


def _velocity_vector(velocity: Velocity | None, res: FunctionResolution) -> str | None:
    """Build a formalism vector-expression `"[vx, vy(, vz)]"` from a VCell ``Velocity``
    (the species' advection field → the template's ``relative_advection`` slot), or
    ``None`` if there is no velocity. A 2D model (no z component) yields two components."""

    if velocity is None:
        return None
    components = [velocity.x, velocity.y]
    if velocity.z is not None:
        components.append(velocity.z)
    if all(c is None for c in components):
        return None
    rendered = [translate_expression(res.inline(c) or "") if c is not None else "0" for c in components]
    return "[" + ", ".join(rendered) + "]"


def _translate_parameters(vcml: VcmlMathDescription, res: FunctionResolution) -> list[Parameter]:
    """VCell ``Constant``s → parameters; **pure** ``MathFunction``s (no variable reference)
    → ``ParameterExpression``. Variable-referencing functions are not parameters — they are
    inlined into equations and surfaced as observables instead.

    Pure functions that nothing in the model references are *not* imported: VCell emits region-size
    bookkeeping (``Size_<compartment>``, ``vobj_<region>_size``) calling geometric built-ins like
    ``vcRegionVolume('domain')`` that our expression formalism does not model. They are provably dead
    (no equation, boundary, or other parameter reaches them), so dropping them removes no model
    semantics — and a *referenced* such function is still emitted and rejected loudly at validation."""

    reachable = _reachable_names(vcml)
    parameters: list[Parameter] = []
    for constant in vcml.constants:
        numeric = _as_float(constant.exp)
        if numeric is not None:
            parameters.append(ParameterConstant(name=constant.name, value=numeric))
        else:
            parameters.append(
                ParameterExpression(name=constant.name, expression=translate_expression(res.inline(constant.exp) or ""))
            )
    for function in vcml.functions:
        if function.name in res.constant_function_names and function.name in reachable:
            parameters.append(
                ParameterExpression(
                    name=function.name,
                    expression=translate_expression(function.exp or ""),
                    subdomain=function.domain,
                )
            )
    return parameters


def _reachable_names(vcml: VcmlMathDescription) -> frozenset[str]:
    """The names transitively referenced by the model's equations, boundary values, and constant /
    function expressions — the live set. Used to drop dead pure functions (see
    :func:`_translate_parameters`). Roots are every PDE/ODE rate, diffusion, and initial expression,
    each per-face boundary value, each membrane jump flux, and each constant expression; the closure
    then follows function bodies. Identifier matching reuses the inliner's dotted-name-aware,
    call-excluding regex, so a built-in call like ``vcRegionVolume(...)`` is not itself a name."""

    bodies = {f.name: (f.exp or "") for f in vcml.functions}
    roots: list[str | None] = []
    subdomains: list[VcmlSubDomain] = [*vcml.compartment_subdomains, *vcml.membrane_subdomains]
    for subdomain in subdomains:
        for pde in subdomain.pde_equations:
            roots += [pde.rate, pde.diffusion, pde.initial]
            if pde.boundaries is not None:
                roots += [getattr(pde.boundaries, face) for face in ("xm", "xp", "ym", "yp", "zm", "zp")]
        for ode in subdomain.ode_equations:
            roots += [ode.rate, ode.initial]
    for membrane in vcml.membrane_subdomains:
        for jc in membrane.jump_conditions:
            roots += [jc.in_flux, jc.out_flux]
    roots += [constant.exp for constant in vcml.constants]

    seen: set[str] = set()
    stack = [name for raw in roots if raw for name in _IDENT_RE.findall(raw)]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        body = bodies.get(name)
        if body:
            stack.extend(_IDENT_RE.findall(body))
    return frozenset(seen)


def _build_observables(vcml: VcmlMathDescription, res: FunctionResolution) -> list[Observable]:
    """Surface the non-constant (variable- / coordinate- / time-referencing) functions as
    observables, with bodies fully inlined (so they reference only variables, parameters,
    coordinates, and time)."""

    observables: list[Observable] = []
    for function in vcml.functions:
        body = res.inlined_function_bodies.get(function.name)
        if body is not None:
            observables.append(
                Observable(
                    name=function.name,
                    expression=translate_expression(body),
                    subdomain=function.domain,
                )
            )
    return observables


def _as_float(expression: str) -> float | None:
    try:
        return float(expression)
    except (TypeError, ValueError):
        return None


def _reject_stochastic(subdomain: VcmlSubDomain) -> None:
    """Loudly reject the stochastic / particle constructs (§2.6.3) — dropping them would
    silently change the model from what VCell specified."""

    # `particle_*` are on both subdomain kinds; `jump_processes` / `variable_initial_counts` are
    # CompartmentSubDomain-only, so read those defensively (coerced to `object`, not `Any`).
    constructs: list[tuple[object, str]] = [
        (subdomain.particle_jump_processes, "particle (Smoldyn) jump processes"),
        (subdomain.particle_properties, "particle (Smoldyn) properties"),
        (getattr(subdomain, "jump_processes", None), "stochastic jump processes"),
        (getattr(subdomain, "variable_initial_counts", None), "stochastic initial counts"),
    ]
    for value, what in constructs:
        if value:
            raise VcellImportError(
                f"subdomain {subdomain.name!r} uses {what}, which are out of scope for the formalism "
                f"(§2.6.3) and cannot be imported"
            )


# VCell box-face names → the formalism's named faces (the realization produces these, §3).
_FACE_NAME_MAP = {"Xm": "x_minus", "Xp": "x_plus", "Ym": "y_minus", "Yp": "y_plus", "Zm": "z_minus", "Zp": "z_plus"}
# Which box faces are real at each geometry dimension (VCell stores all six even for 2D).
_BOX_FACES_BY_DIM: dict[int, tuple[str, ...]] = {
    0: (),
    1: ("Xm", "Xp"),
    2: ("Xm", "Xp", "Ym", "Yp"),
    3: ("Xm", "Xp", "Ym", "Yp", "Zm", "Zp"),
}


def _translate_boundaries(
    pde: PdeEquation, subdomain: VcmlSubDomain, kind: str, expr: Callable[[str | None], str | None], dim: int | None
) -> list[BoundaryCondition]:
    """Per-face boundary conditions (§2.6.2). VCell's `boundary_types` are model-wide (per face, not
    per species), so the type comes from the subdomain and the value from this PDE; the formalism's
    BCs are per-variable, which is strictly more general. Mapping per real face:

    - ``Flux`` + value → Neumann; ``Flux`` + none → natural (no-flux) default, omitted.
    - ``Value`` + value → Dirichlet(value); ``Value`` + none → **Dirichlet(initial condition)** (VCell's
      default-Dirichlet rule).

    Faces outside the geometry dimension (`dim`) are skipped (VCell lists all six). Without `dim` we
    cannot tell which faces are real, so any non-default boundary is rejected. Surface (membrane)
    box-face BCs are not realized yet — a membrane PDE with explicit boundary values raises."""

    boundaries = pde.boundaries
    has_explicit = boundaries is not None and any(
        getattr(boundaries, face) is not None for face in ("xm", "xp", "ym", "yp", "zm", "zp")
    )
    face_type = {bt.boundary: bt.type for bt in subdomain.boundary_types}

    if dim is None:
        if has_explicit or any(t != "Flux" for t in face_type.values()):
            raise NotImplementedError(
                f"PDE {pde.name!r} on {subdomain.name!r} carries non-default boundary conditions (§2.6.2); "
                f"pass dim= (the geometry dimension) to import per-face boundary conditions"
            )
        return []
    if kind != "volume":
        if has_explicit:
            raise NotImplementedError(
                f"membrane PDE {pde.name!r} on {subdomain.name!r} has explicit box-face boundary values; "
                f"surface boundary-condition import is a follow-up increment"
            )
        return []

    bcs: list[BoundaryCondition] = []
    for face in _BOX_FACES_BY_DIM.get(dim, ()):
        boundary_type = face_type.get(face, "Flux")  # an unlisted face is VCell's default no-flux
        if boundary_type not in ("Flux", "Value"):
            raise NotImplementedError(
                f"subdomain {subdomain.name!r} has a {boundary_type!r} boundary on face {face!r} (§2.6.2); "
                f"only Flux (Neumann) and Value (Dirichlet) are imported — periodic is a follow-up increment"
            )
        value = getattr(boundaries, face.lower()) if boundaries is not None else None
        name = _FACE_NAME_MAP[face]
        if boundary_type == "Value":
            value_expr = expr(value) if value is not None else expr(pde.initial)
            if value_expr is not None:  # no boundary value and no initial condition — nothing to impose
                bcs.append(BCDirichlet(variable=pde.name, boundary=name, expression=value_expr))
        elif value is not None:  # Flux with an explicit value → Neumann; bare Flux is the natural default
            bcs.append(BCNeumann(variable=pde.name, boundary=name, expression=expr(value) or "0"))
    return bcs


def _translate_jump_conditions(
    membrane: MembraneSubDomain,
    bulk_species: set[str],
    species_compartments: dict[str, set[str]],
    res: FunctionResolution,
) -> list[BoundaryCondition]:
    """Map a membrane's VCell ``JumpCondition``s to formalism Neumann BCs at the membrane (the
    composable bulk-surface pattern, §1.6.5). A jump condition is two *independent* Neumann
    conditions for one bulk species: ``in_flux`` on the inside compartment, ``out_flux`` on the
    outside. Each flux may reference ``geom.x`` / ``sim.t``, membrane species (directly), and volume
    species on either side (wrapped in ``trace(·)`` — a bulk variable is only defined on the membrane
    through its trace). This is **not** ``interface_flux_balance`` (§1.6.2): the two sides are
    specified independently, not as one implicit equal-and-opposite expression.

    A species present in *both* compartments would need two BCs distinguished by side, which the
    per-variable ``BCNeumann(variable, boundary)`` cannot express yet — that case is rejected."""

    inside, outside = membrane.inside_compartment, membrane.outside_compartment

    def flux_expr(raw: str | None) -> str | None:
        if raw is None or raw.strip() in ("0.0", "0"):
            return None  # the natural no-flux default
        return _trace_wrap(translate_expression(res.inline(raw) or ""), bulk_species)

    bcs: list[BoundaryCondition] = []
    for jc in membrane.jump_conditions:
        inside_flux, outside_flux = flux_expr(jc.in_flux), flux_expr(jc.out_flux)
        if inside_flux is None and outside_flux is None:
            continue
        compartments = species_compartments.get(jc.name, set())
        if len(compartments) > 1:
            raise NotImplementedError(
                f"jump condition for {jc.name!r} on membrane {membrane.name!r}: the species lives in "
                f"both compartments, so its inside / outside membrane fluxes need per-side BCs — a "
                f"follow-up increment (our BC identifies a variable by name + boundary only)"
            )
        # The Neumann condition applies on the side where the species lives.
        if inside in compartments:
            flux = inside_flux
        elif outside in compartments:
            flux = outside_flux
        else:
            flux = inside_flux if inside_flux is not None else outside_flux
        if flux is not None:
            bcs.append(BCNeumann(variable=jc.name, boundary=membrane.name, expression=flux))
    return bcs


_IDENTIFIER = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(?![\w(])")


def _trace_wrap(expression: str, bulk_species: set[str]) -> str:
    """Wrap each bare *bulk*-species reference in ``trace(·)``. In a membrane-level expression a
    volume variable is only well-defined through its trace on the membrane (§1.6.5); membrane species
    and parameters are referenced directly and left untouched. A single ``re.sub`` pass does not
    re-scan its own substitutions, so an inserted ``trace(name)`` is never re-wrapped."""

    if not bulk_species:
        return expression
    return _IDENTIFIER.sub(lambda m: f"trace({m.group(1)})" if m.group(1) in bulk_species else m.group(1), expression)

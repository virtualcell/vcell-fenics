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
six faces are real. Membrane ``JumpCondition``s map to a single-sided ``interface_flux`` on the bulk
species when the membrane is between two modelled compartments (the §1.6.2 internal interface, solved
by the coupled backend), or a ``Neumann`` BC when the other side is an unmodelled reservoir (the
composable bulk-surface pattern §1.6.5). In **both** the jump-condition fluxes and the membrane PDE
**reactions**, cross-membrane volume-species references are wrapped in ``trace(·)`` — a bulk variable
is only defined on the membrane through its trace (§1.6.5/§1.8.2); membrane species stay direct.

**Loud rejection.** Constructs that would change the math if dropped are never dropped silently:
stochastic / particle dynamics (§2.6.3) raise :class:`VcellImportError`; a boundary-bearing PDE
without ``dim``, a non-Flux/Value face type (periodic), and a jump-condition species living in both
compartments raise :class:`NotImplementedError` pointing at the follow-up.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from vcell_fenics.formalism.schema import (
    REGION_SPACE,
    BCDirichlet,
    BCInterfaceFlux,
    BCNeumann,
    BoundaryCondition,
    Equation,
    MathDescription,
    MotionNone,
    MotionPrescribedVelocity,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.pyvcell_bridge.expression import translate_expression
from vcell_fenics.pyvcell_bridge.inlining import _IDENT_RE, FunctionResolution, referenced_names, resolve_functions

if TYPE_CHECKING:
    from pyvcell.vcml.models_app import FrontVelocity
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


def import_model(
    vcml: VcmlMathDescription,
    *,
    geometry: str | None = None,
    dim: int | None = None,
    front_velocity: FrontVelocity | None = None,
) -> ImportResult:
    """Translate a pyvcell ``MathDescription`` into a formalism ``MathDescription`` plus its
    observables. ``geometry`` defaults to the VCell math description's ``name`` (it is
    cross-checked against a real Geometry at solve time, not here).

    ``dim`` is the spatial dimension of the geometry (0–3). It is required to import per-face
    boundary conditions (§2.6.2): VCell's `boundary_types` are model-wide and list all six box
    faces (`Xm…Zp`) even for a 2D model, so the dimension says which faces are real (`x_minus` /
    `x_plus` for 2D, plus `z_*` for 3D). Without ``dim`` a PDE that carries any non-default
    boundary is rejected rather than mis-imported.

    ``front_velocity`` is the application's moving-boundary front kinematics (``app.front_velocity``
    — *not* part of the lowered ``MathDescription``, so the caller threads it in). It moves a
    geometry *surface* class; we attach it as a prescribed-velocity motion (§1.10) on the volume
    subdomain that surface encloses (its membrane's ``inside_compartment``). That motion is only the
    **frame**: VCell's moving-boundary solver is Eulerian — a fixed grid, a moving front, and each
    species' own lab-frame velocity (its PDE ``<Velocity>``, zero when absent). So the moving
    compartment's species take that velocity as the lab-frame ``advection`` slot, transported relative
    to the moving mesh with zero total flux at the front (the Rankine–Hugoniot condition); a species
    whose velocity equals the front's rides with the cell, and one with none is swept by the front.
    See ``cross_validation/mb_translation.py`` and ``mb_swept.py``."""

    variable_names = _collect_variable_names(vcml)
    resolution = resolve_functions(list(vcml.functions), variable_names)
    # Bulk (volume) species → the compartment(s) each lives in: the species set drives trace-wrapping
    # of cross-membrane flux references (§1.6.5); the compartment map detects a species on both sides
    # of a membrane (which our per-variable BC cannot yet disambiguate).
    species_compartments: dict[str, set[str]] = {}
    for compartment in vcml.compartment_subdomains:
        # A volume-region variable (a well-mixed species) is bulk too: its membrane jump conditions feed
        # its region balance from its own side, and a membrane expression sees it through its trace.
        for name in [pde.name for pde in compartment.pde_equations] + [
            eq.name for eq in compartment.volume_region_equations
        ]:
            species_compartments.setdefault(name, set()).add(compartment.name)
    bulk_species = set(species_compartments)

    subdomains: list[Subdomain] = []
    variables: list[Variable] = []
    equations: list[Equation] = []
    boundary_conditions: list[BoundaryCondition] = []

    motion_by_compartment = _front_motion(vcml, front_velocity, resolution, dim)

    for compartment in vcml.compartment_subdomains:
        _reject_stochastic(compartment)
        motion = motion_by_compartment.get(compartment.name, MotionNone())
        subdomains.append(Subdomain(name=compartment.name, kind="volume", motion=motion))
        first = len(equations)
        _translate_subdomain_equations(
            compartment, "volume", variables, equations, boundary_conditions, resolution, dim, bulk_species
        )
        if compartment.name in motion_by_compartment:
            equations[first:] = [_lab_frame(eq, dim) for eq in equations[first:]]

    for membrane in vcml.membrane_subdomains:
        _reject_stochastic(membrane)
        subdomains.append(Subdomain(name=membrane.name, kind="surface", motion=MotionNone()))
        _translate_subdomain_equations(
            membrane, "surface", variables, equations, boundary_conditions, resolution, dim, bulk_species
        )
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


def _lab_frame(equation: Equation, dim: int | None) -> Equation:
    """A species of a moving compartment: its VCell velocity is lab-frame (Eulerian), so it becomes the
    ``advection`` slot — an explicit zero when VCell gives none (the species is swept by the front, not
    carried with the cell, which is what the formalism's default would do)."""

    if not isinstance(equation, TemplateEquation) or equation.template != "bulk_radv_diff":
        return equation
    terms = dict(equation.terms)
    velocity = terms.pop("relative_advection", None)
    terms["advection"] = velocity if velocity is not None else "[" + ", ".join(["0.0"] * (dim or 2)) + "]"
    return dataclasses.replace(equation, terms=terms)


def _is_zero_neumann(bc: BoundaryCondition, numeric: dict[str, float]) -> bool:
    """Whether ``bc`` is a Neumann flux that resolves to zero — a literal ``0`` or a single numeric
    constant equal to 0 (VCell's per-face no-flux default, e.g. ``u_boundaryXm = 0``)."""

    if not isinstance(bc, BCNeumann):
        return False
    expression = bc.expression.strip()
    literal = _as_float(expression)
    return (literal == 0.0) if literal is not None else (numeric.get(expression) == 0.0)


def import_math_description(
    vcml: VcmlMathDescription,
    *,
    geometry: str | None = None,
    dim: int | None = None,
    front_velocity: FrontVelocity | None = None,
) -> MathDescription:
    """Convenience wrapper returning just the formalism ``MathDescription`` (the observables
    sidecar is dropped). See :func:`import_model`."""

    return import_model(vcml, geometry=geometry, dim=dim, front_velocity=front_velocity).math


def _collect_variable_names(vcml: VcmlMathDescription) -> set[str]:
    """The state-variable names — the equation-governed names (plus any declared
    ``MathVariable``s). Used to decide which functions reference variables."""

    names: set[str] = {v.name for v in vcml.variables}
    for compartment in vcml.compartment_subdomains:
        names.update(p.name for p in compartment.pde_equations)
        names.update(o.name for o in compartment.ode_equations)
        names.update(r.name for r in compartment.volume_region_equations)
    for membrane in vcml.membrane_subdomains:
        names.update(p.name for p in membrane.pde_equations)
        names.update(o.name for o in membrane.ode_equations)
        names.update(r.name for r in membrane.membrane_region_equations)
    return names


def _translate_subdomain_equations(
    subdomain: VcmlSubDomain,
    kind: str,
    variables: list[Variable],
    equations: list[Equation],
    boundary_conditions: list[BoundaryCondition],
    res: FunctionResolution,
    dim: int | None,
    bulk_species: set[str],
) -> None:
    """Append the variables + template equations for one subdomain's PDEs and ODEs (inlining
    variable-referencing functions into each expression), plus the per-face boundary conditions.

    On a **surface** (membrane) subdomain a reaction may reference a volume species adjacent to the
    membrane; a bulk variable is only defined on the membrane through its trace, so its references are
    wrapped in ``trace(·)`` (§1.6.5/§1.8.2) — the same wrapping the jump-condition fluxes get. Volume
    equations reference their own species directly and are left untouched."""

    pde_template = "bulk_radv_diff" if kind == "volume" else "surface_pde_with_dilution"

    def expr(raw: str | None) -> str | None:
        if raw is None:
            return None
        translated = translate_expression(res.inline(raw) or "")
        return _trace_wrap(translated, bulk_species) if kind == "surface" else translated

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

    # Region variables (VCell's VolumeRegion/MembraneRegion equations: a well-mixed species, the membrane
    # potential) — one value per connected region, the `region_ode` template (§1.4.2 T5). A zero rate is
    # the template's default, so it is left out.
    regions: list[tuple[str, str | None, str | None, str | None]] = (
        [
            (r.name, r.uniform_rate, r.volume_rate, r.initial)
            for r in cast("CompartmentSubDomain", subdomain).volume_region_equations
        ]
        if kind == "volume"
        else [
            (r.name, r.uniform_rate, r.membrane_rate, r.initial)
            for r in cast("MembraneSubDomain", subdomain).membrane_region_equations
        ]
    )
    for name, uniform_rate, region_rate, initial in regions:
        region_terms: dict[str, str] = {}
        for slot, raw in (("uniform_rate", uniform_rate), ("region_rate", region_rate)):
            if raw is not None and _as_float(raw) != 0.0:
                region_terms[slot] = expr(raw)  # type: ignore[assignment]
        variables.append(Variable(name=name, subdomain=subdomain.name, type="scalar", space=REGION_SPACE))
        equations.append(
            TemplateEquation(
                template="region_ode",
                variable=name,
                subdomain=subdomain.name,
                temporality="time_dependent",
                terms=region_terms,
                initial_condition=expr(initial),
            )
        )


def _front_motion(
    vcml: VcmlMathDescription,
    front_velocity: FrontVelocity | None,
    res: FunctionResolution,
    dim: int | None,
) -> dict[str, MotionPrescribedVelocity]:
    """Resolve the application's moving-boundary ``front_velocity`` to a prescribed-velocity motion
    on the volume subdomain it moves. Returns ``{compartment_name: MotionPrescribedVelocity}`` (empty
    when there is no front).

    The front moves a geometry *surface* class (``FrontVelocity.surface_name``); the moving volume is
    that surface's ``inside_compartment`` (the cell interior — the bulk it encloses rides along, the
    ``v = v_b`` carry). With no ``surface_name`` we require the model to have exactly one membrane so
    the target is unambiguous."""

    if front_velocity is None:
        return {}

    membranes = list(vcml.membrane_subdomains)
    if not membranes:
        raise VcellImportError("a moving-boundary front_velocity was given but the model has no membrane subdomain")

    name = front_velocity.surface_name
    if name is not None:
        moving = next((m for m in membranes if m.name == name), None)
        if moving is None:
            raise VcellImportError(f"front_velocity.surface_name {name!r} matches no membrane subdomain")
    elif len(membranes) == 1:
        moving = membranes[0]
    else:
        raise VcellImportError(
            "front_velocity has no surface_name and the model has multiple membranes; "
            "set surface_name to say which surface moves"
        )

    interior = moving.inside_compartment
    if interior is None:
        raise VcellImportError(
            f"membrane {moving.name!r} has no inside_compartment, so the moving (interior) volume is undefined"
        )

    components = [front_velocity.velocity_x, front_velocity.velocity_y]
    if dim == 3:
        components.append(front_velocity.velocity_z)
    rendered = [translate_expression(_inline_every_function(str(c), vcml)) for c in components]
    return {interior: MotionPrescribedVelocity(velocity="[" + ", ".join(rendered) + "]")}


def _body_or_name(match: re.Match[str], names: frozenset[str], bodies: dict[str, str]) -> str:
    name = match.group(1)
    return f"({bodies[name]})" if name in names else name


def _inline_every_function(expr: str, vcml: VcmlMathDescription) -> str:
    """``expr`` with *every* MathFunction name replaced by its body, recursively — constant ones too.

    Equations keep constant functions as parameters (emitted when an equation reaches them), but a
    front velocity is not an equation, so a constant component (``velocityY = 0``, reached through
    ``sobj_…_velY`` → ``sproc_….velocityY``) would otherwise name a parameter the model never defines."""

    bodies = {f.name: (f.exp or "0") for f in vcml.functions}
    for _ in range(len(bodies) + 1):  # each pass removes one level; more passes than functions = a cycle
        names = referenced_names(expr) & bodies.keys()
        if not names:
            return expr
        expr = _IDENT_RE.sub(functools.partial(_body_or_name, names=frozenset(names), bodies=bodies), expr)
    raise VcellImportError(f"cyclic MathFunction references in the front velocity {expr!r}")


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
    :func:`_translate_parameters`). Roots are every PDE/ODE/region-equation rate, diffusion, and initial
    expression,
    each PDE velocity component (VCell routes a species velocity through functions, e.g.
    ``vobj_Cyt1_velX`` → ``vproc_1.velocityX``),
    each per-face boundary value, each membrane jump flux, and each constant expression; the closure
    then follows function bodies. Identifier matching reuses the inliner's dotted-name-aware,
    call-excluding regex, so a built-in call like ``vcRegionVolume(...)`` is not itself a name."""

    bodies = {f.name: (f.exp or "") for f in vcml.functions}
    roots: list[str | None] = []
    subdomains: list[VcmlSubDomain] = [*vcml.compartment_subdomains, *vcml.membrane_subdomains]
    for subdomain in subdomains:
        for pde in subdomain.pde_equations:
            roots += [pde.rate, pde.diffusion, pde.initial]
            if pde.velocity is not None:
                roots += [pde.velocity.x, pde.velocity.y, pde.velocity.z]
            if pde.boundaries is not None:
                roots += [getattr(pde.boundaries, face) for face in ("xm", "xp", "ym", "yp", "zm", "zp")]
        for ode in subdomain.ode_equations:
            roots += [ode.rate, ode.initial]
    for compartment in vcml.compartment_subdomains:  # region equations (T5): their rates and initial values
        for volume_region in compartment.volume_region_equations:
            roots += [volume_region.uniform_rate, volume_region.volume_rate, volume_region.initial]
    for membrane in vcml.membrane_subdomains:
        for membrane_region in membrane.membrane_region_equations:
            roots += [membrane_region.uniform_rate, membrane_region.membrane_rate, membrane_region.initial]
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
    cannot tell which faces are real, so any non-default boundary is rejected. A surface (membrane) PDE's
    box-face BCs are dropped: a membrane is realized as the *closed* interface curve between two
    subvolumes (`realize_interface_coupled`), which never touches the box, so VCell's per-face values —
    boilerplate it emits for every species (`R_boundaryXm` → `0.0` for a cell membrane) — have no face to
    apply to. (A membrane that genuinely spans to the box edge is not a realizable geometry here.)"""

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
        return []  # a closed membrane has no box faces — VCell's per-face values have nothing to apply to

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
    """Map a membrane's VCell ``JumpCondition``s to formalism interface BCs. A jump condition is a
    **single-sided** Neumann flux for a bulk species; the flux may reference ``geom.x`` / ``sim.t``,
    membrane species (directly), and volume species on either side (wrapped in ``trace(·)`` — a bulk
    variable is only defined on the membrane through its trace).

    The key fact about VCell math: a modern volume variable is **domain-restricted** (it lives in one
    subvolume — ``calcium_inside`` vs ``calcium_outside``), so its jump condition has exactly one
    well-posed side — ``in_flux`` if the species lives on the membrane's inside, ``out_flux`` if
    outside — and the other side is zero. We therefore take each species' own-side flux verbatim and
    never pair it with another variable: a species crossing a membrane is two independent jump
    conditions (one per domain-restricted species), each its own independent ``BCInterfaceFlux`` entry.
    The two sides need not be equal-and-opposite — VCell's ``in_flux`` / ``out_flux`` are independent
    Neumann fluxes (mass conservation is whatever the modeller writes), e.g. a permeability pair
    ``P·(s_outer − s_inner)`` into the inside and its negation into the outside (§1.6.2).

    Routing turns on whether the membrane is *internal* (both compartments carry bulk species, both
    modelled as PDEs) or *external* (the other side is an unmodelled reservoir):

    - **Internal interface** → ``BCInterfaceFlux`` on the species' own side, solved by the coupled
      assembler (which binds both compartments' traces so the flux can reference either; §1.6.2).
    - **External boundary** → ``BCNeumann`` on the modelled side (§1.6.5).

    The legacy case — a single volume variable with *no* domain, defined on *both* sides of the
    membrane, where VCell's ``JumpCondition`` genuinely carried two meaningful per-side fluxes — is
    rare and still raises (its per-side fluxes need a side-tagged BC; our BC keys on name + boundary)."""

    inside, outside = membrane.inside_compartment, membrane.outside_compartment
    # The membrane is an *internal interface* when both its compartments carry bulk species (both are
    # modelled as PDEs): then each jump-condition side is a single-sided ``BCInterfaceFlux`` on an
    # internal boundary, solved by the coupled assembler (which binds the adjacent compartment's
    # trace). When
    # only one side is modelled (the other is an unmodelled reservoir) the membrane is an *external*
    # boundary of that compartment → ``BCNeumann`` (the single-compartment path; §1.6.5).
    modeled = {compartment for comps in species_compartments.values() for compartment in comps}
    internal = inside in modeled and outside in modeled

    def flux_expr(raw: str | None) -> str | None:
        if raw is None or raw.strip() in ("0.0", "0"):
            return None  # the natural no-flux default
        return _trace_wrap(translate_expression(res.inline(raw) or ""), bulk_species)

    bcs: list[BoundaryCondition] = []
    for jc in membrane.jump_conditions:
        compartments = species_compartments.get(jc.name, set())
        if len(compartments) > 1:
            raise NotImplementedError(
                f"jump condition for {jc.name!r} on membrane {membrane.name!r}: this is a legacy "
                f"domain-less volume variable defined on both sides of the membrane, so its inside / "
                f"outside fluxes are both well-posed and need per-side BCs — a follow-up increment "
                f"(our BC identifies a variable by name + boundary only). Modern VCell math assigns "
                f"each volume variable a single domain, giving exactly one well-posed flux side."
            )
        # The flux into the species' own side: in_flux if it lives inside, out_flux if outside.
        if inside in compartments:
            flux = flux_expr(jc.in_flux)
        elif outside in compartments:
            flux = flux_expr(jc.out_flux)
        else:
            flux = flux_expr(jc.in_flux) or flux_expr(jc.out_flux)
        if flux is None:
            continue
        if internal:
            # A single-sided flux at the internal membrane — VCell's in_flux / out_flux are independent
            # (not an enforced equal-and-opposite balance); the other side is its own BCInterfaceFlux.
            bcs.append(BCInterfaceFlux(variable=jc.name, boundary=membrane.name, expression=flux))
        else:
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

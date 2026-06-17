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

The translator is **duck-typed** over the pydantic object's attributes, so it needs no
import of ``pyvcell`` itself — the caller supplies the
``pyvcell.vcml.models_math.MathDescription`` (a clean ``from pyvcell.vcml.models_math
import ...``, which is lazy and pulls none of pyvcell's heavy stack).

**Loud rejection.** Constructs that would change the math if dropped are never dropped
silently: stochastic / particle dynamics (§2.6.3) raise :class:`VcellImportError`, and
the §2.6.2 constructs not yet implemented here — per-face boundary conditions and
membrane jump conditions — raise :class:`NotImplementedError` pointing at the follow-up.
A model whose PDEs rely on VCell's *default* no-flux faces (all ``Flux`` boundary types,
no boundary expressions) imports cleanly: that default equals the formalism's natural
zero-Neumann boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vcell_fenics.formalism.schema import (
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
from vcell_fenics.pyvcell_bridge.inlining import FunctionResolution, resolve_functions


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


def import_model(vcml: Any, *, geometry: str | None = None) -> ImportResult:
    """Translate a pyvcell ``MathDescription`` into a formalism ``MathDescription`` plus its
    observables. ``geometry`` defaults to the VCell math description's ``name`` (it is
    cross-checked against a real Geometry at solve time, not here)."""

    variable_names = _collect_variable_names(vcml)
    resolution = resolve_functions(list(getattr(vcml, "functions", [])), variable_names)

    subdomains: list[Subdomain] = []
    variables: list[Variable] = []
    equations: list[Equation] = []

    for compartment in vcml.compartment_subdomains:
        _reject_stochastic(compartment)
        subdomains.append(Subdomain(name=compartment.name, kind="volume", motion=MotionNone()))
        _translate_subdomain_equations(compartment, "volume", variables, equations, resolution)

    for membrane in vcml.membrane_subdomains:
        _reject_stochastic(membrane)
        if getattr(membrane, "jump_conditions", None):
            raise NotImplementedError(
                f"membrane {membrane.name!r} has jump conditions (interface flux balance, §2.6.2); "
                f"jump-condition import is a follow-up increment"
            )
        subdomains.append(Subdomain(name=membrane.name, kind="surface", motion=MotionNone()))
        _translate_subdomain_equations(membrane, "surface", variables, equations, resolution)

    parameters = _translate_parameters(vcml, resolution)
    observables = _build_observables(vcml, resolution)

    math = MathDescription(
        geometry=geometry if geometry is not None else vcml.name,
        subdomains=subdomains,
        variables=variables,
        equations=equations,
        parameters=parameters,
        boundary_conditions=list[BoundaryCondition](),
    )
    return ImportResult(math=math, observables=tuple(observables))


def import_math_description(vcml: Any, *, geometry: str | None = None) -> MathDescription:
    """Convenience wrapper returning just the formalism ``MathDescription`` (the observables
    sidecar is dropped). See :func:`import_model`."""

    return import_model(vcml, geometry=geometry).math


def _collect_variable_names(vcml: Any) -> set[str]:
    """The state-variable names — the equation-governed names (plus any declared
    ``MathVariable``s). Used to decide which functions reference variables."""

    names: set[str] = {v.name for v in getattr(vcml, "variables", [])}
    for sub in [*vcml.compartment_subdomains, *vcml.membrane_subdomains]:
        names.update(p.name for p in sub.pde_equations)
        names.update(o.name for o in sub.ode_equations)
    return names


def _translate_subdomain_equations(
    subdomain: Any, kind: str, variables: list[Variable], equations: list[Equation], res: FunctionResolution
) -> None:
    """Append the variables + template equations for one subdomain's PDEs and ODEs,
    inlining variable-referencing functions into each expression."""

    pde_template = "bulk_radv_diff" if kind == "volume" else "surface_pde_with_dilution"

    def expr(raw: str | None) -> str | None:
        if raw is None:
            return None
        return translate_expression(res.inline(raw) or "")

    for pde in subdomain.pde_equations:
        _reject_boundaries(pde, subdomain)
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


def _velocity_vector(velocity: Any, res: FunctionResolution) -> str | None:
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


def _translate_parameters(vcml: Any, res: FunctionResolution) -> list[Parameter]:
    """VCell ``Constant``s → parameters; **pure** ``MathFunction``s (no variable reference)
    → ``ParameterExpression``. Variable-referencing functions are not parameters — they are
    inlined into equations and surfaced as observables instead."""

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
        if function.name in res.pure_function_names:
            parameters.append(
                ParameterExpression(
                    name=function.name,
                    expression=translate_expression(function.exp or ""),
                    subdomain=getattr(function, "domain", None),
                )
            )
    return parameters


def _build_observables(vcml: Any, res: FunctionResolution) -> list[Observable]:
    """Surface the variable-referencing functions as observables, with bodies fully inlined
    (so they reference only variables, parameters, coordinates, and time)."""

    observables: list[Observable] = []
    for function in vcml.functions:
        body = res.var_function_bodies.get(function.name)
        if body is not None:
            observables.append(
                Observable(
                    name=function.name,
                    expression=translate_expression(body),
                    subdomain=getattr(function, "domain", None),
                )
            )
    return observables


def _as_float(expression: str) -> float | None:
    try:
        return float(expression)
    except (TypeError, ValueError):
        return None


def _reject_stochastic(subdomain: Any) -> None:
    """Loudly reject the stochastic / particle constructs (§2.6.3) — dropping them would
    silently change the model from what VCell specified."""

    for attr, what in (
        ("jump_processes", "stochastic jump processes"),
        ("particle_jump_processes", "particle (Smoldyn) jump processes"),
        ("particle_properties", "particle (Smoldyn) properties"),
        ("variable_initial_counts", "stochastic initial counts"),
    ):
        if getattr(subdomain, attr, None):
            raise VcellImportError(
                f"subdomain {subdomain.name!r} uses {what}, which are out of scope for the formalism "
                f"(§2.6.3) and cannot be imported"
            )


def _reject_boundaries(pde: Any, subdomain: Any) -> None:
    """Per-face boundary conditions (§2.6.2) are a follow-up increment. A PDE that
    relies on VCell's default no-flux faces (no boundary expressions, only ``Flux``
    boundary types) imports cleanly — that default is the formalism's natural
    zero-Neumann BC. Anything else raises rather than silently dropping a BC."""

    boundaries = getattr(pde, "boundaries", None)
    if boundaries is not None and any(
        getattr(boundaries, face) is not None for face in ("xm", "xp", "ym", "yp", "zm", "zp")
    ):
        raise NotImplementedError(
            f"PDE for {pde.name!r} on {subdomain.name!r} has explicit boundary expressions (§2.6.2); "
            f"per-face boundary-condition import is a follow-up increment"
        )
    for bt in getattr(subdomain, "boundary_types", []):
        if bt.type.lower() != "flux":
            raise NotImplementedError(
                f"subdomain {subdomain.name!r} has a {bt.type!r} boundary on face {bt.boundary!r} (§2.6.2); "
                f"only the default no-flux (natural) boundary is imported in this increment"
            )

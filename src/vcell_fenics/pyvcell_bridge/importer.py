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
- numeric ``Constant`` → ``ParameterConstant``; symbolic ``Constant`` and
  ``MathFunction`` → ``ParameterExpression``.

Expression strings are run through :func:`translate_expression` (coordinate/time
namespacing, ``^``→``**``).

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


class VcellImportError(Exception):
    """A VCell ``MathDescription`` uses a construct that cannot be imported into the
    formalism (an out-of-scope §2.6.3 construct, or a malformed equation)."""


def import_math_description(vcml: Any, *, geometry: str | None = None) -> MathDescription:
    """Translate a pyvcell ``MathDescription`` into a formalism ``MathDescription``.

    ``vcml`` is a ``pyvcell.vcml.models.MathDescription`` (duck-typed). ``geometry`` is
    the name of the formalism Geometry this math will be paired with at solve time
    (cross-checked then, not now); it defaults to the VCell math description's ``name``.
    The result is a structurally complete ``MathDescription`` ready for
    ``validate_or_raise`` / the backend.
    """

    subdomains: list[Subdomain] = []
    variables: list[Variable] = []
    equations: list[Equation] = []
    boundary_conditions: list[BoundaryCondition] = []

    for compartment in vcml.compartment_subdomains:
        _reject_stochastic(compartment)
        subdomains.append(Subdomain(name=compartment.name, kind="volume", motion=MotionNone()))
        _translate_subdomain_equations(compartment, kind="volume", variables=variables, equations=equations)

    for membrane in vcml.membrane_subdomains:
        _reject_stochastic(membrane)
        if getattr(membrane, "jump_conditions", None):
            raise NotImplementedError(
                f"membrane {membrane.name!r} has jump conditions (interface flux balance, §2.6.2); "
                f"jump-condition import is a follow-up increment"
            )
        subdomains.append(Subdomain(name=membrane.name, kind="surface", motion=MotionNone()))
        _translate_subdomain_equations(membrane, kind="surface", variables=variables, equations=equations)

    parameters = _translate_parameters(vcml)

    return MathDescription(
        geometry=geometry if geometry is not None else vcml.name,
        subdomains=subdomains,
        variables=variables,
        equations=equations,
        parameters=parameters,
        boundary_conditions=boundary_conditions,
    )


def _translate_subdomain_equations(
    subdomain: Any, *, kind: str, variables: list[Variable], equations: list[Equation]
) -> None:
    """Append the variables + template equations for one subdomain's PDEs and ODEs."""

    pde_template = "bulk_radv_diff" if kind == "volume" else "surface_pde_with_dilution"

    for pde in subdomain.pde_equations:
        _reject_boundaries(pde, subdomain)
        terms: dict[str, str] = {}
        if pde.diffusion is not None:
            terms["diffusion"] = translate_expression(pde.diffusion)
        if pde.rate is not None:
            terms["source"] = translate_expression(pde.rate)
        advection = _velocity_vector(pde.velocity)
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
                initial_condition=(translate_expression(pde.initial) if pde.initial is not None else None),
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
                terms={"rate": translate_expression(ode.rate)},
                initial_condition=(translate_expression(ode.initial) if ode.initial is not None else None),
            )
        )


def _velocity_vector(velocity: Any) -> str | None:
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
    rendered = [translate_expression(c) if c is not None else "0" for c in components]
    return "[" + ", ".join(rendered) + "]"


def _translate_parameters(vcml: Any) -> list[Parameter]:
    """Translate VCell ``Constant``s and ``MathFunction``s into formalism parameters.
    A numeric constant becomes a `ParameterConstant`; a symbolic constant or a named
    function becomes a `ParameterExpression`."""

    parameters: list[Parameter] = []
    for constant in vcml.constants:
        numeric = _as_float(constant.exp)
        if numeric is not None:
            parameters.append(ParameterConstant(name=constant.name, value=numeric))
        else:
            parameters.append(ParameterExpression(name=constant.name, expression=translate_expression(constant.exp)))
    for function in vcml.functions:
        parameters.append(
            ParameterExpression(
                name=function.name,
                expression=translate_expression(function.exp),
                subdomain=getattr(function, "domain", None),
            )
        )
    return parameters


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

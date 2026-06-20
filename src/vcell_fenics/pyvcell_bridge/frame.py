"""Normalize an imported VCell model to its geometry's coordinate frame.

VCell tolerates an out-of-dimension coordinate in a lower-dim model's expressions — most often a 3D
sphere from ``add_sphere`` in a *2D* geometry, but it can also appear in a math expression (an initial
condition, rate, or boundary value). VCell's convention is a unit slice: the geometry is one cell
thick in the missing axes, so that coordinate is the constant ``origin[axis]`` (usually 0).

The two importers are deliberately decoupled (`import_geometry` / `import_math_description`) and a
formalism ``MathDescription`` references its geometry only by *name* — it does not itself know it is
2D. So neither importer can, alone, bind ``z`` to ``origin.z``. This post-step takes **both**
descriptions once both are imported and binds every out-of-dimension ``geom.x[axis]`` (``axis >= dim``)
to ``origin[axis]``, using the geometry as the single source of truth for the frame (ADR 007). After
it, every expression references only the in-plane coordinates, so ``realize`` and ``assemble`` never
have to tolerate ``z`` — keeping that VCell quirk contained at the import boundary.

Run it after importing and before realizing / assembling:

    geometry = import_geometry(vcml_geometry)
    math = import_math_description(vcml_math, geometry=geometry.name, dim=geometry.dim)
    geometry, math = normalize_to_geometry_frame(geometry, math)
"""

from __future__ import annotations

import re
from dataclasses import replace

from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.schema import (
    BCRobin,
    BoundaryCondition,
    Equation,
    MathDescription,
    Parameter,
    ParameterExpression,
    TemplateEquation,
    WeakFormEquation,
)

# A coordinate access in the (already translated) formalism syntax: `geom.x[<axis>]`.
_GEOM_X = re.compile(r"geom\.x\[(\d+)\]")


def normalize_to_geometry_frame(
    geometry: GeometryDescription, math: MathDescription
) -> tuple[GeometryDescription, MathDescription]:
    """Bind every out-of-dimension coordinate (``geom.x[axis]`` for ``axis >= geometry.dim``) to the
    constant ``geometry.origin[axis]``, in both the geometry's subvolume expressions and the math's
    expressions. Returns normalized copies (the inputs are frozen and unchanged)."""

    dim, origin = geometry.dim, geometry.origin
    geo = replace(
        geometry,
        subvolumes=tuple(replace(sv, expression=_bind(sv.expression, dim, origin)) for sv in geometry.subvolumes),
    )
    md = replace(
        math,
        parameters=[_bind_parameter(p, dim, origin) for p in math.parameters],
        equations=[_bind_equation(e, dim, origin) for e in math.equations],
        boundary_conditions=[_bind_bc(bc, dim, origin) for bc in math.boundary_conditions],
    )
    return geo, md


def _bind(expression: str | None, dim: int, origin: tuple[float, float, float]) -> str | None:
    """Substitute each ``geom.x[axis]`` with ``axis >= dim`` by the literal ``origin[axis]``."""

    if expression is None:
        return None

    def substitute(match: re.Match[str]) -> str:
        axis = int(match.group(1))
        return f"({origin[axis]})" if axis >= dim else match.group(0)

    return _GEOM_X.sub(substitute, expression)


def _bind_parameter(parameter: Parameter, dim: int, origin: tuple[float, float, float]) -> Parameter:
    # Only a ParameterExpression carries a coordinate expression (a Constant is a number, a RegionMap a
    # subdomain reference).
    if isinstance(parameter, ParameterExpression):
        return replace(parameter, expression=_bind(parameter.expression, dim, origin) or "")
    return parameter


def _bind_equation(equation: Equation, dim: int, origin: tuple[float, float, float]) -> Equation:
    if isinstance(equation, TemplateEquation):
        terms = {slot: _bind(value, dim, origin) or "" for slot, value in equation.terms.items()}
        return replace(equation, terms=terms, initial_condition=_bind(equation.initial_condition, dim, origin))
    if isinstance(equation, WeakFormEquation):
        return replace(
            equation,
            form=_bind(equation.form, dim, origin) or "",
            initial_condition=_bind(equation.initial_condition, dim, origin),
        )
    return equation


def _bind_bc(bc: BoundaryCondition, dim: int, origin: tuple[float, float, float]) -> BoundaryCondition:
    # Every BoundaryCondition kind has an `expression`; Robin additionally has `alpha` / `beta`.
    if isinstance(bc, BCRobin):
        return replace(
            bc,
            alpha=_bind(bc.alpha, dim, origin) or "",
            beta=_bind(bc.beta, dim, origin) or "",
            expression=_bind(bc.expression, dim, origin) or "",
        )
    return replace(bc, expression=_bind(bc.expression, dim, origin) or "")

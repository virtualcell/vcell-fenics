"""Approach-agnostic primitives shared across the membrane representations.

Unlike `backend/` (DOLFINx-specific) and `formalism/` (the declarative model
layer), `core/` holds the pieces every approach needs in the same form — starting
with the conservative surface-density remap (`docs/modeling/conservative-surface-remap.md`),
which Approach A needs on remesh and Approaches B/D need on their own surface
representations.

The remap kernel (`surface_remap.py`) is deliberately pure NumPy: conservation
must be provably exact, so the arithmetic is isolated from the DOLFINx
mesh/`Function` machinery and can be tested without it. The DOLFINx bridge
(`surface_remap_mesh.py`) layers the loop-ordering and `Function`↔array transfer
on top of that kernel.
"""

from vcell_fenics.core.surface_remap import (
    arclength_parameterization,
    project_points_to_polyline_arclength,
    supermesh_remap_1d,
)
from vcell_fenics.core.surface_remap_mesh import ordered_membrane_loop, remap_surface_function

__all__ = [
    "arclength_parameterization",
    "ordered_membrane_loop",
    "project_points_to_polyline_arclength",
    "remap_surface_function",
    "supermesh_remap_1d",
]

"""Approach-agnostic primitives shared across the membrane representations.

Unlike `backend/` (DOLFINx-specific) and `formalism/` (the declarative model
layer), `core/` holds the pieces every approach needs in the same form — starting
with the conservative surface-density remap (`docs/modeling/conservative-surface-remap.md`),
which Approach A needs on remesh and Approaches B/D need on their own surface
representations.

The remap kernel here is deliberately pure NumPy: conservation must be provably
exact, so the arithmetic is isolated from the DOLFINx mesh/`Function` machinery.
The geometry→arc-length and `Function`↔array bridges are later increments built
on top of this kernel.
"""

from vcell_fenics.core.surface_remap import arclength_parameterization, supermesh_remap_1d

__all__ = [
    "arclength_parameterization",
    "supermesh_remap_1d",
]

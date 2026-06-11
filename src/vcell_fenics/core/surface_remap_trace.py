"""Approach-A trace correction: conserve surface mass of a bulk boundary trace.

In Approach A (ALE explicit membrane) the surface density ρ is not an independent
field — it is the *trace* of a bulk function `u` on the boundary facets ∂Ω. On a
remesh, the bulk field is conservatively interpolated, which preserves the *volume*
integral ∫_Ω u dx but gives no guarantee on the *surface* integral ∫_Γ ρ ds — a
different integral over a different-dimensional measure. So the membrane species'
mass drifts unless it is corrected separately (`docs/modeling/conservative-surface-remap.md`,
"The Approach-A-specific wrinkle").

The correction reuses the surface remap as a post-pass on the boundary DOFs:

1. **Gather** the old boundary trace into a surface `Function` on Γ_old (a codim-1
   submesh of the old bulk boundary).
2. **Surface-remap** it conservatively onto Γ_new (`remap_surface_function`).
3. **Scatter** the result back into the new bulk function's boundary DOFs,
   overwriting whatever the bulk interpolation left there. Interior DOFs untouched.

The bridge between a bulk space's boundary-trace DOFs and a surface space on that
boundary is a coordinate-matched bijection — for P1 the bulk boundary vertices are
exactly the submesh vertices — built with the same cKDTree pattern the backend's
mesh-motion uses. Surface mass measured on the bulk boundary (`∫ u ds`) equals the
submesh integral (`∫ ρ dx`) to round-off, so the remap's submesh conservation
transfers exactly to the trace.

Scope (v1): serial, P1, a single closed membrane in 2D — the same envelope as
`surface_remap_mesh`. This operates on a standalone bulk `Function`; wiring it into
a full ALE remesh pipeline (which does not exist yet) is downstream work.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from vcell_fenics.core.surface_remap_mesh import remap_surface_function

Ints = NDArray[np.intp]


class BulkBoundaryTrace:
    """Maps a bulk P1 space's boundary-trace DOFs to a surface P1 space on the
    bulk's boundary, so the trace can be read out as a surface `Function`
    (`gather`) and a corrected surface field written back into the bulk's boundary
    DOFs (`scatter`). Interior bulk DOFs are never touched.
    """

    def __init__(self, V_bulk: fem.FunctionSpace) -> None:
        bulk = V_bulk.mesh
        if bulk.comm.size != 1:
            raise NotImplementedError("BulkBoundaryTrace is serial-only in v1 (single MPI rank)")
        tdim = bulk.topology.dim
        bulk.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = dmesh.exterior_facet_indices(bulk.topology)
        submesh, *_ = dmesh.create_submesh(bulk, tdim - 1, boundary_facets)

        self.V_surf = fem.functionspace(submesh, ("Lagrange", 1))
        self._bulk_dofs: Ints = fem.locate_dofs_topological(V_bulk, tdim - 1, boundary_facets)
        # Match each bulk-boundary dof to its surface dof by coordinate (P1: the
        # two vertex sets coincide, so the match is an exact bijection).
        bulk_xy = V_bulk.tabulate_dof_coordinates()[self._bulk_dofs]
        surf_xy = self.V_surf.tabulate_dof_coordinates()
        self._surf_for_bulk: Ints = cKDTree(surf_xy).query(bulk_xy)[1]

    def gather(self, u_bulk: fem.Function) -> fem.Function:
        """Read the boundary trace of `u_bulk` into a surface `Function` on Γ."""
        rho = fem.Function(self.V_surf, name=u_bulk.name)
        rho.x.array[self._surf_for_bulk] = u_bulk.x.array[self._bulk_dofs]
        return rho

    def scatter(self, rho_surf: fem.Function, u_bulk: fem.Function) -> None:
        """Write a surface `Function` on Γ into `u_bulk`'s boundary DOFs in place."""
        u_bulk.x.array[self._bulk_dofs] = rho_surf.x.array[self._surf_for_bulk]


def correct_surface_trace(u_bulk_old: fem.Function, u_bulk_new: fem.Function, *, conserve: bool = True) -> None:
    """Correct `u_bulk_new`'s boundary trace to conserve surface mass, in place.

    After a bulk remesh + bulk-conservative interpolation has populated
    `u_bulk_new`, its boundary trace generally does not conserve ∫_Γ ρ ds. This
    replaces the boundary DOFs with the conservative surface remap of `u_bulk_old`'s
    trace, so that (with `conserve=True`) ∫_Γnew ρ ds == ∫_Γold ρ ds exactly.
    Interior DOFs of `u_bulk_new` are left untouched.
    """

    old_trace = BulkBoundaryTrace(u_bulk_old.function_space)
    new_trace = BulkBoundaryTrace(u_bulk_new.function_space)
    rho_old = old_trace.gather(u_bulk_old)
    rho_new = remap_surface_function(rho_old, new_trace.V_surf, conserve=conserve)
    new_trace.scatter(rho_new, u_bulk_new)

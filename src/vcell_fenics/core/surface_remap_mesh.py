"""DOLFINx bridge for the conservative surface-density remap.

`surface_remap.py` is a pure-NumPy kernel that operates on arc-length coordinates.
This module connects it to real DOLFINx objects: it orders the vertices of a
closed codim-1 membrane mesh into a loop, reads a P1 `fem.Function`'s nodal values
in that order, runs the conservative remap, and writes the result into a Function
on the target mesh.

Scope (v1): serial only, a single closed-loop P1 membrane in 2D (a 1D mesh of
interval cells embedded in the plane). The loop ordering walks the cell→dof edge
list — for P1 each interval cell carries exactly its two endpoint dofs, so the
cell list *is* the edge list and a closed membrane has every dof at degree 2.
Multi-rank partitioning, higher-order spaces, and open arcs are deferred.

The new mesh is lifted onto the *old* mesh's arc-length frame by closest-point
projection (`project_points_to_polyline_arclength`), so the remap is conservative
in that frame exactly. Because the two polylines are slightly different polygonal
approximations of the same physical curve, the surface integral measured on the
*new* mesh can differ from the old by the geometric discretization gap; with
`conserve=True` (the default) a single global rescale closes that gap so total mass
on the new mesh equals the old exactly. This is the "non-co-located" correction the
design note (`docs/modeling/conservative-surface-remap.md`) calls for.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.core.surface_remap import (
    arclength_parameterization,
    project_points_to_polyline_arclength,
    supermesh_remap_1d,
)

Floats = NDArray[np.float64]
Ints = NDArray[np.intp]


def ordered_membrane_loop(V: fem.FunctionSpace) -> tuple[Floats, Ints]:
    """Order a closed P1 membrane's dofs into a single loop.

    Returns `(coords, order)` where `order` is a permutation of the dof indices
    visiting the loop in traversal order and `coords[k]` is the (gdim,) position of
    dof `order[k]`. `coords` is exactly the ordered node array
    `arclength_parameterization` expects.
    """

    mesh = V.mesh
    if mesh.comm.size != 1:
        raise NotImplementedError("ordered_membrane_loop is serial-only in v1 (single MPI rank)")
    if mesh.topology.dim != 1:
        raise ValueError(f"expected a codim-1 membrane (1D interval mesh), got topology dim {mesh.topology.dim}")

    cell_dofs = np.asarray(V.dofmap.list)
    if cell_dofs.ndim != 2 or cell_dofs.shape[1] != 2:
        raise ValueError("expected a P1 space on interval cells (two dofs per cell)")

    coords_all: Floats = V.tabulate_dof_coordinates()
    n = coords_all.shape[0]

    neighbors: list[list[int]] = [[] for _ in range(n)]
    for a, b in cell_dofs:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    if any(len(nb) != 2 for nb in neighbors):
        raise ValueError("membrane is not a single closed loop (a dof has other than 2 incident edges)")

    order: list[int] = [0]
    prev, cur = -1, 0
    for _ in range(n - 1):
        nxt = neighbors[cur][0] if neighbors[cur][0] != prev else neighbors[cur][1]
        order.append(nxt)
        prev, cur = cur, nxt
    if len(set(order)) != n or cur not in neighbors[0]:
        raise ValueError("membrane is not a single closed loop (walk did not visit every dof and close)")

    ordering: Ints = np.asarray(order, dtype=np.intp)
    gdim = mesh.geometry.dim
    return coords_all[ordering, :gdim], ordering


def remap_surface_function(u_old: fem.Function, V_new: fem.FunctionSpace, *, conserve: bool = True) -> fem.Function:
    """Conservatively transfer a P1 surface density to the mesh of `V_new`.

    Reads `u_old` (a P1 Function on a closed membrane), remaps it onto `V_new`'s
    membrane via the supermesh kernel in the old mesh's arc-length frame, and
    returns a new Function on `V_new`. With `conserve=True` a final global rescale
    makes ∫_Γnew ρ_new ds equal ∫_Γold ρ_old ds exactly (as measured on each mesh);
    with `conserve=False` the field is left as the raw projection-frame remap, which
    preserves constants exactly but lets the surface integral drift by the geometric
    gap between the two polylines.
    """

    old_coords, old_order = ordered_membrane_loop(u_old.function_space)
    new_coords, new_order = ordered_membrane_loop(V_new)

    s_old, length = arclength_parameterization(old_coords, closed=True)
    rho_old: Floats = u_old.x.array[old_order]

    s_new = project_points_to_polyline_arclength(new_coords, old_coords, s_old, length)
    sort = np.argsort(s_new)
    rho_sorted = supermesh_remap_1d(s_old, rho_old, s_new[sort], length)

    rho_walk = np.empty_like(rho_sorted)
    rho_walk[sort] = rho_sorted

    u_new = fem.Function(V_new, name=u_old.name)
    u_new.x.array[new_order] = rho_walk

    if conserve:
        mass_old = _total_mass(u_old)
        mass_new = _total_mass(u_new)
        if mass_new != 0.0:
            u_new.x.array[:] *= mass_old / mass_new
    return u_new


def _total_mass(u: fem.Function) -> float:
    local = fem.assemble_scalar(fem.form(u * ufl.dx))
    return float(u.function_space.mesh.comm.allreduce(local, op=MPI.SUM))

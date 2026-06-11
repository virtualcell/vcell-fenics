"""DOLFINx bridge for the conservative bulk (2D area) remap.

`bulk_remap.py` is a pure-NumPy kernel (`supermesh_project_2d`) that operates on
plain mesh arrays (vertices, triangles, nodal values). This module connects it to
real DOLFINx objects: it reads a P1 `fem.Function`'s nodal values and the
triangulation off its mesh, runs the supermesh projection onto a target mesh, and
writes the result into a Function on that mesh.

This is the area analogue of `surface_remap_mesh.py`, but markedly simpler — the
surface case needs to order a closed membrane's dofs into a loop, whereas here no
ordering is required. For a P1 space on a 2D triangle mesh the dof index *is* the
row index in both `tabulate_dof_coordinates()` and `x.array`, and `V.dofmap.list`
is the (n_cells, 3) vertex-index array the kernel expects directly. (P1-on-interval
gives a (n, 2) edge list; P1-on-triangle gives a (n, 3) triangle list.)

Scope (v1): serial only, a P1 space on a 2D triangle mesh. Higher-order spaces,
multi-rank partitioning, and 3D are deferred.

With `conserve=True` (the default) a single global rescale makes ∫_Ωnew c_new dx
equal ∫_Ωold c_old dx exactly (as measured on each mesh); this closes the geometric
gap when the two meshes triangulate slightly different polygonal approximations of
the same domain (e.g. a disk at two resolutions). With `conserve=False` the field
is left as the raw supermesh projection, which preserves constants and linears
exactly but lets the volume integral drift by that geometric gap.
"""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.core.bulk_remap import supermesh_project_2d

Floats = NDArray[np.float64]
Ints = NDArray[np.intp]


def remap_bulk_function(u_old: fem.Function, V_new: fem.FunctionSpace, *, conserve: bool = True) -> fem.Function:
    """Conservatively transfer a P1 bulk field to the mesh of `V_new`.

    Reads `u_old` (a P1 Function on a 2D triangle mesh), projects it onto `V_new`'s
    mesh via the supermesh kernel, and returns a new Function on `V_new`. With
    `conserve=True` a final global rescale makes ∫_Ωnew c_new dx equal ∫_Ωold c_old dx
    exactly (as measured on each mesh); with `conserve=False` the field is left as the
    raw supermesh projection, which preserves constants/linears exactly but lets the
    volume integral drift by the geometric gap between the two domains' polygons.
    """

    old_verts, old_tris = _p1_mesh_arrays(u_old.function_space)
    new_verts, new_tris = _p1_mesh_arrays(V_new)

    c_old: Floats = np.asarray(u_old.x.array, dtype=np.float64)
    c_new = supermesh_project_2d(old_verts, old_tris, c_old, new_verts, new_tris)

    u_new = fem.Function(V_new, name=u_old.name)
    u_new.x.array[:] = c_new

    if conserve:
        mass_old = _total_mass(u_old)
        mass_new = _total_mass(u_new)
        if mass_new != 0.0:
            u_new.x.array[:] *= mass_old / mass_new
    return u_new


def _p1_mesh_arrays(V: fem.FunctionSpace) -> tuple[Floats, Ints]:
    """Extract `(verts, tris)` for the supermesh kernel from a P1 space on a 2D
    triangle mesh. For P1 the dof index equals the vertex-array row, so `verts[k]`
    is the position of dof `k` and `tris` is the dof-index triangle list."""

    mesh = V.mesh
    if mesh.comm.size != 1:
        raise NotImplementedError("remap_bulk_function is serial-only in v1 (single MPI rank)")
    if mesh.topology.dim != 2:
        raise ValueError(f"expected a 2D triangle mesh, got topology dim {mesh.topology.dim}")

    cell_dofs = np.asarray(V.dofmap.list)
    if cell_dofs.ndim != 2 or cell_dofs.shape[1] != 3:
        raise ValueError("expected a P1 space on triangle cells (three dofs per cell)")

    verts: Floats = np.asarray(V.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    tris: Ints = np.asarray(cell_dofs, dtype=np.intp)
    return verts, tris


def _total_mass(u: fem.Function) -> float:
    local = fem.assemble_scalar(fem.form(u * ufl.dx))
    return float(u.function_space.mesh.comm.allreduce(local.real, op=MPI.SUM))

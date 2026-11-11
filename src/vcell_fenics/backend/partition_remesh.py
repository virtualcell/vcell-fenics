"""Remeshing a moving partition (spike for ADR 016): the whole partition from its own boundary skeleton.

1. **The skeleton.** Every parent facet between two regions, and every box facet, as a `LabelBoundary` segment
   built from the parent's own vertices: its sides the regions beside it (subvolume index = tag − 1; a box
   facet's outside negative), ordered so that `pairs[:, 1]` lies on its left (a geometric side test: the
   mesh's cells are not all oriented alike).
2. **The mesh.** `realize._netgen_from_curves` at the run's `h`: every region keeps its polygon (Netgen keeps
   every vertex and adds points only on segments); its cells tagged by region; `_tag_facets` and the
   realizer's tail (`partition_geometry`) give the tags, submeshes and maps.
3. **The transfer.** A compartment species by the supermesh L2 projection from its old compartment submesh to
   the new one (same polygon, so its total is kept), or its lumped variant where the projection adds
   undershoot; a membrane species by interpolation along the old segments (the new membrane's nodes are the
   old ones plus points on old segments), exact.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from vcell_fenics.backend.label_surfaces import LabelBoundary
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, partition_geometry
from vcell_fenics.backend.realize import _FACE_NAMES_2D, _mesh_on_rank0, _netgen_from_curves, _tag_facets
from vcell_fenics.formalism.geometry_schema import GeometryDescription


def skeleton(geometry: MultiCompartmentGeometry) -> LabelBoundary:
    """The parent's region boundaries and box facets as an oriented, labelled 2D boundary (serial)."""

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    parent.topology.create_connectivity(tdim - 1, 0)
    f2c = parent.topology.connectivity(tdim - 1, tdim)
    cell_map = parent.topology.index_map(tdim)
    region = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=np.int64)
    region[geometry.cell_tags.indices] = geometry.cell_tags.values
    x = parent.geometry.x[:, :2]
    # topological vertex -> geometry node (P1: one each)
    from dolfinx.mesh import entities_to_geometry

    n_facets = parent.topology.index_map(tdim - 1).size_local
    facets = np.arange(n_facets, dtype=np.int32)
    nodes = entities_to_geometry(parent, tdim - 1, facets)
    cell_nodes = parent.geometry.dofmap
    elements, pairs = [], []
    for f in facets:
        cells = f2c.links(f)
        p, q = int(nodes[f, 0]), int(nodes[f, 1])
        if len(cells) == 1:
            sides = [(int(cells[0]), int(region[cells[0]]) - 1), (None, -1)]
        else:
            r0, r1 = int(region[cells[0]]), int(region[cells[1]])
            if r0 == r1:
                continue
            sides = [(int(cells[0]), r0 - 1), (int(cells[1]), r1 - 1)]
        # the left side of p -> q: the side whose cell centroid has a positive cross product
        c0 = x[cell_nodes[sides[0][0]]].mean(axis=0)
        d, e = x[q] - x[p], c0 - x[p]
        first_left = d[0] * e[1] - d[1] * e[0] > 0.0
        left, right = (sides[0][1], sides[1][1]) if first_left else (sides[1][1], sides[0][1])
        elements.append((p, q))
        pairs.append((right, left))
    used = np.unique(np.asarray(elements, dtype=np.int64))
    renumber = -np.ones(x.shape[0], dtype=np.int64)
    renumber[used] = np.arange(used.size)
    return LabelBoundary(
        dim=2,
        points=np.ascontiguousarray(x[used], dtype=np.float64),
        elements=renumber[np.asarray(elements, dtype=np.int64)],
        pairs=np.asarray(pairs, dtype=np.int64),
    )


def remesh_partition(
    geometry: MultiCompartmentGeometry,
    description: GeometryDescription,
    *,
    h: float,
    compartments: Collection[str],
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MultiCompartmentGeometry:
    """A fresh mesh of the moved partition (serial, 2D): the skeleton meshed by Netgen, retagged, rebuilt."""

    if description.dim != 2 or comm.size != 1:
        raise NotImplementedError("remeshing a moving partition is 2D and serial (ADR 016 decision 6)")
    boundary = skeleton(geometry)
    parent, material = _mesh_on_rank0(comm, lambda: _netgen_from_curves(boundary, h), cell="triangle", gdim=2)
    cell_values = np.zeros(material.indices.size, dtype=np.int32)
    cell_values[material.indices] = material.values
    origin = tuple(float(description.origin[i]) for i in range(2))
    extent = tuple(float(description.extent[i]) for i in range(2))
    tagging = _tag_facets(parent, description, cell_values, origin=origin, extent=extent)
    return partition_geometry(description, parent, tagging, _FACE_NAMES_2D, compartments=compartments, comm=comm)


def _total(u: fem.Function) -> float:
    import ufl

    return float(np.real(fem.assemble_scalar(fem.form(u * ufl.dx))))


def transfer_bulk(u_old: fem.Function, V_new: fem.FunctionSpace) -> tuple[fem.Function, str]:
    """The supermesh L2 projection, or its lumped variant where the projection adds undershoot."""

    from vcell_fenics.core.bulk_remap import _mixed_mass_matrix, _new_mass_matrix
    from vcell_fenics.core.bulk_remap_mesh import _p1_mesh_arrays

    old_verts, old_tris = _p1_mesh_arrays(u_old.function_space)
    new_verts, new_tris = _p1_mesh_arrays(V_new)
    mass = _new_mass_matrix(new_verts, new_tris)
    mixed = _mixed_mass_matrix(old_verts, old_tris, new_verts, new_tris)
    c_old = np.asarray(u_old.x.array, dtype=np.float64)
    moments = mixed @ c_old
    from scipy.sparse.linalg import spsolve

    c_new = spsolve(mass, moments)
    kind = "projection"
    old_min, old_max = float(c_old.min()), float(np.abs(c_old).max())
    if c_new.min() < old_min - 1e-8 * max(old_max, 1e-300):
        lumped = np.asarray(mass.sum(axis=1)).ravel()
        c_new = moments / lumped
        kind = "lumped"
    u_new = fem.Function(V_new, name=u_old.name)
    u_new.x.array[:] = c_new
    return u_new, kind


def transfer_membrane(u_old: fem.Function, V_new: fem.FunctionSpace) -> fem.Function:
    """Interpolation along the old membrane's segments: exact when the new nodes lie on them."""

    old_mesh = u_old.function_space.mesh
    xo = u_old.function_space.tabulate_dof_coordinates()[:, :2]
    seg = np.asarray(u_old.function_space.dofmap.list)  # P1 on intervals: each cell's two dofs
    xn = V_new.tabulate_dof_coordinates()[:, :2]
    a, b = xo[seg[:, 0]], xo[seg[:, 1]]
    mid = 0.5 * (a + b)
    tree = cKDTree(mid)
    k = min(8, len(seg))
    _, cand = tree.query(xn, k=k)
    cand = np.atleast_2d(cand)
    out = np.empty(xn.shape[0])
    for i, p in enumerate(xn):
        best, best_d, best_s = -1, np.inf, 0.0
        for c in cand[i]:
            ab = b[c] - a[c]
            s = float(np.clip(np.dot(p - a[c], ab) / max(np.dot(ab, ab), 1e-300), 0.0, 1.0))
            d = float(np.linalg.norm(a[c] + s * ab - p))
            if d < best_d:
                best, best_d, best_s = c, d, s
        if best_d > 1e-9:
            raise RuntimeError(f"a new membrane node lies {best_d:.3g} off the old membrane")
        out[i] = (1 - best_s) * u_old.x.array[seg[best, 0]] + best_s * u_old.x.array[seg[best, 1]]
    del old_mesh
    u_new = fem.Function(V_new, name=u_old.name)
    u_new.x.array[:] = out
    return u_new


def lumped_check(values: NDArray[np.float64]) -> None:  # pragma: no cover - spike helper
    assert np.isfinite(values).all()

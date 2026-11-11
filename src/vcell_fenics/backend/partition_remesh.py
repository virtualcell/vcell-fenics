"""Remeshing a moving partition (ADR 016 decisions 1, 3 and 4): the whole partition, meshed afresh from its own
boundary skeleton, with every species transferred and every total kept.

1. **The skeleton** (decision 1). Every parent facet between two regions, and every box facet, becomes a
   `LabelBoundary` segment built from the parent's own vertex array, so junction vertices are shared. Its sides
   are the regions of the cells beside it (subvolume index = region tag − 1; a box facet's outside is negative),
   and it is ordered so that `pairs[:, 1]` lies on its left, by a geometric side test (a mesh generator does not
   orient all cells alike).
2. **The mesh** (decisions 1 and 4). `realize._netgen_from_curves` meshes the skeleton at the run's `h`. Netgen
   keeps every vertex and adds points only on the segments, so every region keeps exactly its polygon; its cells
   come back tagged by region, and the realizer's tail (`partition_geometry`) rebuilds the tags, submeshes, maps
   and sizes. A Netgen failure stops the run: smoothing the skeleton, as the image path falls back to, would
   change the regions.
3. **The transfer** (decision 3).
   - A compartment's species: the supermesh L2 projection from the old compartment submesh to the new one
     (`core.supermesh`, libsupermesh). Both meshes cover the same polygon, so its total is kept to the solve's
     round-off and a linear field is reproduced. A field that had no negative values and that the projection
     would make negative (below `-1e-12` of its largest magnitude) takes the lumped variant instead, non-negative
     where the old field was and exact in total too (ADR 016 decision 3, as amended 2026-11-11).
     The matrices are built once per compartment and shared by its species.
   - A membrane's species: the new membrane's nodes are the old ones plus points on the old segments, so a kept
     node takes its value by coordinate and an added node interpolates along the old segment it lies on. Values
     and totals are exact.
   - The check: a total that changes by more than `1e-12` of its scale (`∫ |u|`) stops the run.

2D, serial (decision 8). The runner drives it in segments (decisions 2, 4, 6: `runner._run_moving_partition`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from dolfinx import fem
from dolfinx.mesh import entities_to_geometry
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from vcell_fenics.backend.implicit_fields import RealizationError
from vcell_fenics.backend.label_surfaces import LabelBoundary
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, partition_geometry
from vcell_fenics.backend.realize import _FACE_NAMES_2D, _mesh_on_rank0, _netgen_from_curves, _tag_facets
from vcell_fenics.core.supermesh import mass_matrix, mixed_mass_matrix, project
from vcell_fenics.formalism.geometry_schema import GeometryDescription

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

# a remeshed total may differ from the old one by this much of the field's scale, ∫ |u| (round-off)
TOTAL_TOLERANCE = 1.0e-12
# a projected field "goes negative" below this (relative to max |u|): a smaller negative is round-off
NEGATIVE = 1.0e-12
# a new membrane node coincides with an old one, or lies on an old segment, to this (relative to the box)
_ON_OLD = 1.0e-9


class PartitionRemeshError(RuntimeError):
    """The moved partition could not be remeshed, or a remesh failed its checks: the run stops."""


class RemeshLostMass(PartitionRemeshError):
    """A species' total changed across a remesh by more than round-off: with every region unchanged, a bug."""


def skeleton(geometry: MultiCompartmentGeometry) -> LabelBoundary:
    """The parent's region boundaries and box facets as an oriented, labelled 2D boundary (decision 1)."""

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    if tdim != 2:
        raise NotImplementedError("the partition's skeleton is 2D (ADR 016 decision 8)")
    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    cell_map = parent.topology.index_map(tdim)
    region = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=np.int64)
    region[geometry.cell_tags.indices] = geometry.cell_tags.values
    x = np.asarray(parent.geometry.x[:, :2], dtype=np.float64)
    n_facets = parent.topology.index_map(tdim - 1).size_local
    facets = np.arange(n_facets, dtype=np.int32)
    nodes = np.asarray(entities_to_geometry(parent, tdim - 1, facets), dtype=np.int64)
    cell_nodes = np.asarray(parent.geometry.dofmap, dtype=np.int64)
    offsets, links = np.asarray(facet_to_cell.offsets), np.asarray(facet_to_cell.array)

    count = offsets[1:] - offsets[:-1]
    first = links[offsets[:-1]]
    second = np.where(count == 2, links[np.minimum(offsets[:-1] + 1, len(links) - 1)], -1)
    region_first = region[first] - 1
    region_second = np.where(count == 2, region[np.maximum(second, 0)] - 1, -1)  # a box facet: outside is -1
    keep = (count == 1) | (region_first != region_second)
    facets, nodes = facets[keep], nodes[keep]
    first, region_first, region_second = first[keep], region_first[keep], region_second[keep]

    # the left of p -> q: the side whose cell centroid gives a positive cross product
    p, q = x[nodes[:, 0]], x[nodes[:, 1]]
    centroid = x[cell_nodes[first]].mean(axis=1)
    d, e = q - p, centroid - p
    first_left = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0] > 0.0
    left = np.where(first_left, region_first, region_second)
    right = np.where(first_left, region_second, region_first)

    used = np.unique(nodes)
    renumber = -np.ones(x.shape[0], dtype=np.int64)
    renumber[used] = np.arange(used.size)
    boundary = LabelBoundary(
        dim=2,
        points=np.ascontiguousarray(x[used]),
        elements=renumber[nodes],
        pairs=np.stack([right, left], axis=1).astype(np.int64),
    )
    corners = x[cell_nodes]
    cell_area = 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )
    _check_orientation(boundary, region[: len(cell_area)] - 1, cell_area)
    return boundary


def _check_orientation(boundary: LabelBoundary, region_of_cell: Ints, cell_area: Floats) -> None:
    """Every region's oriented boundary must enclose exactly its cells' area (Green's theorem: the region lies on
    the left of the segments naming it on their left). A skeleton that fails this would make Netgen hang rather
    than fail, so it is refused here."""
    p, q = boundary.points[boundary.elements[:, 0]], boundary.points[boundary.elements[:, 1]]
    signed = 0.5 * (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])  # each segment's term of the shoelace sum
    right, left = boundary.pairs[:, 0], boundary.pairs[:, 1]
    for r in np.unique(region_of_cell):
        enclosed = float(signed[left == r].sum() - signed[right == r].sum())
        area = float(cell_area[region_of_cell == r].sum())
        if abs(enclosed - area) > 1e-9 * max(area, 1e-300) + 1e-12 * float(cell_area.sum()):
            raise PartitionRemeshError(
                f"the partition's skeleton is inconsistent: region {int(r)} encloses {enclosed:.12g} by its boundary "
                f"but its cells cover {area:.12g}"
            )


def remesh_partition(
    geometry: MultiCompartmentGeometry,
    description: GeometryDescription,
    *,
    h: float,
    compartments: Collection[str] | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MultiCompartmentGeometry:
    """A fresh mesh of the (moved) partition: its skeleton meshed by Netgen at `h`, retagged and rebuilt
    (decisions 1 and 4). Compartments as for `realize_multi_compartment` (default: those `geometry` has)."""

    if description.dim != 2 or comm.size != 1:
        raise NotImplementedError("remeshing a moving partition is 2D and serial (ADR 016 decision 8)")
    boundary = skeleton(geometry)
    try:
        parent, material = _mesh_on_rank0(comm, lambda: _netgen_from_curves(boundary, h), cell="triangle", gdim=2)
    except RealizationError as error:
        raise PartitionRemeshError(
            f"Netgen could not mesh the moved partition's boundaries at h = {h:g}; the boundaries are kept exactly, "
            "so the run stops (vcell-fenics ADR 016 decision 1)"
        ) from error
    cell_values = np.zeros(material.indices.size, dtype=np.int32)
    cell_values[material.indices] = material.values
    origin = tuple(float(description.origin[i]) for i in range(2))
    extent = tuple(float(description.extent[i]) for i in range(2))
    tagging = _tag_facets(parent, description, cell_values, origin=origin, extent=extent)
    kept = list(geometry.compartments) if compartments is None else compartments
    return partition_geometry(description, parent, tagging, _FACE_NAMES_2D, h=h, compartments=kept, comm=comm)


def _p1_arrays(V: fem.FunctionSpace) -> tuple[Floats, Ints]:
    """A P1 space's dof coordinates and per-cell dofs (for P1, the dofs are the mesh's vertices)."""
    gdim = V.mesh.topology.dim
    return (
        np.asarray(V.tabulate_dof_coordinates()[:, :gdim], dtype=np.float64),
        np.asarray(V.dofmap.list, dtype=np.int64),
    )


def _scale(mass: csr_matrix, values: Floats) -> tuple[float, float]:
    """A P1 field's total ∫ u and its scale ∫ |u| (with the nodal |u|)."""
    weights = np.asarray(mass.sum(axis=0)).ravel()
    return float(weights @ values), float(weights @ np.abs(values))


def transfer_compartment(
    fields: Mapping[str, fem.Function], V_new: fem.FunctionSpace
) -> tuple[dict[str, fem.Function], dict[str, str]]:
    """Every species of one compartment, from its old submesh's P1 space to `V_new` (decision 3): the L2
    projection, or the lumped variant for a non-negative field it would make negative. Returns the new fields and, per
    field, which transfer it took ("projection" or "lumped")."""

    if not fields:
        return {}, {}
    V_old = next(iter(fields.values())).function_space
    old_x, old_cells = _p1_arrays(V_old)
    new_x, new_cells = _p1_arrays(V_new)
    mixed, _ = mixed_mass_matrix(old_x, old_cells, new_x, new_cells)
    old_mass, new_mass = mass_matrix(old_x, old_cells), mass_matrix(new_x, new_cells)
    out: dict[str, fem.Function] = {}
    kinds: dict[str, str] = {}
    for name, u in fields.items():
        if u.function_space is not V_old and u.function_space.mesh is not V_old.mesh:
            raise ValueError(f"the fields of one compartment must share its mesh ({name!r} does not)")
        c_old = np.asarray(u.x.array, dtype=np.float64)
        c_new = project(old_x, old_cells, c_old, new_x, new_cells, mixed=mixed, mass=new_mass)
        kind = "projection"
        largest = float(np.abs(c_old).max()) if c_old.size else 0.0
        # positivity, not undershoot: a projection undershoots a smooth field's minimum by its interpolation error
        # (1e-5 to 2e-3 of the maximum, measured), which is no reason to give up its second order; a negative
        # concentration where there was none is
        if c_new.size and c_old.min() >= 0.0 and c_new.min() < -NEGATIVE * largest:
            c_new = project(old_x, old_cells, c_old, new_x, new_cells, mixed=mixed, mass=new_mass, lumped=True)
            kind = "lumped"
        _check_total(name, _scale(old_mass, c_old), _scale(new_mass, c_new))
        v = fem.Function(V_new, name=u.name)
        v.x.array[:] = c_new
        out[name], kinds[name] = v, kind
    return out, kinds


def transfer_membrane(fields: Mapping[str, fem.Function], V_new: fem.FunctionSpace) -> dict[str, fem.Function]:
    """Every species of one membrane, from its old submesh's P1 space to `V_new` (decision 3): a node the remesh
    kept takes its old value, a node it added interpolates along the old segment it lies on. Exact."""

    if not fields:
        return {}
    V_old = next(iter(fields.values())).function_space
    old_x = np.asarray(V_old.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    segments = np.asarray(V_old.dofmap.list, dtype=np.int64)
    new_x = np.asarray(V_new.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    scale = float(np.ptp(np.vstack([old_x, new_x]), axis=0).max()) or 1.0
    tolerance = _ON_OLD * scale

    a, b = old_x[segments[:, 0]], old_x[segments[:, 1]]
    tree = cKDTree(0.5 * (a + b))
    k = min(8, len(segments))
    _, candidates = tree.query(new_x, k=k)
    candidates = np.asarray(candidates).reshape(len(new_x), k)
    ab = b[candidates] - a[candidates]  # (new nodes, k, 2)
    ap = new_x[:, None, :] - a[candidates]
    s = np.clip(np.einsum("nkd,nkd->nk", ap, ab) / np.maximum(np.einsum("nkd,nkd->nk", ab, ab), 1e-300), 0.0, 1.0)
    distance = np.linalg.norm(ap - s[..., None] * ab, axis=2)
    best = np.argmin(distance, axis=1)
    rows = np.arange(len(new_x))
    off = distance[rows, best]
    if off.size and off.max() > tolerance:
        raise PartitionRemeshError(
            f"a remeshed membrane node lies {off.max():.3g} off the old membrane: the remesh changed a region"
        )
    segment, fraction = candidates[rows, best], s[rows, best]
    # a node that coincides with an old one takes its value exactly, not by a fraction rounded to 0 or 1
    fraction = np.where(np.linalg.norm(new_x - a[segment], axis=1) <= tolerance, 0.0, fraction)
    fraction = np.where(np.linalg.norm(new_x - b[segment], axis=1) <= tolerance, 1.0, fraction)

    old_mass = mass_matrix_1d(old_x, segments)
    new_mass = mass_matrix_1d(new_x, np.asarray(V_new.dofmap.list, dtype=np.int64))
    out: dict[str, fem.Function] = {}
    for name, u in fields.items():
        c_old = np.asarray(u.x.array, dtype=np.float64)
        c_new = (1.0 - fraction) * c_old[segments[segment, 0]] + fraction * c_old[segments[segment, 1]]
        _check_total(name, _scale(old_mass, c_old), _scale(new_mass, c_new))
        v = fem.Function(V_new, name=u.name)
        v.x.array[:] = c_new
        out[name] = v
    return out


def mass_matrix_1d(x: Floats, segments: Ints) -> csr_matrix:
    """The P1 mass matrix of a polyline (segments in the plane)."""
    from scipy.sparse import coo_matrix

    length = np.linalg.norm(x[segments[:, 1]] - x[segments[:, 0]], axis=1)
    local = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
    rows = np.repeat(segments, 2, axis=1).ravel()
    cols = np.tile(segments, (1, 2)).ravel()
    return coo_matrix(((length[:, None, None] * local).ravel(), (rows, cols)), shape=(len(x), len(x))).tocsr()


def _check_total(name: str, old: tuple[float, float], new: tuple[float, float]) -> None:
    (total_old, scale), (total_new, _) = old, new
    if abs(total_new - total_old) > TOTAL_TOLERANCE * max(scale, 1e-300):
        raise RemeshLostMass(
            f"a remesh changed the total of {name!r} from {total_old:.15g} to {total_new:.15g} "
            f"(relative {abs(total_new - total_old) / max(scale, 1e-300):.2e}); the regions are unchanged, so this "
            "is a bug (vcell-fenics ADR 016 decision 3)"
        )

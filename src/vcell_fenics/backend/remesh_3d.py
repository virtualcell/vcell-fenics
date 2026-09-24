"""Remeshing a moving 3D region: re-tetrahedralize the deformed configuration of a tetrahedral mesh.

The 2D remesher meshes the deformed boundary polygon directly (``core.region_remesh_netgen``). In 3D the
boundary is a triangle surface the motion has degraded — thin triangles at a pinching neck — and Netgen's
STL surface mesher, fed that surface, hangs or segfaults. So the region is rebuilt **implicitly**, with
the machinery the image-geometry realization already relies on (ADR 012):

1. **A lattice** over the region's bounding box at half the mesh size ``s = min(h, thinnest / 3)`` (the
   mesh size itself down to ``h / 2``): thin places — a closing neck — are resolved by the lattice. The
   refinement is global (the whole region is rebuilt at that size), so it is bounded: a region thinner
   than ``h`` (twice the floor) stops the run with :class:`PinchOffError` — a clear message, since
   Netgen would otherwise segfault — and a finer ``h`` runs further.
2. **Inside/outside** at each node by exact point location in the current tetrahedral mesh, and the
   **signed distance** to its boundary surface (densely sampled).
3. **The boundary** by SurfaceNets on that label grid, projected onto the zero level set of the signed
   distance (``label_surfaces.extract_boundary`` with the distance as the indicator) — so the new surface
   lies on the old one to O(s²).
4. **Netgen** fills it directly (one ``FaceDescriptor``, ``GenerateVolumeMesh`` at the mesh size), the
   route that is robust for surfaces at lattice resolution.
5. **The new boundary is snapped onto the old one**: each vertex moves to its closest point on the old
   surface triangles. The lattice's signed distance rounds off features narrower than a couple of lattice
   cells — the furrow's sharp groove lost depth at every remesh — and snapping keeps them better (the 3D
   furrow's waist error at t = 2, h = 1: 0.18 with, 0.36–0.41 without).
6. **The volume is restored**: rebuilding a curved surface still cuts its corners slightly; the new
   boundary is offset along its normals to the old volume exactly.

The field transfer (``core.bulk_remap_mesh.remap_bulk_function_3d``) then interpolates and rescales the
mass, so the small volume change of step 3 costs no conservation.
"""

from __future__ import annotations

import numpy as np
from dolfinx import geometry as dgeometry
from dolfinx.mesh import Mesh, create_mesh, entities_to_geometry, exterior_facet_indices
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from vcell_fenics.backend.label_surfaces import extract_boundary
from vcell_fenics.backend.labels import LabelGrid, repair_pinches
from vcell_fenics.core.region_remesh_netgen import PinchOffError, local_thickness

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]


def remesh_region_3d(mesh: Mesh, h: float, *, min_h: float | None = None, lattice: float = 0.5) -> Mesh:
    """A fresh tetrahedral mesh of the region ``mesh`` covers now (see the module docstring). Serial."""

    import basix.ufl
    import ufl

    points, triangles = outward_boundary(mesh)
    floor = h / 2.0 if min_h is None else min_h
    thinnest = float(local_thickness(points, triangles, radius=3.0 * h).min())
    if thinnest < 2.0 * floor:
        raise PinchOffError(
            f"the region is {thinnest:.3g} thick somewhere, under twice the finest mesh size {floor:.3g} at h = "
            f"{h:g} — a neck closing toward a split, narrower than this mesh can follow (a finer h runs further)"
        )
    spacing = float(np.clip(thinnest / 3.0, floor, h))  # the volume mesh size (Netgen maxh)
    grid = _signed_distance_grid(mesh, points, triangles, lattice * spacing)  # the surface: finer
    extent = tuple(float(grid.spacing[0] * (n - 1)) for n in grid.labels.shape)
    boundary = extract_boundary(grid, extent=extent)
    inside = (boundary.pairs == (0, 1)).all(axis=1)  # outside (0) | region (1); no box faces (it is padded)
    triangles_new = boundary.elements[inside][:, ::-1]  # the normal pointed into the region: turn it outward
    surface_points = snap_to_surface(boundary.points, points, triangles)
    new_points, cells = _netgen_fill(surface_points, triangles_new, spacing)
    _restore_volume(new_points, cells, _volume(np.asarray(mesh.geometry.x), np.asarray(mesh.geometry.dofmap)))
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    return create_mesh(mesh.comm, cells, domain, new_points)


def _volume(x: Floats, cells: NDArray[np.integer]) -> float:
    a, b, c, d = (x[cells[:, k]] for k in range(4))
    return float(np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))).sum() / 6.0)


def _restore_volume(points: Floats, cells: Ints, target: float) -> None:
    """Move the boundary nodes of the new mesh along their outward normals, in place, so its volume equals
    ``target`` (the old mesh's). Rebuilding a curved surface cuts its corners — a systematic inward bias of
    O(s²) per remesh that would otherwise accumulate over many remeshes; the offset
    ``δ = (V_target − V) / A`` is a small fraction of a cell, iterated until the volume is exact (a few passes)."""

    faces = np.concatenate([cells[:, [1, 2, 3]], cells[:, [0, 3, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 1]]])
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    boundary = faces[first[counts == 1]]
    # orient each boundary face outward (away from its tetrahedron's centroid)
    owner = np.concatenate([np.arange(len(cells))] * 4)[first[counts == 1]]
    for _ in range(8):
        if abs(target - _volume(points, cells)) <= 1e-13 * target:
            return
        a, b, c = points[boundary[:, 0]], points[boundary[:, 1]], points[boundary[:, 2]]
        normal = np.cross(b - a, c - a)
        centroid = points[cells[owner]].mean(axis=1)
        flip = np.einsum("ij,ij->i", normal, (a + b + c) / 3.0 - centroid) < 0.0
        normal[flip] *= -1.0
        area = 0.5 * float(np.linalg.norm(normal, axis=1).sum())
        vertex_normal = np.zeros_like(points)
        for k in range(3):
            np.add.at(vertex_normal, boundary[:, k], normal)
        on_boundary = np.unique(boundary)
        vertex_normal[on_boundary] /= np.linalg.norm(vertex_normal[on_boundary], axis=1, keepdims=True)
        delta = (target - _volume(points, cells)) / area
        points[on_boundary] += delta * vertex_normal[on_boundary]


def outward_boundary(mesh: Mesh) -> tuple[Floats, Ints]:
    """The boundary of a tetrahedral mesh as ``(points, triangles)``, each triangle's normal pointing out
    of the mesh (away from its tetrahedron's fourth vertex)."""

    mesh.topology.create_connectivity(2, 3)
    facets = exterior_facet_indices(mesh.topology)
    triangles = np.asarray(entities_to_geometry(mesh, 2, facets), dtype=np.int64)
    facet_to_cell = mesh.topology.connectivity(2, 3)
    cells = np.asarray([facet_to_cell.links(f)[0] for f in facets.tolist()], dtype=np.int64)
    corners = np.asarray(mesh.geometry.dofmap)[cells]
    x = np.asarray(mesh.geometry.x, dtype=np.float64)
    opposite = np.array(
        [next(v for v in c if v not in t) for c, t in zip(corners.tolist(), triangles.tolist(), strict=True)]
    )
    a, b, c = x[triangles[:, 0]], x[triangles[:, 1]], x[triangles[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), x[opposite] - a) > 0.0
    triangles[inward] = triangles[inward][:, ::-1]
    used, local = np.unique(triangles, return_inverse=True)
    return x[used], local.reshape(-1, 3).astype(np.int64)


def snap_to_surface(query: Floats, points: Floats, triangles: Ints, candidates: int = 12) -> Floats:
    """Each ``query`` point moved to its closest point on the triangle surface ``(points, triangles)`` —
    exact point-triangle projection over the ``candidates`` triangles with the nearest centroids."""

    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    k = min(candidates, len(triangles))
    _, near = cKDTree((a + b + c) / 3.0).query(query, k=k)
    near = np.asarray(near).reshape(len(query), k)
    best = np.empty_like(query)
    best_distance = np.full(len(query), np.inf)
    for j in range(k):
        t = near[:, j]
        closest = _closest_on_triangles(query, a[t], b[t], c[t])
        distance = np.linalg.norm(closest - query, axis=1)
        better = distance < best_distance
        best[better], best_distance[better] = closest[better], distance[better]
    return best


def _closest_on_triangles(p: Floats, a: Floats, b: Floats, c: Floats) -> Floats:
    """The closest point on triangle (a, b, c) to p, row by row (Ericson, *Real-Time Collision Detection*
    §5.1.5, vectorized)."""

    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        out = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]  # the face interior
        t_ab = d1 / (d1 - d3)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out[on_ab] = (a + ab * t_ab[:, None])[on_ab]
        t_ac = d2 / (d2 - d6)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out[on_ac] = (a + ac * t_ac[:, None])[on_ac]
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        on_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        out[on_bc] = (b + (c - b) * t_bc[:, None])[on_bc]
    at_a = (d1 <= 0) & (d2 <= 0)  # the vertices last: they override the edge tests at the corners
    at_b = (d3 >= 0) & (d4 <= d3)
    at_c = (d6 >= 0) & (d5 <= d6)
    out[at_a], out[at_b], out[at_c] = a[at_a], b[at_b], c[at_c]
    return np.asarray(out, dtype=np.float64)


def _signed_distance_grid(mesh: Mesh, points: Floats, triangles: Ints, spacing: float) -> LabelGrid:
    """The region as a label grid (1 inside, 0 outside) on a lattice padded around its bounding box, with
    the signed distance to its boundary as the indicators (``I_1 = −φ``, ``I_0 = φ``: their equality is the
    surface)."""

    lo = points.min(axis=0) - 3.0 * spacing
    counts = np.ceil((points.max(axis=0) + 3.0 * spacing - lo) / spacing).astype(np.int64) + 1
    axes = [lo[k] + spacing * np.arange(counts[k]) for k in range(3)]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    tree = dgeometry.bb_tree(mesh, mesh.topology.dim)
    colliding = dgeometry.compute_colliding_cells(mesh, dgeometry.compute_collisions_points(tree, nodes), nodes)
    inside = np.diff(np.asarray(colliding.offsets)) > 0

    # the distance to the surface, from a dense sampling of its triangles (spacing / 4 apart)
    samples = _sample_surface(points, triangles, spacing / 4.0)
    distance, _ = cKDTree(samples).query(nodes)
    phi = np.where(inside, -distance, distance).reshape(tuple(counts))
    labels = repair_pinches(inside.astype(np.int32).reshape(tuple(counts)))
    return LabelGrid(
        labels=labels,
        origin=(float(lo[0]), float(lo[1]), float(lo[2])),
        spacing=(spacing, spacing, spacing),
        indicators=(phi.astype(np.float32), (-phi).astype(np.float32)),
    )


def _sample_surface(points: Floats, triangles: Ints, gap: float) -> Floats:
    """Points covering the triangle surface at most ``gap`` apart (a barycentric lattice per triangle)."""

    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    edges = [np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)]
    longest = np.max(np.stack(edges), axis=0)
    samples = [points]
    for n in np.unique(np.maximum(1, np.ceil(longest / gap)).astype(np.int64)):
        chosen = np.ceil(longest / gap).clip(min=1).astype(np.int64) == n
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = (i[keep] / n)[:, None, None], (j[keep] / n)[:, None, None]
        samples.append((a[chosen] + u * (b[chosen] - a[chosen]) + v * (c[chosen] - a[chosen])).reshape(-1, 3))
    return np.concatenate(samples)


def _netgen_fill(points: Floats, triangles: Ints, h: float) -> tuple[Floats, Ints]:
    """Tetrahedralize the closed outward surface ``(points, triangles)`` directly with Netgen."""

    from vcell_fenics.backend.realize import netgen_fill_surface

    return netgen_fill_surface(points, triangles, h)

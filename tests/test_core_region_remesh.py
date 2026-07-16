"""Verification of the region remesher (core/region_remesh).

`mesh_region(loop, h)` takes an ordered closed polyline (a deformed boundary) and
produces a fresh good-quality 2D mesh of the region it encloses — step (b) of the
ALE remesh routine (`docs/modeling/ale-remesh-driver.md`). The checks:

1. **Region preserved** — the meshed area equals the input polygon's shoelace area
   to round-off (the boundary nodes lie on the polyline, so the region is exact),
   for arbitrary (non-circular) loops.
2. **Quality** — the fresh mesh has well-shaped triangles (min angle comfortably
   above degeneracy), which is the whole point of remeshing a tangled mesh.
3. **fix_boundary_nodes** — the new boundary is exactly the input loop (Γ_new ⊂
   Γ_old), the interior-only fast path that lets `correct_surface_trace` be skipped.
4. **Default resampling** — without that flag the boundary is refined at `h`, but
   the inserted nodes still lie on the polyline and the area is still exact.
5. **Resolution** — smaller `h` yields more cells.
6. **Integration** — extract a real bulk mesh's boundary loop via
   `BulkBoundaryTrace.boundary_loop()` and remesh it; area matches the loop polygon.
7. **Input guards** — malformed loops are rejected loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dmesh
from numpy.typing import NDArray

from tests.gmsh_meshers.region_remesh import mesh_region
from tests.gmsh_meshers.submesh.geometry import create_disk_with_membrane
from vcell_fenics.core import BulkBoundaryTrace

Floats = NDArray[np.float64]


def _circle_loop(n: int, *, radius: float = 1.0) -> Floats:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((radius * np.cos(t), radius * np.sin(t)))


def _ellipse_loop(n: int, *, a: float = 1.5, b: float = 0.6) -> Floats:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))


def _shoelace(loop: Floats) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _tris_and_verts(mesh: dmesh.Mesh) -> tuple[Floats, NDArray[np.intp]]:
    V = fem.functionspace(mesh, ("Lagrange", 1))
    verts = np.asarray(V.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    tris = np.asarray(V.dofmap.list, dtype=np.intp)
    return verts, tris


def _mesh_area(mesh: dmesh.Mesh) -> float:
    verts, tris = _tris_and_verts(mesh)
    total = 0.0
    for tri in tris:
        a, b, c = verts[tri]
        total += 0.5 * abs(float(np.cross(b - a, c - a)))
    return total


def _min_angle_deg(mesh: dmesh.Mesh) -> float:
    verts, tris = _tris_and_verts(mesh)
    smallest = 180.0
    for tri in tris:
        p = verts[tri]
        for i in range(3):
            u = p[(i + 1) % 3] - p[i]
            v = p[(i + 2) % 3] - p[i]
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            smallest = min(smallest, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return smallest


def _boundary_coords(mesh: dmesh.Mesh) -> Floats:
    V = fem.functionspace(mesh, ("Lagrange", 1))
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dmesh.exterior_facet_indices(mesh.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    return np.asarray(V.tabulate_dof_coordinates()[dofs, :2], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. region preserved (arbitrary polygon)
# ---------------------------------------------------------------------------


def test_area_matches_polygon() -> None:
    loop = _ellipse_loop(80)
    mesh = mesh_region(loop, h=0.12)

    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. quality
# ---------------------------------------------------------------------------


def test_fresh_mesh_is_well_shaped() -> None:
    mesh = mesh_region(_circle_loop(64), h=0.15)

    verts, tris = _tris_and_verts(mesh)
    assert len(tris) > 0
    # No inverted / degenerate cells, and angles comfortably away from a sliver.
    for tri in tris:
        a, b, c = verts[tri]
        assert abs(float(np.cross(b - a, c - a))) > 1e-12
    assert _min_angle_deg(mesh) > 20.0


# ---------------------------------------------------------------------------
# 3. fix_boundary_nodes ⇒ Γ_new is exactly the input loop
# ---------------------------------------------------------------------------


def test_fixed_boundary_nodes_preserve_the_loop() -> None:
    loop = _circle_loop(48)
    mesh = mesh_region(loop, h=0.2, fix_boundary_nodes=True)

    boundary = _boundary_coords(mesh)
    assert boundary.shape[0] == loop.shape[0]
    # Every input vertex is reproduced exactly (set equality up to ordering).
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(boundary).query(loop)
    assert float(np.max(dist)) < 1e-10


# ---------------------------------------------------------------------------
# 4. default resampling refines the boundary but stays on the polyline
# ---------------------------------------------------------------------------


def test_default_resamples_boundary_on_the_polyline() -> None:
    loop = _circle_loop(12)  # coarse loop, fine target size ⇒ boundary gets refined
    mesh = mesh_region(loop, h=0.1, fix_boundary_nodes=False)

    boundary = _boundary_coords(mesh)
    assert boundary.shape[0] > loop.shape[0]  # boundary was subdivided

    # Inserted nodes lie on the polyline, so the region (area) is unchanged.
    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)
    # Each boundary node is on some segment of the loop.
    for q in boundary:
        assert _dist_to_polyline(q, loop) < 1e-9


def _dist_to_polyline(q: Floats, loop: Floats) -> float:
    n = loop.shape[0]
    best = np.inf
    for i in range(n):
        a, b = loop[i], loop[(i + 1) % n]
        ab = b - a
        t = float(np.clip(np.dot(q - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(q - (a + t * ab))))
    return best


# ---------------------------------------------------------------------------
# 5. resolution
# ---------------------------------------------------------------------------


def test_finer_h_gives_more_cells() -> None:
    loop = _circle_loop(64)
    coarse = mesh_region(loop, h=0.3)
    fine = mesh_region(loop, h=0.12)

    _, coarse_tris = _tris_and_verts(coarse)
    _, fine_tris = _tris_and_verts(fine)
    assert len(fine_tris) > len(coarse_tris)


# ---------------------------------------------------------------------------
# 6. integration — remesh a real bulk mesh's boundary loop
# ---------------------------------------------------------------------------


def test_remesh_boundary_loop_of_a_real_mesh() -> None:
    bulk = create_disk_with_membrane(radius=1.0, h=0.4).bulk_mesh
    V = fem.functionspace(bulk, ("Lagrange", 1))
    loop = BulkBoundaryTrace(V).boundary_loop()

    # The extracted loop is a closed polygon inscribed in the unit circle.
    assert np.allclose(np.linalg.norm(loop, axis=1), 1.0, atol=0.05)

    mesh = mesh_region(loop, h=0.2)
    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)
    # A coarse inscribed polygon underestimates π·r² only slightly.
    assert _mesh_area(mesh) == pytest.approx(np.pi, rel=0.05)


# ---------------------------------------------------------------------------
# 7. input guards
# ---------------------------------------------------------------------------


def test_rejects_malformed_loops() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        mesh_region(_circle_loop(2), h=0.2)
    with pytest.raises(ValueError, match="repeat its first point"):
        loop = _circle_loop(8)
        mesh_region(np.vstack([loop, loop[:1]]), h=0.2)
    with pytest.raises(ValueError, match="no area"):
        collinear = np.column_stack((np.linspace(0.0, 1.0, 5), np.zeros(5)))
        mesh_region(collinear, h=0.2)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        mesh_region(np.zeros((5, 3)), h=0.2)

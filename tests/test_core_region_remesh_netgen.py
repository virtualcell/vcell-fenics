"""Verification of the Netgen region remesher (core/region_remesh_netgen).

`mesh_region_netgen(loop, h)` is the LGPL, import-boundary analogue of the gmsh
`mesh_region` (ADR 008): same contract — a fresh good-quality 2D DOLFINx mesh of the
region enclosed by a deformed boundary polyline — backed by Netgen instead of gmsh.
The checks mirror `test_core_region_remesh.py`, with one honest divergence:

- **Region preserved / quality / default resampling / resolution / integration** —
  identical expectations to the gmsh remesher; Netgen resamples the boundary on the
  polyline, so Γ_new ⊂ Γ_old and area is exact.
- **fix_boundary_nodes** — NOT supported by the high-level Netgen mesher (it resamples
  regardless), so it raises `NotImplementedError`. This test pins that documented gap
  rather than exact-vertex preservation (which the gmsh remesher provides).
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.core import BulkBoundaryTrace
from vcell_fenics.core.region_remesh_netgen import mesh_region_netgen

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
# 1. region preserved (arbitrary polygon)
# ---------------------------------------------------------------------------


def test_area_matches_polygon() -> None:
    loop = _ellipse_loop(80)
    mesh = mesh_region_netgen(loop, h=0.12)

    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. quality
# ---------------------------------------------------------------------------


def test_fresh_mesh_is_well_shaped() -> None:
    mesh = mesh_region_netgen(_circle_loop(64), h=0.15)

    verts, tris = _tris_and_verts(mesh)
    assert len(tris) > 0
    for tri in tris:
        a, b, c = verts[tri]
        assert abs(float(np.cross(b - a, c - a))) > 1e-12
    assert _min_angle_deg(mesh) > 20.0


# ---------------------------------------------------------------------------
# 3. fix_boundary_nodes ⇒ unsupported by the high-level Netgen mesher (ADR 008)
# ---------------------------------------------------------------------------


def test_fix_boundary_nodes_is_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="fix_boundary_nodes"):
        mesh_region_netgen(_circle_loop(48), h=0.2, fix_boundary_nodes=True)


# ---------------------------------------------------------------------------
# 4. default resampling refines the boundary but stays on the polyline
# ---------------------------------------------------------------------------


def test_default_resamples_boundary_on_the_polyline() -> None:
    loop = _circle_loop(12)  # coarse loop, fine target size ⇒ boundary gets refined
    mesh = mesh_region_netgen(loop, h=0.1)

    boundary = _boundary_coords(mesh)
    assert boundary.shape[0] > loop.shape[0]  # boundary was subdivided

    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)
    for q in boundary:
        assert _dist_to_polyline(q, loop) < 1e-9


# ---------------------------------------------------------------------------
# 5. resolution
# ---------------------------------------------------------------------------


def test_finer_h_gives_more_cells() -> None:
    loop = _circle_loop(64)
    coarse = mesh_region_netgen(loop, h=0.3)
    fine = mesh_region_netgen(loop, h=0.12)

    _, coarse_tris = _tris_and_verts(coarse)
    _, fine_tris = _tris_and_verts(fine)
    assert len(fine_tris) > len(coarse_tris)


# ---------------------------------------------------------------------------
# 6. integration — remesh a real bulk mesh's boundary loop.
# Deliberately gmsh-free: a DOLFINx built-in mesh, not the gmsh geometry helpers.
# The netgen remesher must not require gmsh, and mixing gmsh-OCC meshing with netgen
# in one long-lived process has shown instability (ADR 008) — kept out of this suite.
# ---------------------------------------------------------------------------


def test_remesh_boundary_loop_of_a_real_mesh() -> None:
    bulk = dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8, dmesh.CellType.triangle)
    V = fem.functionspace(bulk, ("Lagrange", 1))
    loop = BulkBoundaryTrace(V).boundary_loop()

    # The extracted loop is the unit square's boundary polygon.
    assert _shoelace(loop) == pytest.approx(1.0, rel=1e-9)

    mesh = mesh_region_netgen(loop, h=0.15)
    assert _mesh_area(mesh) == pytest.approx(_shoelace(loop), rel=1e-9)
    assert _mesh_area(mesh) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. input guards (identical messages to the gmsh remesher)
# ---------------------------------------------------------------------------


def test_rejects_malformed_loops() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        mesh_region_netgen(_circle_loop(2), h=0.2)
    with pytest.raises(ValueError, match="repeat its first point"):
        loop = _circle_loop(8)
        mesh_region_netgen(np.vstack([loop, loop[:1]]), h=0.2)
    with pytest.raises(ValueError, match="no area"):
        collinear = np.column_stack((np.linspace(0.0, 1.0, 5), np.zeros(5)))
        mesh_region_netgen(collinear, h=0.2)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        mesh_region_netgen(np.zeros((5, 3)), h=0.2)

"""Netgen region remesher — the LGPL remesher the ALE driver uses at runtime, and an
import-boundary alternative to the (now test-only) gmsh `mesh_region`
(`tests/gmsh_meshers/region_remesh.py`; LICENSING.md).

Both functions do the same job — step (b) of the ALE remesh routine
(`docs/modeling/ale-remesh-driver.md`): take the current (deformed) boundary loop
and generate a fresh good-quality bulk mesh of the region it encloses. They differ
only in the engine and, therefore, in their **license reach** (ADR 008):

- gmsh is **GPL-2.0**, so the gmsh `mesh_region` is kept out of `src/` entirely — it
  lives under `tests/` as a reference mesher and never ships in this package.
- Netgen (`netgen.geom2d`) is **LGPL-2.1**, the same weak-copyleft tier as DOLFINx
  itself, so this function may be imported and called **in-process** from permissive
  code with no copyleft obligation on the caller — no isolation, no plugin. It is what
  `backend/ale.py` calls to remesh.

The mesh is handed to DOLFINx directly through `create_mesh` on Netgen's point/cell
arrays; no `.msh` file and no ngsPETSc bridge is involved.

**Boundary handling.** Netgen's high-level 2D mesher always *resamples* the boundary
at `maxh`, but the inserted nodes lie on the straight input segments, so Γ_new ⊂ Γ_old
and the enclosed region (hence area) is preserved to round-off. This matches
`mesh_region`'s *default* (`fix_boundary_nodes=False`) path. The exact-vertex
`fix_boundary_nodes=True` fast path (used to skip `correct_surface_trace` when only
the interior degraded) has **no high-level Netgen equivalent** — the `SplineGeometry`
API resamples regardless of per-segment `maxh` / `MeshingParameters`. Reproducing it
would require Netgen's low-level `Element1D` construction; that is deferred (ADR 008),
so this function raises `NotImplementedError` for it rather than silently resampling.

Scope (v1): serial, 2D, a single simple (non-self-intersecting) closed loop — same as
the gmsh remesher; the caller owns the self-intersection / pinch-off guard.

**Opt-in import.** This module is deliberately *not* re-exported from ``vcell_fenics.core``:
importing it loads Netgen and caps its thread pool (below), and interleaving gmsh-OCC
meshing with Netgen in one long-lived process has shown instability (ADR 008). Keeping it
opt-in means gmsh-only code paths never load Netgen. Import it directly where you want it.
"""

from __future__ import annotations

from pathlib import Path

import basix.ufl
import numpy as np
import pyngcore
import ufl
from dolfinx.mesh import Mesh, create_mesh
from mpi4py import MPI
from numpy.typing import NDArray

# Netgen's default multi-threaded TaskManager keeps a worker pool that busy-waits in a
# long-lived process (e.g. a whole pytest session or a running solver), pinning cores and
# stalling progress. Our remeshes are small serial 2D meshes, so single-threaded netgen is
# both correct and faster. Cap the pool *before* importing the mesher (which starts it).
pyngcore.SetNumThreads(1)

from netgen.geom2d import SplineGeometry  # noqa: E402  (must follow SetNumThreads)

Floats = NDArray[np.float64]


def mesh_region_netgen(
    loop: Floats,
    h: float,
    *,
    fix_boundary_nodes: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Mesh:
    """Mesh the region enclosed by the closed polyline `loop` at target size `h`.

    Drop-in analogue of the gmsh `tests/gmsh_meshers/region_remesh.mesh_region` backed by
    LGPL Netgen (ADR 008). `loop` is an ordered ``(N, 2)`` array of boundary vertices in traversal
    order, *without* repeating the first vertex. Returns a 2D DOLFINx triangle mesh of
    the enclosed polygon whose boundary lies on `loop`.

    `fix_boundary_nodes=True` is not supported by the high-level Netgen mesher and
    raises `NotImplementedError`; use the default resampling path (and the full surface
    remap) or the gmsh `mesh_region` when the interior-only fast path is required.
    """

    loop = np.asarray(loop, dtype=np.float64)
    if loop.ndim != 2 or loop.shape[1] != 2:
        raise ValueError("loop must be an (N, 2) array of ordered boundary points")
    n = loop.shape[0]
    if n < 3:
        raise ValueError("loop must have at least 3 points to enclose an area")
    if bool(np.allclose(loop[0], loop[-1])):
        raise ValueError("loop must not repeat its first point as a closing vertex")
    if abs(_signed_area(loop)) < 1e-14:
        raise ValueError("loop encloses no area (degenerate or collinear)")
    if fix_boundary_nodes:
        raise NotImplementedError(
            "fix_boundary_nodes is not available in the high-level Netgen mesher "
            "(it resamples the boundary at h); it needs the low-level Element1D path "
            "(deferred, ADR 008). Use the gmsh mesh_region for the interior-only fast path."
        )

    # Netgen's `SplineGeometry` meshes the region to the **left** of each directed segment (default
    # `leftdomain=1, rightdomain=0`), so the boundary must run counter-clockwise (positive signed area).
    # A clockwise loop puts the meshed domain on the *outside* — Netgen then tries to mesh the unbounded
    # exterior and `GenerateMesh` spins indefinitely. The ALE driver's `ordered_membrane_loop` can hand us
    # either orientation depending on the facet ordering of the (remeshed) membrane, so normalise to CCW
    # here. (gmsh's plane-surface mesher is orientation-agnostic; this is a Netgen-specific requirement.)
    if _signed_area(loop) < 0.0:
        loop = loop[::-1]

    geo = SplineGeometry()
    pids = [geo.AppendPoint(float(x), float(y)) for x, y in loop]
    for i in range(n):
        geo.Append(["line", pids[i], pids[(i + 1) % n]], bc="bdry")
    ngmesh = geo.GenerateMesh(maxh=float(h))

    # Netgen point/cell arrays → DOLFINx mesh directly (no .msh, no ngsPETSc). Netgen
    # PointIds are 1-based; its 2D points carry a z=0 we drop.
    points = np.array([list(p.p)[:2] for p in ngmesh.Points()], dtype=np.float64)
    cells = np.array([[v.nr - 1 for v in el.vertices] for el in ngmesh.Elements2D()], dtype=np.int64)
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    return create_mesh(comm, cells, domain, points)


def _signed_area(loop: Floats) -> float:
    """Shoelace signed area of the closed polygon `loop` (CCW positive)."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class PinchOffError(RuntimeError):
    """The region is thinner somewhere than the finest allowed mesh can resolve (a neck closing toward a
    topological split) — raised *before* Netgen, which segfaults on such surfaces rather than failing."""


def local_thickness(points: Floats, triangles: NDArray[np.int64], radius: float) -> Floats:
    """At each surface vertex, the distance across the region to the facing side of the surface: the
    nearest vertex within ``radius`` that lies along this vertex's inward normal and whose own outward
    normal points on along that line (so a sharp crease, whose flanks also face each other, is not a neck).
    ``inf`` where the region is at least ``radius`` thick. A cheap medial-axis proxy — what a pinching neck
    needs, not an exact thickness."""

    from scipy.spatial import cKDTree

    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    face_normals = np.cross(b - a, c - a)  # area-weighted, outward
    normals = np.zeros_like(points)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    tree = cKDTree(points)
    thickness = np.full(len(points), np.inf)
    for i, near in enumerate(tree.query_ball_point(points, r=radius)):
        near = np.asarray(near, dtype=np.int64)
        near = near[near != i]
        if not near.size:
            continue
        offset = points[near] - points[i]
        distance = np.linalg.norm(offset, axis=1)
        direction = offset / distance[:, None]
        # across the region: the partner lies along this vertex's inward normal, and the partner's own
        # outward normal points on along that line — true across a neck, false across a sharp crease
        # (where neighbours on the two flanks also have opposing normals, but sideways)
        across = (direction @ -normals[i] > 0.8) & (np.einsum("ij,ij->i", direction, normals[near]) > 0.8)
        if across.any():
            thickness[i] = float(distance[across].min())
    return thickness


def write_stl(path: Path, points: Floats, triangles: NDArray[np.int64]) -> None:
    """Write an ASCII STL of the triangle surface ``(points, triangles)`` with per-facet normals."""

    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    lines = ["solid s"]
    for n, p, q, r in zip(normals, a, b, c, strict=True):
        lines.append(f"facet normal {n[0]:.17g} {n[1]:.17g} {n[2]:.17g}")
        lines.append("outer loop")
        lines.extend(f"vertex {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}" for v in (p, q, r))
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid s")
    path.write_text("\n".join(lines) + "\n")

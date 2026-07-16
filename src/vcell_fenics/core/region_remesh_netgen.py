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

"""Region remesher: a fresh quality mesh of the area enclosed by a polyline.

The ALE remesh driver (`docs/modeling/ale-remesh-driver.md`) survives large
deformation by *remeshing and continuing* instead of failing when the mesh tangles.
Step (b) of its remesh routine is `mesh_region(boundary, h)` — take the current
(deformed) boundary loop recovered from the tangled mesh and generate a fresh,
good-quality bulk mesh of the region it encloses. This module is that piece.

`mesh_region` drives gmsh from an explicit ordered polyline (geo kernel: one point
per loop vertex, straight line segments, one plane surface), so it meshes an
*arbitrary* deformed shape, not just the analytic disk the geometry helpers build.
Because the boundary segments are straight, any nodes gmsh inserts along them stay
on the polyline — so the meshed region is exactly the input polygon and its area is
preserved to round-off, regardless of `h`.

`fix_boundary_nodes=True` forces exactly the input vertices onto the boundary (two
nodes per segment, no subdivision), so Γ_new is the input polyline *exactly*
(Γ_new ⊂ Γ_old). That is the interior-only fast path of the driver sketch
(subtlety 3): when only the interior degraded, regenerate it with the boundary held
fixed and the surface trace is unchanged, so `correct_surface_trace` can be skipped.

Scope (v1): serial, 2D, a single simple (non-self-intersecting) closed loop. The
caller is responsible for the self-intersection / pinch-off guard (in Approach A a
pinch-off genuinely breaks the explicit-membrane representation). Higher dimensions,
multiple loops / holes, and MPI partitioning are deferred.
"""

from __future__ import annotations

import gmsh
import numpy as np
from dolfinx import mesh as dmesh
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI
from numpy.typing import NDArray

Floats = NDArray[np.float64]


def mesh_region(
    loop: Floats,
    h: float,
    *,
    fix_boundary_nodes: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> dmesh.Mesh:
    """Mesh the region enclosed by the closed polyline `loop` at target size `h`.

    `loop` is an ordered (N, 2) array of the boundary vertices in traversal order,
    *without* repeating the first vertex as a closing point. Returns a 2D DOLFINx
    triangle mesh of the enclosed polygon; its boundary (recoverable via
    `dmesh.exterior_facet_indices`) lies on `loop`.

    With `fix_boundary_nodes=True` the boundary carries exactly the input vertices
    (no inserted nodes), so the new boundary equals `loop` exactly — the
    interior-only fast path. By default gmsh resamples the boundary at `h` (the
    inserted nodes still lie on the polyline, so the region is unchanged).
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

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # Make `h` the authoritative uniform target size: don't inherit the (possibly
    # very non-uniform) spacing of the deformed input boundary, and don't refine by
    # curvature — a remesh wants a clean uniform mesh, not the old mesh's density.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    try:
        gmsh.model.add("region")
        pts = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in loop]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % n]) for i in range(n)]
        if fix_boundary_nodes:
            # Two nodes per segment = just its endpoints, so no boundary subdivision:
            # Γ_new is exactly the input polyline.
            for line in lines:
                gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)
        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="bulk")
        gmsh.model.addPhysicalGroup(1, lines, tag=2, name="boundary")

        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(2)

        data = model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
    finally:
        gmsh.finalize()

    return data.mesh


def _signed_area(loop: Floats) -> float:
    """Shoelace signed area of the closed polygon `loop` (CCW positive)."""
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

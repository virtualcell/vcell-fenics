"""Conservative bulk (2D area) interpolation between triangulations.

The area sibling of the surface remap (`surface_remap.py`): transfer a P1 field `c`
from one 2D triangulation to another while preserving the volume integral ∫_Ω c dx
*exactly* (to linear-solve round-off), by local Galerkin projection over the
**supermesh** — the common refinement of the two meshes (Farrell–Maddison 2011).

Where the 1D surface remap merges interval breakpoints, the 2D supermesh is built
by clipping every new triangle against the old triangles it overlaps (two convex
triangles → a convex intersection polygon, Sutherland–Hodgman), and integrating
products of P1 basis functions over those polygons. The assembled pieces are:

    M  = the true new-mesh mass matrix          (∫ φ_new_j φ_new_k dx, over full new triangles)
    B  = the mixed mass matrix over the supermesh (∫ φ_new_j φ_old_i dx, over new∩old)

and `c_new = M⁻¹ B c_old`. Conservation is structural, exactly as in 1D: the new P1
basis is a partition of unity, so 1ᵀ M c_new = 1ᵀ B c_old = ∫_Ω c_old dx. (When the
two meshes triangulate the *same* polygon the supermesh tiles each new triangle
exactly and conservation is to round-off; when their boundaries differ slightly,
the uncovered slivers are the geometric gap, the area analogue of the surface case.)

This is the kernel `transfer_bulk` referenced by the ALE remesh driver sketch
(`docs/modeling/ale-remesh-driver.md`). It is pure NumPy + scipy.sparse and operates
on plain mesh arrays; the DOLFINx `Function` bridge (extract arrays from a mesh,
write `c_new` back) is a later increment, as for the surface remap. P1 only;
broad-phase is bounding-box rejection (fine for the v1 mesh sizes, not optimized for
large meshes).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

Floats = NDArray[np.float64]
Ints = NDArray[np.intp]

_EPS = 1e-12
# Edge-midpoint quadrature on a triangle, in barycentric coordinates — exact for
# degree 2, which is the degree of a P1×P1 basis product. Weights 1/3 each.
_MIDPOINTS: Floats = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])


def supermesh_project_2d(old_verts: Floats, old_tris: Ints, c_old: Floats, new_verts: Floats, new_tris: Ints) -> Floats:
    """Conservatively project a P1 field from one 2D triangulation to another.

    `old_verts` / `new_verts` are (N, 2) vertex arrays, `old_tris` / `new_tris` are
    (M, 3) vertex-index arrays, `c_old` holds the P1 nodal values at `old_verts`.
    Returns `c_new` at `new_verts` with ∫_Ω c_new dx == ∫_Ω c_old dx (exactly when
    the two meshes triangulate the same polygon).
    """

    if c_old.shape != (old_verts.shape[0],):
        raise ValueError("c_old must have one value per old vertex")
    if old_verts.shape[1] != 2 or new_verts.shape[1] != 2:
        raise ValueError("supermesh_project_2d is 2D: vertex arrays must be (N, 2)")

    mass = _new_mass_matrix(new_verts, new_tris)
    mixed = _mixed_mass_matrix(old_verts, old_tris, new_verts, new_tris)
    return cast(Floats, spsolve(mass, mixed @ c_old))


def _new_mass_matrix(verts: Floats, tris: Ints) -> Any:
    """The true P1 mass matrix ∫ φ_j φ_k dx on (verts, tris), assembled per triangle
    from the analytic local matrix (Area/12)·[[2,1,1],[1,2,1],[1,1,2]]."""

    local = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    rows, cols, vals = [], [], []
    for tri in tris:
        a, b, c = verts[tri]
        area = 0.5 * abs(float(np.cross(b - a, c - a)))
        block = local * (area / 12.0)
        for i in range(3):
            for j in range(3):
                rows.append(int(tri[i]))
                cols.append(int(tri[j]))
                vals.append(block[i, j])
    n = verts.shape[0]
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def _mixed_mass_matrix(old_verts: Floats, old_tris: Ints, new_verts: Floats, new_tris: Ints) -> Any:
    """The mixed mass matrix ∫ φ_new_j φ_old_i dx over the supermesh, assembled by
    clipping each new triangle against the old triangles it overlaps."""

    old_lo = np.stack([old_verts[tri].min(axis=0) for tri in old_tris])
    old_hi = np.stack([old_verts[tri].max(axis=0) for tri in old_tris])

    rows, cols, vals = [], [], []
    for new_tri in new_tris:
        n_pts = new_verts[new_tri]  # (3, 2)
        n_lo, n_hi = n_pts.min(axis=0), n_pts.max(axis=0)
        # Broad phase: old triangles whose bounding box overlaps this new triangle's.
        overlap = ~(
            (old_hi[:, 0] < n_lo[0]) | (old_lo[:, 0] > n_hi[0]) | (old_hi[:, 1] < n_lo[1]) | (old_lo[:, 1] > n_hi[1])
        )
        for old_tri in old_tris[overlap]:
            o_pts = old_verts[old_tri]
            polygon = _clip_triangle(n_pts, _ccw(o_pts))
            if len(polygon) < 3:
                continue
            for tri in _fan(polygon):
                area = 0.5 * abs(float(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                if area < _EPS:
                    continue
                for bary in _MIDPOINTS:
                    point = bary @ tri
                    lam = _barycentric(point, n_pts)  # P1 new basis at the quad point
                    mu = _barycentric(point, o_pts)  # P1 old basis at the quad point
                    weight = area / 3.0
                    for i in range(3):
                        for j in range(3):
                            rows.append(int(new_tri[i]))
                            cols.append(int(old_tri[j]))
                            vals.append(weight * lam[i] * mu[j])
    return coo_matrix((vals, (rows, cols)), shape=(new_verts.shape[0], old_verts.shape[0])).tocsr()


def _ccw(tri: Floats) -> Floats:
    """Triangle vertices reordered counter-clockwise (so the clip half-planes point
    inward for the Sutherland–Hodgman inside test)."""
    return tri[::-1] if float(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 0.0 else tri


def _clip_triangle(subject: Floats, clip: Floats) -> list[Floats]:
    """Sutherland–Hodgman intersection of a subject triangle with a CCW clip
    triangle. Returns the convex intersection polygon's vertices (≤ 6)."""

    output: list[Floats] = list(subject)
    for k in range(3):
        a, b = clip[k], clip[(k + 1) % 3]
        edge = b - a
        current, output = output, []
        if not current:
            break
        prev = current[-1]
        prev_in = float(np.cross(edge, prev - a)) >= -_EPS
        for point in current:
            point_in = float(np.cross(edge, point - a)) >= -_EPS
            if point_in:
                if not prev_in:
                    output.append(_line_intersect(a, b, prev, point))
                output.append(point)
            elif prev_in:
                output.append(_line_intersect(a, b, prev, point))
            prev, prev_in = point, point_in
    return output


def _line_intersect(a: Floats, b: Floats, s: Floats, e: Floats) -> Floats:
    """Intersection of the infinite line a→b with the segment s→e (which is known to
    cross it, from the Sutherland–Hodgman caller)."""
    edge, seg = b - a, e - s
    t = float(np.cross(edge, s - a)) / -float(np.cross(edge, seg))
    return s + t * seg


def _fan(polygon: list[Floats]) -> list[Floats]:
    """Fan-triangulate a convex polygon from its first vertex."""
    p = polygon
    return [np.stack((p[0], p[m], p[m + 1])) for m in range(1, len(p) - 1)]


def _barycentric(point: Floats, tri: Floats) -> Floats:
    """Barycentric coordinates of `point` in triangle `tri` (the P1 hat values at
    `point`), ordered to match `tri`'s vertices. Valid for either orientation."""
    a, b, c = tri
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = float(v0 @ v0), float(v0 @ v1), float(v1 @ v1)
    d20, d21 = float(v2 @ v0), float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])

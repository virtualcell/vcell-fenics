"""Verification of the conservative bulk (2D) supermesh projection.

Two triangulations of the *same* unit square (so the supermesh tiles each new
triangle exactly and conservation is to round-off), following the design's
verification plan:

1. **Conservation** — ∫_Ω c dx is preserved to round-off for arbitrary fields.
2. **Constant preservation** — a uniform field stays uniform (partition of unity).
3. **Linear preservation** — a globally linear field (in both P1 spaces) is
   reproduced exactly at the new nodes.
4. **Accuracy (refinement)** — for a smooth field, the L2 error falls ~2nd order.
5. **Negative control** — a nearest-vertex copy drifts mass.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.core import supermesh_project_2d

Floats = np.ndarray


def _square_mesh(n: int) -> tuple[Floats, np.ndarray]:
    """Triangulation of the unit square [0,1]^2 on an n x n grid (two triangles per
    cell). The geometric boundary is the same square for every n, so two such meshes
    triangulate an identical domain."""
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    verts = np.column_stack((gx.ravel(), gy.ravel()))

    def vid(i: int, j: int) -> int:
        return j * (n + 1) + i

    tris = []
    for j in range(n):
        for i in range(n):
            v00, v10, v01, v11 = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    return verts, np.asarray(tris, dtype=np.intp)


def _mass(verts: Floats, tris: np.ndarray, c: Floats) -> float:
    """∫_Ω c dx for a P1 field (sum of Area/3 * mean of the three nodal values)."""
    total = 0.0
    for tri in tris:
        a, b, cc = verts[tri]
        area = 0.5 * abs(float(np.cross(b - a, cc - a)))
        total += area * float(np.mean(c[tri]))
    return total


# ---------------------------------------------------------------------------
# 1. conservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 4])
def test_mass_conserved_to_roundoff(seed: int) -> None:
    ov, ot = _square_mesh(5)
    nv, nt = _square_mesh(7)
    rng = np.random.default_rng(seed)
    c_old = np.asarray(rng.uniform(0.1, 5.0, size=ov.shape[0]))

    c_new = supermesh_project_2d(ov, ot, c_old, nv, nt)

    assert _mass(nv, nt, c_new) == pytest.approx(_mass(ov, ot, c_old), rel=1e-11)


# ---------------------------------------------------------------------------
# 2 + 3. constant and linear preservation
# ---------------------------------------------------------------------------


def test_constant_preserved_exactly() -> None:
    ov, ot = _square_mesh(4)
    nv, nt = _square_mesh(6)
    c_new = supermesh_project_2d(ov, ot, np.full(ov.shape[0], 3.25), nv, nt)
    assert np.allclose(c_new, 3.25, atol=1e-11)


def test_linear_field_reproduced_exactly() -> None:
    ov, ot = _square_mesh(5)
    nv, nt = _square_mesh(8)

    def field(xy: Floats) -> Floats:
        return 0.7 + 1.3 * xy[:, 0] - 0.4 * xy[:, 1]

    c_new = supermesh_project_2d(ov, ot, field(ov), nv, nt)
    assert np.allclose(c_new, field(nv), atol=1e-10)


# ---------------------------------------------------------------------------
# 4. accuracy under refinement
# ---------------------------------------------------------------------------


def test_smooth_field_converges_second_order() -> None:
    def f(xy: Floats) -> Floats:
        return np.sin(2.0 * xy[:, 0]) * np.cos(1.5 * xy[:, 1])

    errors = []
    for n in (6, 12, 24):
        ov, ot = _square_mesh(n)
        nv, nt = _square_mesh(3 * n // 2)
        c_new = supermesh_project_2d(ov, ot, f(ov), nv, nt)
        errors.append(float(np.max(np.abs(c_new - f(nv)))))

    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert min(rates) > 1.7  # ~second order


# ---------------------------------------------------------------------------
# 5. negative control
# ---------------------------------------------------------------------------


def test_nearest_vertex_copy_drifts_mass() -> None:
    ov, ot = _square_mesh(5)
    nv, nt = _square_mesh(8)
    rng = np.random.default_rng(2)
    c_old = np.asarray(rng.uniform(0.5, 4.0, size=ov.shape[0]))

    nearest = np.argmin(np.linalg.norm(nv[:, None, :] - ov[None, :, :], axis=2), axis=1)
    c_nearest = c_old[nearest]
    c_conservative = supermesh_project_2d(ov, ot, c_old, nv, nt)

    m_old = _mass(ov, ot, c_old)
    assert _mass(nv, nt, c_conservative) == pytest.approx(m_old, rel=1e-11)
    assert abs(_mass(nv, nt, c_nearest) - m_old) / m_old > 1e-3

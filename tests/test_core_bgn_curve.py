"""Verification of BGN tangential mesh redistribution (core/bgn_curve).

`bgn_curvature_flow_step(points, mobility=m, dt=dt)` advances a closed membrane
polyline by one semi-implicit BGN step of mean-curvature flow (V = −m κ), with the
tangential node motion that keeps the mesh well-distributed falling out of the
coupled (position, curvature) solve. The checks pin known-answer geometry:

1. **Circle shrink law** — a circle stays a circle and shrinks per the exact
   `r² = r₀² − 2 m t`; the nodes stay equidistributed by symmetry (r_std ≈ 0).
2. **Curve-shortening area law** — *any* convex curve loses area at the constant
   rate `dA/dt = −2π m`, independent of shape (the classic result); checked on an
   ellipse, which is the discriminating non-circular case.
3. **Redistribution is the point** — on an ellipse the BGN edge-length ratio stays
   bounded (mesh maintained) where a naive normal-only step bunches nodes at the
   high-curvature tips an order of magnitude worse.
4. **Input guards** — degenerate loops / parameters are rejected loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray

from vcell_fenics.core import bgn_curvature_flow_step, polygon_area

Floats = NDArray[np.float64]


def _ellipse(n: int, a: float, b: float) -> Floats:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))


def _edge_ratio(points: Floats) -> float:
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float(lengths.max() / lengths.min())


def _naive_normal_step(points: Floats, mobility: float, dt: float) -> Floats:
    """Explicit normal-only motion X ← X − m·dt·κ⃗ (the no-redistribution baseline):
    the same discrete curvature vector κ⃗ = M⁻¹ A X, moved without the tangential
    freedom BGN adds. Used only to demonstrate the bunching BGN avoids."""

    n = len(points)
    nxt = np.roll(np.arange(n), -1)
    prv = np.roll(np.arange(n), 1)
    elen = np.linalg.norm(points[nxt] - points, axis=1)
    mass = 0.5 * (elen + elen[prv])
    w = 1.0 / elen
    rows = np.concatenate([np.arange(n), nxt, np.arange(n)])
    cols = np.concatenate([np.arange(n), np.arange(n), nxt])
    vals = np.concatenate([w + w[prv], -w, -w])
    stiffness = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    kappa_vec = np.asarray(stiffness @ points) / mass[:, None]  # κ⃗ = κ·outward-normal
    moved: Floats = points - mobility * dt * kappa_vec
    return moved


def test_circle_shrinks_per_exact_law() -> None:
    n, mob, dt, steps = 80, 0.1, 0.01, 20
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = np.column_stack((np.cos(t), np.sin(t)))  # unit circle, r0 = 1
    for _ in range(steps):
        points = bgn_curvature_flow_step(points, mobility=mob, dt=dt)

    radii = np.linalg.norm(points, axis=1)
    expected = np.sqrt(1.0 - 2.0 * mob * dt * steps)  # r² = r₀² − 2 m t
    assert radii.mean() == pytest.approx(expected, rel=1e-3)
    assert radii.std() < 1e-10  # stays perfectly round — symmetry preserved


def test_convex_area_decreases_at_curve_shortening_rate() -> None:
    # dA/dt = −2π m for any convex curve, regardless of shape (here an ellipse).
    points = _ellipse(160, a=2.0, b=0.8)
    mob, dt, steps = 0.05, 0.005, 40
    area0 = polygon_area(points)
    for _ in range(steps):
        points = bgn_curvature_flow_step(points, mobility=mob, dt=dt)

    expected_drop = -2.0 * np.pi * mob * (dt * steps)
    assert polygon_area(points) - area0 == pytest.approx(expected_drop, rel=1e-2)


def test_redistribution_keeps_mesh_bounded_where_naive_bunches() -> None:
    # The reason BGN exists: pure normal motion crowds nodes at the ellipse tips,
    # while BGN's tangential freedom keeps the spacing bounded.
    start = _ellipse(120, a=2.0, b=0.6)
    mob, dt, steps = 0.1, 0.01, 150
    ratio0 = _edge_ratio(start)

    bgn = start.copy()
    naive = start.copy()
    for _ in range(steps):
        bgn = bgn_curvature_flow_step(bgn, mobility=mob, dt=dt)
        naive = _naive_normal_step(naive, mobility=mob, dt=dt)

    bgn_ratio, naive_ratio = _edge_ratio(bgn), _edge_ratio(naive)
    assert bgn_ratio <= ratio0  # BGN does not worsen the distribution (it improves it)
    assert naive_ratio > 3.0 * bgn_ratio  # naive bunches dramatically worse


def test_input_guards() -> None:
    good = _ellipse(20, 1.0, 1.0)
    with pytest.raises(ValueError, match="N >= 3"):
        bgn_curvature_flow_step(good[:2], mobility=0.1, dt=0.01)
    with pytest.raises(ValueError, match="positive"):
        bgn_curvature_flow_step(good, mobility=-0.1, dt=0.01)
    with pytest.raises(ValueError, match="positive"):
        bgn_curvature_flow_step(good, mobility=0.1, dt=0.0)
    with pytest.raises(ValueError, match="coincident-node"):
        coincident = np.vstack([good[0], good])  # a duplicated first node → zero edge
        bgn_curvature_flow_step(coincident, mobility=0.1, dt=0.01)

"""Verification of the 1D conservative surface-density remap (core/surface_remap).

Follows the verification plan in docs/modeling/conservative-surface-remap.md and
the project's established pattern (analytical check + refinement + negative
control):

1. **Conservation** — total mass ∫_Γ ρ ds is preserved to round-off, for arbitrary
   old→new node distributions.
2. **Constant preservation** — a uniform ρ remaps to the same uniform value
   exactly (the partition-of-unity property).
3. **Refinement exactness / accuracy** — if the new mesh refines the old (node set
   a superset), the remap is exact; for a smooth field the L2 error falls under
   refinement.
4. **Negative control** — the naive nearest-node copy is *not* conservative, so the
   supermesh step is demonstrably load-bearing.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from vcell_fenics.core import arclength_parameterization, supermesh_remap_1d

Floats = NDArray[np.float64]


def _circle_nodes(n: int, *, phase: float = 0.0, radius: float = 1.0) -> Floats:
    """n points evenly (in angle) around a circle, offset by `phase` radians."""
    theta = phase + np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return radius * np.column_stack((np.cos(theta), np.sin(theta)))


def _uneven_loop_coords(n: int, length: float, *, seed: int) -> Floats:
    """n strictly-increasing arc-length coordinates in [0, length), non-uniform."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.5, 1.5, size=n)
    s = np.concatenate(([0.0], np.cumsum(gaps)[:-1]))
    return cast(Floats, s / s[-1] * length * (n - 1) / n)  # keep the last node strictly < length


def _mass(s: Floats, rho: Floats, length: float) -> float:
    """∫_Γ ρ ds for P1 ρ on a closed loop (trapezoid over each segment incl. wrap)."""
    s_ext = np.concatenate((s, [s[0] + length]))
    rho_ext = np.concatenate((rho, [rho[0]]))
    seg = np.diff(s_ext)
    return float(np.sum(0.5 * (rho_ext[:-1] + rho_ext[1:]) * seg))


# ---------------------------------------------------------------------------
# 0. arc-length parameterization
# ---------------------------------------------------------------------------


def test_arclength_of_circle_matches_perimeter() -> None:
    n = 256
    s, length = arclength_parameterization(_circle_nodes(n), closed=True)
    assert s.shape == (n,)
    assert s[0] == 0.0
    assert s[-1] < length
    # A fine inscribed polygon's perimeter approaches 2πr.
    assert length == pytest.approx(2.0 * np.pi, rel=1e-3)


# ---------------------------------------------------------------------------
# 1. conservation (round-off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_mass_conserved_to_roundoff(seed: int) -> None:
    length = 10.0
    s_old = _uneven_loop_coords(40, length, seed=seed)
    s_new = _uneven_loop_coords(57, length, seed=seed + 100)
    rng = np.random.default_rng(seed)
    rho_old = rng.uniform(0.1, 5.0, size=s_old.size)

    rho_new = supermesh_remap_1d(s_old, rho_old, s_new, length)

    assert _mass(s_new, rho_new, length) == pytest.approx(_mass(s_old, rho_old, length), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. constant preservation
# ---------------------------------------------------------------------------


def test_constant_field_preserved_exactly() -> None:
    length = 7.5
    s_old = _uneven_loop_coords(31, length, seed=3)
    s_new = _uneven_loop_coords(48, length, seed=99)
    rho_old = np.full(s_old.size, 2.5)

    rho_new = supermesh_remap_1d(s_old, rho_old, s_new, length)

    assert np.allclose(rho_new, 2.5, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. refinement exactness and accuracy
# ---------------------------------------------------------------------------


def test_refinement_is_exact() -> None:
    # New node set is a superset of the old (each old segment bisected), so the old
    # P1 field lies in the new space and the remap reproduces it exactly.
    length = 6.0
    s_old = np.linspace(0.0, length, 12, endpoint=False)
    midpoints = 0.5 * (s_old + np.concatenate((s_old[1:], [s_old[0] + length])))
    s_new = np.sort(np.concatenate((s_old, midpoints % length)))
    rng = np.random.default_rng(5)
    rho_old = rng.uniform(0.0, 3.0, size=s_old.size)

    rho_new = supermesh_remap_1d(s_old, rho_old, s_new, length)

    # Old nodes survive in the new mesh: their values must be unchanged.
    idx = np.searchsorted(s_new, s_old)
    assert np.allclose(rho_new[idx], rho_old, atol=1e-12)


def test_smooth_field_converges_second_order() -> None:
    # Refine *both* meshes together: the remap of a smooth field then converges to
    # the exact field at the new nodes. (Refining only the target floors the error
    # at the source-mesh interpolation error — it is the source resolution, not the
    # target's, that bounds accuracy.) Expect ~O(h^2) for P1.
    length = 2.0 * np.pi

    errors = []
    for n in (24, 48, 96):
        s_old = np.linspace(0.0, length, n, endpoint=False)
        rho_old = np.sin(s_old)
        # A distinct, phase-shifted target mesh of comparable resolution.
        s_new = np.sort(np.linspace(0.013, length + 0.013, 3 * n // 2, endpoint=False) % length)
        rho_new = supermesh_remap_1d(s_old, rho_old, s_new, length)
        errors.append(float(np.max(np.abs(rho_new - np.sin(s_new)))))

    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert min(rates) > 1.8  # second-order convergence


# ---------------------------------------------------------------------------
# 4. negative control — nearest-node copy is not conservative
# ---------------------------------------------------------------------------


def test_nearest_node_copy_drifts_mass() -> None:
    length = 10.0
    s_old = _uneven_loop_coords(20, length, seed=11)
    s_new = _uneven_loop_coords(33, length, seed=222)
    rng = np.random.default_rng(11)
    rho_old = rng.uniform(0.5, 4.0, size=s_old.size)

    # Naive transfer: each new node takes the value of the nearest old node.
    nearest = np.argmin(np.abs(s_new[:, None] - s_old[None, :]), axis=1)
    rho_nearest = rho_old[nearest]
    conservative = supermesh_remap_1d(s_old, rho_old, s_new, length)

    m_old = _mass(s_old, rho_old, length)
    assert _mass(s_new, conservative, length) == pytest.approx(m_old, rel=1e-12)
    # The naive copy visibly fails to conserve — the supermesh step is load-bearing.
    assert abs(_mass(s_new, rho_nearest, length) - m_old) / m_old > 1e-3

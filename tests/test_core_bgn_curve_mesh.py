"""Verification of the BGN DOLFINx mesh bridge (core/bgn_curve_mesh).

`bgn_redistribute_membrane(mesh, mobility=m, dt=dt)` advances a live closed P1
membrane `Mesh` by one BGN curvature-flow step, in place — the bridge from the
pure-NumPy `bgn_curve` kernel to real DOLFINx mesh motion. The checks run it on a
membrane extracted from a disk geometry:

1. **Circle shrink law on a real mesh** — the membrane shrinks per the exact
   `r² = r₀² − 2 m t`, stays round, keeps its nodes equidistributed, and runs at a
   `dt` well past where the velocity-based curvature path degrades — with no
   `MeshQualityError`.
2. **Redistribution on a deformed (elliptical) membrane** — the edge-length ratio
   stays bounded over a long run where pure normal motion would bunch the tips.
3. **Guard** — a non-membrane (2D) mesh is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem

from vcell_fenics.backend import make_disk_geometry, make_disk_membrane_geometry
from vcell_fenics.core import bgn_redistribute_membrane, ordered_membrane_loop


def _membrane_mesh(h: float = 0.06):  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=h).mesh_of("mem")


def _edge_ratio(mesh) -> float:  # type: ignore[no-untyped-def]
    loop, _ = ordered_membrane_loop(fem.functionspace(mesh, ("Lagrange", 1)))
    lengths = np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1)
    return float(lengths.max() / lengths.min())


def test_circle_shrinks_on_real_mesh() -> None:
    mesh = _membrane_mesh()
    mob, dt, steps = 0.1, 0.02, 20  # dt = 4× what the velocity-based path needed
    for _ in range(steps):
        bgn_redistribute_membrane(mesh, mobility=mob, dt=dt)

    radii = np.linalg.norm(mesh.geometry.x[:, :2], axis=1)
    expected = np.sqrt(1.0 - 2.0 * mob * dt * steps)  # r² = r₀² − 2 m t
    assert radii.mean() == pytest.approx(expected, rel=1e-3)
    assert radii.std() < 1e-9  # stays round
    assert _edge_ratio(mesh) == pytest.approx(1.0, abs=1e-6)  # nodes stay equidistributed


def test_redistribution_keeps_deformed_membrane_bounded() -> None:
    mesh = _membrane_mesh(h=0.05)
    mesh.geometry.x[:, 0] *= 2.0  # stretch the circle into an ellipse (a=2, b=1)
    ratio0 = _edge_ratio(mesh)
    assert ratio0 > 1.5  # the stretch genuinely un-equidistributes the nodes

    for _ in range(120):
        bgn_redistribute_membrane(mesh, mobility=0.1, dt=0.01)

    # BGN does not let the distortion grow — it improves it (where naive bunches).
    assert _edge_ratio(mesh) <= ratio0


def test_rejects_non_membrane_mesh() -> None:
    bulk = make_disk_geometry("g", volume_subdomain="cell", radius=1.0, h=0.2).mesh_of("cell")
    with pytest.raises(ValueError, match="codim-1 membrane"):
        bgn_redistribute_membrane(bulk, mobility=0.1, dt=0.01)

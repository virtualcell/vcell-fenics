"""Verification of the prescribed-motion FSI loop (backend/fsi) — the dynamic integration.

`step_prescribed_fsi` advances one step of a moving membrane driving a conservative H(div)
bulk: solve the exactly-divergence-free fluid (`v·n = w·n`), then move the ALE mesh by
`dt·w`. The whole point of building it on H(div) is that incompressibility holds *as the
domain deforms*, where the Taylor–Hood Nitsche slip leaked ~3%.

1. **Exactly incompressible throughout** — `∇·v` is zero to round-off at *every* step of a
   deforming loop, not just the first.
2. **Volume conserved** — a divergence-free prescribed motion (`∮ w·n = 0` on any shape)
   keeps the enclosed area constant to the O(dt) forward-Euler integration error (and it
   refines with `dt`), where the leaky bulk drifted percent-level.
3. **Stable** — the mesh stays valid (no inversion) over the loop.
"""

from __future__ import annotations

import ufl
from dolfinx import fem

from vcell_fenics.backend import enclosed_volume, make_disk_geometry, step_prescribed_fsi


def _divergence_free_strain(mesh, rate: float) -> fem.Function:  # type: ignore[no-untyped-def]
    # w = rate·(x, −y): ∇·w = rate − rate = 0 ⇒ ∮ w·n = 0 on any shape (area-preserving).
    space = fem.functionspace(mesh, ("Lagrange", 2, (2,)))
    x = ufl.SpatialCoordinate(mesh)
    w = fem.Function(space)
    w.interpolate(fem.Expression(ufl.as_vector([rate * x[0], -rate * x[1]]), space.element.interpolation_points))
    return w


def _max_cell_volume_ratio(mesh) -> float:  # type: ignore[no-untyped-def]
    dg0 = fem.functionspace(mesh, ("DG", 0))
    volumes = fem.assemble_vector(fem.form(ufl.TestFunction(dg0) * ufl.dx(domain=mesh))).array
    return float(volumes.min())  # >0 ⇒ no inverted cell


def _run(dt: float, n_steps: int, *, rate: float = 0.3, h: float = 0.06) -> tuple[float, float, bool]:
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=h).mesh_of("c")
    volume0 = enclosed_volume(mesh)
    max_div = 0.0
    valid = True
    for _ in range(n_steps):
        v, _ = step_prescribed_fsi(mesh, boundary_velocity=_divergence_free_strain(mesh, rate), dt=dt)
        div = fem.assemble_scalar(fem.form(ufl.div(v) ** 2 * ufl.dx(domain=v.function_space.mesh)))
        max_div = max(max_div, float(div.real) ** 0.5)
        if _max_cell_volume_ratio(mesh) <= 0.0:
            valid = False
            break
    return max_div, abs(enclosed_volume(mesh) - volume0) / volume0, valid


def test_fsi_loop_is_exactly_incompressible_as_the_domain_deforms() -> None:
    # ∇·v at round-off at every step of the deforming loop — the H(div) win, dynamically.
    max_div, _, valid = _run(dt=0.02, n_steps=15)
    assert valid
    assert max_div < 1e-8  # exact incompressibility throughout (Taylor-Hood leaked ~3%)


def test_fsi_loop_conserves_volume() -> None:
    # A divergence-free prescribed motion preserves the enclosed area to the forward-Euler
    # O(dt) integration error over a fixed total time — halving dt (same total time) halves
    # the drift, confirming it is the time-stepping error, not a fluid mass leak.
    coarse = _run(dt=0.02, n_steps=10)[1]
    fine = _run(dt=0.01, n_steps=20)[1]
    assert coarse < 5e-3  # far below the percent-level drift of the leaky (Taylor-Hood) bulk
    assert fine < 0.6 * coarse  # O(dt): refining dt reduces the drift

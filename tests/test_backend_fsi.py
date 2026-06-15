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

`step_force_balance_fsi` is the **closure**: the membrane moves under its *own* surface
tension and the bulk pressure (no prescribed motion), on Taylor–Hood so the ALE mesh moves
by the solved `v` directly with `∮ v·n = 0` exact.

4. **Laplace fixed point** — a circle under uniform tension is an equilibrium: the bulk
   returns `p = γ/R` and `v ≈ 0`, so the shape (and area) barely moves.
5. **Relaxation** — a perturbed (elliptical) shape relaxes toward the minimal-perimeter
   circle: the perimeter decreases monotonically and the area is conserved to O(dt), with the
   pressure enforcing `∮ v·n = 0` automatically (no prescribed-motion consistency to arrange).
"""

from __future__ import annotations

import math

import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    enclosed_volume,
    make_disk_geometry,
    step_force_balance_fsi,
    step_prescribed_fsi,
)


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


# --- force-balance closure: the membrane moves under its own tension (no prescribed motion) ---


def _perimeter(mesh) -> float:  # type: ignore[no-untyped-def]
    from mpi4py import MPI

    local = fem.assemble_scalar(fem.form(1.0 * ufl.ds(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))


def _stretch_into_ellipse(mesh, a: float) -> None:  # type: ignore[no-untyped-def]
    # (x, y) → (a·x, y/a): area-preserving, so any area drift is the solver's, not the seed.
    mesh.geometry.x[:, 0] *= a
    mesh.geometry.x[:, 1] /= a


def test_force_balance_circle_is_a_laplace_fixed_point() -> None:
    # A circle under uniform tension is an equilibrium: the bulk returns the Laplace pressure
    # p = γ/R (here γ=0.5, R=1 ⇒ 0.5) and the velocity is ~0, so the shape barely moves.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    area0 = enclosed_volume(mesh)
    v, p = step_force_balance_fsi(mesh, tension=0.5, dt=0.02)
    p_integral = fem.assemble_scalar(fem.form(p * ufl.dx(domain=p.function_space.mesh)))
    p_mean = float(p_integral.real) / area0
    max_speed = float(abs(v.x.array).max())
    assert abs(p_mean - 0.5) < 2e-2  # Laplace law p = γ/R
    assert max_speed < 5e-2  # equilibrium: essentially no flow
    assert abs(enclosed_volume(mesh) - area0) / area0 < 1e-3  # fixed point, area held


def test_force_balance_ellipse_relaxes_toward_the_circle() -> None:
    # Out of equilibrium, tension drives the shape toward minimal perimeter at conserved area:
    # the perimeter decreases monotonically toward the circle's, and the area is held to O(dt).
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    area0 = enclosed_volume(mesh)
    circle_perimeter = 2.0 * math.pi * math.sqrt(area0 / math.pi)

    perimeter = _perimeter(mesh)
    for _ in range(25):
        step_force_balance_fsi(mesh, tension=0.5, dt=0.02)
        nxt = _perimeter(mesh)
        assert nxt < perimeter + 1e-9  # monotone decrease toward the minimal-perimeter circle
        perimeter = nxt

    assert perimeter < _perimeter_of_initial_ellipse(area0, a=1.3)  # actually relaxed
    assert perimeter > circle_perimeter - 1e-3  # toward, not past, the circle
    assert abs(enclosed_volume(mesh) - area0) / area0 < 1e-2  # area conserved to O(dt)


def _perimeter_of_initial_ellipse(area0: float, a: float) -> float:
    # Ramanujan's approximation, just as a strictly-greater-than reference for "relaxed".
    semi_major, semi_minor = a * math.sqrt(area0 / math.pi), math.sqrt(area0 / math.pi) / a
    h = ((semi_major - semi_minor) / (semi_major + semi_minor)) ** 2
    return math.pi * (semi_major + semi_minor) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))

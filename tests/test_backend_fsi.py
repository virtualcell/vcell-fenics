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

`step_two_phase_fsi` is the same closure for a **two-phase** (network + solvent) incompressible
mixture with interphase drag: the membrane tension loads the mixture, the mesh follows a chosen
*frame* phase (the mixture average conserves volume via the shared pressure).

6. **Two-phase Laplace fixed point** — a circle is an equilibrium: the *physical* mixture
   pressure `2p = γ/R` (p is the multiplier conjugate to the total velocity), both phases at
   rest, area held.
7. **Two-phase relaxation** — a perturbed ellipse relaxes toward the circle, area conserved.
8. **Drag locks the phases** — with asymmetric phase viscosities the relative slip
   `|v_n − v_s|` shrinks as the interphase drag `ξ` grows.

`step_two_phase_fsi_with_species` carries a **co-moving volume species** `c` through the moving
cell: the species rides a physical phase (`carrier`) while the mesh moves at the bookkeeping
velocity, transported by `relative_advection` + the bulk dilution `c ∇·v_carrier`.

9.  **Uniform stays uniform on the incompressible mixture** — the divergence-free mixture
    carrier injects no spurious source, so a constant species stays constant (round-off).
10. **Conserved during relaxation** — `∫c` is held to O(dt) as the membrane relaxes.
11. **Dilution does real work (bulk analogue of `ρ ∇_Γ·v_Γ`)** — a uniform species on the
    *compressible* network phase concentrates/dilutes (mass conserved); the negative control
    that drops `c ∇·v_carrier` leaves it spuriously uniform.

`step_two_phase_fsi_with_reacting_species` carries **several reacting volume species** (a vector
space) coupled by a linear reaction `R(c)`, transported by the same flow:

12. **Conservative conversion `A ⇌ B`** — the total `∫(A + B)` is conserved while the reaction
    shifts mass A→B (the headline multi-species + reaction check on the moving cell).
13. **Detailed-balance equilibrium** — a uniform `A, B` relaxes to `A/B = k_off/k_on`.
14. **Linear decay matches backward Euler** — a uniform decaying species follows the exact
    `c₀/(1 + k·dt)^steps`, staying uniform.
"""

from __future__ import annotations

import math

import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from vcell_fenics.backend import (
    enclosed_volume,
    make_disk_geometry,
    step_force_balance_fsi,
    step_prescribed_fsi,
    step_two_phase_fsi,
    step_two_phase_fsi_with_reacting_species,
    step_two_phase_fsi_with_species,
)
from vcell_fenics.backend.fsi import _apply_displacement, _harmonic_displacement, _phase_velocity
from vcell_fenics.backend.multiphase import solve_two_phase_stokes_surface_tension


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


# --- two-phase force-balance closure: a tense membrane bounding a network + solvent mixture ---


def test_two_phase_circle_is_a_laplace_fixed_point() -> None:
    # A circle bounding the two-phase mixture is an equilibrium. The *physical* mixture pressure
    # is 2p = γ/R (p is the multiplier conjugate to the TOTAL velocity, so each phase feels −p·I
    # and the two phases together give the Laplace stress); both phases are at rest.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    area0 = enclosed_volume(mesh)
    v_n, v_s, p = step_two_phase_fsi(mesh, tension=0.5, drag=1.0, dt=0.02)
    p_integral = fem.assemble_scalar(fem.form(p * ufl.dx(domain=p.function_space.mesh)))
    physical_pressure = 2.0 * float(p_integral.real) / area0
    assert abs(physical_pressure - 0.5) < 2e-2  # Laplace law: 2p = γ/R
    assert max(abs(v_n.x.array).max(), abs(v_s.x.array).max()) < 5e-2  # equilibrium, no flow
    assert abs(v_n.x.array - v_s.x.array).max() < 1e-10  # symmetric load ⇒ phases identical
    assert abs(enclosed_volume(mesh) - area0) / area0 < 1e-3  # fixed point, area held


def test_two_phase_ellipse_relaxes_toward_the_circle() -> None:
    # The two-phase mixture relaxes exactly like the single phase: tension drives the shape to
    # minimal perimeter, the mixture pressure enforces ∮(v_n+v_s)·n=0 ⇒ the mixture-average
    # frame conserves area to O(dt).
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    area0 = enclosed_volume(mesh)
    circle_perimeter = 2.0 * math.pi * math.sqrt(area0 / math.pi)

    perimeter = _perimeter(mesh)
    for _ in range(25):
        step_two_phase_fsi(mesh, tension=0.5, drag=1.0, dt=0.02)
        nxt = _perimeter(mesh)
        assert nxt < perimeter + 1e-9  # monotone decrease toward the circle
        perimeter = nxt

    assert perimeter < _perimeter_of_initial_ellipse(area0, a=1.3)  # actually relaxed
    assert perimeter > circle_perimeter - 1e-3  # toward, not past, the circle
    assert abs(enclosed_volume(mesh) - area0) / area0 < 1e-2  # area conserved to O(dt)


def _relative_phase_slip(drag: float) -> float:
    # Asymmetric phase viscosities make the two phases *want* to move differently; the
    # interphase drag opposes that. Returns |v_n − v_s| / |v|, the relative slip after one step.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    v_n, v_s, _ = step_two_phase_fsi(mesh, tension=0.5, drag=drag, dt=0.02, viscosity_n=2.0, viscosity_s=0.4)
    speed = max(abs(v_n.x.array).max(), abs(v_s.x.array).max())
    return float(abs(v_n.x.array - v_s.x.array).max() / speed)


def test_two_phase_drag_locks_the_phases() -> None:
    # The interphase drag is the genuine two-phase coupling: as ξ grows, the phases lock
    # together (relative slip → 0). Asymmetric viscosities are needed — the symmetric ½/½
    # tension load alone gives v_n ≡ v_s for any drag.
    weak = _relative_phase_slip(drag=0.1)
    strong = _relative_phase_slip(drag=100.0)
    assert weak > 0.5  # at weak drag the phases slip substantially
    assert strong < 0.5 * weak  # strong drag locks them together


def test_two_phase_frame_selects_the_mesh_velocity() -> None:
    # The mesh frame is the model's one genuine fork (§6); the network frame runs and returns
    # the three fields, and an unknown frame is rejected loudly.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.2)
    v_n, _v_s, p = step_two_phase_fsi(mesh, tension=0.5, drag=1.0, dt=0.02, frame="network")
    assert v_n.x.array.size > 0 and p.x.array.size > 0

    with pytest.raises(ValueError, match="must be 'mixture'"):
        step_two_phase_fsi(mesh, tension=0.5, drag=1.0, dt=0.02, frame="cytoskeleton")


# --- a co-moving volume species carried through the moving two-phase cell ---


def _total_species(c, mesh) -> float:  # type: ignore[no-untyped-def]
    from mpi4py import MPI

    local = fem.assemble_scalar(fem.form(c * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))


def _uniform_species(mesh) -> fem.Function:  # type: ignore[no-untyped-def]
    c = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))
    c.x.array[:] = 1.0
    return c


def test_co_moving_species_uniform_on_mixture_stays_uniform() -> None:
    # The mixture carrier is divergence-free, so its dilution vanishes and a constant species
    # stays constant — the coupling injects no spurious source.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    c = _uniform_species(mesh)
    for _ in range(10):
        step_two_phase_fsi_with_species(mesh, c, tension=0.5, drag=1.0, dt=0.02, diffusivity=0.0)
    assert float(abs(c.x.array - 1.0).max()) < 1e-10  # stays uniform to round-off


def test_co_moving_species_conserved_on_mixture_during_relaxation() -> None:
    # A non-uniform species riding the (incompressible, no-transmembrane-flux) mixture is
    # transported as the membrane relaxes with ∫c held to the O(dt) integration error.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    space = fem.functionspace(mesh, ("Lagrange", 1))
    c = fem.Function(space)
    c.interpolate(fem.Expression(1.0 + 0.5 * ufl.SpatialCoordinate(mesh)[0], space.element.interpolation_points))
    mass0 = _total_species(c, mesh)
    for _ in range(20):
        step_two_phase_fsi_with_species(mesh, c, tension=0.5, drag=1.0, dt=0.02, diffusivity=0.01)
    assert abs(_total_species(c, mesh) - mass0) / mass0 < 5e-3  # ∫c conserved, O(dt)


def _uniform_on_network(*, with_dilution: bool) -> tuple[float, float]:
    # One controlled experiment: a uniform species on the *compressible* network phase (asymmetric
    # viscosities ⇒ ∇·v_n ≠ 0), advanced by an identical loop with the dilution term toggled.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    space = fem.functionspace(mesh, ("Lagrange", 1))
    c = _uniform_species(mesh)
    mass0 = _total_species(c, mesh)
    for _ in range(20):
        v_n, v_s, _ = solve_two_phase_stokes_surface_tension(
            mesh, tension=0.5, drag=0.2, viscosity_n=4.0, viscosity_s=0.25
        )
        carrier = _phase_velocity("network", v_n, v_s)
        displacement = _harmonic_displacement(mesh, carrier, 0.02)
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", domain=mesh)
        a = (trial / 0.02 * test + ufl.dot(carrier - displacement / 0.02, ufl.grad(trial)) * test) * dx
        if with_dilution:
            a = a + trial * ufl.div(carrier) * test * dx
        updated = fem.Function(space)
        LinearProblem(
            a,
            c / 0.02 * test * dx,
            u=updated,
            petsc_options_prefix=f"vcellfenics_test_{id(updated):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        ).solve()
        c.x.array[:] = updated.x.array
        _apply_displacement(mesh, displacement)
    return abs(_total_species(c, mesh) - mass0) / mass0, float(abs(c.x.array - 1.0).max())


def test_co_moving_species_dilutes_under_compressible_network() -> None:
    # The bulk dilution c ∇·v_carrier is the volume analogue of the mandatory surface ρ ∇_Γ·v_Γ:
    # on a compressible phase it makes a uniform species concentrate/dilute while ∫c is conserved.
    # The negative control (drop the dilution) leaves the species spuriously uniform.
    mass_drift, deviation = _uniform_on_network(with_dilution=True)
    assert deviation > 1e-4  # the species responds to the network's compression
    assert mass_drift < 5e-3  # mass conserved — it redistributes, it does not leak

    _, control_deviation = _uniform_on_network(with_dilution=False)
    assert control_deviation < 1e-9  # without the dilution it ignores the compression entirely


def test_co_moving_species_through_public_step_matches_with_dilution() -> None:
    # The public step includes the dilution: a uniform species on the compressible network
    # concentrates (matching the controlled with-dilution loop), not stays uniform.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    c = _uniform_species(mesh)
    for _ in range(20):
        step_two_phase_fsi_with_species(
            mesh,
            c,
            tension=0.5,
            drag=0.2,
            dt=0.02,
            diffusivity=0.0,
            carrier="network",
            frame="network",
            viscosity_n=4.0,
            viscosity_s=0.25,
        )
    assert float(abs(c.x.array - 1.0).max()) > 1e-4  # the public step dilutes (includes c∇·v)


# --- several reacting volume species carried through the moving two-phase cell ---


def _two_species(mesh, ic_a) -> fem.Function:  # type: ignore[no-untyped-def]
    # A vector P1 species: component 0 (A) seeded from ic_a, component 1 (B) starts at zero.
    space = fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    c = fem.Function(space)
    c.sub(0).interpolate(fem.Expression(ic_a(ufl.SpatialCoordinate(mesh)), space.sub(0).element.interpolation_points))
    return c


def _conversion_reaction(k_on: float, k_off: float):  # type: ignore[no-untyped-def]
    # A ⇌ B: net rate r = k_on·A − k_off·B; conservative (the rows sum to zero).
    def reaction(c):  # type: ignore[no-untyped-def]
        r = k_on * c[0] - k_off * c[1]
        return ufl.as_vector([-r, r])

    return reaction


def test_reacting_species_conversion_conserves_total_and_shifts_mass() -> None:
    # A ⇌ B on the moving (incompressible-mixture) cell: the conservative reaction keeps the
    # total ∫(A+B) fixed while shifting mass A→B (k_on > k_off), the §1.4.5 check on a moving cell.
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    _stretch_into_ellipse(mesh, a=1.3)
    c = _two_species(mesh, lambda x: 1.0 + 0.3 * x[0])
    total0 = _total_species(c[0], mesh) + _total_species(c[1], mesh)
    a_initial = _total_species(c[0], mesh)
    for _ in range(20):
        step_two_phase_fsi_with_reacting_species(
            mesh,
            c,
            tension=0.5,
            drag=1.0,
            dt=0.02,
            diffusivities=[0.01, 0.01],
            reaction=_conversion_reaction(2.0, 0.5),
        )
    a_final, b_final = _total_species(c[0], mesh), _total_species(c[1], mesh)
    assert abs((a_final + b_final) - total0) / total0 < 5e-3  # total ∫(A+B) conserved
    assert b_final > 0.5 and a_final < a_initial  # mass shifted A→B


def test_reacting_species_reaches_detailed_balance() -> None:
    # A uniform A, B relaxes to the detailed-balance ratio A/B = k_off/k_on (here 0.5/2 = 0.25).
    k_on, k_off = 2.0, 0.5
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    c = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (2,))))
    c.x.array.reshape(-1, 2)[:, 0] = 1.0  # A = 1
    c.x.array.reshape(-1, 2)[:, 1] = 2.0  # B = 2
    for _ in range(80):
        step_two_phase_fsi_with_reacting_species(
            mesh,
            c,
            tension=0.5,
            drag=1.0,
            dt=0.02,
            diffusivities=[0.01, 0.01],
            reaction=_conversion_reaction(k_on, k_off),
        )
    ratio = _total_species(c[0], mesh) / _total_species(c[1], mesh)
    assert abs(ratio - k_off / k_on) < 1e-2  # detailed balance: A/B → k_off/k_on


def test_reacting_species_linear_decay_matches_backward_euler() -> None:
    # A uniform single decaying species A → ∅ follows the exact backward-Euler solution
    # c₀/(1+k·dt)^steps (and stays uniform — the mixture carrier injects no transport).
    k, dt, steps, c0 = 1.5, 0.02, 20, 2.0
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    c = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (1,))))
    c.x.array[:] = c0
    for _ in range(steps):
        step_two_phase_fsi_with_reacting_species(
            mesh,
            c,
            tension=0.5,
            drag=1.0,
            dt=dt,
            diffusivities=[0.0],
            reaction=lambda field: ufl.as_vector([-k * field[0]]),
        )
    expected = c0 / (1.0 + k * dt) ** steps
    assert abs(float(c.x.array.mean()) - expected) < 1e-3  # backward-Euler exact for linear decay
    assert float(abs(c.x.array - c.x.array.mean()).max()) < 1e-10  # stays uniform


def test_reacting_species_validates_diffusivity_count() -> None:
    mesh = make_disk_geometry("g", volume_subdomain="c", radius=1.0, h=0.06).mesh_of("c")
    c = fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (2,))))
    with pytest.raises(ValueError, match="diffusivities has length"):
        step_two_phase_fsi_with_reacting_species(mesh, c, tension=0.5, drag=1.0, dt=0.02, diffusivities=[0.01])

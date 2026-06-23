"""Verification of the Cahn–Hilliard prototype (backend/cahn_hilliard) — diffuse-interface
phase separation, a first test of the "new physics as a template" pipeline.

The 4th-order equation is solved in its **mixed `(φ, μ)`** form — two coupled 2nd-order weak
forms. Three checks, sharing one integration (a module fixture):

1. **Conservation** — `φ` is a conserved order parameter (`∂φ/∂t = ∇·(M∇μ)`, no-flux), so `∫φ`
   is invariant to round-off, however far the field separates.
2. **Free energy relaxes** — the free energy `F = ∫[Wφ²(1−φ)² + ε²/2|∇φ|²]` (the Lyapunov
   functional of the gradient flow) decreases monotonically at a small enough step.
3. **Phase separation** — a near-uniform state in the spinodal band separates toward the two
   wells (`φ ≈ 0` and `φ ≈ 1`) across diffuse interfaces.
"""

from __future__ import annotations

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import cahn_hilliard_free_energy, solve_cahn_hilliard


def _total(phi: fem.Function) -> float:
    mesh = phi.function_space.mesh
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(phi * ufl.dx(domain=mesh))), op=MPI.SUM))


@pytest.fixture(scope="module")
def separation() -> tuple[float, float, fem.Function, list[float]]:
    """One Cahn–Hilliard run from a near-uniform spinodal state; returns the initial `∫φ`, the
    initial peak-to-peak spread, the final `φ`, and the free-energy history."""

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 48, 48)
    initial = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))
    initial.x.array[:] = 0.5 + 0.05 * np.random.default_rng(42).standard_normal(initial.x.array.size)
    total0 = _total(initial)
    spread0 = float(initial.x.array.max() - initial.x.array.min())
    phi, energies = solve_cahn_hilliard(mesh, initial=initial, dt=2e-4, n_steps=80, epsilon=0.05)
    return total0, spread0, phi, energies


def test_cahn_hilliard_conserves_the_order_parameter(separation) -> None:  # type: ignore[no-untyped-def]
    total0, _, phi, _ = separation
    assert abs(_total(phi) - total0) < 1e-12  # ∫φ invariant to round-off — the conservative form


def test_cahn_hilliard_free_energy_decreases_monotonically(separation) -> None:  # type: ignore[no-untyped-def]
    *_, energies = separation
    assert all(energies[i + 1] <= energies[i] + 1e-12 for i in range(len(energies) - 1))  # Lyapunov
    assert energies[-1] < energies[0]  # and it actually relaxes


def test_cahn_hilliard_separates_into_two_phases(separation) -> None:  # type: ignore[no-untyped-def]
    _, spread0, phi, _ = separation
    assert spread0 < 0.5  # started near-uniform (in the spinodal band)
    assert float(phi.x.array.min()) < 0.2  # a phase near the φ = 0 well
    assert float(phi.x.array.max()) > 0.8  # a phase near the φ = 1 well — partially coincident, diffuse


def test_free_energy_helper_is_zero_at_a_well() -> None:
    # A uniform field at a well (φ = 0) has zero free energy (no bulk penalty, no gradient).
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    phi = fem.Function(fem.functionspace(mesh, ("Lagrange", 1)))  # φ = 0 everywhere
    assert abs(cahn_hilliard_free_energy(phi, epsilon=0.1)) < 1e-14


# ---------------------------------------------------------------------------
# Verification against the exact 1D equilibrium interface (the diffuse-interface
# width). For F = ∫[W φ²(1−φ)² + ε²/2 |∇φ|²] the stationary interface is
#     φ(x) = ½[1 + tanh((x − c)/2δ)],   δ = ε/√(2W),
# so the peak slope is max|∂ₓφ| = 1/(4δ). These pin the *physical* interface
# width to theory (it is δ, not a free knob), independently of how many mesh
# cells happen to resolve it.
# ---------------------------------------------------------------------------

_IFACE_EPS, _IFACE_WELL = 0.05, 1.0
_IFACE_DELTA = _IFACE_EPS / np.sqrt(2.0 * _IFACE_WELL)


def _strip(nx: int, ny: int = 4):  # type: ignore[no-untyped-def]
    # A thin strip so φ varies only across x — a 1D interface embedded in the 2D solver.
    return dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 0.2]], [nx, ny])


def _tanh_interface(x: np.ndarray, center: float, delta: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((x - center) / (2.0 * delta)))


def _measured_delta(phi: fem.Function) -> float:
    # Read the interface scale from the peak slope: max|∂ₓφ| = 1/(4δ) at the tanh's centre.
    v0 = fem.functionspace(phi.function_space.mesh, ("DG", 0))
    gx = fem.Function(v0)
    gx.interpolate(fem.Expression(ufl.grad(phi)[0], v0.element.interpolation_points))
    return 0.25 / float(np.abs(gx.x.array).max())


def _relax_interface(nx: int, init_delta: float, n_steps: int) -> tuple[fem.Function, float]:
    mesh = _strip(nx)
    space = fem.functionspace(mesh, ("Lagrange", 1))
    xc = space.tabulate_dof_coordinates()[:, 0]
    phi = fem.Function(space)
    phi.x.array[:] = _tanh_interface(xc, 0.5, init_delta)
    phi, _ = solve_cahn_hilliard(
        mesh, initial=phi, dt=2e-4, n_steps=n_steps, epsilon=_IFACE_EPS, well_height=_IFACE_WELL
    )
    analytic = _tanh_interface(xc, 0.5, _IFACE_DELTA)
    rms = float(np.sqrt(np.mean((phi.x.array - analytic) ** 2)))
    return phi, rms


def test_cahn_hilliard_reproduces_the_analytic_tanh_interface() -> None:
    # The exact equilibrium interface is held stationary: initialized as the analytic tanh it stays put
    # (small rms to the profile) and its width, read from the peak slope, matches δ = ε/√(2W) to ~1%.
    phi, rms = _relax_interface(160, _IFACE_DELTA, 150)
    assert rms < 0.003  # reproduced (held stationary) to discretization error
    assert _measured_delta(phi) == pytest.approx(_IFACE_DELTA, rel=0.03)  # width matches ε/√(2W)


def test_cahn_hilliard_interface_error_decreases_under_refinement() -> None:
    # h-refinement: the L2 distance from the analytic interface to the discrete solution falls as the
    # strip is refined — the discretisation converges to the exact interface.
    _, coarse = _relax_interface(80, _IFACE_DELTA, 150)
    _, fine = _relax_interface(160, _IFACE_DELTA, 150)
    assert fine < 0.6 * coarse  # error at least halves when h halves


def test_cahn_hilliard_relaxes_a_wrong_width_interface_to_delta() -> None:
    # The equilibrium width is an attractor of the dynamics, not just an IC we imposed: start from a
    # too-diffuse interface (2δ) and the gradient flow sharpens it to the analytic δ (conserving ∫φ).
    phi, rms = _relax_interface(160, 2.0 * _IFACE_DELTA, 1500)
    assert _measured_delta(phi) == pytest.approx(_IFACE_DELTA, rel=0.05)  # sharpened to the analytic width
    assert rms < 0.005  # and the whole profile matches the analytic tanh


# ---------------------------------------------------------------------------
# Verification against the spinodal dispersion relation (the linearised dynamics).
# Linearising CH about the unstable homogeneous state φ̄ = ½ (where f''(½) = −W) a
# single Fourier mode cos(kx) grows/decays exponentially at
#     σ(k) = M k² (W − ε²k²),
# positive (unstable) for k < k* = √W/ε and negative above it — the cutoff that
# selects the spinodal length scale, with fastest growth at k = √(W/2)/ε. Seeding
# admissible modes k_n = nπ/L (zero slope at the no-flux walls) and fitting the
# early-time amplitude growth recovers σ(k).
# ---------------------------------------------------------------------------

_DISP_WELL, _DISP_EPS = 1.0, 0.08  # ⇒ cutoff k* = √W/ε = 12.5, fastest mode at √(W/2)/ε ≈ 8.8


def _dispersion_sigma(k: float) -> float:
    return k**2 * (_DISP_WELL - _DISP_EPS**2 * k**2)  # M = 1


def _mode_amplitude(phi: fem.Function, k: float) -> float:
    # Project φ − ½ onto cos(kx): for a pure mode φ = ½ + a cos(kx) this returns a (since ∫cos² over an
    # admissible mode is half the domain). Isolates the seeded mode from any harmonics.
    mesh = phi.function_space.mesh
    span = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
    x = ufl.SpatialCoordinate(mesh)
    integ = fem.assemble_scalar(fem.form((phi - 0.5) * ufl.cos(k * x[0]) * ufl.dx))
    return 2.0 * float(integ.real) / float(span[0] * span[1])


def _growth_rate(n: int) -> float:
    # Seed mode n, evolve briefly, and fit σ as the slope of ln|amplitude| vs t (the linear regime).
    k = n * np.pi  # L = 1 ⇒ admissible k_n = nπ
    mesh = _strip(80, 8)
    space = fem.functionspace(mesh, ("Lagrange", 1))
    xc = space.tabulate_dof_coordinates()[:, 0]
    phi = fem.Function(space)
    phi.x.array[:] = 0.5 + 0.005 * np.cos(k * xc)  # small amplitude ⇒ stays linear
    times, amps = [0.0], [_mode_amplitude(phi, k)]
    for c in range(5):
        phi, _ = solve_cahn_hilliard(mesh, initial=phi, dt=2e-4, n_steps=50, epsilon=_DISP_EPS, well_height=_DISP_WELL)
        times.append((c + 1) * 50 * 2e-4)
        amps.append(_mode_amplitude(phi, k))
    return float(np.polyfit(times, np.log(np.abs(amps)), 1)[0])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cahn_hilliard_growth_rate_matches_the_dispersion_relation(n: int) -> None:
    # A growing spinodal mode (k = nπ < k* = 12.5): the measured early-time growth rate matches
    # σ(k) = M k²(W − ε²k²) to a few percent.
    assert _growth_rate(n) == pytest.approx(_dispersion_sigma(n * np.pi), rel=0.05)


def test_cahn_hilliard_supercritical_mode_decays() -> None:
    # Above the cutoff k* = √W/ε the gradient penalty wins: the mode decays (σ < 0), and the measured
    # rate matches the (negative) dispersion relation — the sign change that sets the spinodal scale.
    rate = _growth_rate(5)  # k = 5π ≈ 15.7 > k* ≈ 12.5
    assert rate < 0
    assert rate == pytest.approx(_dispersion_sigma(5 * np.pi), rel=0.05)


# ---------------------------------------------------------------------------
# Verification against the curvature (Gibbs-Thomson) relation for a droplet.
# A curved interface raises the chemical potential by the Laplace/Gibbs-Thomson
# amount μ = σκ (κ = 1/R in 2D, σ = ε√(2W)/6 the surface tension), so a circular
# drop of radius R sets the matrix just outside it to φ_out = μ/f''(0) = σ/(2WR),
# i.e. the supersaturation scales as the curvature 1/R. The drop also shrinks
# (curvature drives dissolution). `min φ` reads φ_out directly. (In a closed box,
# conservation slowly raises the matrix above the ideal value, so the clean
# window is just after the interface locally equilibrates.)
# ---------------------------------------------------------------------------

_DROP_WELL, _DROP_EPS = 1.0, 0.06
_DROP_SIGMA = _DROP_EPS * np.sqrt(2.0 * _DROP_WELL) / 6.0  # CH surface tension ε√(2W)/6
_DROP_GT = _DROP_SIGMA / (2.0 * _DROP_WELL)  # φ_out · R should equal this (the Gibbs-Thomson coefficient)


def _relax_droplet(r0: float, nx: int = 64, n_steps: int = 250) -> tuple[float, float]:
    # Seed a circular drop (φ=1 inside, 0 outside, tanh interface) in a unit box, relax briefly, and
    # return (radius from the φ>½ area, matrix concentration = min φ).
    delta = _DROP_EPS / np.sqrt(2.0 * _DROP_WELL)
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [1.0, 1.0]], [nx, nx])
    space = fem.functionspace(mesh, ("Lagrange", 1))
    xy = space.tabulate_dof_coordinates()
    r = np.sqrt((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2)
    phi = fem.Function(space)
    phi.x.array[:] = 0.5 * (1.0 - np.tanh((r - r0) / (2.0 * delta)))
    phi, _ = solve_cahn_hilliard(mesh, initial=phi, dt=2e-4, n_steps=n_steps, epsilon=_DROP_EPS, well_height=_DROP_WELL)
    area = fem.assemble_scalar(fem.form(ufl.conditional(phi > 0.5, 1.0, 0.0) * ufl.dx))
    return float(np.sqrt(float(area.real) / np.pi)), float(phi.x.array.min())


def test_cahn_hilliard_droplet_obeys_gibbs_thomson() -> None:
    # Two drops verify the curvature law on three counts: both shrink (curvature drives dissolution);
    # the tighter-curvature (smaller) drop sets a higher matrix supersaturation (the 1/R direction); and
    # the product φ_out·R matches the Gibbs-Thomson coefficient σ/(2W) for each (the magnitude).
    r_small, min_small = _relax_droplet(0.30)
    r_large, min_large = _relax_droplet(0.35)
    assert r_small < 0.30 and r_large < 0.35  # both shrank
    assert min_small > min_large  # tighter curvature ⇒ more supersaturation, φ_out ∝ 1/R
    assert min_small * r_small == pytest.approx(_DROP_GT, rel=0.12)  # φ_out·R = σ/(2W)
    assert min_large * r_large == pytest.approx(_DROP_GT, rel=0.12)


# --- the solidified formal template: a MathDescription with a `cahn_hilliard` equation ---

from vcell_fenics.backend import iter_cahn_hilliard, make_disk_geometry, run_cahn_hilliard  # noqa: E402
from vcell_fenics.formalism import load_yaml, validate  # noqa: E402

_CH_MODEL = """
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: none }} }}
  variables:
    - {{ name: c, subdomain: cyto, type: scalar }}
  equations:
    - template: cahn_hilliard
      variable: c
      subdomain: cyto
      temporality: {temporality}
      terms: {{ interface_width: "0.08" }}
      initial_condition: "0.5 + 0.1 * cos(6.0*geom.x[0]) * cos(6.0*geom.x[1])"
"""


def test_cahn_hilliard_template_validates() -> None:
    # The solidified template is recognised by the generic registry-driven validator — a correct
    # model has no errors. (The order parameter is `c`: `phi` is a reserved name, the azimuthal
    # angle, so the chemical-potential split needs no reserved name either — μ stays internal.)
    errors = [d for d in validate(load_yaml(_CH_MODEL.format(temporality="time_dependent"))) if d.severity == "error"]
    assert errors == []


def test_cahn_hilliard_template_requires_time_dependent() -> None:
    # Cahn–Hilliard is always transient; a steady-state declaration is rejected by the template.
    errors = [d for d in validate(load_yaml(_CH_MODEL.format(temporality="steady_state"))) if d.severity == "error"]
    assert any("temporality" in d.message and "cahn_hilliard" in d.message for d in errors)


def test_run_cahn_hilliard_drives_the_model_conserving_and_separating() -> None:
    # Driven from the validated model: the order parameter is conserved to round-off (the
    # conservative form survives the formalism path) and the field separates into two phases.
    md = load_yaml(_CH_MODEL.format(temporality="time_dependent"))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.06)
    total0 = _total(run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.0))  # 0 steps ⇒ the IC's ∫c
    phi = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.06)
    assert abs(_total(phi) - total0) < 1e-10  # ∫c conserved through the template driver
    assert float(phi.x.array.min()) < 0.3 and float(phi.x.array.max()) > 0.7  # separated into phases


def test_iter_cahn_hilliard_streams_snapshots_consistent_with_run() -> None:
    # The time-lapse driver yields (t, φ) snapshots from the same declarative model — at the initial
    # state, every `every` steps, and the final step. The snapshots conserve ∫c across the stream, the
    # free energy relaxes, and the final snapshot equals run_cahn_hilliard's final φ (the two entry
    # points share one expansion) — so a time-resolved demo needs no solver internals.
    md = load_yaml(_CH_MODEL.format(temporality="time_dependent"))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.06)
    snapshots = list(iter_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.02, every=5))

    times = [t for t, _ in snapshots]
    assert times[0] == 0.0 and times[-1] == pytest.approx(0.02)
    assert len(snapshots) == 5  # 20 steps, every 5 ⇒ steps 0, 5, 10, 15, 20

    total0 = _total(snapshots[0][1])
    assert all(abs(_total(phi) - total0) < 1e-10 for _, phi in snapshots)  # conserved across the stream
    energies = [cahn_hilliard_free_energy(phi, epsilon=0.08) for _, phi in snapshots]
    assert energies[-1] < energies[0]  # the gradient flow ran downhill

    final = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.02)
    assert np.allclose(snapshots[-1][1].x.array, final.x.array, atol=1e-9)  # same final state, two entry points


_NOISE_MODEL = """
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: none }} }}
  variables:
    - {{ name: c, subdomain: cyto, type: scalar }}
  equations:
    - template: cahn_hilliard
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ interface_width: "0.08" }}
      initial_condition: "{ic}"
"""


def test_random_initial_condition_is_a_reproducible_stored_field() -> None:
    # A declarative spinodal seed `mean + normal(0, σ)` is realized ONCE into a stored field: same seed
    # reproduces it exactly, a different seed differs, and the drawn field matches the requested
    # distribution. Storing the realization (not drawing per evaluation) is what makes it a fixed
    # function of space — re-interpolation/substitution returns the same value at the same point.
    md = load_yaml(_NOISE_MODEL.format(ic="0.5 + normal(0, 0.05)"))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.08)

    a = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.0, seed=7)  # 0 steps ⇒ just the realized IC
    b = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.0, seed=7)
    c = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.0, seed=8)
    assert np.array_equal(a.x.array, b.x.array)  # same seed ⇒ identical realization
    assert not np.allclose(a.x.array, c.x.array)  # different seed ⇒ different realization
    assert a.x.array.mean() == pytest.approx(0.5, abs=0.02) and a.x.array.std() == pytest.approx(0.05, abs=0.01)


def test_spinodal_via_random_primitive_separates_and_conserves() -> None:
    # The narrow `spinodal_noise` IC is unnecessary: `0.5 + normal(0, …)` seeds spinodal decomposition
    # through the general primitive — ∫c is conserved through the run and the field separates.
    md = load_yaml(_NOISE_MODEL.format(ic="0.5 + normal(0, 0.05)"))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.06)
    total0 = _total(run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.0, seed=3))
    phi = run_cahn_hilliard(md, geometry, dt=1e-3, t_final=0.05, seed=3)
    assert abs(_total(phi) - total0) < 1e-10  # conserved
    assert float(phi.x.array.min()) < 0.3 and float(phi.x.array.max()) > 0.7  # separated into two phases

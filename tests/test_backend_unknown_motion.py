"""Verification of unknown (mechanics-driven) membrane motion — the §1.10.8 model.

The membrane velocity is *solved* from a weak-form force balance (not prescribed), the
membrane moves under it, and a receptor species dilutes with the motion — the staggered
scheme (solve force → move → advance species). The checks pin a known-answer force
(`η v = f₀ x/r` ⇒ `v = (f₀/η)·x̂`, a radial expansion at speed `f₀/η`):

1. **The solved velocity is right** — `|v| = f₀/η`, radial.
2. **It drives the mesh** — the membrane radius grows at `f₀/η`.
3. **The receptor dilutes** — ∫_Γ ρ ds is conserved across the (solved) expansion;
   without the dilution term it would grow with the membrane perimeter.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import make_disk_membrane_geometry
from vcell_fenics.backend.unknown_motion import UnknownMotionProblem, assemble_unknown_motion
from vcell_fenics.formalism import load_yaml

_ETA, _F0 = 1.0, 0.5

_MODEL = f"""
math_description:
  geometry: g
  subdomains:
    - name: mem
      kind: surface
      motion: {{ kind: unknown, variable: v }}
  variables:
    - {{ name: v, subdomain: mem, type: vector }}
    - {{ name: rho, subdomain: mem }}
  equations:
    - template: weak_form
      variable: v
      subdomain: mem
      temporality: steady_state
      form: "(eta*inner(v, v_test) - inner(f0 * geom.x / geom.radius, v_test)) * dx_Gamma"
      initial_condition: "0"
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: mem
      temporality: time_dependent
      terms: {{ diffusion: "0.02" }}
      initial_condition: "1.0 + 0.3*cos(2*geom.azimuth)"
  parameters:
    - {{ name: eta, value: {_ETA} }}
    - {{ name: f0, value: {_F0} }}
"""


def _problem(dt: float = 0.04, h: float = 0.06) -> UnknownMotionProblem:
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=h)
    return assemble_unknown_motion(load_yaml(_MODEL), geometry, dt=dt)


def _mean_radius(problem: UnknownMotionProblem) -> float:
    return float(np.linalg.norm(problem.velocity.function_space.mesh.geometry.x[:, :2], axis=1).mean())


def _receptor_mass(problem: UnknownMotionProblem) -> float:
    assert problem.receptor is not None
    mesh = problem.receptor.unknown.function_space.mesh
    return float(fem.assemble_scalar(fem.form(problem.receptor.unknown * ufl.dx(domain=mesh))).real)


def test_solved_velocity_matches_force_balance() -> None:
    problem = _problem()
    problem.step()  # one solve of the force balance

    speeds = np.linalg.norm(problem.velocity.x.array.reshape(-1, 2), axis=1)
    assert speeds.mean() == pytest.approx(_F0 / _ETA, rel=1e-3)  # |v| = f₀/η
    assert speeds.std() < 1e-3  # uniform magnitude (radial)


def test_solved_velocity_drives_membrane_motion() -> None:
    dt, n_steps = 0.04, 15
    problem = _problem(dt=dt)
    r0 = _mean_radius(problem)
    for _ in range(n_steps):
        problem.step()

    # The membrane expands at the solved speed f₀/η.
    assert _mean_radius(problem) == pytest.approx(r0 + dt * (_F0 / _ETA) * n_steps, rel=1e-3)


def test_receptor_dilutes_under_solved_motion() -> None:
    dt, n_steps = 0.04, 15
    problem = _problem(dt=dt)
    mass0, r0 = _receptor_mass(problem), _mean_radius(problem)
    for _ in range(n_steps):
        problem.step()

    assert _mean_radius(problem) > 1.2 * r0  # the membrane genuinely expanded
    # The mandatory dilution term keeps ∫_Γ ρ ds invariant under the solved expansion.
    assert _receptor_mass(problem) == pytest.approx(mass0, rel=1e-2)


# --- curvature forces: n(x) / H(x) drive mean-curvature flow -------------------

_SIGMA = 0.1

_CURVATURE_MODEL = f"""
math_description:
  geometry: g
  subdomains:
    - name: mem
      kind: surface
      motion: {{ kind: unknown, variable: v }}
  variables:
    - {{ name: v, subdomain: mem, type: vector }}
  equations:
    - template: weak_form
      variable: v
      subdomain: mem
      temporality: steady_state
      form: "(eta*inner(v, v_test) + sigma*geom.mean_curvature*inner(geom.normal, v_test)) * dx_Gamma"
      initial_condition: "0"
  parameters:
    - {{ name: eta, value: {_ETA} }}
    - {{ name: sigma, value: {_SIGMA} }}
"""


def _curvature_problem(dt: float, h: float = 0.05, *, redistribute: bool = False) -> UnknownMotionProblem:
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=h)
    return assemble_unknown_motion(load_yaml(_CURVATURE_MODEL), geometry, dt=dt, redistribute=redistribute)


def test_curvature_force_gives_inward_velocity() -> None:
    # Surface tension `η v + σ H n = 0` on a unit circle: H = 1/r = 1, n outward, so
    # v = -(σ/η) n̂ — radially inward at speed σ/η. This pins n(x)/H(x) → the correct
    # weak curvature vector (κ = H·n), not just its magnitude.
    problem = _curvature_problem(dt=0.005)
    problem._curvature.project()  # type: ignore[union-attr]
    problem._force_balance.solve()

    coords = problem.velocity.function_space.tabulate_dof_coordinates()[:, :2]
    v = problem.velocity.x.array.reshape(-1, 2)
    speeds = np.linalg.norm(v, axis=1)
    assert speeds.mean() == pytest.approx(_SIGMA / _ETA, rel=2e-2)  # |v| = σ/(η r), r = 1

    # The velocity is anti-parallel to the outward position (points inward).
    radial = (v * coords).sum(axis=1) / np.linalg.norm(coords, axis=1)
    assert np.all(radial < 0)
    assert radial.mean() == pytest.approx(-_SIGMA / _ETA, rel=2e-2)


def test_surface_tension_shrinks_circle() -> None:
    # Mean-curvature flow of a circle: r² = r₀² − 2σt/η. The default (non-redistribute)
    # path needs a small dt to keep the membrane from degenerating.
    dt, n_steps = 0.005, 20
    problem = _curvature_problem(dt=dt)
    r0 = _mean_radius(problem)
    for _ in range(n_steps):
        problem.step()

    expected = np.sqrt(r0**2 - 2 * _SIGMA / _ETA * dt * n_steps)
    assert _mean_radius(problem) == pytest.approx(expected, rel=2e-3)


def test_redistribute_runs_curvature_flow_at_large_dt() -> None:
    # With redistribute=True the membrane uses the BGN scheme: it follows the same
    # `r² = r₀² − 2σt/η` flow but with tangential redistribution, so it runs cleanly at
    # a dt several× larger than the velocity-based path tolerates — and the mobility
    # σ/η is read straight off the force balance (no parameter is passed to the bridge).
    dt, n_steps = 0.02, 20
    problem = _curvature_problem(dt=dt, redistribute=True)
    r0 = _mean_radius(problem)
    for _ in range(n_steps):
        problem.step()  # no MeshQualityError despite the 4× step

    expected = np.sqrt(r0**2 - 2 * _SIGMA / _ETA * dt * n_steps)
    assert _mean_radius(problem) == pytest.approx(expected, rel=2e-3)
    radii = np.linalg.norm(problem.velocity.function_space.mesh.geometry.x[:, :2], axis=1)
    assert radii.std() < 1e-3  # stays round


def test_redistribute_requires_a_curvature_force_balance() -> None:
    # The known-answer `η v = f₀ x/r` force balance is not curvature flow, so there is
    # no curvature projection to drive BGN — redistribute must refuse it.
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=0.1)
    with pytest.raises(NotImplementedError, match="curvature"):
        assemble_unknown_motion(load_yaml(_MODEL), geometry, dt=0.04, redistribute=True)


# --- redistribution WITH a co-moving receptor: the ALE coupling --------------------

_CURVATURE_RECEPTOR_MODEL = f"""
math_description:
  geometry: g
  subdomains:
    - {{ name: mem, kind: surface, motion: {{ kind: unknown, variable: v }} }}
  variables:
    - {{ name: v, subdomain: mem, type: vector }}
    - {{ name: rho, subdomain: mem }}
  equations:
    - {{ template: weak_form, variable: v, subdomain: mem, temporality: steady_state,
        form: "(eta*inner(v, v_test) + sigma*geom.mean_curvature*inner(geom.normal, v_test)) * dx_Gamma",
        initial_condition: "0" }}
    - {{ template: surface_pde_with_dilution, variable: rho, subdomain: mem, temporality: time_dependent,
        terms: {{ diffusion: "0.001" }}, initial_condition: "1.0 + 0.5*cos(2*geom.azimuth)" }}
  parameters:
    - {{ name: eta, value: {_ETA} }}
    - {{ name: sigma, value: {_SIGMA} }}
"""


def _curvature_receptor_problem(dt: float, *, deform: float = 1.0) -> UnknownMotionProblem:
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.2, h=0.05)
    problem = assemble_unknown_motion(load_yaml(_CURVATURE_RECEPTOR_MODEL), geometry, dt=dt, redistribute=True)
    if deform != 1.0:  # squash the circle into an ellipse so nodes genuinely slide
        problem.receptor.unknown.function_space.mesh.geometry.x[:, 1] *= deform  # type: ignore[union-attr]
    return problem


def test_redistribute_conserves_receptor_mass_under_renoding() -> None:
    # The crux of the ALE coupling: on a redistributing (re-noded) membrane the surface
    # PDE picks up an advection term; realised here as a conservative remap, total
    # surface mass ∫_Γ ρ ds stays invariant even with a non-uniform ρ and heavy
    # tangential node motion — at a dt where the non-redistribute path tangles.
    problem = _curvature_receptor_problem(dt=0.02, deform=0.6)
    mass0 = _receptor_mass(problem)
    for _ in range(25):
        problem.step()  # decomposed normal-flow + re-node-remap; no MeshQualityError

    assert _receptor_mass(problem) == pytest.approx(mass0, rel=5e-3)


def test_redistribute_with_receptor_uses_the_decomposed_path() -> None:
    # A receptor on a redistributing membrane wires up the decomposed BGN+remap motion
    # (not the motion-only combined-BGN path, and not a refusal).
    problem = _curvature_receptor_problem(dt=0.02)
    assert problem._bgn_receptor is not None
    assert problem._bgn is None


def test_redistribute_requires_a_curvature_force_balance_even_with_receptor() -> None:
    # The known-answer `η v = f₀ x/r` model is not curvature flow — redistribute refuses
    # it whether or not a receptor is present (it carries one).
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=0.1)
    with pytest.raises(NotImplementedError, match="curvature"):
        assemble_unknown_motion(load_yaml(_MODEL), geometry, dt=0.04, redistribute=True)


def test_curvature_unavailable_outside_mechanics() -> None:
    # n(x)/H(x) are only bound where the curvature projection lives (a mechanics solve).
    # A plain weak-form surface PDE that references them must fail to compile.
    from vcell_fenics.backend import CompileError
    from vcell_fenics.backend.weakform import assemble_weak_form

    model = """
math_description:
  geometry: g
  subdomains: [ { name: mem, kind: surface, motion: { kind: none } } ]
  variables: [ { name: rho, subdomain: mem } ]
  equations:
    - template: weak_form
      variable: rho
      subdomain: mem
      temporality: steady_state
      form: "(rho*rho_test - geom.mean_curvature*rho_test) * dx_Gamma"
"""
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=0.2)
    with pytest.raises(CompileError, match=r"curvature projection|mechanics"):
        assemble_weak_form(load_yaml(model), geometry, dt=0.04)


def test_requires_an_unknown_motion_subdomain() -> None:
    # A plain static surface-diffusion model has no unknown-motion subdomain.
    static = """
math_description:
  geometry: g
  subdomains: [ { name: mem, kind: surface, motion: { kind: none } } ]
  variables: [ { name: rho, subdomain: mem } ]
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: mem
      temporality: time_dependent
      terms: { diffusion: "0.1" }
      initial_condition: "1.0"
"""
    geometry = make_disk_membrane_geometry("g", surface_subdomain="mem", radius=1.0, h=0.2)
    with pytest.raises(NotImplementedError, match="unknown-motion subdomain"):
        assemble_unknown_motion(load_yaml(static), geometry, dt=0.04)

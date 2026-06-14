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
      form: "(eta*inner(v, v_test) - inner(f0 * x / r(x), v_test)) * dx_Gamma"
      initial_condition: "0"
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: mem
      temporality: time_dependent
      terms: {{ diffusion: "0.02" }}
      initial_condition: "1.0 + 0.3*cos(2*theta(x))"
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

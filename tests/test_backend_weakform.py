"""Verification of the weak-form escape hatch (§1.5) — the route for membrane forces.

A `weak_form` equation lets the user write the residual UFL directly (the equation is
`form = 0` for all admissible test functions), trading the template guardrails for full
expressiveness. The checks pin three analytically-known cases:

1. **Time-dependent reproduces a template** — surface diffusion written as a weak form
   (`partial_t(ρ)·w + D ∇ρ·∇w`) decays the cos(kθ) eigenmode at the analytical rate and
   conserves mass, matching the T2 surface-PDE template.
2. **Membrane force balance (scalar, steady)** — a tensioned membrane on an elastic
   foundation, `α u − σ Δ_Γ u = f`, has the analytic response `A = f₀/(α + 4σ/r²)` to a
   `cos(2θ)` load.
3. **Membrane force balance (vector, steady)** — a viscous balance `η v = f_active`
   gives `v = f_active/η` (exercising a vector unknown and a vector literal `[·, ·]`).

Plus the structural shape and the v1 guards (BCs on a weak-form variable, multiple
equations).
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import make_disk_membrane_geometry
from vcell_fenics.backend.weakform import WeakFormProblem, assemble_weak_form
from vcell_fenics.formalism import load_yaml


def _geom(h: float = 0.05, radius: float = 1.0):  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("g", surface_subdomain="mem", radius=radius, h=h)


def _model(
    *, temporality: str, form: str, variable: str = "rho", vtype: str = "scalar", ic: str = "", params: str = ""
) -> str:
    var_type = f", type: {vtype}" if vtype != "scalar" else ""
    ic_line = f'\n      initial_condition: "{ic}"' if ic else ""
    return f"""
math_description:
  geometry: g
  subdomains: [ {{ name: mem, kind: surface, motion: {{ kind: none }} }} ]
  variables: [ {{ name: {variable}, subdomain: mem{var_type} }} ]
  equations:
    - template: weak_form
      variable: {variable}
      subdomain: mem
      temporality: {temporality}
      form: "{form}"{ic_line}
  parameters: [ {params} ]
"""


# ---------------------------------------------------------------------------
# 1. time-dependent weak form reproduces the T2 surface diffusion template
# ---------------------------------------------------------------------------


def test_time_dependent_weak_form_reproduces_surface_diffusion() -> None:
    d, k, r, dt, n_steps = 0.1, 2, 1.0, 0.01, 50
    model = _model(
        temporality="time_dependent",
        form="(partial_t(rho) * rho_test + D * inner(grad(rho), grad(rho_test))) * dx_Gamma",
        ic="1.0 + 0.5 * cos(2 * geom.azimuth)",
        params="{ name: D, value: 0.1 }",
    )
    problem = assemble_weak_form(load_yaml(model), _geom(0.05, r), dt=dt)

    def mass() -> float:
        return float(fem.assemble_scalar(fem.form(problem.solution * ufl.dx(domain=problem.V.mesh))).real)

    def amplitude() -> float:
        a = problem.solution.x.array
        return 0.5 * float(a.max() - a.min())

    m0, amp0 = mass(), amplitude()
    for _ in range(n_steps):
        problem.step()

    expected = np.exp(-d * k**2 / r**2 * dt * n_steps)  # cos(kθ) decays as exp(−D k²/r² t)
    assert amplitude() / amp0 == pytest.approx(expected, rel=2e-2)
    assert mass() == pytest.approx(m0, rel=1e-10)  # closed manifold conserves mass


# ---------------------------------------------------------------------------
# 2. scalar membrane force balance (steady) — tension + elastic foundation
# ---------------------------------------------------------------------------


def test_membrane_elastic_foundation_force_balance() -> None:
    # α u − σ Δ_Γ u = f₀ cos(2θ) on the unit circle ⇒ u = A cos(2θ), A = f₀/(α + 4σ/r²).
    alpha, sigma, f0, r = 2.0, 0.5, 1.0, 1.0
    model = _model(
        variable="u",
        temporality="steady_state",
        form="(alpha*u*u_test + sigma*inner(grad(u),grad(u_test)) - f0*cos(2*geom.azimuth)*u_test) * dx_Gamma",
        params="{ name: alpha, value: 2.0 }, { name: sigma, value: 0.5 }, { name: f0, value: 1.0 }",
    )
    problem = assemble_weak_form(load_yaml(model), _geom(0.04, r), dt=1.0)
    problem.step()

    amplitude = 0.5 * (problem.solution.x.array.max() - problem.solution.x.array.min())
    assert amplitude == pytest.approx(f0 / (alpha + 4 * sigma / r**2), rel=2e-2)


# ---------------------------------------------------------------------------
# 3. vector membrane force balance (steady) — viscous drag vs active traction
# ---------------------------------------------------------------------------


def test_vector_viscous_force_balance() -> None:
    # η v = f_active with f_active = [f₀ cos(θ), 0] ⇒ v = [f₀ cos(θ)/η, 0].
    eta, f0 = 1.0, 0.3
    model = _model(
        variable="v",
        vtype="vector",
        temporality="steady_state",
        form="(eta*inner(v, v_test) - inner([f0*cos(geom.azimuth), 0], v_test)) * dx_Gamma",
        params="{ name: eta, value: 1.0 }, { name: f0, value: 0.3 }",
    )
    problem = assemble_weak_form(load_yaml(model), _geom(0.05), dt=1.0)
    problem.step()

    vx, vy = problem.solution.x.array[0::2], problem.solution.x.array[1::2]
    assert 0.5 * (vx.max() - vx.min()) == pytest.approx(f0 / eta, rel=3e-2)  # |v_x| amplitude = f₀/η
    assert float(np.abs(vy).max()) < 1e-9  # no y-component


# ---------------------------------------------------------------------------
# 4. structure + guards
# ---------------------------------------------------------------------------


def test_problem_space_matches_variable_type() -> None:
    scalar = assemble_weak_form(
        load_yaml(_model(variable="u", temporality="steady_state", form="(u*u_test - u_test) * dx_Gamma")),
        _geom(0.2),
        dt=1.0,
    )
    vector = assemble_weak_form(
        load_yaml(
            _model(variable="w", vtype="vector", temporality="steady_state", form="(inner(w, w_test)) * dx_Gamma")
        ),
        _geom(0.2),
        dt=1.0,
    )
    assert isinstance(scalar, WeakFormProblem)
    assert scalar.V.num_sub_spaces == 0  # scalar
    assert vector.V.num_sub_spaces == 2  # vector in R²
    assert scalar.previous is None  # steady-state has no previous step


def test_bc_on_weak_form_variable_is_rejected() -> None:
    model = """
math_description:
  geometry: g
  subdomains: [ { name: mem, kind: surface, motion: { kind: none } } ]
  variables: [ { name: u, subdomain: mem } ]
  equations:
    - template: weak_form
      variable: u
      subdomain: mem
      temporality: steady_state
      form: "(u*u_test - u_test) * dx_Gamma"
  boundary_conditions:
    - { kind: dirichlet, variable: u, boundary: edge, expression: "0" }
"""
    with pytest.raises(NotImplementedError, match="weak-form variable"):
        assemble_weak_form(load_yaml(model), _geom(0.2), dt=1.0)

"""Verification of boundary conditions beyond the zero-Neumann default (§1.6).

The backend now assembles external Dirichlet / Neumann / Robin BCs on a labelled
boundary; the implicit default for an unlabelled boundary remains zero-Neumann
(conservative, no-flux). Each kind is checked against its analytical signature, with
the zero-Neumann default as the discriminating control:

1. **Dirichlet** u = g — the boundary value is pinned exactly, and the field relaxes
   to a constant g everywhere; with g = 0 the domain drains where the no-BC default
   conserves.
2. **Neumann** D∇u·n = h — a constant influx adds mass at exactly the rate ∫_Γ h ds
   (the discrete mass balance is exact for backward Euler).
3. **Robin** αu + βD∇u·n = h — the field relaxes to the uniform steady state h/α.

Plus: an unlabelled boundary and the (unsupported) interface kinds are rejected,
the coupled vector path pins one component, and a BC problem cannot be remeshed.
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    TermKind,
    assemble,
    make_disk_geometry,
    rebuild_on_mesh,
)
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.validator import FormalismValidationError


def _model(*, bcs: str = "", ic: str = "1.0", diffusion: str = "0.5", variables: str = "", equations: str = "") -> str:
    vars_block = variables or "    - { name: c, subdomain: cyto }"
    eqs_block = equations or f"""    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "{ic}\""""
    return f"""
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: none }} }}
  variables:
{vars_block}
  equations:
{eqs_block}
{bcs}
"""


def _geom(*, boundary: str | None = "wall", h: float = 0.12):  # type: ignore[no-untyped-def]
    return make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary=boundary, radius=1.0, h=h)


def _mass(dp) -> float:  # type: ignore[no-untyped-def]
    return float(fem.assemble_scalar(fem.form(dp.unknown * dp.dx)).real)


def _perimeter(dp) -> float:  # type: ignore[no-untyped-def]
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.ds(domain=dp.V.mesh))).real)


def _dirichlet(value: str) -> str:
    return f"""
  boundary_conditions:
    - {{ kind: dirichlet, variable: c, boundary: wall, expression: "{value}" }}
"""


def _eq(variable: str, *, diffusion: str = "0.1", ic: str = "1.0") -> str:
    return f"""    - template: bulk_radv_diff
      variable: {variable}
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "{ic}\""""


_TWO_VARS = "    - { name: c, subdomain: cyto }\n    - { name: d, subdomain: cyto }"


# ---------------------------------------------------------------------------
# 1. Dirichlet
# ---------------------------------------------------------------------------


def test_dirichlet_pins_boundary_and_relaxes_to_constant() -> None:
    dp = assemble(load_yaml(_model(bcs=_dirichlet("3.0"), ic="0.0")), _geom(), dt=0.05)
    for _ in range(200):
        dp.step()

    # The whole field relaxes to the boundary value, pinned exactly on the boundary.
    assert dp.unknown.x.array.min() == pytest.approx(3.0, abs=1e-6)
    assert dp.unknown.x.array.max() == pytest.approx(3.0, abs=1e-6)


def test_dirichlet_zero_drains_where_default_conserves() -> None:
    drained = assemble(load_yaml(_model(bcs=_dirichlet("0.0"), ic="2.0")), _geom(), dt=0.02)
    conserved = assemble(load_yaml(_model(ic="2.0")), _geom(boundary=None), dt=0.02)  # no BC ⇒ zero-Neumann
    m0 = _mass(drained)
    for _ in range(50):
        drained.step()
        conserved.step()

    assert _mass(drained) < 0.1 * m0  # drained through the Dirichlet boundary
    assert _mass(conserved) == pytest.approx(m0, rel=1e-9)  # no-flux default conserves
    assert drained.bcs  # the Dirichlet BC is present (strong, in bcs)
    assert conserved.boundary_kinds() == set()  # the default adds no boundary term


# ---------------------------------------------------------------------------
# 2. Neumann
# ---------------------------------------------------------------------------


def test_neumann_influx_adds_mass_at_predicted_rate() -> None:
    bc = """
  boundary_conditions:
    - { kind: neumann, variable: c, boundary: wall, expression: "0.5" }
"""
    dp = assemble(load_yaml(_model(bcs=bc, ic="1.0")), _geom(), dt=0.01)
    assert dp.boundary_kinds() == {TermKind.NEUMANN}

    h, n_steps, dt = 0.5, 50, 0.01
    m0 = _mass(dp)
    flux_per_step = h * _perimeter(dp)  # d(mass)/dt = ∫_Γ h ds
    for _ in range(n_steps):
        dp.step()

    # Exact discrete mass balance: each backward-Euler step adds dt·∫_Γ h ds.
    assert _mass(dp) - m0 == pytest.approx(flux_per_step * dt * n_steps, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Robin
# ---------------------------------------------------------------------------


def test_robin_relaxes_to_h_over_alpha() -> None:
    bc = """
  boundary_conditions:
    - { kind: robin, variable: c, boundary: wall, alpha: "1.0", beta: "1.0", expression: "2.0" }
"""
    dp = assemble(load_yaml(_model(bcs=bc, ic="0.0")), _geom(), dt=0.05)
    assert dp.boundary_kinds() == {TermKind.ROBIN}

    for _ in range(300):
        dp.step()
    # Steady state of αu + βD∇u·n = h with no source is the uniform u = h/α.
    assert dp.unknown.x.array.min() == pytest.approx(2.0, abs=1e-3)
    assert dp.unknown.x.array.max() == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. rejection paths
# ---------------------------------------------------------------------------


def test_unlabelled_boundary_is_rejected() -> None:
    bc = """
  boundary_conditions:
    - { kind: dirichlet, variable: c, boundary: nowhere, expression: "0.0" }
"""
    with pytest.raises(FormalismValidationError, match="nowhere"):
        assemble(load_yaml(_model(bcs=bc)), _geom(), dt=0.05)


def test_interface_bc_is_not_supported() -> None:
    # Two variables on the shared subdomain so the partner resolves; an interface BC
    # needs an internal boundary between two subdomains (multi-compartment), deferred.
    equations = f"{_eq('c')}\n{_eq('d')}"
    bc = """
  boundary_conditions:
    - { kind: interface_value_equality, variable: c, partner_variable: d, boundary: wall, expression: "1" }
"""
    with pytest.raises(NotImplementedError, match="interface"):
        assemble(load_yaml(_model(variables=_TWO_VARS, equations=equations, bcs=bc)), _geom(), dt=0.05)


def test_rebuild_rejects_a_bc_problem() -> None:
    dp = assemble(load_yaml(_model(bcs=_dirichlet("0.0"), ic="2.0")), _geom(), dt=0.05)
    new_mesh = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2).mesh_of("cyto")

    with pytest.raises(NotImplementedError, match="BCs"):
        rebuild_on_mesh(dp, load_yaml(_model(bcs=_dirichlet("0.0"), ic="2.0")), new_mesh)


# ---------------------------------------------------------------------------
# 5. coupled vector path — Dirichlet on one component
# ---------------------------------------------------------------------------


def test_dirichlet_on_one_component_of_a_coupled_solve() -> None:
    equations = f"{_eq('c', diffusion='0.3')}\n{_eq('d', diffusion='0.3')}"
    bc = """
  boundary_conditions:
    - { kind: dirichlet, variable: c, boundary: wall, expression: "0.0" }
"""
    dp = assemble(load_yaml(_model(variables=_TWO_VARS, equations=equations, bcs=bc)), _geom(), dt=0.02)
    for _ in range(50):
        dp.step()

    # Component c (constrained) drains toward 0; component d (no BC) is conserved.
    c_mass = float(fem.assemble_scalar(fem.form(dp.unknown.sub(0) * dp.dx)).real)
    d_mass = float(fem.assemble_scalar(fem.form(dp.unknown.sub(1) * dp.dx)).real)
    assert c_mass < 0.5  # drained via its Dirichlet boundary
    assert d_mass == pytest.approx(np.pi, rel=2e-2)  # ∫ d ≈ area·1 = π, conserved (zero-Neumann)

"""Verification of the §1.6.6 mixed-dimensional ligand-receptor binding solver.

This is the first cross-mesh coupled solve: extracellular ligand L (bulk) binds to
membrane free receptor ρ_f → bound ρ_b (surface), coupled by the binding rate
b = k_on·trace(L)·ρ_f − k_off·ρ_b, with a reservoir Dirichlet on L. The checks are
the model's analytical signatures:

1. **Receptor conservation** — binding only interconverts free ↔ bound, so the total
   ∫_Γ (ρ_f + ρ_b) ds is invariant (to round-off).
2. **Binding occurs and stays physical** — ρ_b rises from zero, and ρ_f never goes
   negative (the semi-implicit-in-membrane linearisation is unconditionally stable).
3. **Detailed balance** — at steady state b → 0, i.e. k_on·trace(L)·ρ_f = k_off·ρ_b
   pointwise on the membrane.
4. **Reservoir Dirichlet** — L equals L_reservoir on the outer boundary throughout.
5. **Parameters come from the §1.6.6 MathDescription.**
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from vcell_fenics.approaches.multicompartment.geometry import create_extracellular_annulus
from vcell_fenics.backend import BindingParameters, LigandReceptorBinding
from vcell_fenics.formalism import load_yaml

# Faster rates than the §1.6.6 defaults so the test equilibrates in few steps; the
# coupling mechanics are identical. (from_math_description is tested separately.)
_PARAMS = BindingParameters(k_on=0.5, k_off=0.1, l_reservoir=1.0, d_ligand=0.5, d_surface=0.05)


def _solver(params: BindingParameters = _PARAMS, *, dt: float = 0.05, h: float = 0.16) -> LigandReceptorBinding:
    annulus = create_extracellular_annulus(inner_radius=0.5, outer_radius=1.0, h=h)
    return LigandReceptorBinding(annulus, params, dt=dt, ligand_ic=1.0, free_ic=0.5, bound_ic=0.0)


def _membrane_trace_of_ligand(solver: LigandReceptorBinding) -> np.ndarray:
    """trace(L) at the membrane dofs, by matching membrane dof coordinates to the
    nearest bulk dof (P1: the membrane vertices coincide with bulk boundary vertices)."""
    state = solver.state
    membrane_xy = state.free_receptor.function_space.tabulate_dof_coordinates()[:, :2]
    bulk_xy = state.ligand.function_space.tabulate_dof_coordinates()[:, :2]
    nearest = cKDTree(bulk_xy).query(membrane_xy)[1]
    return np.asarray(state.ligand.x.array[nearest])


# ---------------------------------------------------------------------------
# 1. receptor conservation
# ---------------------------------------------------------------------------


def test_total_receptor_is_conserved() -> None:
    solver = _solver()
    total0 = solver.receptor_total()
    for _ in range(60):
        solver.step()

    assert solver.receptor_total() == pytest.approx(total0, rel=1e-10)


# ---------------------------------------------------------------------------
# 2. binding occurs and stays physical
# ---------------------------------------------------------------------------


def test_binding_proceeds_and_free_receptor_stays_nonnegative() -> None:
    solver = _solver()
    bound_before = solver.state.bound_receptor.x.array.mean()

    free_min_seen = solver.state.free_receptor.x.array.min()
    for _ in range(60):
        solver.step()
        free_min_seen = min(free_min_seen, float(solver.state.free_receptor.x.array.min()))

    assert solver.state.bound_receptor.x.array.mean() > bound_before + 0.1  # ρ_b rose from 0
    assert solver.state.free_receptor.x.array.mean() < 0.5  # ρ_f fell
    assert free_min_seen >= -1e-9  # never negative (stability), modulo round-off


# ---------------------------------------------------------------------------
# 3. detailed balance at steady state
# ---------------------------------------------------------------------------


def test_reaches_detailed_balance_equilibrium() -> None:
    solver = _solver()
    for _ in range(250):
        solver.step()

    state = solver.state
    residual = _PARAMS.k_on * _membrane_trace_of_ligand(solver) * state.free_receptor.x.array - (
        _PARAMS.k_off * state.bound_receptor.x.array
    )
    assert float(np.abs(residual).max()) < 1e-2  # b = k_on·trace(L)·ρ_f − k_off·ρ_b → 0


# ---------------------------------------------------------------------------
# 4. reservoir Dirichlet
# ---------------------------------------------------------------------------


def test_reservoir_dirichlet_is_held_on_the_outer_boundary() -> None:
    solver = _solver()
    for _ in range(20):
        solver.step()

    # The outer-boundary dofs sit at radius 1; check the ligand there equals L_reservoir.
    state = solver.state
    xy = state.ligand.function_space.tabulate_dof_coordinates()[:, :2]
    on_outer = np.linalg.norm(xy, axis=1) > 0.999
    assert np.allclose(state.ligand.x.array[on_outer], _PARAMS.l_reservoir, atol=1e-9)


# ---------------------------------------------------------------------------
# 5. parameters from the §1.6.6 MathDescription
# ---------------------------------------------------------------------------

_SECTION_1_6_6 = """
math_description:
  geometry: cell_with_extracellular
  subdomains:
    - { name: extracellular, kind: volume, motion: { kind: none } }
    - { name: membrane, kind: surface, motion: { kind: prescribed, velocity: "0" } }
  variables:
    - { name: L, subdomain: extracellular }
    - { name: rho_f, subdomain: membrane }
    - { name: rho_b, subdomain: membrane }
  equations:
    - template: bulk_radv_diff
      variable: L
      subdomain: extracellular
      temporality: time_dependent
      terms: { diffusion: "0.5" }
      initial_condition: "1.0"
    - template: surface_pde_with_dilution
      variable: rho_f
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.05", source: "-(k_on * trace(L) * rho_f - k_off * rho_b)" }
      initial_condition: "0.5"
    - template: surface_pde_with_dilution
      variable: rho_b
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.05", source: "k_on * trace(L) * rho_f - k_off * rho_b" }
      initial_condition: "0.0"
  parameters:
    - { name: k_on, value: 0.10 }
    - { name: k_off, value: 0.02 }
    - { name: L_reservoir, value: 1.00 }
"""


def test_parameters_extracted_from_math_description() -> None:
    params = BindingParameters.from_math_description(load_yaml(_SECTION_1_6_6))

    assert params.k_on == pytest.approx(0.10)
    assert params.k_off == pytest.approx(0.02)
    assert params.l_reservoir == pytest.approx(1.00)
    assert params.d_ligand == pytest.approx(0.5)
    assert params.d_surface == pytest.approx(0.05)

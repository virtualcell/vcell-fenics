"""Verification of the formalism-driven mixed-dimensional coupled assembly (§1.6.6).

The §1.6.6 ligand-receptor model runs **through `assemble()`** (dispatched to the
coupled path by a `CoupledGeometry`): the block system, the bulk↔surface coupling, the
semi-implicit lag, BCs, and — when the membrane moves — the conservative-ALE dilution
terms are all derived from the MathDescription's expressions. The checks:

1. **Dispatch** — `assemble()` on a `CoupledGeometry` returns a `CoupledProblem`.
2. **Receptor conservation** — ∫_Γ (ρ_f + ρ_b) is invariant.
3. **Binding + stability** — ρ_b rises, ρ_f stays non-negative.
4. **Detailed balance** — k_on·trace(L)·ρ_f = k_off·ρ_b at steady state.
5. **Reservoir Dirichlet** — L = L_reservoir on the outer boundary.
6. **A purely-surface source stays local** — a non-`trace` source is not mistaken for
   coupling.
7. **Moving membrane (conservative ALE)** — under membrane expansion the receptor total
   and (in a closed cell) the total ligand are conserved, and consumption keeps L below
   the reservoir. The closed-ligand check is the mass-balance gate on the moving-boundary
   flux.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from tests.gmsh_meshers.multicompartment.geometry import make_extracellular_annulus_geometry
from vcell_fenics.backend import CoupledProblem, assemble
from vcell_fenics.formalism import load_yaml


def _model(
    *,
    k_on: float = 0.5,
    k_off: float = 0.1,
    free_source: str | None = None,
    velocity: str = "0",
    reservoir: bool = True,
) -> str:
    binding = "k_on * trace(L) * rho_f - k_off * rho_b"
    rho_f_source = free_source if free_source is not None else f"-({binding})"
    # The binding consumes ligand, so the L Neumann is −(binding): the backend's
    # natural-BC convention is `D∇u·n = h` with positive h an influx, so a sink needs
    # the negated rate (see §1.6 sign note in the doc).
    reservoir_bc = (
        '    - { kind: dirichlet, variable: L, boundary: outer, expression: "L_reservoir" }\n' if reservoir else ""
    )
    return f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: ext, kind: volume, motion: {{ kind: none }} }}
    - {{ name: mem, kind: surface, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: L, subdomain: ext }}
    - {{ name: rho_f, subdomain: mem }}
    - {{ name: rho_b, subdomain: mem }}
  equations:
    - template: bulk_radv_diff
      variable: L
      subdomain: ext
      temporality: time_dependent
      terms: {{ diffusion: "0.5" }}
      initial_condition: "1.0"
    - template: surface_pde_with_dilution
      variable: rho_f
      subdomain: mem
      temporality: time_dependent
      terms: {{ diffusion: "0.05", source: "{rho_f_source}" }}
      initial_condition: "0.5"
    - template: surface_pde_with_dilution
      variable: rho_b
      subdomain: mem
      temporality: time_dependent
      terms: {{ diffusion: "0.05", source: "{binding}" }}
      initial_condition: "0.0"
  boundary_conditions:
{reservoir_bc}    - {{ kind: neumann, variable: L, boundary: interface, expression: "-({binding})" }}
  parameters:
    - {{ name: k_on, value: {k_on} }}
    - {{ name: k_off, value: {k_off} }}
    - {{ name: L_reservoir, value: 1.0 }}
"""


def _geom(h: float = 0.16):  # type: ignore[no-untyped-def]
    return make_extracellular_annulus_geometry(
        "cell", extracellular="ext", membrane="mem", interface="interface", outer="outer", h=h
    )


def _solve(model: str, *, steps: int, dt: float = 0.05, h: float = 0.16) -> CoupledProblem:
    problem = assemble(load_yaml(model), _geom(h), dt=dt)
    assert isinstance(problem, CoupledProblem)
    for _ in range(steps):
        problem.step()
    return problem


def _receptor_total(problem: CoupledProblem) -> float:
    return problem.integral(problem.surface[0] + problem.surface[1], surface=True)


# ---------------------------------------------------------------------------
# 1. dispatch
# ---------------------------------------------------------------------------


def test_assemble_dispatches_on_coupled_geometry() -> None:
    problem = assemble(load_yaml(_model()), _geom(), dt=0.05)
    assert isinstance(problem, CoupledProblem)
    assert problem.bulk_var == "L"
    assert problem.surface_vars == ["rho_f", "rho_b"]


# ---------------------------------------------------------------------------
# 2. receptor conservation
# ---------------------------------------------------------------------------


def test_total_receptor_conserved() -> None:
    problem = assemble(load_yaml(_model()), _geom(), dt=0.05)
    assert isinstance(problem, CoupledProblem)
    total0 = _receptor_total(problem)
    for _ in range(60):
        problem.step()

    assert _receptor_total(problem) == pytest.approx(total0, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. binding proceeds and stays physical
# ---------------------------------------------------------------------------


def test_binding_proceeds_and_free_receptor_nonnegative() -> None:
    problem = assemble(load_yaml(_model()), _geom(), dt=0.05)
    assert isinstance(problem, CoupledProblem)

    free_min = float(problem.field("rho_f").x.array.min())
    for _ in range(60):
        problem.step()
        free_min = min(free_min, float(problem.field("rho_f").x.array.min()))

    assert problem.field("rho_b").x.array.mean() > 0.1  # ρ_b rose from 0
    assert problem.field("rho_f").x.array.mean() < 0.5  # ρ_f fell
    assert free_min >= -1e-9


# ---------------------------------------------------------------------------
# 4. detailed balance
# ---------------------------------------------------------------------------


def test_detailed_balance_at_steady_state() -> None:
    problem = _solve(_model(), steps=250)

    rho_f = problem.field("rho_f")
    rho_b = problem.field("rho_b")
    membrane_xy = rho_f.function_space.tabulate_dof_coordinates()[:, :2]
    bulk_xy = problem.bulk.function_space.tabulate_dof_coordinates()[:, :2]
    trace_l = problem.bulk.x.array[cKDTree(bulk_xy).query(membrane_xy)[1]]

    residual = 0.5 * trace_l * rho_f.x.array - 0.1 * rho_b.x.array  # k_on·trace(L)·ρ_f − k_off·ρ_b
    assert float(np.abs(residual).max()) < 1e-2


# ---------------------------------------------------------------------------
# 5. reservoir Dirichlet
# ---------------------------------------------------------------------------


def test_reservoir_dirichlet_held() -> None:
    problem = _solve(_model(), steps=20)

    xy = problem.bulk.function_space.tabulate_dof_coordinates()[:, :2]
    on_outer = np.linalg.norm(xy, axis=1) > 0.999
    assert np.allclose(problem.bulk.x.array[on_outer], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. a purely-surface source is not treated as coupling
# ---------------------------------------------------------------------------


def test_local_surface_source_is_not_coupling() -> None:
    # rho_f decays at a fixed rate (no trace(L)); rho_b still binds. The local-vs-
    # coupling split must route this source to the surface mesh, not the bulk ds.
    problem = _solve(_model(free_source="-0.3 * rho_f"), steps=40)

    # Receptor total is no longer conserved (rho_f has a sink with no matching source),
    # but the solve runs and stays finite — exercising the local-source path.
    assert np.all(np.isfinite(problem.field("rho_f").x.array))
    assert problem.field("rho_f").x.array.mean() < 0.5  # decayed


# ---------------------------------------------------------------------------
# 7. moving coupled surface — conservative ALE
# ---------------------------------------------------------------------------
#
# When the membrane moves, both fields co-move with the deforming domain: the surface
# species get the dilution term ρ ∇_Γ·v_Γ and the bulk ligand the volumetric L ∇·v_mesh;
# the consumption flux at the co-moving membrane is the ordinary diffusive Neumann (no
# moving-boundary relative-flux correction). The defining checks are conservation across
# the motion — receptor total, and (in a *closed* extracellular space) total ligand.

_MOVING = "0.2 * geom.x / geom.radius"  # radial membrane expansion


def _membrane_length(problem: CoupledProblem) -> float:
    import ufl

    return problem.integral(ufl.as_ufl(1.0), surface=True)


def test_moving_membrane_dilutes_and_conserves_receptor() -> None:
    problem = assemble(load_yaml(_model(velocity=_MOVING)), _geom(h=0.13), dt=0.04)
    assert isinstance(problem, CoupledProblem)
    receptor0, length0 = _receptor_total(problem), _membrane_length(problem)
    for _ in range(15):
        problem.step()

    assert _membrane_length(problem) > 1.15 * length0  # the membrane genuinely stretched
    # The mandatory dilution term keeps ∫_Γ (ρ_f + ρ_b) invariant under the stretch;
    # without it the total would grow with the membrane area.
    assert _receptor_total(problem) == pytest.approx(receptor0, rel=5e-3)
    assert problem.field("rho_f").x.array.min() >= -1e-9


def test_moving_membrane_conserves_total_ligand_in_a_closed_cell() -> None:
    # The mass-balance gate: in a closed extracellular space (no reservoir) the ligand
    # only moves between free-in-bulk and membrane-bound, so ∫_Ω L + ∫_Γ ρ_b is invariant
    # — and stays so *under membrane motion* only if the moving-boundary flux (bulk
    # dilution + co-moving Neumann) is handled correctly.
    problem = assemble(load_yaml(_model(velocity=_MOVING, reservoir=False)), _geom(h=0.13), dt=0.04)
    assert isinstance(problem, CoupledProblem)

    def total_ligand() -> float:
        return problem.integral(problem.bulk, surface=False) + problem.integral(problem.surface[1], surface=True)

    ligand0, length0 = total_ligand(), _membrane_length(problem)
    for _ in range(15):
        problem.step()

    assert _membrane_length(problem) > 1.15 * length0
    assert total_ligand() == pytest.approx(ligand0, rel=5e-3)


def test_moving_membrane_consumption_keeps_ligand_below_reservoir() -> None:
    # With the correct consumption sign, binding is a *sink* on the ligand, so L never
    # exceeds the reservoir it is fed from (a maximum-principle check that also guards
    # the §1.6 Neumann sign convention).
    problem = _solve(_model(velocity=_MOVING), steps=15, dt=0.04, h=0.13)

    assert problem.bulk.x.array.max() <= 1.0 + 1e-2
    assert problem.field("rho_b").x.array.mean() > 0.05  # binding still proceeded

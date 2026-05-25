"""Inc-3b: the §1.4.5 two-species reference model, coupled and moving.

This is the CLAUDE.md first concrete goal: a 2D single cell with two receptor
species (active / inactive) on a prescribed-motion (radially expanding) membrane,
coupled by mass-action interconversion (k_on·inactive − k_off·active) and both
diffusing. It runs end-to-end through the formalism as one coupled solve over a
two-component space.

Two checks, both physical:

1. **Total mass conserved.** The reaction conserves active+inactive pointwise
   (its two source terms sum to zero) and dilution conserves each species'
   integral on the moving membrane, so ∫(active+inactive) is conserved.
2. **Reaction direction.** k_on > k_off drives inactive → active, so the active
   integral rises and the inactive integral falls — confirming the cross-variable
   coupling is wired (and signed) correctly.
"""

from __future__ import annotations

from pathlib import Path

from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import TermKind, assemble, make_disk_membrane_geometry
from vcell_fenics.formalism import load_yaml

_FIXTURE = Path(__file__).parent / "fixtures" / "section_1_4_5.yaml"


def _species_mass(dp: object, k: int) -> float:
    # ∫ of the k-th species over the membrane's current configuration.
    component = dp.unknown.sub(k)  # type: ignore[attr-defined]
    local = fem.assemble_scalar(fem.form(component * dp.dx))  # type: ignore[attr-defined]
    return float(dp.V.mesh.comm.allreduce(local, op=MPI.SUM))  # type: ignore[attr-defined]


def test_section_1_4_5_assembles_as_a_coupled_moving_problem() -> None:
    md = load_yaml(_FIXTURE)
    geometry = make_disk_membrane_geometry("disk_radius_1", surface_subdomain="membrane", radius=1.0, h=0.1)
    dp = assemble(md, geometry, dt=0.01)
    assert dp.term_kinds() == {
        TermKind.TIME_DERIVATIVE,
        TermKind.DIFFUSION,
        TermKind.DILUTION,
        TermKind.SOURCE,
    }


def test_section_1_4_5_conserves_total_species() -> None:
    md = load_yaml(_FIXTURE)
    geometry = make_disk_membrane_geometry("disk_radius_1", surface_subdomain="membrane", radius=1.0, h=0.1)
    dp = assemble(md, geometry, dt=0.01)

    total0 = _species_mass(dp, 0) + _species_mass(dp, 1)
    for _ in range(100):  # ṙ = 1, T = 1 ⇒ membrane expands r: 1 → 2
        dp.step()
    total1 = _species_mass(dp, 0) + _species_mass(dp, 1)

    assert abs(total1 - total0) / total0 < 0.02


def test_section_1_4_5_reaction_shifts_toward_active() -> None:
    md = load_yaml(_FIXTURE)
    geometry = make_disk_membrane_geometry("disk_radius_1", surface_subdomain="membrane", radius=1.0, h=0.1)
    dp = assemble(md, geometry, dt=0.01)

    active0, inactive0 = _species_mass(dp, 0), _species_mass(dp, 1)
    for _ in range(100):
        dp.step()
    active1, inactive1 = _species_mass(dp, 0), _species_mass(dp, 1)

    # k_on (0.10) > k_off (0.05): net conversion inactive → active.
    assert active1 > active0
    assert inactive1 < inactive0

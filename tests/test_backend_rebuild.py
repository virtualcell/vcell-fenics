"""Verification of the DiscreteProblem rebuild path (backend.rebuild_on_mesh).

The ALE remesh driver (`docs/modeling/ale-remesh-driver.md`) survives mesh tangling
by swapping to a fresh mesh and continuing. Because the IR is build-once (ADR 004),
that swap is teardown + reassemble with conservative state transfer, which is what
`rebuild_on_mesh` does. The checks follow the driver's verification plan:

1. **Structural** — the rebuilt problem has the same term kinds, lives on the new
   mesh, and preserves motion (a moving membrane stays moving).
2. **Conservation** — across the swap, ∫ of the state is preserved to round-off on
   surface (T2), bulk (T1), and coupled vector (§1.4.5) problems.
3. **Transparency** — a problem forced to remesh mid-run onto an equivalent mesh
   tracks a never-remesh reference (mass exactly, field to remap accuracy).
4. **u_prev is load-bearing** — a rebuild that leaves `previous` un-transferred
   takes a visibly wrong first step, proving step (d)-on-`u_prev` matters.
5. **Guard** — a non-P1 problem is rejected (the remaps require P1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import (
    DiscreteProblem,
    TermKind,
    assemble,
    make_disk_geometry,
    make_disk_membrane_geometry,
    rebuild_on_mesh,
)
from vcell_fenics.formalism import load_yaml

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

_SURFACE_DIFFUSION = """
math_description:
  geometry: disk_membrane
  subdomains:
    - { name: membrane, kind: surface, motion: { kind: none } }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.1" }
      initial_condition: "1.0 + 0.5 * cos(2 * geom.azimuth)"
"""

_MOVING_MEMBRANE = """
math_description:
  geometry: disk_membrane
  subdomains:
    - { name: membrane, kind: surface, motion: { kind: prescribed, velocity: "r_dot * geom.x / geom.radius" } }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      initial_condition: "1.0 + 0.3 * cos(2 * geom.azimuth)"
  parameters:
    - { name: r_dot, value: 1.0 }
"""

_BULK_DIFFUSION = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cytoplasm, kind: volume }
  variables:
    - { name: c, subdomain: cytoplasm }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cytoplasm
      temporality: time_dependent
      terms: { diffusion: "0.2" }
      initial_condition: "1.0 + 0.3 * geom.x[0]"
"""

_SECTION_1_4_5 = Path(__file__).parent / "fixtures" / "section_1_4_5.yaml"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _membrane_mesh(h: float, *, radius: float = 1.0):  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=radius, h=h).mesh_of(
        "membrane"
    )


def _disk_mesh(h: float, *, radius: float = 1.0):  # type: ignore[no-untyped-def]
    return make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=radius, h=h).mesh_of("cytoplasm")


def _surface_problem(h: float) -> DiscreteProblem:
    geom = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=h)
    return assemble(load_yaml(_SURFACE_DIFFUSION), geom, dt=0.01)


def _integral(f: fem.Function, dp: DiscreteProblem) -> float:
    local = fem.assemble_scalar(fem.form(f * dp.dx))
    return float(dp.V.mesh.comm.allreduce(local, op=MPI.SUM))


def _species_mass(dp: DiscreteProblem, k: int) -> float:
    local = fem.assemble_scalar(fem.form(dp.unknown.sub(k) * dp.dx))
    return float(dp.V.mesh.comm.allreduce(local, op=MPI.SUM))


def _amplitude(dp: DiscreteProblem) -> float:
    arr = dp.unknown.x.array
    return 0.5 * float(arr.max() - arr.min())


# ---------------------------------------------------------------------------
# 1. structural
# ---------------------------------------------------------------------------


def test_rebuild_preserves_structure_and_moves_to_new_mesh() -> None:
    dp = _surface_problem(0.3)
    new_mesh = _membrane_mesh(0.18)

    new = rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), new_mesh)

    assert new.term_kinds() == dp.term_kinds()
    assert new.V.mesh is new_mesh
    assert new.motion_velocity is None  # a static membrane stays static


def test_rebuild_preserves_motion() -> None:
    geom = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.3)
    dp = assemble(load_yaml(_MOVING_MEMBRANE), geom, dt=0.01)

    new = rebuild_on_mesh(dp, load_yaml(_MOVING_MEMBRANE), _membrane_mesh(0.18))

    assert TermKind.DILUTION in new.term_kinds()
    assert new.motion_velocity is not None


# ---------------------------------------------------------------------------
# 2. conservation across the swap
# ---------------------------------------------------------------------------


def test_surface_mass_conserved_across_rebuild() -> None:
    dp = _surface_problem(0.3)
    for _ in range(5):
        dp.step()
    mass_before = dp.total_mass()
    prev_before = _integral(dp.previous, dp)

    new = rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), _membrane_mesh(0.15))

    assert new.total_mass() == pytest.approx(mass_before, rel=1e-11)
    assert _integral(new.previous, new) == pytest.approx(prev_before, rel=1e-11)


def test_bulk_mass_conserved_across_rebuild() -> None:
    geom = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.3)
    dp = assemble(load_yaml(_BULK_DIFFUSION), geom, dt=0.01)
    for _ in range(5):
        dp.step()
    mass_before = dp.total_mass()

    new = rebuild_on_mesh(dp, load_yaml(_BULK_DIFFUSION), _disk_mesh(0.15))

    assert new.V.mesh.topology.dim == 2  # exercised the bulk (2D) transfer path
    assert new.total_mass() == pytest.approx(mass_before, rel=1e-11)


def test_coupled_species_mass_conserved_across_rebuild() -> None:
    geom = make_disk_membrane_geometry("disk_radius_1", surface_subdomain="membrane", radius=1.0, h=0.2)
    dp = assemble(load_yaml(_SECTION_1_4_5), geom, dt=0.01)
    for _ in range(10):
        dp.step()
    active_before, inactive_before = _species_mass(dp, 0), _species_mass(dp, 1)

    new_mesh = make_disk_membrane_geometry("disk_radius_1", surface_subdomain="membrane", radius=1.0, h=0.12).mesh_of(
        "membrane"
    )
    new = rebuild_on_mesh(dp, load_yaml(_SECTION_1_4_5), new_mesh)

    assert new.V.num_sub_spaces == 2  # exercised the per-component (vector) transfer
    assert _species_mass(new, 0) == pytest.approx(active_before, rel=1e-11)
    assert _species_mass(new, 1) == pytest.approx(inactive_before, rel=1e-11)


# ---------------------------------------------------------------------------
# 3. transparency — forced remesh tracks a never-remesh reference
# ---------------------------------------------------------------------------


def test_forced_remesh_tracks_reference() -> None:
    reference = _surface_problem(0.18)
    for _ in range(40):
        reference.step()

    dp = _surface_problem(0.18)
    for _ in range(20):
        dp.step()
    dp = rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), _membrane_mesh(0.18))  # swap onto an equivalent mesh
    for _ in range(20):
        dp.step()

    # Mass is conserved exactly across the swap; the field tracks the reference to
    # remap accuracy (the swapped mesh is an equivalent discretisation).
    assert dp.total_mass() == pytest.approx(reference.total_mass(), rel=1e-9)
    assert _amplitude(dp) == pytest.approx(_amplitude(reference), rel=0.02)


# ---------------------------------------------------------------------------
# 4. u_prev is load-bearing (subtlety 1)
# ---------------------------------------------------------------------------


def test_untransferred_previous_corrupts_the_first_step() -> None:
    dp = _surface_problem(0.2)
    for _ in range(5):
        dp.step()
    new_mesh = _membrane_mesh(0.15)

    correct = rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), new_mesh)
    broken = rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), new_mesh)
    broken.previous.x.array[:] = 0.0  # the mistake the driver warns against

    correct.step()
    broken.step()

    assert not np.allclose(correct.unknown.x.array, broken.unknown.x.array, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. guard — non-P1 is rejected
# ---------------------------------------------------------------------------


def test_rebuild_rejects_non_p1() -> None:
    geom = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.3)
    dp = assemble(load_yaml(_SURFACE_DIFFUSION), geom, dt=0.01, fe_degree=2)

    with pytest.raises(NotImplementedError, match="P1-only"):
        rebuild_on_mesh(dp, load_yaml(_SURFACE_DIFFUSION), _membrane_mesh(0.18))

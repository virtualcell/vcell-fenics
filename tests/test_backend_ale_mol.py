"""Method-of-lines on a moving membrane **with remeshing** (`run_moving_with_remeshing`).

Combines the strided ALE method-of-lines (`integrate_discrete_problem_stride`) with the remesh driver
so a *large* deformation — one that distorts the mesh past the quality budget — runs to completion,
swapping to a fresh mesh of the deformed configuration (conservative transfer) between strides. This
mirrors the backward-Euler remesh tests (`test_backend_ale.py`), with the per-stride solve done by the
adaptive-BDF `TS` instead of one backward-Euler step:

1. **Completes + advances** — a distorting motion runs to `t_final`, recording the inner TS steps.
2. **Remesh fires only when needed** — a distorting motion triggers remeshes; a uniform (scale-
   invariant) expansion never distorts, so it does not.
3. **A single remesh is mass-exact** — `∫ρ ds` is unchanged to round-off across one `remesh()`, the
   conservative remap carrying the MOL-integrated state.
4. **Remeshing is transparent** — a remeshed run tracks a forced-never-remesh run to remap accuracy.
5. **Guard** — a static subdomain is rejected (it uses the fixed-domain integrator).

As in the backward-Euler driver, the dilution scheme conserves `∫ρ ds` tightly only for *uniform*
motion; under non-uniform motion it drifts at O(h), so conservation is asserted across a single remesh,
not across a whole distorting run.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend import (
    ALEState,
    SolverConfiguration,
    assemble,
    integrate_discrete_problem_moving,
    make_disk_membrane_geometry,
    run_moving_with_remeshing,
    stride_with_remeshing,
)
from vcell_fenics.formalism import load_yaml

# A non-uniform radial motion: angle-dependent speed elongates the membrane and distorts the node
# spacing (the curve stays a simple radial graph). Uniform expansion is scale-invariant — never distorts.
_DISTORTING = "(1.0 + 0.8 * cos(2 * geom.azimuth)) * geom.x / geom.radius"
_UNIFORM = "geom.x / geom.radius"


def _model(velocity: str, *, ic: str = "1.0 + 0.5 * cos(geom.azimuth)", diffusion: str = "0.01") -> str:
    return f"""
math_description:
  geometry: disk_membrane
  subdomains:
    - {{ name: membrane, kind: surface, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: rho, subdomain: membrane }}
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "{ic}"
"""


def _geom(h: float = 0.12):  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=h)


def test_mol_run_completes_advances_and_records_steps() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = run_moving_with_remeshing(
        md, _geom(), SolverConfiguration(dt=0.01, t_final=1.5), motion_steps=50, target_h=0.12, quality_limit=1.4
    )

    assert state.t == pytest.approx(1.5, abs=1e-9)
    assert np.all(np.isfinite(state.problem.unknown.x.array))
    assert state.steps > 50  # the adaptive TS sub-stepped within the strides


def test_mol_remesh_fires_under_distortion_but_not_under_uniform_expansion() -> None:
    cfg = SolverConfiguration(dt=0.01, t_final=1.5)
    distorting = run_moving_with_remeshing(
        load_yaml(_model(_DISTORTING)), _geom(), cfg, motion_steps=50, target_h=0.12, quality_limit=1.4
    )
    uniform = run_moving_with_remeshing(
        load_yaml(_model(_UNIFORM)), _geom(), cfg, motion_steps=50, target_h=0.12, quality_limit=1.4
    )

    assert distorting.remesh_count >= 1  # the distorting motion needed remeshing
    assert uniform.remesh_count == 0  # scale-invariant expansion never crosses the budget


def test_mol_single_remesh_conserves_surface_mass() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = ALEState.initial(md, _geom(), SolverConfiguration(dt=0.01, t_final=1.5))
    state.problem.dt.value = 0.03  # the per-stride move magnitude
    for i in range(20):  # MOL-stride into a distorted configuration without auto-remeshing
        stride_with_remeshing(state, h=0.03, t_start=i * 0.03, quality_limit=99.0, target_h=0.12)

    assert state.problem.mesh_quality_growth() > 1.3  # genuinely distorted
    mass_before = state.problem.total_mass()

    state.remesh(0.12)

    assert state.problem.total_mass() == pytest.approx(mass_before, rel=1e-11)  # remap conserves ∫ρ ds
    assert state.problem.mesh_quality_growth() == pytest.approx(1.0, abs=0.1)  # budget reset


def test_mol_remeshed_run_tracks_forced_no_remesh_reference() -> None:
    md = load_yaml(_model(_DISTORTING))
    cfg = SolverConfiguration(dt=0.01, t_final=1.5)
    remeshed = run_moving_with_remeshing(md, _geom(), cfg, motion_steps=50, target_h=0.12, quality_limit=1.4)
    reference = run_moving_with_remeshing(md, _geom(), cfg, motion_steps=50, target_h=0.12, quality_limit=1e9)

    assert remeshed.remesh_count >= 1
    assert reference.remesh_count == 0
    # Both carry the same O(h) dilution drift; the conservative remap adds only remap-accuracy error.
    assert remeshed.problem.total_mass() == pytest.approx(reference.problem.total_mass(), rel=5e-3)


def test_mol_remeshing_driver_rejects_a_static_subdomain() -> None:
    static = """
math_description:
  geometry: disk_membrane
  subdomains:
    - { name: membrane, kind: surface }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: { diffusion: "0.01" }
      initial_condition: "1.0"
"""
    with pytest.raises(NotImplementedError, match="moving subdomain"):
        run_moving_with_remeshing(
            load_yaml(static), _geom(), SolverConfiguration(dt=0.01, t_final=0.1), motion_steps=4, target_h=0.12
        )


def test_no_remesh_mol_leaves_the_mesh_more_distorted() -> None:
    # The point of remeshing: without it the moving mesh degrades. A forced-no-remesh run ends far more
    # distorted than the remeshed run (which resets the budget each remesh), on the same motion.
    md = load_yaml(_model(_DISTORTING))
    problem = assemble(md, _geom(), dt=0.01)
    no_remesh = integrate_discrete_problem_moving(problem, t_final=1.5, motion_steps=50)
    assert no_remesh.time == pytest.approx(1.5)
    assert problem.mesh_quality_growth() > 2.0  # mesh badly distorted without remeshing

    remeshed = run_moving_with_remeshing(
        md, _geom(), SolverConfiguration(dt=0.01, t_final=1.5), motion_steps=50, target_h=0.12, quality_limit=1.4
    )
    assert remeshed.problem.mesh_quality_growth() < 1.6  # remeshing keeps it bounded near the limit

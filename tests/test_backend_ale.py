"""Verification of the ALE remesh driver (backend.ale).

The driver turns the v1 backend's "fail loudly when the mesh tangles" into "remesh
at the step boundary and continue" (`docs/modeling/ale-remesh-driver.md`). These
checks exercise the loop on a moving 1D membrane, following the sketch's plan:

1. **Completes + advances** — a distorting motion runs to t_final.
2. **Remesh fires only when needed** — a non-uniform (distorting) motion triggers
   remeshes; a uniform radial expansion (scale-invariant, never distorts) does not.
3. **Remesh is mass-exact** — a single remesh leaves ∫ρ ds unchanged to round-off
   and resets the distortion budget.
4. **Remeshing is transparent** — a remeshed run tracks a forced-never-remesh run
   to remap accuracy; with no remesh the driver reproduces `run` exactly.
5. **StepTooLarge** — a step that tangles even a fresh mesh is surfaced, not looped.

The dilution scheme itself only conserves ∫ρ ds tightly for *uniform* motion (it is
scale-invariant there); under non-uniform motion the scheme drifts at O(dt), so the
remesh-conservation check isolates a single `remesh()` rather than asserting tight
conservation across a whole distorting run.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend import (
    ALEState,
    SolverConfiguration,
    StepTooLarge,
    make_disk_membrane_geometry,
    run,
    run_with_remeshing,
    step_with_remeshing,
)
from vcell_fenics.formalism import load_yaml


def _model(velocity: str, *, ic: str = "1.0") -> str:
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
      terms: {{ diffusion: "0.01" }}
      initial_condition: "{ic}"
"""


# A non-uniform radial motion: speed varies with angle, so node spacing along the
# membrane diverges and the mesh distorts (the curve stays a simple radial graph).
_DISTORTING = "(1.0 + 0.8 * cos(2 * theta(x))) * x / r(x)"
# Uniform radial expansion: scale-invariant, so the mesh never distorts.
_UNIFORM = "x / r(x)"


def _geom(h: float = 0.12):  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=h)


# ---------------------------------------------------------------------------
# 1. completes and advances time
# ---------------------------------------------------------------------------


def test_run_completes_and_advances_to_t_final() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = run_with_remeshing(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0), target_h=0.12, quality_limit=1.4)

    assert state.t == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.isfinite(state.problem.unknown.x.array))


# ---------------------------------------------------------------------------
# 2. remesh fires only when the mesh actually distorts
# ---------------------------------------------------------------------------


def test_remesh_fires_under_distortion() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = run_with_remeshing(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0), target_h=0.12, quality_limit=1.4)

    assert state.remesh_count >= 1


def test_no_remesh_under_uniform_expansion() -> None:
    md = load_yaml(_model(_UNIFORM))
    state = run_with_remeshing(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0), target_h=0.12, quality_limit=1.4)

    # Uniform radial expansion preserves cell-size ratios exactly, so the budget is
    # never crossed — remeshing only fires when it is genuinely needed.
    assert state.remesh_count == 0


# ---------------------------------------------------------------------------
# 3. a single remesh conserves mass and resets the distortion budget
# ---------------------------------------------------------------------------


def test_remesh_conserves_mass_and_resets_budget() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = ALEState.initial(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0))
    # Distort without auto-remeshing (a budget large enough to never trip).
    for _ in range(20):
        step_with_remeshing(state, quality_limit=99.0, target_h=0.12)

    assert state.problem.mesh_quality_growth() > 1.3  # genuinely distorted
    mass_before = state.problem.total_mass()

    state.remesh(0.12)

    assert state.problem.total_mass() == pytest.approx(mass_before, rel=1e-11)
    assert state.problem.mesh_quality_growth() == pytest.approx(1.0, abs=0.1)  # budget reset


# ---------------------------------------------------------------------------
# 4. remeshing is transparent
# ---------------------------------------------------------------------------


def test_remeshed_run_tracks_forced_no_remesh_reference() -> None:
    md = load_yaml(_model(_DISTORTING))
    cfg = SolverConfiguration(dt=0.02, t_final=1.0)
    remeshed = run_with_remeshing(md, _geom(), cfg, target_h=0.12, quality_limit=1.4)
    reference = run_with_remeshing(md, _geom(), cfg, target_h=0.12, quality_limit=1e9)

    assert remeshed.remesh_count >= 1
    assert reference.remesh_count == 0
    # Both carry the same O(dt) dilution-scheme drift; the conservative remap adds
    # only remap-accuracy error, so the totals stay close.
    assert remeshed.problem.total_mass() == pytest.approx(reference.problem.total_mass(), rel=5e-3)


def test_driver_reproduces_run_when_no_remesh_needed() -> None:
    md = load_yaml(_model(_UNIFORM))
    cfg = SolverConfiguration(dt=0.02, t_final=1.0)
    driven = run_with_remeshing(md, _geom(), cfg, target_h=0.12, quality_limit=1e9)
    plain = run(md, _geom(), cfg)

    assert driven.remesh_count == 0
    assert np.allclose(driven.problem.unknown.x.array, plain.unknown.x.array, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. a step that tangles even a fresh mesh is surfaced as StepTooLarge
# ---------------------------------------------------------------------------


def test_step_too_large_is_raised() -> None:
    # Inward motion with dt large enough to collapse the membrane onto the origin in
    # a single step — remeshing cannot rescue it.
    md = load_yaml(_model("-1.0 * x / r(x)"))
    state = ALEState.initial(md, _geom(0.3), SolverConfiguration(dt=1.0, t_final=1.0))

    with pytest.raises(StepTooLarge):
        step_with_remeshing(state, quality_limit=2.0, target_h=0.3)

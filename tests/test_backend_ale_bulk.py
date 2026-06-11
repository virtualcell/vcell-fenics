"""Verification of the ALE driver on a moving *bulk* region (codim-0).

The driver (`backend.ale`) now handles a moving bulk subdomain as well as a
membrane: `_remesh` dispatches on codimension, and for a bulk mesh it recovers the
deformed boundary loop (`BulkBoundaryTrace.boundary_loop()`) and meshes the region
directly (`mesh_region` returns the new bulk mesh), transferring the field with the
conservative bulk remap. Interior nodes are carried by harmonic-extension
mesh-motion, which stays smooth but — under a non-affine boundary motion — still
distorts enough over time to need remeshing.

Checks, mirroring the membrane driver tests:

1. **Completes + remeshes** — a distorting bulk motion runs to t_final and triggers
   at least one remesh, on a mesh that stays 2D (the bulk path, not the membrane one).
2. **A single bulk remesh is mass-exact** — ∫c dx is unchanged to round-off and the
   distortion budget resets, on a still-2D mesh.
3. **Remeshing is transparent** — a remeshed run tracks a forced-never-remesh run.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend import (
    ALEState,
    SolverConfiguration,
    make_disk_geometry,
    run_with_remeshing,
    step_with_remeshing,
)
from vcell_fenics.formalism import load_yaml


def _model(velocity: str, *, ic: str = "1.0 + 0.3 * x[0]") -> str:
    return f"""
math_description:
  geometry: disk_2d
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "0.02" }}
      initial_condition: "{ic}"
"""


# Angle-dependent radial speed: a non-affine bulk motion whose harmonic interior
# fill still distorts over time (the curve stays a simple radial graph).
_DISTORTING = "(1.0 + 0.7 * cos(2 * theta(x))) * x / r(x)"


def _geom(h: float = 0.18):  # type: ignore[no-untyped-def]
    return make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=h)


# ---------------------------------------------------------------------------
# 1. completes, remeshes, and stays a bulk mesh
# ---------------------------------------------------------------------------


def test_bulk_run_completes_and_remeshes() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = run_with_remeshing(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0), target_h=0.18, quality_limit=2.0)

    assert state.t == pytest.approx(1.0, abs=1e-9)
    assert state.remesh_count >= 1
    assert state.problem.V.mesh.topology.dim == 2  # remeshed via the bulk path, not the membrane one
    assert np.all(np.isfinite(state.problem.unknown.x.array))


# ---------------------------------------------------------------------------
# 2. a single bulk remesh conserves mass and resets the budget
# ---------------------------------------------------------------------------


def test_bulk_remesh_conserves_mass_and_resets_budget() -> None:
    md = load_yaml(_model(_DISTORTING))
    state = ALEState.initial(md, _geom(), SolverConfiguration(dt=0.02, t_final=1.0))
    for _ in range(25):  # distort without auto-remeshing
        step_with_remeshing(state, quality_limit=999.0, target_h=0.18)

    assert state.problem.mesh_quality_growth() > 1.3  # genuinely distorted
    mass_before = state.problem.total_mass()

    state.remesh(0.18)

    assert state.problem.V.mesh.topology.dim == 2
    assert state.problem.total_mass() == pytest.approx(mass_before, rel=1e-11)
    assert state.problem.mesh_quality_growth() == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# 3. remeshing is transparent
# ---------------------------------------------------------------------------


def test_bulk_remeshed_run_tracks_forced_no_remesh_reference() -> None:
    md = load_yaml(_model(_DISTORTING))
    cfg = SolverConfiguration(dt=0.02, t_final=1.0)
    remeshed = run_with_remeshing(md, _geom(), cfg, target_h=0.18, quality_limit=2.0)
    reference = run_with_remeshing(md, _geom(), cfg, target_h=0.18, quality_limit=1e9)

    assert remeshed.remesh_count >= 1
    assert reference.remesh_count == 0
    # Conservative bulk transfer adds only remap-accuracy error, so the totals agree.
    assert remeshed.problem.total_mass() == pytest.approx(reference.problem.total_mass(), rel=5e-3)

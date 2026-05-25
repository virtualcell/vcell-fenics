"""Tests for the SolverConfiguration + run driver (the (MathDescription,
Geometry, SolverConfiguration) triple of §3.4).

`run` should advance a model to t_final and return the solved problem; these
check it drives a real model to the same end state as a manual step loop, on
both the scalar and the moving-membrane paths.
"""

from __future__ import annotations

import numpy as np

from vcell_fenics.backend import (
    SolverConfiguration,
    assemble,
    make_disk_geometry,
    make_disk_membrane_geometry,
    run,
)
from vcell_fenics.formalism import load_yaml

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
      initial_condition: "1.0 + 0.3 * x[0]"
"""

_MOVING_MEMBRANE = """
math_description:
  geometry: disk_membrane
  subdomains:
    - name: membrane
      kind: surface
      motion: { kind: prescribed, velocity: "r_dot * x / r(x)" }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      initial_condition: "1.0"
  parameters:
    - { name: r_dot, value: 1.0 }
"""


def test_run_reaches_same_state_as_manual_loop() -> None:
    md = load_yaml(_BULK_DIFFUSION)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)

    driven = run(md, geometry, SolverConfiguration(dt=0.01, t_final=0.5))

    manual = assemble(load_yaml(_BULK_DIFFUSION), geometry, dt=0.01)
    for _ in range(50):  # 0.5 / 0.01
        manual.step()

    assert np.allclose(driven.unknown.x.array, manual.unknown.x.array)


def test_run_drives_a_moving_membrane_and_conserves_mass() -> None:
    md = load_yaml(_MOVING_MEMBRANE)
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.1)

    dp = run(md, geometry, SolverConfiguration(dt=0.01, t_final=1.0))

    # Initial mass was 2π (ρ = 1 on the unit circle); auto-dilution conserves it
    # to O(dt) even though the membrane doubles in length over the run.
    assert abs(dp.total_mass() - 2.0 * np.pi) / (2.0 * np.pi) < 0.02

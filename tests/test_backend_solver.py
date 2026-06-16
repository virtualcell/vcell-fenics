"""Tests for the SolverConfiguration + run driver (the (MathDescription,
Geometry, SolverConfiguration) triple of §3.4).

`run` should advance a model to t_final and return the solved problem; these
check it drives a real model to the same end state as a manual step loop, on
both the scalar and the moving-membrane paths, and that the `method_of_lines`
time-integration choice (PETSc `TS`) matches backward Euler on a linear model
and additionally handles a nonlinear source that backward Euler cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

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


# A nonlinear (logistic) source `growth·c·(1 − c)` — bilinear in c, so backward Euler's
# lhs/rhs split cannot assemble it; the method-of-lines TS integrator handles it via the
# inner Newton. The spatially-uniform steady state is the carrying capacity c = 1.
_LOGISTIC = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cytoplasm, kind: volume }
  variables:
    - { name: c, subdomain: cytoplasm }
  parameters:
    - { name: growth, value: 3.0 }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cytoplasm
      temporality: time_dependent
      terms: { diffusion: "0.05", source: "growth * c * (1.0 - c)" }
      initial_condition: "0.3"
"""


def test_method_of_lines_matches_backward_euler_on_a_linear_model() -> None:
    md, geometry = (
        load_yaml(_BULK_DIFFUSION),
        make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2),
    )
    backward_euler = run(md, geometry, SolverConfiguration(dt=0.01, t_final=0.5))
    mol_config = SolverConfiguration(dt=0.01, t_final=0.5, time_integration="method_of_lines")
    method_of_lines = run(load_yaml(_BULK_DIFFUSION), geometry, mol_config)
    # Same continuous solution by two time schemes — they agree to the discretisation error.
    assert float(np.abs(backward_euler.unknown.x.array - method_of_lines.unknown.x.array).max()) < 5e-3


def test_method_of_lines_handles_a_nonlinear_source() -> None:
    # The bilinear logistic source backward Euler cannot represent, integrated by TS to the
    # nonlinear steady state c = 1 (a fixed point, so robust to the transient time accuracy).
    md = load_yaml(_LOGISTIC)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)
    solved = run(md, geometry, SolverConfiguration(dt=0.05, t_final=5.0, time_integration="method_of_lines"))
    values = solved.unknown.x.array
    assert abs(float(values.mean()) - 1.0) < 1e-2  # reaches the carrying capacity
    assert float(np.abs(values - values.mean()).max()) < 1e-6  # stays uniform


def test_method_of_lines_rejects_a_moving_subdomain() -> None:
    # Method-of-lines is fixed-domain; a prescribed-motion model must use the per-step path.
    md = load_yaml(_MOVING_MEMBRANE)
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.1)
    with pytest.raises(NotImplementedError, match="fixed-domain"):
        run(md, geometry, SolverConfiguration(dt=0.01, t_final=0.1, time_integration="method_of_lines"))


def test_run_rejects_unknown_time_integration() -> None:
    md = load_yaml(_BULK_DIFFUSION)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)
    with pytest.raises(ValueError, match="time_integration must be"):
        run(md, geometry, SolverConfiguration(dt=0.01, t_final=0.1, time_integration="rk4"))

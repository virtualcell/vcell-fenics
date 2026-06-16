"""Runtime-failure translation (backend/diagnostics) — the runtime half of
docs/modeling/validation-and-diagnostics.md §5.

When a solve fails deep inside PETSc, the integrators should raise a `SolveError` whose message
names *what about the model* went wrong (the time, the likely cause, a fix), with the original
low-level exception chained — not let a bare reason code surface.
"""

from __future__ import annotations

import pytest
from petsc4py import PETSc

from vcell_fenics.backend import SolveError, SolverConfiguration, make_disk_geometry, run
from vcell_fenics.backend.diagnostics import linear_step_failure_message
from vcell_fenics.formalism import load_yaml

# An autocatalytic source dc/dt = +k·c² grows without bound and blows up in finite time
# (t* = 1/(k·c₀) = 0.2 here), so the residual goes non-finite mid-integration.
_BLOWUP = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  parameters:
    - { name: k, value: 5.0 }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.0", source: "k * c * c" }
      initial_condition: "1.0"
"""


def test_method_of_lines_blowup_raises_a_translated_solve_error() -> None:
    md = load_yaml(_BLOWUP)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    with pytest.raises(SolveError) as excinfo:
        run(md, geometry, SolverConfiguration(dt=0.05, t_final=2.0, time_integration="method_of_lines"))

    message = str(excinfo.value)
    assert "non-finite" in message  # the cause, in model terms
    assert "t ≈" in message  # localised in time (the blow-up is before t_final)
    assert "source" in message  # points at the term class to check
    assert isinstance(excinfo.value.__cause__, PETSc.Error)  # the raw failure is chained, not lost


def test_linear_step_failure_message_is_actionable() -> None:
    # The backward-Euler translation names a singular system and a concrete fix.
    message = linear_step_failure_message(0.42)
    assert "t ≈ 0.42" in message
    assert "singular" in message
    assert "constraint" in message or "Dirichlet" in message

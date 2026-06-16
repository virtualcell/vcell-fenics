"""Diagnostics (backend/diagnostics) — docs/modeling/validation-and-diagnostics.md §5 (runtime
failure translation) and the registry-#7 build-time check.

When a solve fails deep inside PETSc, the integrators raise a `SolveError` whose message names
*what about the model* went wrong, with the original exception chained. And a nonlinear term
under backward Euler is caught *before* form compilation with a named fix (`NonlinearTermError`)
rather than a deep UFL arity mismatch.
"""

from __future__ import annotations

import pytest
from petsc4py import PETSc
from ufl.algorithms.check_arities import ArityMismatch

from vcell_fenics.backend import (
    NonlinearTermError,
    SolveError,
    SolverConfiguration,
    make_disk_geometry,
    run,
)
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


# A nonlinear (logistic) source growth·c·(1 − c) — bilinear in c, so backward Euler's lhs/rhs
# split cannot assemble it (the form is non-affine in the trial).
_NONLINEAR = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  parameters:
    - { name: growth, value: 3.0 }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.05", source: "growth * c * (1.0 - c)" }
      initial_condition: "0.3"
"""


def test_backward_euler_rejects_a_nonlinear_source_with_a_named_fix() -> None:
    # The registry-#7 build-time check: a nonlinear source under backward Euler raises a clear
    # NonlinearTermError naming the method-of-lines fix — caught in pure UFL before form
    # compilation, so no raw arity-mismatch traceback (and no JIT-cache poisoning).
    md = load_yaml(_NONLINEAR)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    with pytest.raises(NonlinearTermError) as excinfo:
        run(md, geometry, SolverConfiguration(dt=0.05, t_final=0.1))  # backward_euler default

    message = str(excinfo.value)
    assert "nonlinear" in message  # names the cause
    assert "method-of-lines" in message or "method_of_lines" in message  # names the fix
    assert isinstance(excinfo.value.__cause__, ArityMismatch)  # the UFL detail is chained, not lost


def test_method_of_lines_still_accepts_the_same_nonlinear_source() -> None:
    # The check is integrator-specific: the very model backward Euler rejects integrates fine by
    # method-of-lines (the inner Newton), reaching the carrying-capacity equilibrium c = 1.
    md = load_yaml(_NONLINEAR)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    solved = run(md, geometry, SolverConfiguration(dt=0.05, t_final=5.0, time_integration="method_of_lines"))
    assert abs(float(solved.unknown.x.array.mean()) - 1.0) < 1e-2

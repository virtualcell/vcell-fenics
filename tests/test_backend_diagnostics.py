"""Diagnostics (backend/diagnostics) — docs/modeling/validation-and-diagnostics.md §5 (runtime
failure translation) and the registry-#7 build-time check.

When a solve fails deep inside PETSc, the integrators raise a `SolveError` whose message names
*what about the model* went wrong, with the original exception chained. And a nonlinear term
under backward Euler is caught *before* form compilation with a named fix (`NonlinearTermError`)
rather than a deep UFL arity mismatch.
"""

from __future__ import annotations

import dolfinx.mesh
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from ufl.algorithms.check_arities import ArityMismatch

from vcell_fenics.backend import (
    NonlinearTermError,
    SolveError,
    SolverConfiguration,
    integrate_reaction_diffusion,
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


# A *rational* nonlinearity — a trial function in a denominator (a Hill / GTPase switch). Unlike the
# polynomial case above (which `ufl.lhs` mis-buckets, caught by the arity check), `ufl.lhs` raises a
# ValueError on this one. The backward-Euler split is composed lazily so that even a rational-nonlinear
# model can be assembled and handed to the method-of-lines integrator — this is what lets the real Rac/Rho
# public BioModel (source `a²/(1+a²)·b − a`) solve through the formalism pipeline.
_RATIONAL_NONLINEAR = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  parameters:
    - { name: rate, value: 3.0 }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.05", source: "rate * c * c / (1.0 + c * c) - c" }
      initial_condition: "0.6"
"""


def test_method_of_lines_accepts_a_rational_nonlinear_source() -> None:
    # The key check for the lazy backward-Euler composition: a rational source (a trial in a denominator)
    # makes `ufl.lhs` raise outright — so eagerly composing the BE split at assembly used to crash this
    # model even for method-of-lines. Deferring the split lets it assemble and integrate via the Newton
    # solve, settling to a finite bistable-switch steady state (c² − 3c + 1 = 0 ⇒ c ≈ 0.38 or 2.62).
    md = load_yaml(_RATIONAL_NONLINEAR)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    solved = run(md, geometry, SolverConfiguration(dt=0.05, t_final=8.0, time_integration="method_of_lines"))
    mean = float(solved.unknown.x.array.mean())
    assert 0.0 < mean < 5.0  # solved to a finite, positive steady state (did not crash or blow up)


def test_backward_euler_rejects_a_rational_nonlinear_source() -> None:
    # Backward Euler still rejects it loudly — now via the ValueError `ufl.lhs` raises on the denominator,
    # reported as the same NonlinearTermError naming the method-of-lines fix.
    md = load_yaml(_RATIONAL_NONLINEAR)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    with pytest.raises(NonlinearTermError) as excinfo:
        run(md, geometry, SolverConfiguration(dt=0.05, t_final=0.1))  # backward_euler default
    assert "method-of-lines" in str(excinfo.value) or "method_of_lines" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)  # the UFL 'Argument in denominator' detail


# ---------------------------------------------------------------------------
# t = 0 pre-flight (validation-and-diagnostics.md §5.3): a model already broken
# at the initial condition (a non-finite residual) is caught before any step.
# ---------------------------------------------------------------------------


def _unit_square_species() -> tuple[dolfinx.mesh.Mesh, fem.Function]:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    return mesh, fem.Function(fem.functionspace(mesh, ("Lagrange", 1, (1,))))


def test_preflight_catches_a_nonfinite_residual_at_the_ic() -> None:
    # A reaction 1/c with the initial condition c = 0 is non-finite at t = 0; the pre-flight
    # raises before the adaptive solve grinds on it. The message names the initial condition
    # (distinguishing it from a mid-solve blow-up, which reports "at t ≈ …").
    mesh, c = _unit_square_species()  # c = 0
    with pytest.raises(SolveError) as excinfo:
        integrate_reaction_diffusion(
            mesh, c, diffusivities=[0.1], t_final=1.0, reaction=lambda u: ufl.as_vector([1.0 / u[0]])
        )
    assert "non-finite" in str(excinfo.value)
    assert "initial condition" in str(excinfo.value)


def test_preflight_catches_a_nonfinite_initial_condition() -> None:
    mesh, c = _unit_square_species()
    c.x.array[:] = 1.0
    c.x.array[3] = float("inf")  # a broken IC value
    with pytest.raises(SolveError, match="initial condition"):
        integrate_reaction_diffusion(mesh, c, diffusivities=[0.1], t_final=1.0)


def test_preflight_lets_a_well_posed_model_through() -> None:
    # The pre-flight is transparent to a finite model — it integrates normally.
    mesh, c = _unit_square_species()
    c.x.array[:] = 2.0
    integrate_reaction_diffusion(
        mesh, c, diffusivities=[0.1], t_final=0.5, reaction=lambda u: ufl.as_vector([-0.5 * u[0]])
    )
    assert float(c.x.array.mean()) < 2.0  # it decayed; no spurious pre-flight failure


# A singular forcing (1/0.0) is non-finite at t=0 — but only the SOURCE term, not diffusion —
# so the per-term probe (ADR-004 tagged terms) can attribute the failure to it.
_SINGULAR_SOURCE = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.1", source: "1.0 / 0.0" }
      initial_condition: "1.0"
"""


def test_preflight_localizes_a_nonfinite_term_to_the_source() -> None:
    # The pre-flight failure is attributed to the specific tagged term that is non-finite — the
    # source — rather than the generic "the residual", since the state and diffusion are finite.
    md = load_yaml(_SINGULAR_SOURCE)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.2)
    with pytest.raises(SolveError, match="the source term is non-finite") as excinfo:
        run(md, geometry, SolverConfiguration(dt=0.05, t_final=1.0, time_integration="method_of_lines"))
    assert "initial condition" in str(excinfo.value)  # still localised in time

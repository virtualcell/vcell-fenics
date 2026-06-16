"""Translate low-level solver failures into model-level diagnostics.

The build-time validator (`formalism/validator.py`) catches what can be known a priori; this
is the runtime half of `docs/modeling/validation-and-diagnostics.md` (§5). When a solve fails
deep inside PETSc — a Newton divergence, a non-finite residual, a singular Jacobian — the raw
exception is a reason code that names nothing in the model. These helpers catch the known
failure types and re-express each in terms a modeler can act on: *when* it failed, the likely
*cause*, and a concrete *fix*. The original PETSc/DOLFINx exception is always chained
(`raise SolveError(...) from original`) so the low-level detail is still there for debugging.
"""

from __future__ import annotations

from petsc4py import PETSc


class SolveError(RuntimeError):
    """A solver failure translated into model-level terms (the original low-level exception is
    chained). Raised by the integrators when a PETSc solve fails, so callers see *what about the
    model* went wrong rather than a bare reason code."""


class NonlinearTermError(RuntimeError):
    """A model term is nonlinear in the unknown, but the chosen integrator (backward Euler) can
    only assemble terms *affine* in the unknown. A build-time check (registry #7) — raised before
    form compilation, so the modeler sees a named fix instead of a deep UFL/FFCx arity mismatch."""


def nonlinear_backward_euler_message() -> str:
    """The fix-oriented message for a nonlinear term under backward Euler."""

    return (
        "a term is nonlinear in the unknown (e.g. a product like c*c, or a function of c such as "
        "exp(c) or c/(1+c)) — backward Euler can only assemble terms affine in the unknown, so it "
        "cannot lower this model. Use the method-of-lines integrator "
        "(SolverConfiguration(time_integration='method_of_lines'), which handles nonlinear "
        "reactions via Newton), or linearise / lag the term."
    )


def preflight_failure_message() -> str:
    """The message for a residual that is already non-finite at the initial condition — caught by
    the `t = 0` pre-flight, before any step is attempted."""

    return (
        "the residual is non-finite (NaN/Inf) at the initial condition (t = 0), before any step — "
        "the initial condition, or a term evaluated at it (a division by a quantity that is zero "
        "at t=0, or a fractional power / logarithm of a non-positive value), produced NaN/Inf. "
        "Check the initial condition and the source / reaction terms."
    )


def ts_failure_message(ts: PETSc.TS) -> str:
    """A model-level message for a failed PETSc `TS` (method-of-lines) solve, from its and its
    inner `SNES`'s converged reasons and the time it reached."""

    time = float(ts.getTime())
    ts_reason = ts.getConvergedReason()
    snes = ts.getSNES()
    snes_reason = snes.getConvergedReason()
    iterations = snes.getIterationNumber()
    snes_reasons = PETSc.SNES.ConvergedReason
    ts_reasons = PETSc.TS.ConvergedReason
    at = f"at t ≈ {time:.4g}"

    if snes_reason in (snes_reasons.DIVERGED_FUNCTION_NANORINF, snes_reasons.DIVERGED_OBJECTIVE_NANORINF):  # type: ignore[comparison-overlap]  # petsc4py stubs the reason as int
        return (
            f"the residual became non-finite (NaN/Inf) {at} — likely a blow-up (e.g. an "
            f"autocatalytic source growing without bound), a division by a quantity that reached "
            f"zero, or a fractional power of a negative value. Check the reaction / source terms."
        )
    if snes_reason == snes_reasons.DIVERGED_LINEAR_SOLVE:  # type: ignore[comparison-overlap]  # petsc4py stubs the reason as int
        return (
            f"the linear solve inside Newton failed {at} — the Jacobian is likely singular or "
            f"ill-conditioned, e.g. an unconstrained null space (a rigid-body or constant mode). "
            f"Add a screening term, a Dirichlet condition, or another constraint."
        )
    if ts_reason == ts_reasons.DIVERGED_STEP_REJECTED:  # type: ignore[comparison-overlap]  # petsc4py stubs the reason as int
        return (
            f"the adaptive integrator could not take a stable step {at} — the problem is likely "
            f"too stiff there, or the solution near-singular. Check the model, or adjust the "
            f"solver tolerances."
        )
    return (
        f"the nonlinear (Newton) solve failed to converge {at} (after {iterations} iterations) — "
        f"the step may be too stiff or the problem ill-posed. Try a smaller dt, or check the "
        f"reaction rate constants and the model's well-posedness."
    )


def linear_step_failure_message(time: float) -> str:
    """A model-level message for a failed backward-Euler linear solve at `time`. The direct
    solve fails when the system is singular — typically an unconstrained null space."""

    return (
        f"the backward-Euler linear solve failed at t ≈ {time:.4g} — the system is likely "
        f"singular (an unconstrained null space, e.g. a constant or rigid-body mode) or badly "
        f"ill-conditioned. Add a constraint (a Dirichlet condition or screening term); for a "
        f"nonlinear or stiff source use the method-of-lines integrator instead."
    )

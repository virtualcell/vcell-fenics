"""Record an adaptive PETSc ``TS`` solution at prescribed output times without disturbing its steps.

VCell asks for results at fixed output times (uniform or explicit `OutputOptions`), while the
method-of-lines integrators step adaptively. Stopping the integrator on every output time
(``ExactFinalTime.MATCHSTEP`` per interval) would restart BDF at order 1 each time; instead a ``TS``
monitor watches the accepted steps and, as each output time is crossed, fills a work vector with
``TSInterpolate`` (the integrator's own dense output) — or a plain copy when a step lands on the time.
The step sequence is exactly the one an unmonitored run takes (`docs/decisions/010-results-bundle-vtu-zarr.md`
§6(e): interpolated values carry the same error as the steps themselves).

The monitor is collective: ``TS`` calls it on every rank after every step, so ``emit`` may assemble,
interpolate or gather.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from petsc4py import PETSc


class OutputMonitor:
    """A ``TS`` monitor that calls ``emit(t_k, work)`` once per output time ``t_k`` and ``progress(t)``
    after every accepted step. Install with ``ts.setMonitor(monitor)``; call :meth:`finish` after
    ``ts.solve`` so an output time on the final step is never missed.

    An output time equal to ``t_start`` is the initial state: ``TS`` calls its monitors once before
    the first step (step 0, ``t = t_start``), and that call records it. Earlier times are skipped.
    Times are matched with a tolerance of ``1e-9`` of the interval, so a planned time computed as
    ``k·Δt`` still matches ``t_final`` despite round-off.
    """

    def __init__(
        self,
        output_times: Sequence[float],
        *,
        t_start: float,
        t_final: float,
        work: PETSc.Vec,
        emit: Callable[[float, PETSc.Vec], None],
        progress: Callable[[float], None] | None = None,
    ) -> None:
        span = t_final - t_start
        if span <= 0.0:
            raise ValueError(f"empty integration interval [{t_start}, {t_final}]")
        self._tol = 1.0e-9 * span
        late = [t for t in output_times if t > t_final + self._tol]
        if late:
            raise ValueError(f"output times {late} lie beyond t_final={t_final}")
        self._pending = sorted({float(t) for t in output_times if t >= t_start - self._tol})
        self._work = work
        self._emit = emit
        self._progress = progress

    def __call__(self, ts: PETSc.TS, step: int, t: float, u: PETSc.Vec) -> None:
        while self._pending and self._pending[0] <= t + self._tol:
            self._record(self._pending.pop(0), ts, t, u)
        if self._progress is not None:
            self._progress(t)

    def _record(self, t_out: float, ts: PETSc.TS, t: float, u: PETSc.Vec) -> None:
        if abs(t_out - t) <= self._tol:
            u.copy(self._work)  # the step landed on the output time: its own state, not an interpolant
        else:
            ts.interpolate(t_out, self._work)
        self._emit(t_out, self._work)

    def finish(self, ts: PETSc.TS, t: float, u: PETSc.Vec) -> None:
        """Emit any output time the monitor did not reach, which is only legitimate at the final time
        ``t`` (within tolerance); anything later means the solve stopped short of the schedule."""

        while self._pending:
            t_out = self._pending.pop(0)
            if t_out > t + self._tol:
                raise RuntimeError(f"the integrator stopped at t={t} before output time {t_out}")
            self._record(t_out, ts, t, u)

    @property
    def remaining(self) -> tuple[float, ...]:
        return tuple(self._pending)

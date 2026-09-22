"""Method-of-lines reaction–diffusion–advection on a **fixed** mesh, via PETSc `TS`.

The fixed-domain RDA integrator — the apples-to-apples analogue of VCell's finite-volume
reaction-diffusion solver, which uses method-of-lines + SUNDIALS/CVODE. Here the FEM spatial
discretisation turns the PDE system into an implicit ODE/DAE `M ċ = G(c)`, and PETSc `TS`
integrates it with an **adaptive-order, adaptive-step BDF** (`TSBDF`) — the same strategy as
CVODE: L-stable, high-order, with error-controlled step selection that absorbs stiff reaction
kinetics without an explicit-stability `dt` cap.

This is "Option 2" of the time-integration options (see the FSI `step_two_phase_fsi_with_*`
docstrings for "Option 1", the per-step Newton used on the *moving* cell). MOL fits a *fixed*
domain cleanly — the spatial operators do not change in time, so the whole trajectory is one
`TS.solve`. For a moving ALE subdomain the operators change with the geometry, so `TS` cannot run
the whole trajectory; `integrate_discrete_problem_moving` instead **strides** — move the mesh
discretely, then `TS`-integrate each inter-move interval on the now-fixed configuration (first-order
in the mesh motion, adaptive-high-order in the reaction–diffusion within a stride).

`n` species live in one vector P1 space (component `k` ↔ species `k`). Each obeys

    ∂c_k/∂t = D_k ∇²c_k − u·∇c_k + R_k(c)

with a per-species diffusivity `D_k`, an optional prescribed advection velocity `u`, and an
optional reaction `R(c)` that **may be nonlinear** (the inner SNES Newton handles it). The weak
implicit residual `F(c, ċ) = ∫[ ċ·v + Σ_k(D_k ∇c_k·∇v_k + (u·∇c_k) v_k) − R(c)·v ] dx` and its
exact Jacobian `σ M + ∂F/∂c` (`σ` the `TS` shift, `M` the mass matrix) feed the `TS`
`IFunction`/`IJacobian` callbacks. A no-flux (natural Neumann) boundary is assumed, so `∫Σc` is
conserved by transport and changes only through a non-conservative reaction.

Verified (`tests/test_backend_reaction_diffusion.py`): a diffusion eigenmode decays at the
analytical rate; a stiff reaction is integrated accurately in far fewer steps than an explicit
stability bound would force (the adaptive-BDF win); a nonlinear `A + B ⇌ C` reaches detailed
balance; and a bump advects at the prescribed speed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.diagnostics import SolveError, preflight_failure_message, ts_failure_message
from vcell_fenics.backend.discrete import DiscreteProblem, TermKind
from vcell_fenics.backend.linear_solvers import set_preconditioner
from vcell_fenics.backend.output_times import OutputMonitor


@dataclass
class IntegrationResult:
    """The outcome of a `TS` integration: the final-time `solution` (the same `Function` passed
    in, mutated in place), the number of `steps` the adaptive integrator took, and the `time`
    actually reached."""

    solution: fem.Function
    steps: int
    time: float


def integrate_reaction_diffusion(
    mesh: Mesh,
    initial: fem.Function,
    *,
    diffusivities: list[float] | tuple[float, ...],
    t_final: float,
    reaction: Callable[[fem.Function], UflExpr] | None = None,
    advection: UflExpr | None = None,
    dt_initial: float | None = None,
    ts_type: str = "bdf",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    ksp_rtol: float = 1.0e-9,
) -> IntegrationResult:
    """Integrate the `n`-species reaction–diffusion–advection system on `mesh` to `t_final`
    with PETSc `TS` (adaptive BDF by default), method-of-lines.

    `initial` is a vector P1 `Function` (its `n` components are the species) holding the initial
    condition; it is **mutated in place** to the final state and returned in the result.
    `diffusivities` are the per-species `D_k` (length `n`). `reaction` is an optional callable
    `c ↦ R(c)` returning a length-`n` UFL vector — it may be nonlinear (e.g. bilinear mass
    action), handled by the integrator's inner Newton. `advection` is an optional velocity UFL
    expression `u` giving the `−u·∇c_k` term (a fixed Eulerian flow). `dt_initial` seeds the
    adaptive controller (defaults to `t_final / 100`); `rtol`/`atol` are its error tolerances.

    `ksp_type`/`pc_type`/`ksp_rtol` configure the inner Newton linear solver — by default a
    **GMRES + ILU** Krylov solve (the scalable VCell-style choice: ILU preconditioning is what
    keeps the per-step cost low as the system grows; a direct `"preonly"`/`"lu"` is fine for
    small problems). No-flux (natural) boundary. Returns an `IntegrationResult`.
    """

    space = initial.function_space
    n_species = space.dofmap.index_map_bs  # block size = component count (num_sub_spaces is 0 for n=1)
    if len(diffusivities) != n_species:
        raise ValueError(f"diffusivities has length {len(diffusivities)}, but there are {n_species} species")

    state = initial  # the unknown c; TS owns its vector and updates it in place
    rate = fem.Function(space)  # ċ, supplied by TS each callback
    test = ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=mesh)

    residual = ufl.inner(rate, test)  # mass term ċ·v
    for k in range(n_species):
        residual += diffusivities[k] * ufl.dot(ufl.grad(state[k]), ufl.grad(test[k]))  # D_k ∇c_k·∇v_k
        if advection is not None:
            residual += ufl.dot(advection, ufl.grad(state[k])) * test[k]  # u·∇c_k (Eulerian advection)
    if reaction is not None:
        residual -= ufl.inner(reaction(state), test)  # − R(c)·v  (may be nonlinear)

    options = _TimeStepperOptions(t_final, dt_initial, ts_type, rtol, atol, ksp_type, pc_type, ksp_rtol)
    steps, final_time = _run_time_stepper(state, rate, residual * dx, options)
    return IntegrationResult(solution=state, steps=steps, time=final_time)


@dataclass(frozen=True)
class _TimeStepperOptions:
    t_final: float
    dt_initial: float | None
    ts_type: str
    rtol: float
    atol: float
    ksp_type: str
    pc_type: str
    ksp_rtol: float


def _run_time_stepper(
    state: fem.Function,
    rate: fem.Function,
    residual: UflExpr,
    options: _TimeStepperOptions,
    bcs: Sequence[fem.DirichletBC] = (),
    on_time: Callable[[float], None] | None = None,
    localize: Callable[[], str | None] | None = None,
    t_start: float = 0.0,
    output_times: Sequence[float] = (),
    on_output: Callable[[float, fem.Function], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[int, float]:
    """Integrate the implicit residual `F(state, rate) = 0` to `options.t_final` with PETSc `TS`,
    mutating `state` in place. `residual` is the UFL form of `F` (`rate` the time derivative `ċ`,
    a `Function` TS supplies each step); the exact Jacobian `σ ∂F/∂rate + ∂F/∂state` comes from
    `ufl.derivative`. Returns `(steps, final_time)`. Shared by the explicit-API integrator and
    the DiscreteProblem (formalism) driver, so the `TS` wiring lives in exactly one place.

    Strong Dirichlet `bcs` enter as algebraic constraints `x = g` on the boundary dofs: the IC
    is seeded to satisfy them, the residual's boundary rows are overwritten with `x − g` (so the
    inner Newton drives them to `g`), and the Jacobian is assembled with the bcs (boundary
    rows/columns zeroed, unit diagonal). The boundary dofs then stay at `g` for the whole run.

    With `on_output`, the solution at each of `output_times` in [`t_start`, `t_final`] is handed to it
    as a snapshot `Function` (`t_start` itself being the initial state), recorded from a `TS` monitor
    by interpolation so the adaptive step sequence is exactly an unmonitored run's
    (`backend/output_times.py`); the snapshot is reused between calls. `on_progress(t)` follows every
    accepted step."""

    space = state.function_space
    mesh = space.mesh
    residual_form = fem.form(residual)
    shift = fem.Constant(mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # petsc4py stubs ScalarType as a dtype
    jacobian = shift * ufl.derivative(residual, rate) + ufl.derivative(residual, state)
    jacobian_form = fem.form(jacobian)  # σ M + ∂F/∂c — exact (ufl.derivative)
    jacobian_matrix = fem_petsc.create_matrix(jacobian_form)
    # Dirichlet dofs zero only the Jacobian *rows* (not columns) so the Jacobian stays
    # consistent with the un-lifted residual when a boundary value moves (a time-dependent
    # g(t) leaves x_bc ≠ g at the start of each step). Symmetric (row+column) elimination
    # would need the residual lifted and breaks the line search here.
    bc_dofs = np.concatenate([bc.dof_indices()[0] for bc in bcs]).astype(np.int32) if bcs else None

    def _set_state(x: PETSc.Vec, x_dot: PETSc.Vec) -> None:
        x.copy(state.x.petsc_vec)
        state.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x_dot.copy(rate.x.petsc_vec)
        rate.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def evaluate_residual(_ts: PETSc.TS, t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        if on_time is not None:  # advance g(t) etc. to the stage time before assembling/applying BCs
            on_time(t)
        _set_state(x, x_dot)
        with result.localForm() as local:
            local.set(0.0)
            fem_petsc._assemble_vector_array(local.array_w, residual_form)  # type: ignore[attr-defined]
        result.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        if bcs:  # boundary residual rows ← x − g, so Newton holds the dofs at g
            fem_petsc.set_bc(result, bcs, x0=state.x.petsc_vec, alpha=-1.0)

    def evaluate_jacobian(
        _ts: PETSc.TS, t: float, x: PETSc.Vec, x_dot: PETSc.Vec, sigma: float, mat: PETSc.Mat, _pre: PETSc.Mat
    ) -> None:
        if on_time is not None:
            on_time(t)
        _set_state(x, x_dot)
        shift.value = sigma
        mat.zeroEntries()
        fem_petsc.assemble_matrix(mat, jacobian_form)  # type: ignore[arg-type]  # in-place into a preallocated Mat
        mat.assemble()
        if bc_dofs is not None:
            mat.zeroRowsLocal(bc_dofs, diag=1.0)  # type: ignore[arg-type]  # ndarray of local rows → identity (cols kept)

    ts = PETSc.TS().create(mesh.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType(options.ts_type)
    ts.setIFunction(evaluate_residual, fem.Function(space).x.petsc_vec)
    ts.setIJacobian(evaluate_jacobian, jacobian_matrix)
    # BDF starts at order 1 (= backward Euler), and that first step's truncation error is *not*
    # caught by the adaptive controller (it has no history to estimate it from), so it persists as a
    # constant over-diffusion ≈ the initial step — independent of `rtol`. The remedy is a small
    # startup step the controller then grows: `t_final / 1e4` drives the cold-start error below the
    # spatial floor here while adding only a handful of steps (the controller ramps geometrically).
    ts.setTimeStep(options.dt_initial if options.dt_initial is not None else (options.t_final - t_start) / 1.0e4)
    ts.setTime(t_start)  # integrate the sub-interval [t_start, t_final] (the moving driver strides over these)
    ts.setMaxTime(options.t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(options.atol, options.rtol)
    ts.setMaxSNESFailures(-1)  # let the adaptive controller cut the step and retry when Newton fails
    # Inner Newton (SNES) linear solver. The MOL default is a **Krylov solve (GMRES) with an
    # ILU preconditioner** — VCell's reaction-diffusion solver uses CVODE with SPGMR + ILU for
    # exactly this: ILU keeps the per-iteration cost low and convergence fast as the system
    # grows (in 3D a direct LU does not scale). `"preonly"`/`"lu"` gives a robust direct solve
    # for small problems. Eisenstat–Walker is disabled and a fixed `ksp_rtol` used so the linear
    # solve is tight enough not to pollute the time-integration accuracy.
    snes = ts.getSNES()
    snes.setUseEW(False)
    ksp = snes.getKSP()
    ksp.setType(options.ksp_type)
    ksp.setTolerances(rtol=options.ksp_rtol)
    set_preconditioner(ksp, options.pc_type)  # ILU → block-Jacobi/ILU(0) under MPI (backend/linear_solvers.py)
    ts.setFromOptions()

    if on_time is not None:
        on_time(t_start)  # evaluate g(t), time-dependent terms etc. at the interval's start time
    if bcs:  # make the initial state consistent with the Dirichlet boundary values g(t=0)
        fem_petsc.set_bc(state.x.petsc_vec, bcs)
        state.x.scatter_forward()

    # t = 0 pre-flight: catch a model that is already broken at the initial condition (a non-finite
    # residual — a divide-by-zero, a fractional power / log of a non-positive value at the IC)
    # before the adaptive solve grinds on it. (For a time-dependent problem the implicit operator
    # is mass-regularised, so the null-space classes — registry #3/#4 — belong to a steady solver.)
    preflight = fem_petsc.assemble_vector(residual_form)
    preflight.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
    finite = bool(mesh.comm.allreduce(bool(np.isfinite(preflight.array).all()), op=MPI.LAND))
    preflight.destroy()
    if not finite:
        culprit = localize() if localize is not None else None  # attribute it to a term, if we can
        raise SolveError(preflight_failure_message(culprit) if culprit else preflight_failure_message())

    monitor: OutputMonitor | None = None
    if on_output is not None or on_progress is not None:
        snapshot = fem.Function(space)

        def emit(t: float, work: PETSc.Vec) -> None:
            snapshot.x.scatter_forward()  # `work` is the snapshot's own vector; refresh its ghosts
            if on_output is not None:
                on_output(t, snapshot)

        monitor = OutputMonitor(
            output_times if on_output is not None else (),
            t_start=t_start,
            t_final=options.t_final,
            work=snapshot.x.petsc_vec,
            emit=emit,
            progress=on_progress,
        )
        ts.setMonitor(monitor)

    try:
        ts.solve(state.x.petsc_vec)
    except PETSc.Error as original:  # re-express the failure in model terms (see backend/diagnostics)
        raise SolveError(ts_failure_message(ts)) from original
    state.x.scatter_forward()
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    if monitor is not None:
        monitor.finish(ts, final_time, state.x.petsc_vec)
    ts.destroy()
    return steps, final_time


def _localize_nonfinite_term(problem: DiscreteProblem) -> str | None:
    """Attribute a non-finite residual to a tagged term (ADR-004): assemble each term's
    contribution at the current state and name the first that is non-finite. If the state (the
    initial condition) is itself non-finite, *that* is the culprit, not a term. Returns a phrase
    like `"the source term"` / `"the initial condition"`, or None if nothing localises (the
    caller then falls back to the generic message). Cheap, and only run on a pre-flight failure."""

    mesh = problem.V.mesh

    def finite(array: np.typing.NDArray[Any]) -> bool:
        return bool(mesh.comm.allreduce(bool(np.isfinite(array).all()), op=MPI.LAND))

    if not finite(problem.unknown.x.array):
        return "the initial condition"
    state, trial, dx = problem.unknown, problem.trial, problem.dx
    for term in problem.terms:
        if term.kind is TermKind.TIME_DERIVATIVE or term.integrand is None:
            continue
        contribution = fem_petsc.assemble_vector(fem.form(ufl.replace(term.integrand, {trial: state}) * dx))
        contribution.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        term_finite = finite(contribution.array)
        contribution.destroy()
        if not term_finite:
            return f"the {term.kind.value} term"
    return None


def integrate_discrete_problem(
    problem: DiscreteProblem,
    *,
    t_final: float,
    dt_initial: float | None = None,
    ts_type: str = "bdf",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    ksp_rtol: float = 1.0e-9,
    output_times: Sequence[float] = (),
    on_output: Callable[[float, fem.Function], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> IntegrationResult:
    """Integrate an assembled `DiscreteProblem` (a formalism MathDescription, via
    `backend.assemble`) with the method-of-lines `TS` integrator instead of backward Euler.

    `on_output(t, snapshot)` receives the solution at each of `output_times` in [0, `t_final`] — a
    `Function` on the unknown's space, valid only for the duration of the call; t = 0 is the initial
    condition — without changing the adaptive steps; `on_progress(t)` follows every accepted step.

    The IR already carries the tagged spatial terms (diffusion, advection, source, …) built
    against the trial function; this reuses them — substituting `trial → unknown` so the source
    becomes a genuine nonlinear residual — and swaps the backward-Euler time term for `ċ·v`. So a
    **nonlinear** reaction in the model's `source` slot is handled by the `TS` inner Newton,
    where the backward-Euler driver would need it linear (or lagged). The `unknown` carries the
    IC `assemble` applied and is integrated in place to `t_final`.

    Fixed-domain only: a prescribed `motion_velocity` (moving subdomain) raises — use
    `integrate_discrete_problem_moving` (strided ALE-MOL). All boundary kinds are supported:
    natural / Neumann / Robin enter the residual, and strong **Dirichlet** BCs (`problem.bcs`)
    are imposed as algebraic constraints in the `TS` callbacks. Returns an `IntegrationResult`.
    """

    if problem.motion_velocity is not None:
        raise NotImplementedError(
            "method-of-lines is fixed-domain; a moving subdomain uses integrate_discrete_problem_moving"
        )
    if TermKind.TIME_DERIVATIVE not in problem.term_kinds():
        raise NotImplementedError("method-of-lines integrates time-dependent problems only")

    state, rate, residual = _mol_residual(problem)
    options = _TimeStepperOptions(t_final, dt_initial, ts_type, rtol, atol, ksp_type, pc_type, ksp_rtol)
    steps, final_time = _run_time_stepper(
        state,
        rate,
        residual,
        options,
        bcs=problem.bcs,
        on_time=problem.set_time,
        localize=lambda: _localize_nonfinite_term(problem),  # attribute a pre-flight failure to a tagged term
        output_times=output_times,
        on_output=on_output,
        on_progress=on_progress,
    )
    problem.previous.x.array[:] = state.x.array
    return IntegrationResult(solution=state, steps=steps, time=final_time)


def _mol_residual(problem: DiscreteProblem) -> tuple[fem.Function, fem.Function, UflExpr]:
    """The method-of-lines implicit residual `F(c, ċ)` for an assembled `DiscreteProblem`: the IR's
    tagged spatial terms (diffusion, advection, **dilution**, source, …) with `trial → unknown` (so a
    nonlinear source is a genuine residual), the backward-Euler time term swapped for `ċ·v`, plus the
    boundary terms. Returns `(state, rate, residual)` — `rate` is the fresh `ċ` Function `TS` drives.
    The DILUTION term is kept, so on a moving mesh the residual is the ALE referential form."""

    state, trial, test, dx = problem.unknown, problem.trial, problem.test, problem.dx
    rate = fem.Function(state.function_space)  # ċ
    residual = ufl.inner(rate, test) * dx  # mass term ċ·v
    for term in problem.terms:
        if term.kind is TermKind.TIME_DERIVATIVE or term.integrand is None:
            continue
        integrand = ufl.replace(term.integrand, {trial: state})  # trial → unknown ⇒ a nonlinear residual
        # Same signs as backward Euler: source on the PDE RHS ⇒ −source in the residual.
        residual = residual - integrand * dx if term.kind is TermKind.SOURCE else residual + integrand * dx
    for boundary in problem.boundary_terms:
        residual = residual + ufl.replace(boundary.integrand, {trial: state}) * boundary.measure
    return state, rate, residual


def integrate_discrete_problem_moving(
    problem: DiscreteProblem,
    *,
    t_final: float,
    motion_steps: int,
    dt_initial: float | None = None,
    ts_type: str = "bdf",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    ksp_rtol: float = 1.0e-9,
) -> IntegrationResult:
    """Method-of-lines on a **moving** subdomain (strided ALE), to `t_final`.

    PETSc `TS` integrates a fixed spatial operator, but an ALE mesh moves the operator in time. This
    mirrors what the backward-Euler `step` does — move the mesh discretely, then solve — except each
    interval is integrated with the adaptive-BDF `TS` instead of a single backward-Euler step. The run
    is split into `motion_steps` equal strides of length `h = t_final / motion_steps`; each stride:

    1. **moves the mesh** by `h·v` (`advance_mesh`: nodes carried in the material frame, the field
       values riding along, harmonic-extended in the interior for a bulk);
    2. **`TS`-integrates** the reaction–diffusion–**dilution** residual over `[t, t+h]` on the
       now-fixed configuration — the DILUTION term `(∇·v)c` makes this the ALE referential form, so a
       volume-changing motion dilutes the field correctly.

    So it is **first-order in the mesh motion** (the mesh is frozen within a stride — refine
    `motion_steps` to tighten that split) but keeps MOL's **adaptive high-order** time integration of
    the reaction–diffusion *within* each stride: the win is a stiff reaction under a slow/smooth motion,
    where few strides suffice yet the kinetics need fine, error-controlled sub-stepping. The mesh-move
    magnitude per stride is set through `problem.dt`. No remeshing — a motion that tangles the mesh
    raises `MeshQualityError`; for a **large deformation** use `backend.ale.run_moving_with_remeshing`,
    which remeshes between strides. Returns an `IntegrationResult` whose `steps` is the total `TS`
    steps across all strides.
    """

    if problem.motion_velocity is None:
        raise NotImplementedError(
            "integrate_discrete_problem_moving needs a moving subdomain; the fixed-domain "
            "case uses integrate_discrete_problem"
        )
    if TermKind.TIME_DERIVATIVE not in problem.term_kinds():
        raise NotImplementedError("method-of-lines integrates time-dependent problems only")
    if motion_steps < 1:
        raise ValueError(f"motion_steps must be >= 1, got {motion_steps}")

    h = t_final / motion_steps
    problem.dt.value = h  # the per-stride mesh-move magnitude (advance_mesh moves by dt·v)
    total_steps = 0
    for i in range(motion_steps):
        problem.advance_mesh()  # move the mesh by h·v, carrying the field (material frame)
        total_steps += integrate_discrete_problem_stride(
            problem,
            t_start=i * h,
            t_final=(i + 1) * h,
            dt_initial=dt_initial,
            ts_type=ts_type,
            rtol=rtol,
            atol=atol,
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_rtol=ksp_rtol,
        )
    return IntegrationResult(solution=problem.unknown, steps=total_steps, time=motion_steps * h)


def integrate_discrete_problem_stride(
    problem: DiscreteProblem,
    *,
    t_start: float,
    t_final: float,
    dt_initial: float | None = None,
    ts_type: str = "bdf",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    ksp_rtol: float = 1.0e-9,
) -> int:
    """Method-of-lines over **one fixed-mesh interval** `[t_start, t_final]`, returning the `TS` step
    count. The IR's tagged terms (incl. DILUTION) build the residual, so on a moving subdomain's mesh
    this is the ALE referential form. This primitive neither moves the mesh nor remeshes — the caller
    owns that: `integrate_discrete_problem_moving` strides it after each `advance_mesh`; the ALE
    remesh driver (`backend.ale.run_moving_with_remeshing`) strides it with a quality-triggered remesh
    between strides, so it composes with remeshing for large deformations. Mutates `problem.unknown`
    (and `previous`) in place."""

    state, rate, residual = _mol_residual(problem)
    options = _TimeStepperOptions(t_final, dt_initial, ts_type, rtol, atol, ksp_type, pc_type, ksp_rtol)
    steps, _ = _run_time_stepper(
        state,
        rate,
        residual,
        options,
        bcs=problem.bcs,
        on_time=problem.set_time,
        localize=lambda: _localize_nonfinite_term(problem),
        t_start=t_start,
    )
    problem.previous.x.array[:] = state.x.array
    return steps

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
`TS.solve`. For the moving ALE cell the operators change with the geometry, which is why that
path stays a per-step implicit instead.

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

import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.discrete import DiscreteProblem, TermKind


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
) -> tuple[int, float]:
    """Integrate the implicit residual `F(state, rate) = 0` to `options.t_final` with PETSc `TS`,
    mutating `state` in place. `residual` is the UFL form of `F` (`rate` the time derivative `ċ`,
    a `Function` TS supplies each step); the exact Jacobian `σ ∂F/∂rate + ∂F/∂state` comes from
    `ufl.derivative`. Returns `(steps, final_time)`. Shared by the explicit-API integrator and
    the DiscreteProblem (formalism) driver, so the `TS` wiring lives in exactly one place.

    Strong Dirichlet `bcs` enter as algebraic constraints `x = g` on the boundary dofs: the IC
    is seeded to satisfy them, the residual's boundary rows are overwritten with `x − g` (so the
    inner Newton drives them to `g`), and the Jacobian is assembled with the bcs (boundary
    rows/columns zeroed, unit diagonal). The boundary dofs then stay at `g` for the whole run."""

    space = state.function_space
    mesh = space.mesh
    residual_form = fem.form(residual)
    shift = fem.Constant(mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # petsc4py stubs ScalarType as a dtype
    jacobian = shift * ufl.derivative(residual, rate) + ufl.derivative(residual, state)
    jacobian_form = fem.form(jacobian)  # σ M + ∂F/∂c — exact (ufl.derivative)
    jacobian_matrix = fem_petsc.create_matrix(jacobian_form)

    def _set_state(x: PETSc.Vec, x_dot: PETSc.Vec) -> None:
        x.copy(state.x.petsc_vec)
        state.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x_dot.copy(rate.x.petsc_vec)
        rate.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def evaluate_residual(_ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        _set_state(x, x_dot)
        with result.localForm() as local:
            local.set(0.0)
            fem_petsc._assemble_vector_array(local.array_w, residual_form)  # type: ignore[attr-defined]
        result.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        if bcs:  # boundary residual rows ← x − g, so Newton holds the dofs at g
            fem_petsc.set_bc(result, bcs, x0=state.x.petsc_vec, alpha=-1.0)

    def evaluate_jacobian(
        _ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, sigma: float, mat: PETSc.Mat, _pre: PETSc.Mat
    ) -> None:
        _set_state(x, x_dot)
        shift.value = sigma
        mat.zeroEntries()
        fem_petsc.assemble_matrix(mat, jacobian_form, bcs=bcs)  # type: ignore[arg-type, misc]  # in-place; bcs zero boundary rows/cols
        mat.assemble()

    ts = PETSc.TS().create(mesh.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType(options.ts_type)
    ts.setIFunction(evaluate_residual, fem.Function(space).x.petsc_vec)
    ts.setIJacobian(evaluate_jacobian, jacobian_matrix)
    ts.setTimeStep(options.dt_initial if options.dt_initial is not None else options.t_final / 100.0)
    ts.setMaxTime(options.t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(options.atol, options.rtol)
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
    ksp.getPC().setType(options.pc_type)
    ts.setFromOptions()

    if bcs:  # make the initial state consistent with the Dirichlet boundary values
        fem_petsc.set_bc(state.x.petsc_vec, bcs)
        state.x.scatter_forward()
    ts.solve(state.x.petsc_vec)
    state.x.scatter_forward()
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    ts.destroy()
    return steps, final_time


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
) -> IntegrationResult:
    """Integrate an assembled `DiscreteProblem` (a formalism MathDescription, via
    `backend.assemble`) with the method-of-lines `TS` integrator instead of backward Euler.

    The IR already carries the tagged spatial terms (diffusion, advection, source, …) built
    against the trial function; this reuses them — substituting `trial → unknown` so the source
    becomes a genuine nonlinear residual — and swaps the backward-Euler time term for `ċ·v`. So a
    **nonlinear** reaction in the model's `source` slot is handled by the `TS` inner Newton,
    where the backward-Euler driver would need it linear (or lagged). The `unknown` carries the
    IC `assemble` applied and is integrated in place to `t_final`.

    Fixed-domain only: a prescribed `motion_velocity` (moving subdomain) raises — the moving
    cell uses the per-step Newton path (`backend/fsi.py`). All boundary kinds are supported:
    natural / Neumann / Robin enter the residual, and strong **Dirichlet** BCs (`problem.bcs`)
    are imposed as algebraic constraints in the `TS` callbacks. Returns an `IntegrationResult`.
    """

    if problem.motion_velocity is not None:
        raise NotImplementedError("method-of-lines is fixed-domain; a moving subdomain uses the per-step Newton path")
    if TermKind.TIME_DERIVATIVE not in problem.term_kinds():
        raise NotImplementedError("method-of-lines integrates time-dependent problems only")

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

    options = _TimeStepperOptions(t_final, dt_initial, ts_type, rtol, atol, ksp_type, pc_type, ksp_rtol)
    steps, final_time = _run_time_stepper(state, rate, residual, options, bcs=problem.bcs)
    problem.previous.x.array[:] = state.x.array
    return IntegrationResult(solution=state, steps=steps, time=final_time)

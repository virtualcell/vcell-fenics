"""Backward-Euler assembly for two volume compartments coupled across a membrane (§1.6.2).

Two bulk diffusion species — one per compartment — coupled by **single-sided interface flux** BCs at
their shared membrane: `D ∇u·n = f(trace(u_inner), trace(u_outer), params)` on each side, with the two
sides carrying *independent* `BCInterfaceFlux`es (VCell's in_flux / out_flux are not an enforced
equal-and-opposite balance — each side gets its own Neumann flux). The common case is a permeability
flux `P·(u_outer − u_inner)` into the inner side and its negation into the outer side, which together
drive the two compartments toward equilibrium and conserve mass; `f` may be any expression in the two
interface traces (and, in a follow-up, an adjacent membrane species).

Cross-mesh mechanics (the `InterfaceCoupledGeometry` substrate):

- A block form needs ONE reference mesh that is also an integration domain, so *every* term is
  integrated on the shared **parent** mesh: the per-compartment mass + diffusion over the parent's
  region cells (`dx(parent)(region_tag)`), and the coupling flux over the parent's interior interface
  facets (`dS(parent)(interface_tag)`). Each compartment's P1 function lives on its own submesh and is
  pulled into the parent integral through its `EntityMap`.
- The interface trace of a bulk variable is `membrane_trace(f) = f('+') + f('-')` — each compartment's
  function is non-zero only on its own side, so the sum is its boundary value regardless of the `dS`
  side labelling. Both trials and tests are restricted this way in the coupling term.
- A single-sided permeability flux is **linear in the unknowns**, so it is fully implicit — assembled
  into the block bilinear form via `ufl.extract_blocks`/`ufl.lhs`, with no lagging. The two-block
  system over `MixedFunctionSpace(V_inner, V_outer)` is solved each step.

Scope (this increment): two bulk diffusion species + single-sided interface flux BCs, static geometry,
**backward Euler**. The method-of-lines coupled integrator, a membrane species coupled to both bulks,
`BCInterfaceValueEquality` (the `u_inner = k·u_outer` constraint), and a reservoir Dirichlet on the
outer boundary are follow-ups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.geometry import InterfaceCoupledGeometry, membrane_trace
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    MathDescription,
    ParameterConstant,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import validate_or_raise


@dataclass
class InterfaceCoupledProblem:
    """A two-bulk interface-coupled solve, advanced by backward-Euler `step()`s. `inner` / `outer` are
    the per-compartment solution `Function`s (on their own submeshes); `_advance` is the per-step
    closure the assembler builds over the block system."""

    inner_var: str
    outer_var: str
    inner: fem.Function
    outer: fem.Function
    _advance: Callable[[], None]

    def step(self) -> None:
        self._advance()

    def field(self, name: str) -> fem.Function:
        if name == self.inner_var:
            return self.inner
        if name == self.outer_var:
            return self.outer
        raise KeyError(f"{name!r} is not a variable of this coupled problem")

    def total_mass(self) -> float:
        """∫u over both compartments (conserved by an equal-and-opposite interface-flux pair, no external flux)."""
        return _total_mass(self.inner, self.outer)


@dataclass
class InterfaceCoupledResult:
    """The result of a **method-of-lines** integration of an interface-coupled system to `t_final`
    (PETSc TS adaptive BDF): the integrated per-compartment solution `Function`s, the step count, and
    the final time. Unlike `InterfaceCoupledProblem` (backward-Euler, stepped) the solve is one call."""

    inner: fem.Function
    outer: fem.Function
    steps: int
    time: float

    def total_mass(self) -> float:
        return _total_mass(self.inner, self.outer)


def _total_mass(inner: fem.Function, outer: fem.Function) -> float:
    """∫u over both compartments (conserved by an equal-and-opposite interface-flux pair, no external flux)."""
    mass = 0.0
    for field in (inner, outer):
        mesh = field.function_space.mesh
        local = fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh)))
        mass += mesh.comm.allreduce(local.real, op=MPI.SUM)
    return mass


def _const_params(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    """The model's constant parameters as `fem.Constant`s on `mesh` (the parent integration mesh)."""
    return {
        p.name: fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
        for p in md.parameters
        if isinstance(p, ParameterConstant)
    }


def assemble_interface_coupled(
    md: MathDescription, geometry: InterfaceCoupledGeometry, *, dt: float
) -> InterfaceCoupledProblem:
    """Assemble a backward-Euler step for two bulk compartments coupled by single-sided interface flux
    BCs (see the module docstring). Returns an `InterfaceCoupledProblem` whose `step()` advances `dt`."""

    validate_or_raise(md)
    equations = {eq.subdomain: eq for eq in md.equations if isinstance(eq, TemplateEquation)}
    inner_eq = _require_equation(equations, geometry.inner_subdomain)
    outer_eq = _require_equation(equations, geometry.outer_subdomain)
    inner_var, outer_var = inner_eq.variable, outer_eq.variable

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1))
    u_in_fn, u_in_prev = fem.Function(V_in, name=inner_var), fem.Function(V_in)
    u_out_fn, u_out_prev = fem.Function(V_out, name=outer_var), fem.Function(V_out)
    n_in = V_in.dofmap.index_map.size_local
    n_out = V_out.dofmap.index_map.size_local

    mixed = ufl.MixedFunctionSpace(V_in, V_out)
    u_in, u_out = ufl.TrialFunctions(mixed)
    w_in, w_out = ufl.TestFunctions(mixed)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)(geometry.interface_tag)
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map]

    params = _const_params(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    d_in = compile_expression(parse(inner_eq.terms["diffusion"]), ctx)
    d_out = compile_expression(parse(outer_eq.terms["diffusion"]), ctx)

    # Backward-Euler residual over the mixed space: per-compartment mass + diffusion (region dx) plus
    # the single-sided interface fluxes (interface dS). All integrals on the parent → one block form.
    f = (u_in - u_in_prev) * w_in * dx_in + dt * d_in * ufl.dot(ufl.grad(u_in), ufl.grad(w_in)) * dx_in
    f += (u_out - u_out_prev) * w_out * dx_out + dt * d_out * ufl.dot(ufl.grad(u_out), ufl.grad(w_out)) * dx_out

    test_of = {inner_var: w_in, outer_var: w_out}
    # Both interface traces are in scope for every flux (a flux on one side may reference either bulk's
    # trace); each `BCInterfaceFlux` then deposits its flux into its OWN side's test function only.
    coupling_symbols = {
        inner_var: membrane_trace(u_in),
        outer_var: membrane_trace(u_out),
        "geom.x": ufl.SpatialCoordinate(parent),
        **params,
    }
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this assembler handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in test_of:
            continue
        coupling_ctx = CompileContext(parent, coupling_symbols)
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        f += -dt * flux * membrane_trace(test_of[bc.variable]) * ds_int  # var gains the influx

    a_form = fem.form(ufl.extract_blocks(ufl.lhs(f)), entity_maps=emaps)
    rhs_expr = ufl.rhs(f)
    l_form = fem.form(ufl.extract_blocks(rhs_expr), entity_maps=emaps)

    matrix = petsc.assemble_matrix(a_form)  # static geometry + linear coupling ⇒ assemble once
    matrix.assemble()
    ksp = PETSc.KSP().create(parent.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    _interpolate_ic(u_in_fn, inner_eq, ctx)
    _interpolate_ic(u_out_fn, outer_eq, ctx)
    u_in_prev.x.array[:] = u_in_fn.x.array
    u_out_prev.x.array[:] = u_out_fn.x.array

    solution = matrix.createVecRight()

    def advance() -> None:
        rhs = petsc.assemble_vector(l_form)
        ksp.solve(rhs, solution)
        values = solution.array_r
        u_in_fn.x.array[:n_in] = values[:n_in]
        u_out_fn.x.array[:n_out] = values[n_in : n_in + n_out]
        u_in_fn.x.scatter_forward()
        u_out_fn.x.scatter_forward()
        u_in_prev.x.array[:] = u_in_fn.x.array
        u_out_prev.x.array[:] = u_out_fn.x.array
        rhs.destroy()

    return InterfaceCoupledProblem(inner_var, outer_var, u_in_fn, u_out_fn, advance)


def integrate_interface_coupled(
    md: MathDescription,
    geometry: InterfaceCoupledGeometry,
    *,
    t_final: float,
    dt_initial: float | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
) -> InterfaceCoupledResult:
    """Integrate an interface-coupled two-bulk system to `t_final` with the **method-of-lines**
    integrator (PETSc TS adaptive BDF) — the same strategy as the FV solver and the single-mesh
    `integrate_discrete_problem`, here over the blocked two-mesh system. The coupling flux may be
    nonlinear in the unknowns (the inner Newton handles it); the time error is ≈0 (adaptive).

    The residual `F(state, rate) = M·rate + K·state − coupling` is a 2-block form over the two
    compartment spaces (all integrals on the parent: region `dx`, interface `dS`); the exact Jacobian
    `σ ∂F/∂rate + ∂F/∂state` comes from `ufl.derivative`. Both are assembled monolithically
    (`kind="mpi"`) so the TS state is one blocked vector. **Key:** the TS Jacobian holder is built by
    `assemble_matrix` (the full coupling sparsity, including the off-diagonal blocks), not
    `create_matrix` — a mismatched preallocation breaks the in-callback copy.
    """

    validate_or_raise(md)
    equations = {eq.subdomain: eq for eq in md.equations if isinstance(eq, TemplateEquation)}
    inner_eq = _require_equation(equations, geometry.inner_subdomain)
    outer_eq = _require_equation(equations, geometry.outer_subdomain)
    inner_var, outer_var = inner_eq.variable, outer_eq.variable

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1))
    u_in_fn = fem.Function(V_in, name=inner_var)
    u_out_fn = fem.Function(V_out, name=outer_var)
    rate_in, rate_out = fem.Function(V_in), fem.Function(V_out)
    n_in = V_in.dofmap.index_map.size_local
    n_out = V_out.dofmap.index_map.size_local

    w_in, w_out = ufl.TestFunction(V_in), ufl.TestFunction(V_out)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)(geometry.interface_tag)
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map]

    params = _const_params(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    d_in = compile_expression(parse(inner_eq.terms["diffusion"]), ctx)
    d_out = compile_expression(parse(outer_eq.terms["diffusion"]), ctx)

    # The MOL residual uses the state Functions directly (so the coupling can be nonlinear), with the
    # time derivative ċ = rate (a Function TS supplies). One block per compartment, all on the parent.
    residual = [
        rate_in * w_in * dx_in + d_in * ufl.dot(ufl.grad(u_in_fn), ufl.grad(w_in)) * dx_in,
        rate_out * w_out * dx_out + d_out * ufl.dot(ufl.grad(u_out_fn), ufl.grad(w_out)) * dx_out,
    ]
    test_of = {inner_var: w_in, outer_var: w_out}
    index_of = {inner_var: 0, outer_var: 1}
    # Both interface traces are in scope for every flux (a flux on one side may reference either bulk's
    # trace); each `BCInterfaceFlux` then deposits its flux into its OWN side's residual block only.
    coupling_symbols = {
        inner_var: membrane_trace(u_in_fn),
        outer_var: membrane_trace(u_out_fn),
        "geom.x": ufl.SpatialCoordinate(parent),
        **params,
    }
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this integrator handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in test_of:
            continue
        coupling_ctx = CompileContext(parent, coupling_symbols)
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        residual[index_of[bc.variable]] += -flux * membrane_trace(test_of[bc.variable]) * ds_int

    states, rates = [u_in_fn, u_out_fn], [rate_in, rate_out]
    shift = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]  # the TS σ
    jacobian = [
        [shift * ufl.derivative(residual[i], rates[j]) + ufl.derivative(residual[i], states[j]) for j in range(2)]
        for i in range(2)
    ]
    residual_form = fem.form(residual, entity_maps=emaps)
    jacobian_form = fem.form(jacobian, entity_maps=emaps)

    _interpolate_ic(u_in_fn, inner_eq, ctx)
    _interpolate_ic(u_out_fn, outer_eq, ctx)

    def unpack(x: PETSc.Vec) -> None:
        u_in_fn.x.array[:n_in] = x.array_r[:n_in]
        u_out_fn.x.array[:n_out] = x.array_r[n_in : n_in + n_out]
        u_in_fn.x.scatter_forward()
        u_out_fn.x.scatter_forward()

    def evaluate_residual(_ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        unpack(x)
        rate_in.x.array[:n_in] = x_dot.array_r[:n_in]
        rate_out.x.array[:n_out] = x_dot.array_r[n_in : n_in + n_out]
        b = petsc.assemble_vector(residual_form, kind="mpi")
        b.copy(result)
        b.destroy()

    def evaluate_jacobian(
        _ts: PETSc.TS, _t: float, x: PETSc.Vec, _x_dot: PETSc.Vec, sigma: float, mat: PETSc.Mat, _pre: PETSc.Mat
    ) -> None:
        unpack(x)
        shift.value = sigma
        fresh = petsc.assemble_matrix(jacobian_form, kind="mpi")
        fresh.assemble()
        fresh.copy(mat, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        mat.assemble()
        fresh.destroy()

    ts = PETSc.TS().create(parent.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType("bdf")
    state_vec = petsc.create_vector([V_in, V_out], kind="mpi")
    state_vec.array[:n_in] = u_in_fn.x.array[:n_in]
    state_vec.array[n_in : n_in + n_out] = u_out_fn.x.array[:n_out]
    # Jacobian holder via assemble_matrix → the full coupling sparsity (off-diagonal blocks present).
    jacobian_matrix = petsc.assemble_matrix(jacobian_form, kind="mpi")
    jacobian_matrix.assemble()
    ts.setIFunction(evaluate_residual, state_vec.duplicate())
    ts.setIJacobian(evaluate_jacobian, jacobian_matrix, jacobian_matrix)
    # A small startup step (BDF cold-start, see the single-mesh integrator) then adapt.
    ts.setTimeStep(dt_initial if dt_initial is not None else t_final / 1.0e4)
    ts.setMaxTime(t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(atol, rtol)
    ts.setMaxSNESFailures(-1)
    snes = ts.getSNES()
    snes.setUseEW(False)
    # Inner Newton linear solver. The default is a Krylov solve (GMRES) with an incomplete-LU (ILU)
    # preconditioner — the scalable choice (≈O(N) memory and per-iteration cost), as the single-mesh
    # MOL uses and as VCell's CVODE uses SPGMR+ILU. A *direct* sparse LU (`ksp_type="preonly"`,
    # `pc_type="lu"`) is robust for small problems but its 2D fill-in does not scale to large meshes.
    snes.getKSP().setType(ksp_type)
    snes.getKSP().getPC().setType(pc_type)
    ts.setFromOptions()

    ts.solve(state_vec)
    unpack(state_vec)
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    for obj in (ts, state_vec, jacobian_matrix):
        obj.destroy()
    return InterfaceCoupledResult(u_in_fn, u_out_fn, steps, final_time)


def _require_equation(equations: dict[str, TemplateEquation], subdomain: str) -> TemplateEquation:
    eq = equations.get(subdomain)
    if eq is None or "diffusion" not in eq.terms:
        raise NotImplementedError(f"interface coupling needs a bulk diffusion equation on compartment {subdomain!r}")
    return eq


def _interpolate_ic(field: fem.Function, equation: TemplateEquation, ctx: CompileContext) -> None:
    if equation.initial_condition is None:
        return
    mesh = field.function_space.mesh
    local_ctx = CompileContext(
        mesh, {"geom.x": ufl.SpatialCoordinate(mesh), **{k: v for k, v in ctx.symbols.items() if k != "geom.x"}}
    )
    expression = fem.Expression(
        compile_expression(parse(equation.initial_condition), local_ctx),
        field.function_space.element.interpolation_points,
    )
    field.interpolate(expression)

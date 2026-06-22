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

This module also carries the three-region extension where **surface species on the membrane couple to
both bulks** (the receptor–ligand cell): any number of bulk species per compartment and any number of
surface species, coupled by interface fluxes referencing the surface density and surface reactions
referencing the bulk traces. Both a backward-Euler stepper (`assemble_membrane_coupled`, fully-lagged
/IMEX binding) and a method-of-lines integrator (`integrate_membrane_coupled`, fully-implicit binding
via a **matrix-free** Newton — the exact `∂(binding)/∂(membrane species)` Jacobian block is
un-assemblable in DOLFINx 0.10/0.11, so the implicit Jacobian is applied by finite-differencing the
residual, preconditioned by the assembled partial Jacobian) are provided; the MOL one is
unconditionally stable for stiff binding.

Scope: follow-ups are a moving membrane (which makes the `ρ ∇_Γ·v_Γ` dilution term mandatory), in-bulk
reactions, `BCInterfaceValueEquality` (the `u_inner = k·u_outer` constraint), and a reservoir Dirichlet
on the outer boundary.
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
    ParameterExpression,
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
    return sum(_mass_of(field) for field in (inner, outer))


def _mass_of(field: fem.Function) -> float:
    """∫field over its own mesh (an area-integral for a surface field, a volume-integral for a bulk one)."""
    mesh = field.function_space.mesh
    local = fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local.real, op=MPI.SUM))


@dataclass
class _MembraneCoupledFields:
    """The shared state of a three-region bulk–surface–bulk solve: any number of bulk species in each of
    two compartments and any number of surface species on their shared membrane. `inner` / `outer` /
    `membrane` are the per-region **vector** `Function`s whose components are the species named in
    `inner_species` / `outer_species` / `membrane_species` order, with by-name accessors."""

    inner_species: list[str]
    outer_species: list[str]
    membrane_species: list[str]
    inner: fem.Function
    outer: fem.Function
    membrane: fem.Function

    def _locate(self, name: str) -> tuple[fem.Function, int]:
        """The (vector `Function`, component index) holding species `name`."""
        for fn, names in (
            (self.inner, self.inner_species),
            (self.outer, self.outer_species),
            (self.membrane, self.membrane_species),
        ):
            if name in names:
                return fn, names.index(name)
        raise KeyError(f"{name!r} is not a variable of this membrane-coupled solve")

    def field(self, name: str) -> fem.Function:
        """A scalar `Function` holding the named species' values (a copy, for inspection)."""
        fn, k = self._locate(name)
        space = fn.function_space
        if space.dofmap.index_map_bs == 1:  # a one-component region has no `sub(0)` to collapse
            out = fem.Function(fem.functionspace(space.mesh, ("Lagrange", 1)), name=name)
            out.x.array[:] = fn.x.array
            return out
        sub_space, dofs = space.sub(k).collapse()
        out = fem.Function(sub_space, name=name)
        out.x.array[:] = fn.x.array[dofs]
        return out

    def mass(self, name: str) -> float:
        """∫ of the named species over its own mesh."""
        fn, k = self._locate(name)
        mesh = fn.function_space.mesh
        local = fem.assemble_scalar(fem.form(fn[k] * ufl.dx(domain=mesh)))
        return float(mesh.comm.allreduce(local.real, op=MPI.SUM))

    def total_mass(self) -> float:
        """∫ of every species over all three meshes — the conserved total for a single-substance model
        (free in each compartment + membrane-bound), since binding only moves it between those pools."""
        return sum(self.mass(n) for n in (*self.inner_species, *self.outer_species, *self.membrane_species))


@dataclass
class MembraneCoupledProblem(_MembraneCoupledFields):
    """A membrane-coupled solve advanced by **backward-Euler** `step()`s. `_advance` is the per-step
    closure over the three-block system."""

    _advance: Callable[[], None]

    def step(self) -> None:
        self._advance()


@dataclass
class MembraneCoupledResult(_MembraneCoupledFields):
    """The result of a **method-of-lines** integration of a membrane-coupled system to `t_final` (PETSc
    TS adaptive BDF with a matrix-free Newton inner solve): the integrated per-region solution `Function`s,
    the step count, and the final time. The solve is one call (vs `MembraneCoupledProblem`'s stepping)."""

    steps: int
    time: float


def _param_symbols(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    """The model's parameters as symbols on `mesh`: constants as `fem.Constant`, expressions compiled in
    declaration order (the import keeps them dependency-ordered) — so a VCell unit factor like
    `KFlux = AreaPerUnitArea/VolumePerUnitVolume` or `UnitFactor = pow(KMOLE, 1)` binds as a UFL
    expression, not only bare constants. Mirrors the single-mesh `_compile_context` parameter loop."""
    scratch: dict[str, UflExpr] = {
        "geom.x": ufl.SpatialCoordinate(mesh),
        "sim.t": fem.Constant(mesh, PETSc.ScalarType(0.0)),  # type: ignore[operator]
    }
    ctx = CompileContext(mesh=mesh, symbols=scratch)
    params: dict[str, UflExpr] = {}
    for p in md.parameters:
        if isinstance(p, ParameterConstant):
            value: UflExpr = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
        elif isinstance(p, ParameterExpression):
            value = compile_expression(parse(p.expression), ctx)
        else:
            raise NotImplementedError(f"coupled assembly cannot bind parameter {p.name!r} of type {type(p).__name__}")
        scratch[p.name] = params[p.name] = value  # later parameters may reference this one
    return params


def _sum_forms(terms: list[UflExpr]) -> UflExpr:
    """Sum a non-empty list of UFL form terms (sidesteps `sum()`'s int-start typing)."""
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


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

    params = _param_symbols(md, parent)
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

    params = _param_symbols(md, parent)
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


def assemble_membrane_coupled(
    md: MathDescription, geometry: InterfaceCoupledGeometry, *, dt: float
) -> MembraneCoupledProblem:
    """Assemble a backward-Euler step for **membrane species coupled to both bulk compartments**: any
    number of bulk diffusion species in each compartment plus any number of surface species on their
    shared membrane, the full receptor–ligand cell (the §1.6.5/§1.6.6 composable pattern at an *internal*
    interface). Each region is one **vector** P1 space whose components are its species.

    The regions couple at the membrane two ways: (1) each surface species' reaction `source` may
    reference the interface trace of any bulk species in either compartment, and (2) each bulk species'
    `BCInterfaceFlux` may reference any same-compartment trace, any adjacent-compartment trace, and any
    membrane species. Every species is in scope via a single symbol table mapping its name to its
    interface representation. The coupling is **fully lagged (IMEX)**: in the membrane-facet terms the
    bulk traces *and* the surface species use previous-step values, so the binding rate is an explicit
    RHS forcing. The SAME rate is subtracted from the bulk(s) (each flux, tested on its own side) and
    added to the surface (each source, tested on the membrane), so the membrane-facet integrals cancel
    termwise and **mass is conserved to round-off** when the modeller writes the fluxes as the negation
    of the matching production. (Keeping the surface species implicit would need an off-diagonal
    bulk-test × membrane-trial block on `dS` — a codim-0 × codim-1 coupling that leaks the cancellation;
    full lagging sidesteps it, at the cost of conditional stability for stiff binding, which the
    method-of-lines follow-up lifts with a Newton solve.)

    Cross-mesh mechanics (extending the two-bulk substrate, §1.6.2): each species' mass + diffusion is
    integrated on the parent's region cells (`dx(parent)(region)`, per component, with its own diffusion)
    and the surface mass + **intrinsic surface** diffusion on the membrane submesh's own `dx` (its `grad`
    is the tangential ∇_Γ). The coupling is integrated on the parent's interior interface facets (`dS`),
    where each bulk trace is the orientation-robust `membrane_trace(u) = u('+') + u('-')` and a membrane
    species is restricted `ρ('+')` (single-valued on the facet). All three submesh functions are pulled
    into the parent integrals through their `EntityMap`s; with the coupling fully lagged the block matrix
    is the static, block-diagonal mass + diffusion (factorised once) and only the RHS changes each step.

    Scope (this increment): static membrane (**no dilution term** — `ρ ∇_Γ·v_Γ` is zero without motion;
    it becomes mandatory the moment the membrane moves), bulk species are diffusion-only (no in-bulk
    reaction), backward Euler. The fully-implicit method-of-lines integrator (nonlinear binding via
    Newton) and a moving membrane are follow-ups.
    """

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    inner_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.inner_subdomain]
    outer_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.outer_subdomain]
    membrane_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.membrane_subdomain]
    if not inner_eqs or not outer_eqs:
        raise NotImplementedError("membrane coupling needs at least one bulk species in each compartment")
    if not membrane_eqs:
        raise NotImplementedError(
            f"membrane coupling needs a surface equation on the membrane subdomain "
            f"{geometry.membrane_subdomain!r} (the species coupled to both bulks)"
        )
    for eq in (*inner_eqs, *outer_eqs):
        if "diffusion" not in eq.terms:
            raise NotImplementedError(f"membrane coupling needs a bulk diffusion equation for {eq.variable!r}")
    inner_species = [eq.variable for eq in inner_eqs]
    outer_species = [eq.variable for eq in outer_eqs]
    membrane_species = [eq.variable for eq in membrane_eqs]

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1, (len(inner_species),)))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1, (len(outer_species),)))
    V_mem = fem.functionspace(geometry.membrane_mesh, ("Lagrange", 1, (len(membrane_species),)))
    u_in_fn, u_in_prev = fem.Function(V_in), fem.Function(V_in)
    u_out_fn, u_out_prev = fem.Function(V_out), fem.Function(V_out)
    rho_fn, rho_prev = fem.Function(V_mem), fem.Function(V_mem)
    n_in = V_in.dofmap.index_map.size_local * V_in.dofmap.index_map_bs
    n_out = V_out.dofmap.index_map.size_local * V_out.dofmap.index_map_bs
    n_mem = V_mem.dofmap.index_map.size_local * V_mem.dofmap.index_map_bs

    mixed = ufl.MixedFunctionSpace(V_in, V_out, V_mem)
    u_in, u_out, rho = ufl.TrialFunctions(mixed)
    w_in, w_out, w_rho = ufl.TestFunctions(mixed)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    dx_mem = ufl.Measure("dx", domain=geometry.membrane_mesh)
    # A fixed quadrature degree on the interface so the shared binding rate integrates *identically*
    # whether it is tested by the bulk side's `membrane_trace(w_in[k])` (a flux) or the surface side's
    # `w_ρ[k]('+')` (a source) — the termwise cancellation that gives mass conservation relies on both
    # forcing integrals using the same quadrature.
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags, metadata={"quadrature_degree": 4})(
        geometry.interface_tag
    )
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map, geometry.membrane_entity_map]

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    surf_params = _param_symbols(md, geometry.membrane_mesh)
    surf_ctx = CompileContext(
        geometry.membrane_mesh, {"geom.x": ufl.SpatialCoordinate(geometry.membrane_mesh), **surf_params}
    )

    # A single block form cannot mix two meshes (each block needs one integration domain), so the
    # residual splits into a LOCAL part (per-mesh mass + diffusion) and a COUPLING part (all on the
    # parent's interface `dS`). With the coupling fully lagged (below) the bilinear form is the
    # block-diagonal local part — static — so the matrix assembles + factorises once and only the RHS
    # changes each step.
    #
    # Local: each species' mass + diffusion, per component with its own diffusivity. Bulk species on the
    # parent region cells; surface species on the membrane submesh (`grad` there is the tangential ∇_Γ).
    # Static membrane, so no dilution term ρ ∇_Γ·v_Γ.
    terms: list[UflExpr] = []
    for trial, prev, test, eqs, dx, region_ctx in (
        (u_in, u_in_prev, w_in, inner_eqs, dx_in, ctx),
        (u_out, u_out_prev, w_out, outer_eqs, dx_out, ctx),
        (rho, rho_prev, w_rho, membrane_eqs, dx_mem, surf_ctx),
    ):
        for k, eq in enumerate(eqs):
            terms.append((trial[k] - prev[k]) * test[k] * dx)
            diffusion = eq.terms.get("diffusion")
            if diffusion is not None:
                d = compile_expression(parse(diffusion), region_ctx)
                terms.append(dt * d * ufl.dot(ufl.grad(trial[k]), ufl.grad(test[k])) * dx)
    f_local = sum(terms[1:], terms[0])

    # Coupling on the parent's interior interface facets, **fully lagged (IMEX)**: the bulk interface
    # traces *and* the surface species use previous-step values, so each binding rate is an explicit RHS
    # forcing. The SAME rate is subtracted from a bulk (its `BCInterfaceFlux`, tested on its own side)
    # and added to the surface (a `source`, tested on the membrane), so the membrane-facet integrals
    # cancel termwise and mass is conserved to round-off. (Keeping the surface species implicit would
    # need an off-diagonal bulk-test × membrane-trial block on `dS` — a codim-0 × codim-1 coupling that
    # leaks the cancellation; full lagging sidesteps it, at the cost of conditional stability for stiff
    # binding. The method-of-lines follow-up lifts that with a Newton solve.)
    #
    # One symbol table puts EVERY species in scope at the membrane: each bulk species as its lagged
    # interface trace, each membrane species as its lagged '+' restriction. Any flux or source then
    # references any same-compartment, adjacent-compartment, or membrane species by name.
    coupling_symbols: dict[str, UflExpr] = {"geom.x": ufl.SpatialCoordinate(parent), **params}
    for k, name in enumerate(inner_species):
        coupling_symbols[name] = membrane_trace(u_in_prev[k])
    for k, name in enumerate(outer_species):
        coupling_symbols[name] = membrane_trace(u_out_prev[k])
    for k, name in enumerate(membrane_species):
        coupling_symbols[name] = rho_prev[k]("+")
    coupling_ctx = CompileContext(parent, coupling_symbols)

    # Test-only structural zeros keep every test block present in the coupling RHS, even for a species
    # that no flux/source happens to touch (else `extract_blocks` yields a None block).
    zero = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    structural = (
        [membrane_trace(w_in[k]) for k in range(len(inner_species))]
        + [membrane_trace(w_out[k]) for k in range(len(outer_species))]
        + [w_rho[k]("+") for k in range(len(membrane_species))]
    )
    f_coupling = zero * sum(structural[1:], structural[0]) * ds_int
    for k, eq in enumerate(membrane_eqs):
        source = eq.terms.get("source")
        if source is not None:
            reaction = compile_expression(parse(source), coupling_ctx)  # ∂ρ_k/∂t = … + reaction
            f_coupling += -dt * reaction * w_rho[k]("+") * ds_int
    bulk_test_of = {name: (w_in, k) for k, name in enumerate(inner_species)}
    bulk_test_of.update({name: (w_out, k) for k, name in enumerate(outer_species)})
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this assembler handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in bulk_test_of:
            continue
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # D∇u_var·n = flux INTO var's side
        test, k = bulk_test_of[bc.variable]
        f_coupling += -dt * flux * membrane_trace(test[k]) * ds_int

    a_form = fem.form(ufl.extract_blocks(ufl.lhs(f_local)), entity_maps=emaps)  # block-diagonal, static
    rhs_local_form = fem.form(ufl.extract_blocks(ufl.rhs(f_local)), entity_maps=emaps)
    rhs_coupling_form = fem.form(ufl.extract_blocks(ufl.rhs(f_coupling)), entity_maps=emaps)
    matrix = petsc.assemble_matrix(a_form)
    matrix.assemble()
    ksp = PETSc.KSP().create(parent.comm)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")

    for fn, eqs, region_ctx in (
        (u_in_fn, inner_eqs, ctx),
        (u_out_fn, outer_eqs, ctx),
        (rho_fn, membrane_eqs, surf_ctx),
    ):
        _interpolate_component_ics(fn, eqs, region_ctx)
    for fn, prev in ((u_in_fn, u_in_prev), (u_out_fn, u_out_prev), (rho_fn, rho_prev)):
        prev.x.array[:] = fn.x.array
    solution = matrix.createVecRight()

    def advance() -> None:
        rhs = petsc.assemble_vector(rhs_local_form)
        coupling_b = petsc.assemble_vector(rhs_coupling_form)  # the lagged binding forcing
        rhs.axpy(1.0, coupling_b)
        coupling_b.destroy()
        ksp.solve(rhs, solution)
        values = solution.array_r
        u_in_fn.x.array[:n_in] = values[:n_in]
        u_out_fn.x.array[:n_out] = values[n_in : n_in + n_out]
        rho_fn.x.array[:n_mem] = values[n_in + n_out : n_in + n_out + n_mem]
        for fn in (u_in_fn, u_out_fn, rho_fn):
            fn.x.scatter_forward()
        for fn, prev in ((u_in_fn, u_in_prev), (u_out_fn, u_out_prev), (rho_fn, rho_prev)):
            prev.x.array[:] = fn.x.array
        rhs.destroy()

    return MembraneCoupledProblem(inner_species, outer_species, membrane_species, u_in_fn, u_out_fn, rho_fn, advance)


class _MatrixFreeShiftedJacobian:
    """PETSc Python-matrix context for the matrix-free action of the TS implicit Jacobian
    ``σ·∂F/∂u̇ + ∂F/∂u``, computed by finite-differencing the residual at the shifted base state.

    This is the crux of the membrane-coupled MOL solve. The *exact* analytic Jacobian cannot be
    assembled: the coupling's ``∂/∂(membrane species)`` block is a bulk-coefficient × membrane-trial term
    on the interface facet, which assembles to **zero** in DOLFINx 0.10 *and* 0.11 (a mixed
    codim-0/codim-1 restriction-assembly limitation — verified directly). Finite-differencing the
    *residual* (which assembles correctly and is exactly mass-conservative) recovers that block, giving
    quadratic Newton convergence; the assembled partial Jacobian preconditions the matrix-free operator.
    `set_base` is called from the TS IJacobian each Newton update."""

    def __init__(self, residual: Callable[[PETSc.Vec, PETSc.Vec, PETSc.Vec], None], template: PETSc.Vec) -> None:
        self._residual = residual  # residual(state, rate, out): assemble F(state, rate) into out
        self._u, self._udot = template.duplicate(), template.duplicate()
        self._sigma = 0.0
        self._f_base = template.duplicate()
        self._u_pert, self._udot_pert, self._f_pert = template.duplicate(), template.duplicate(), template.duplicate()

    def set_base(self, u: PETSc.Vec, udot: PETSc.Vec, sigma: float) -> None:
        u.copy(self._u)
        udot.copy(self._udot)
        self._sigma = sigma
        self._residual(self._u, self._udot, self._f_base)

    def mult(self, _mat: PETSc.Mat, v: PETSc.Vec, y: PETSc.Vec) -> None:
        # Directional derivative of the (shifted) residual: perturb u by εv and u̇ by εσv.
        # petsc4py types Vec.norm() as float | tuple; with no norm_type it returns a float.
        eps = 1.0e-7 * (1.0 + float(self._u.norm())) / (float(v.norm()) + 1.0e-30)  # type: ignore[arg-type]
        self._u.copy(self._u_pert)
        self._u_pert.axpy(eps, v)
        self._udot.copy(self._udot_pert)
        self._udot_pert.axpy(eps * self._sigma, v)
        self._residual(self._u_pert, self._udot_pert, self._f_pert)
        self._f_pert.copy(y)
        y.axpy(-1.0, self._f_base)
        y.scale(1.0 / eps)


def integrate_membrane_coupled(
    md: MathDescription,
    geometry: InterfaceCoupledGeometry,
    *,
    t_final: float,
    dt_initial: float | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
) -> MembraneCoupledResult:
    """Integrate a **membrane species coupled to both bulks** to `t_final` with the method-of-lines
    integrator (PETSc TS adaptive BDF) — the fully-implicit counterpart of `assemble_membrane_coupled`'s
    backward-Euler/IMEX step. The binding is implicit (no lagging), so it is unconditionally stable for
    stiff kinetics, and the adaptive time error is ≈0; the same multi-species, symbol-table structure as
    the BE assembler (any number of bulk + surface species).

    **Matrix-free Newton.** The coupling is referenced through the *state* functions, so its Jacobian is
    needed — but the block ``∂(binding)/∂(membrane species)`` (a bulk-coefficient × membrane-trial term on
    `dS`) assembles to zero in DOLFINx 0.10/0.11. So the TS implicit Jacobian is applied **matrix-free**
    (`_MatrixFreeShiftedJacobian`: finite differences of the residual, which assembles fine and is exactly
    conservative), preconditioned by the *assembled partial* Jacobian (`σ·mass + diffusion + the
    assemblable coupling blocks`). This recovers quadratic Newton convergence + round-off conservation.

    Cross-mesh mechanics are the BE assembler's (region `dx`, surface `dx`, interface `dS`, `EntityMap`s,
    `membrane_trace`). The residual splits into a per-mesh local part (mass/rate + diffusion) and an
    interface coupling part (all on the parent `dS`), summed monolithically (`kind="mpi"`). Scope: static
    membrane (no `ρ ∇_Γ·v_Γ` dilution), bulk species diffusion-only; a moving membrane is a follow-up.
    """

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    inner_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.inner_subdomain]
    outer_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.outer_subdomain]
    membrane_eqs = [eq for eq in template_eqs if eq.subdomain == geometry.membrane_subdomain]
    if not inner_eqs or not outer_eqs:
        raise NotImplementedError("membrane coupling needs at least one bulk species in each compartment")
    if not membrane_eqs:
        raise NotImplementedError(
            f"membrane coupling needs a surface equation on the membrane subdomain "
            f"{geometry.membrane_subdomain!r} (the species coupled to both bulks)"
        )
    for eq in (*inner_eqs, *outer_eqs):
        if "diffusion" not in eq.terms:
            raise NotImplementedError(f"membrane coupling needs a bulk diffusion equation for {eq.variable!r}")
    inner_species = [eq.variable for eq in inner_eqs]
    outer_species = [eq.variable for eq in outer_eqs]
    membrane_species = [eq.variable for eq in membrane_eqs]

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    membrane_mesh = geometry.membrane_mesh
    V_in = fem.functionspace(geometry.inner_mesh, ("Lagrange", 1, (len(inner_species),)))
    V_out = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1, (len(outer_species),)))
    V_mem = fem.functionspace(membrane_mesh, ("Lagrange", 1, (len(membrane_species),)))
    u_in_fn, u_out_fn, rho_fn = fem.Function(V_in), fem.Function(V_out), fem.Function(V_mem)
    rate_in, rate_out, rate_mem = fem.Function(V_in), fem.Function(V_out), fem.Function(V_mem)
    n_in = V_in.dofmap.index_map.size_local * V_in.dofmap.index_map_bs
    n_out = V_out.dofmap.index_map.size_local * V_out.dofmap.index_map_bs
    n_mem = V_mem.dofmap.index_map.size_local * V_mem.dofmap.index_map_bs

    mixed = ufl.MixedFunctionSpace(V_in, V_out, V_mem)
    u_in, u_out, rho = ufl.TrialFunctions(mixed)
    w_in, w_out, w_rho = ufl.TestFunctions(mixed)
    dx_in = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.inner_region_tag)
    dx_out = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)(geometry.outer_region_tag)
    dx_mem = ufl.Measure("dx", domain=membrane_mesh)
    ds_int = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags, metadata={"quadrature_degree": 4})(
        geometry.interface_tag
    )
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map, geometry.membrane_entity_map]

    params = _param_symbols(md, parent)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    surf_params = _param_symbols(md, membrane_mesh)
    surf_ctx = CompileContext(membrane_mesh, {"geom.x": ufl.SpatialCoordinate(membrane_mesh), **surf_params})

    # --- residual: local (per-mesh mass/rate + diffusion) + coupling (interface dS, implicit) ----------
    # MOL form: the time derivative ċ = rate (a Function the TS supplies); diffusion uses the state.
    shift = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]  # the TS σ on the bulk meshes
    shift_mem = fem.Constant(membrane_mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # σ on the membrane mesh
    local_terms: list[UflExpr] = []
    precond_terms: list[UflExpr] = []
    for state, rate, trial, test, eqs, dx, region_ctx, sigma in (
        (u_in_fn, rate_in, u_in, w_in, inner_eqs, dx_in, ctx, shift),
        (u_out_fn, rate_out, u_out, w_out, outer_eqs, dx_out, ctx, shift),
        (rho_fn, rate_mem, rho, w_rho, membrane_eqs, dx_mem, surf_ctx, shift_mem),
    ):
        for k, eq in enumerate(eqs):
            local_terms.append(rate[k] * test[k] * dx)  # ċ·w
            precond_terms.append(sigma * trial[k] * test[k] * dx)  # σ·mass (the preconditioner's ∂/∂rate)
            diffusion = eq.terms.get("diffusion")
            if diffusion is not None:
                d = compile_expression(parse(diffusion), region_ctx)
                local_terms.append(d * ufl.dot(ufl.grad(state[k]), ufl.grad(test[k])) * dx)
                precond_terms.append(d * ufl.dot(ufl.grad(trial[k]), ufl.grad(test[k])) * dx)
    f_local = _sum_forms(local_terms)
    j_local = _sum_forms(precond_terms)

    # Coupling, FULLY IMPLICIT: every species in scope via its interface representation (state functions).
    coupling_symbols: dict[str, UflExpr] = {"geom.x": ufl.SpatialCoordinate(parent), **params}
    for k, name in enumerate(inner_species):
        coupling_symbols[name] = membrane_trace(u_in_fn[k])
    for k, name in enumerate(outer_species):
        coupling_symbols[name] = membrane_trace(u_out_fn[k])
    for k, name in enumerate(membrane_species):
        coupling_symbols[name] = rho_fn[k]("+")
    coupling_ctx = CompileContext(parent, coupling_symbols)
    zero = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    structural = (
        [membrane_trace(w_in[k]) for k in range(len(inner_species))]
        + [membrane_trace(w_out[k]) for k in range(len(outer_species))]
        + [w_rho[k]("+") for k in range(len(membrane_species))]
    )
    f_coup = zero * _sum_forms(structural) * ds_int
    for k, eq in enumerate(membrane_eqs):
        source = eq.terms.get("source")
        if source is not None:
            reaction = compile_expression(parse(source), coupling_ctx)
            f_coup += -reaction * w_rho[k]("+") * ds_int
    bulk_test_of = {name: (w_in, k) for k, name in enumerate(inner_species)}
    bulk_test_of.update({name: (w_out, k) for k, name in enumerate(outer_species)})
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·u_adjacent) is a follow-up increment; "
                "this integrator handles single-sided interface flux BCs"
            )
        if not isinstance(bc, BCInterfaceFlux) or bc.boundary != geometry.interface or bc.variable not in bulk_test_of:
            continue
        flux = compile_expression(parse(bc.expression), coupling_ctx)
        test, k = bulk_test_of[bc.variable]
        f_coup += -flux * membrane_trace(test[k]) * ds_int
    # The assemblable part of the coupling Jacobian (the bulk-trial columns; the membrane-trial columns
    # are zero — that block is what the matrix-free operator supplies). Used only for preconditioning.
    states, trials = [u_in_fn, u_out_fn, rho_fn], [u_in, u_out, rho]
    j_coup = _sum_forms([ufl.derivative(f_coup, states[j], trials[j]) for j in range(3)])

    f_local_form = fem.form(ufl.extract_blocks(f_local), entity_maps=emaps)
    f_coup_form = fem.form(ufl.extract_blocks(f_coup), entity_maps=emaps)
    j_local_form = fem.form(ufl.extract_blocks(j_local), entity_maps=emaps)
    j_coup_form = fem.form(ufl.extract_blocks(j_coup), entity_maps=emaps)

    for fn, eqs, region_ctx in (
        (u_in_fn, inner_eqs, ctx),
        (u_out_fn, outer_eqs, ctx),
        (rho_fn, membrane_eqs, surf_ctx),
    ):
        _interpolate_component_ics(fn, eqs, region_ctx)

    def unpack(x: PETSc.Vec) -> None:
        u_in_fn.x.array[:n_in] = x.array_r[:n_in]
        u_out_fn.x.array[:n_out] = x.array_r[n_in : n_in + n_out]
        rho_fn.x.array[:n_mem] = x.array_r[n_in + n_out : n_in + n_out + n_mem]
        for fn in (u_in_fn, u_out_fn, rho_fn):
            fn.x.scatter_forward()

    def residual(state: PETSc.Vec, rate: PETSc.Vec, out: PETSc.Vec) -> None:
        unpack(state)
        rate_in.x.array[:n_in] = rate.array_r[:n_in]
        rate_out.x.array[:n_out] = rate.array_r[n_in : n_in + n_out]
        rate_mem.x.array[:n_mem] = rate.array_r[n_in + n_out : n_in + n_out + n_mem]
        b = petsc.assemble_vector(f_local_form, kind="mpi")
        coupling_b = petsc.assemble_vector(f_coup_form, kind="mpi")
        b.axpy(1.0, coupling_b)
        b.copy(out)
        b.destroy()
        coupling_b.destroy()

    def evaluate_residual(_ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        residual(x, x_dot, result)

    # Preconditioner holder with the union sparsity of (σ·mass + diffusion) and the assemblable coupling.
    shift.value, shift_mem.value = 1.0, 1.0
    precond = petsc.assemble_matrix(j_local_form, kind="mpi")
    precond.assemble()
    coupling_pre = petsc.assemble_matrix(j_coup_form, kind="mpi")
    coupling_pre.assemble()
    precond.axpy(1.0, coupling_pre, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    coupling_pre.destroy()

    state_vec = petsc.create_vector([V_in, V_out, V_mem], kind="mpi")
    state_vec.array[:n_in] = u_in_fn.x.array[:n_in]
    state_vec.array[n_in : n_in + n_out] = u_out_fn.x.array[:n_out]
    state_vec.array[n_in + n_out : n_in + n_out + n_mem] = rho_fn.x.array[:n_mem]

    mf = _MatrixFreeShiftedJacobian(residual, state_vec)
    sizes = state_vec.getSizes()  # (local, global); petsc4py types it loosely as int | tuple
    operator = PETSc.Mat().createPython((sizes, sizes), comm=parent.comm)  # type: ignore[arg-type]
    operator.setPythonContext(mf)
    operator.setUp()

    def evaluate_jacobian(
        _ts: PETSc.TS, _t: float, x: PETSc.Vec, x_dot: PETSc.Vec, sigma: float, _mat: PETSc.Mat, pre: PETSc.Mat
    ) -> None:
        mf.set_base(x, x_dot, sigma)  # refresh the matrix-free operator's linearisation point
        unpack(x)
        shift.value, shift_mem.value = sigma, sigma
        fresh = petsc.assemble_matrix(j_local_form, kind="mpi")
        fresh.assemble()
        coupling_fresh = petsc.assemble_matrix(j_coup_form, kind="mpi")
        coupling_fresh.assemble()
        # The coupling adds off-diagonal blocks absent from the local part → its union is `pre`'s sparsity.
        fresh.axpy(1.0, coupling_fresh, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        fresh.copy(pre, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        pre.assemble()
        fresh.destroy()
        coupling_fresh.destroy()

    ts = PETSc.TS().create(parent.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType("bdf")
    ts.setIFunction(evaluate_residual, state_vec.duplicate())
    ts.setIJacobian(evaluate_jacobian, operator, precond)  # matrix-free operator, assembled preconditioner
    ts.setTimeStep(dt_initial if dt_initial is not None else t_final / 1.0e4)
    ts.setMaxTime(t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(atol, rtol)
    ts.setMaxSNESFailures(-1)
    snes = ts.getSNES()
    snes.setUseEW(False)
    snes.getKSP().setType(ksp_type)
    snes.getKSP().getPC().setType(pc_type)
    ts.setFromOptions()

    ts.solve(state_vec)
    unpack(state_vec)
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    for obj in (ts, state_vec, operator, precond):
        obj.destroy()
    return MembraneCoupledResult(
        inner_species, outer_species, membrane_species, u_in_fn, u_out_fn, rho_fn, steps, final_time
    )


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


def _interpolate_component_ics(field: fem.Function, equations: list[TemplateEquation], ctx: CompileContext) -> None:
    """Interpolate the equations' initial conditions into the vector `field` (its components are the
    equations' variables, in order; a missing IC defaults to 0). The whole vector is interpolated at
    once via `as_vector` — a one-component space has no `sub(0)` to interpolate into — with the
    expressions recompiled on `field`'s own mesh (reusing the parameter constants from `ctx`)."""
    mesh = field.function_space.mesh
    local_ctx = CompileContext(
        mesh, {"geom.x": ufl.SpatialCoordinate(mesh), **{k: v for k, v in ctx.symbols.items() if k != "geom.x"}}
    )
    components = [
        compile_expression(parse(eq.initial_condition), local_ctx) if eq.initial_condition is not None else ufl.zero()
        for eq in equations
    ]
    field.interpolate(fem.Expression(ufl.as_vector(components), field.function_space.element.interpolation_points))

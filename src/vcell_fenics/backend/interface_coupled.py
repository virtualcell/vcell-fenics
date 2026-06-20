"""Backward-Euler assembly for two volume compartments coupled across a membrane (§1.6.2).

Two bulk diffusion species — one per compartment — coupled by an **interface flux-balance** BC at
their shared membrane: `D ∇u·n = f(trace(u), trace(partner), params)` on one side, equal-and-opposite
on the other (mass conservation). The common case is a permeability flux `P·(u_outer − u_inner)` that
drives the two compartments toward equilibrium; `f` may be any expression in the two interface traces.

Cross-mesh mechanics (the `InterfaceCoupledGeometry` substrate):

- A block form needs ONE reference mesh that is also an integration domain, so *every* term is
  integrated on the shared **parent** mesh: the per-compartment mass + diffusion over the parent's
  region cells (`dx(parent)(region_tag)`), and the coupling flux over the parent's interior interface
  facets (`dS(parent)(interface_tag)`). Each compartment's P1 function lives on its own submesh and is
  pulled into the parent integral through its `EntityMap`.
- The interface trace of a bulk variable is `membrane_trace(f) = f('+') + f('-')` — each compartment's
  function is non-zero only on its own side, so the sum is its boundary value regardless of the `dS`
  side labelling. Both trials and tests are restricted this way in the coupling term.
- The flux-balance coupling is **linear in the unknowns** (a permeability flux), so it is fully
  implicit — assembled into the block bilinear form via `ufl.extract_blocks`/`ufl.lhs`, with no
  lagging. The two-block system over `MixedFunctionSpace(V_inner, V_outer)` is solved each step.

Scope (this increment): two bulk diffusion species + flux-balance interface BCs, static geometry,
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
    BCInterfaceFluxBalance,
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
        """∫u over both compartments (conserved by a flux-balance coupling with no external flux)."""
        mass = 0.0
        for field in (self.inner, self.outer):
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
    """Assemble a backward-Euler step for two bulk compartments coupled by flux-balance interface BCs
    (see the module docstring). Returns an `InterfaceCoupledProblem` whose `step()` advances `dt`."""

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
    # the flux-balance coupling (interface dS). All integrals on the parent → one block form.
    f = (u_in - u_in_prev) * w_in * dx_in + dt * d_in * ufl.dot(ufl.grad(u_in), ufl.grad(w_in)) * dx_in
    f += (u_out - u_out_prev) * w_out * dx_out + dt * d_out * ufl.dot(ufl.grad(u_out), ufl.grad(w_out)) * dx_out

    trial_of = {inner_var: u_in, outer_var: u_out}
    test_of = {inner_var: w_in, outer_var: w_out}
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError(
                "the interface value-equality constraint (u = k·partner) is a follow-up increment; "
                "this assembler handles flux-balance interface BCs"
            )
        if not isinstance(bc, BCInterfaceFluxBalance) or bc.boundary != geometry.interface:
            continue
        # The flux references the two interface traces; bind each variable to its membrane trace so a
        # linear flux (a permeability P·(partner − var)) lands fully implicit in the block bilinear.
        coupling_ctx = CompileContext(
            parent,
            {
                bc.variable: membrane_trace(trial_of[bc.variable]),
                bc.partner_variable: membrane_trace(trial_of[bc.partner_variable]),
                "geom.x": ufl.SpatialCoordinate(parent),
                **params,
            },
        )
        flux = compile_expression(parse(bc.expression), coupling_ctx)  # flux D∇u_var·n = f INTO var's side
        f += -dt * flux * membrane_trace(test_of[bc.variable]) * ds_int  # var gains the influx
        f += dt * flux * membrane_trace(test_of[bc.partner_variable]) * ds_int  # partner: equal-opposite

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

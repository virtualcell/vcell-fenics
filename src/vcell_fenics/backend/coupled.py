"""Formalism-driven mixed-dimensional assembly for bulk-surface coupled models.

This generalises the hardcoded §1.6.6 solver (`binding.py`) into the `assemble()`
path: a multi-subdomain MathDescription — one `volume` (bulk) subdomain and one
`surface` subdomain that is the bulk's boundary, coupled through `trace(·)` in the
surface sources and a Neumann BC on the bulk variable referencing the surface
variables — is read structurally and assembled as a cross-mesh block system. Any
diffusion / rate / source *expressions* and any number of surface species work,
driven by the model rather than baked in.

The mixed-dimensional mechanics are exactly those validated in `binding.py` (and the
extensive comments there), now expressed declaratively with `ufl.extract_blocks`:

- The full residual is split into a **local** part (per-subdomain mass + diffusion,
  each on its own mesh) and a **coupling** part (the cross-subdomain source and BC
  terms, on the *bulk's* interface facets `ds`, where the bulk variable is native and
  the surface trial/test functions are pulled in via the DOLFINx 0.10 `EntityMap`).
  A single form cannot mix two meshes, so the two are assembled as separate block
  matrices and summed.
- Bilinear coupling (a `trace(L)·ρ` product) is linearised semi-implicitly by
  **lagging the bulk variable** (binding stays implicit in the fast surface species —
  unconditionally stable). The coupling block re-assembles each step as Lⁿ changes; a
  structural-zero `0·L·w_L ds` keeps the lagged bulk column present for block-space
  deduction.
- `fem.Constant` carries a mesh, so coupling-form coefficients (params, dt) are bound
  to the *bulk* mesh (the coupling integration domain); a surface-mesh constant on the
  bulk `ds` would trip ffcx's cross-mesh tabulation.

Scope: one bulk + one surface-on-its-boundary, static (the §1.6.6 class). A moving
surface, >2 subdomains, and interface (bulk-bulk) BCs reuse the same machinery but
are not wired here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.geometry import CoupledGeometry
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Name, Number, UnaryOp
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCNeumann,
    MathDescription,
    ParameterConstant,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import validate_or_raise


@dataclass
class CoupledProblem:
    """A bulk-surface coupled solve, advanced by backward-Euler `step()`s. `bulk` is
    the bulk variable's solution `Function`; `surface` is the vector `Function` whose
    components are the surface variables (in `surface_vars` order). `_advance` is the
    per-step closure the assembler builds over the block system."""

    bulk_var: str
    surface_vars: list[str]
    bulk: fem.Function
    surface: fem.Function
    _advance: Callable[[], None]

    def step(self) -> None:
        self._advance()

    def field(self, name: str) -> fem.Function:
        """The solution `Function` for variable `name` (the bulk function, or a
        collapsed copy of a surface component)."""

        if name == self.bulk_var:
            return self.bulk
        k = self.surface_vars.index(name)
        component, _ = self.surface.function_space.sub(k).collapse()
        out = fem.Function(component, name=name)
        out.x.array[:] = self.surface.x.array[self.surface.function_space.sub(k).collapse()[1]]
        return out

    def integral(self, expr: UflExpr, *, surface: bool) -> float:
        """∫ over the surface (or bulk) of a UFL expression in the solution fields."""

        mesh = (self.surface if surface else self.bulk).function_space.mesh
        local = fem.assemble_scalar(fem.form(expr * ufl.dx(domain=mesh)))
        return float(mesh.comm.allreduce(local.real, op=MPI.SUM))


def _references(node: Expr, names: set[str]) -> bool:
    """Whether the expression AST references any name in `names` (e.g. a variable on
    another subdomain — making a source a cross-subdomain coupling term)."""

    if isinstance(node, Name):
        return node.name in names
    if isinstance(node, UnaryOp):
        return _references(node.operand, names)
    if isinstance(node, BinaryOp):
        return _references(node.left, names) or _references(node.right, names)
    if isinstance(node, IndexAccess):
        return _references(node.base, names)
    if isinstance(node, FunctionCall):
        return any(_references(arg, names) for arg in node.args)
    return False


def _const_params(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    out: dict[str, UflExpr] = {}
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("coupled assembly supports constant parameters only")
        out[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    return out


def assemble_coupled(md: MathDescription, geometry: CoupledGeometry, *, dt: float) -> CoupledProblem:
    """Assemble a bulk-surface coupled MathDescription against a `CoupledGeometry`.

    Exactly one equation on the bulk subdomain and one or more on the surface
    subdomain; the surface sources couple to the bulk via `trace(bulk_var)` and the
    bulk variable carries a Neumann BC at the interface referencing surface variables
    (the §1.6.6 composable pattern). Other constructs raise `NotImplementedError`.
    """

    validate_or_raise(md)
    equations = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    bulk_eqs = [eq for eq in equations if eq.subdomain == geometry.bulk_subdomain]
    surf_eqs = [eq for eq in equations if eq.subdomain == geometry.surface_subdomain]
    if len(bulk_eqs) != 1 or not surf_eqs:
        raise NotImplementedError(
            "coupled assembly handles one bulk equation and ≥1 surface equation (the §1.6.6 class); "
            f"got {len(bulk_eqs)} bulk and {len(surf_eqs)} surface equations"
        )
    bulk_eq = bulk_eqs[0]
    bulk_var = bulk_eq.variable
    surface_vars = [eq.variable for eq in surf_eqs]

    bulk_mesh, surf_mesh = geometry.bulk_mesh, geometry.surface_mesh
    n_surf = len(surf_eqs)
    V_l = fem.functionspace(bulk_mesh, ("Lagrange", 1))
    V_r = fem.functionspace(surf_mesh, ("Lagrange", 1, (n_surf,)))
    n_l = V_l.dofmap.index_map.size_local
    n_r = V_r.dofmap.index_map.size_local * V_r.dofmap.index_map_bs

    ligand, ligand_prev = fem.Function(V_l, name=bulk_var), fem.Function(V_l)
    surface, surface_prev = fem.Function(V_r, name=",".join(surface_vars)), fem.Function(V_r)

    # One mixed space so the bulk + surface arguments are jointly numbered — required
    # for `ufl.extract_blocks` over a multi-space (block) form.
    mixed = ufl.MixedFunctionSpace(V_l, V_r)
    u_l, u_r = ufl.TrialFunctions(mixed)
    w_l, w_r = ufl.TestFunctions(mixed)
    dx = ufl.Measure("dx", domain=bulk_mesh)
    dx_s = ufl.Measure("dx", domain=surf_mesh)
    ds_int = ufl.Measure("ds", domain=bulk_mesh, subdomain_data=geometry.facet_tags)(geometry.interface_tag)
    emaps = [geometry.entity_map]

    # ---- local residual (per-mesh mass + diffusion) --------------------------
    bulk_ctx = CompileContext(bulk_mesh, {"x": ufl.SpatialCoordinate(bulk_mesh), **_const_params(md, bulk_mesh)})
    surf_ctx = CompileContext(surf_mesh, {"x": ufl.SpatialCoordinate(surf_mesh), **_const_params(md, surf_mesh)})
    d_l = compile_expression(parse(bulk_eq.terms["diffusion"]), bulk_ctx)
    f_local = ufl.inner(u_l - ligand_prev, w_l) * dx + dt * d_l * ufl.inner(ufl.grad(u_l), ufl.grad(w_l)) * dx
    f_local += ufl.inner(u_r - surface_prev, w_r) * dx_s
    for k, eq in enumerate(surf_eqs):
        d_k = compile_expression(parse(eq.terms["diffusion"]), surf_ctx)
        f_local += dt * d_k * ufl.inner(ufl.grad(u_r[k]), ufl.grad(w_r[k])) * dx_s
        source = eq.terms.get("source")
        if source is not None and not _references(parse(source), {bulk_var}):
            # A purely-surface source (no bulk trace) stays local.
            local_ctx = CompileContext(surf_mesh, {**surf_ctx.symbols, **_surface_symbols(surface_vars, u_r)})
            f_local += -dt * compile_expression(parse(source), local_ctx) * w_r[k] * dx_s

    # ---- coupling residual (on the bulk interface facets) --------------------
    # Bulk variable lagged (semi-implicit); surface variables implicit (trial).
    coupling_symbols = {
        bulk_var: ligand_prev,
        **_surface_symbols(surface_vars, u_r),
        **_const_params(md, bulk_mesh),
        "x": ufl.SpatialCoordinate(bulk_mesh),
    }
    coupling_ctx = CompileContext(bulk_mesh, coupling_symbols)
    zero = fem.Constant(bulk_mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # bulk-ds, bulk constant ok
    f_coupling = zero * u_l * w_l * ds_int  # structural zero keeps the lagged bulk column
    neumann = _interface_neumann(md, bulk_var, geometry)
    if neumann is not None:
        f_coupling += -dt * compile_expression(parse(neumann.expression), coupling_ctx) * w_l * ds_int
    for k, eq in enumerate(surf_eqs):
        source = eq.terms.get("source")
        if source is not None and _references(parse(source), {bulk_var}):
            f_coupling += -dt * compile_expression(parse(source), coupling_ctx) * w_r[k] * ds_int

    # ---- assemble: A_local (once) + A_coupling (per step) --------------------
    a_local = petsc.assemble_matrix(fem.form(ufl.extract_blocks(ufl.lhs(f_local)), entity_maps=emaps))
    a_local.assemble()
    coupling_lhs = fem.form(ufl.extract_blocks(ufl.lhs(f_coupling)), entity_maps=emaps)
    rhs_local = fem.form(ufl.extract_blocks(ufl.rhs(f_local)), entity_maps=emaps)
    # The coupling is purely bilinear (no constant forcing) for §1.6.6, so its RHS is
    # empty; assemble it only when a source/BC contributes a constant term.
    rhs_coupling_expr = ufl.rhs(f_coupling)
    rhs_coupling = (
        fem.form(ufl.extract_blocks(rhs_coupling_expr), entity_maps=emaps)
        if isinstance(rhs_coupling_expr, ufl.Form) and rhs_coupling_expr.integrals()
        else None
    )

    # ---- reservoir Dirichlet on the bulk variable at the outer boundary ------
    tdim = bulk_mesh.topology.dim
    bulk_mesh.topology.create_connectivity(tdim - 1, tdim)
    outer_dofs, outer_value = _bulk_dirichlet(md, bulk_var, geometry, V_l, tdim)

    _apply_initial_conditions(bulk_eq, surf_eqs, ligand, surface, bulk_ctx, surf_ctx, V_r)
    ligand_prev.x.array[:] = ligand.x.array
    surface_prev.x.array[:] = surface.x.array

    def do_step() -> None:
        coupling = petsc.assemble_matrix(coupling_lhs)
        coupling.assemble()
        matrix = a_local.copy()
        matrix.axpy(1.0, coupling, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        if outer_dofs is not None:
            matrix.zeroRows(outer_dofs, diag=1.0)

        rhs = petsc.assemble_vector(rhs_local)
        coupling_b = petsc.assemble_vector(rhs_coupling) if rhs_coupling is not None else None
        if coupling_b is not None:
            rhs.axpy(1.0, coupling_b)
        if outer_dofs is not None:
            rhs.array[outer_dofs] = outer_value

        solution = matrix.createVecRight()
        ksp = PETSc.KSP().create(bulk_mesh.comm)
        ksp.setOperators(matrix)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.solve(rhs, solution)

        values = solution.array_r
        ligand.x.array[:n_l] = values[:n_l]
        surface.x.array[:n_r] = values[n_l : n_l + n_r]
        ligand_prev.x.array[:] = ligand.x.array
        surface_prev.x.array[:] = surface.x.array
        for obj in (ksp, coupling, coupling_b, matrix, rhs, solution):
            if obj is not None:
                obj.destroy()

    return CoupledProblem(bulk_var, surface_vars, ligand, surface, do_step)


def _surface_symbols(surface_vars: list[str], u_r: UflExpr) -> dict[str, UflExpr]:
    """Bind each surface variable to its component of the surface vector trial."""

    return {name: u_r[k] for k, name in enumerate(surface_vars)}


def _interface_neumann(md: MathDescription, bulk_var: str, geometry: CoupledGeometry) -> BCNeumann | None:
    for bc in md.boundary_conditions:
        if isinstance(bc, BCNeumann) and bc.variable == bulk_var and bc.boundary == geometry.interface:
            return bc
    return None


def _bulk_dirichlet(
    md: MathDescription, bulk_var: str, geometry: CoupledGeometry, V_l: fem.FunctionSpace, tdim: int
) -> tuple[np.ndarray | None, float]:
    """The bulk variable's Dirichlet value + dofs at the outer boundary, if any. The
    value must be a constant parameter or literal in v1 (no spatial reservoir yet)."""

    params = {p.name: p.value for p in md.parameters if isinstance(p, ParameterConstant)}
    for bc in md.boundary_conditions:
        if isinstance(bc, BCDirichlet) and bc.variable == bulk_var and bc.boundary == geometry.outer:
            node = parse(bc.expression)
            if isinstance(node, Name) and node.name in params:
                value = params[node.name]
            elif isinstance(node, Number):
                value = node.value
            else:
                raise NotImplementedError("coupled-path reservoir Dirichlet must be a constant parameter or literal")
            dofs = fem.locate_dofs_topological(V_l, tdim - 1, geometry.facet_tags.find(geometry.outer_tag))
            return dofs.astype(np.int32), float(value)
    return None, 0.0


def _apply_initial_conditions(
    bulk_eq: TemplateEquation,
    surf_eqs: list[TemplateEquation],
    ligand: fem.Function,
    surface: fem.Function,
    bulk_ctx: CompileContext,
    surf_ctx: CompileContext,
    V_r: fem.FunctionSpace,
) -> None:
    if bulk_eq.initial_condition is not None:
        ic = compile_expression(parse(bulk_eq.initial_condition), bulk_ctx)
        ligand.interpolate(fem.Expression(ic, ligand.function_space.element.interpolation_points))
    for k, eq in enumerate(surf_eqs):
        if eq.initial_condition is not None:
            ic = compile_expression(parse(eq.initial_condition), surf_ctx)
            sub = V_r.sub(k)
            surface.sub(k).interpolate(fem.Expression(ic, sub.element.interpolation_points))

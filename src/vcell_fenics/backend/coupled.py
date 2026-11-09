"""Formalism-driven mixed-dimensional assembly for bulk-surface coupled models.

A multi-subdomain MathDescription — one `volume` (bulk) subdomain and one `surface`
subdomain that is the bulk's boundary, coupled through `trace(·)` in the surface
sources and a Neumann BC on the bulk variable referencing the surface variables (the
§1.6.6 composable pattern) — is read structurally and assembled as a cross-mesh block
system. Any diffusion / rate / source *expressions* and any number of surface species
work, driven by the model rather than baked in.

The cross-mesh mechanics, expressed declaratively with `ufl.extract_blocks`:

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

**Moving surface (ADR 013).** When the surface has a prescribed velocity, the membrane and the
bulk's interface boundary move by `dt·v` each step. The surface species ride the membrane (its
material velocity is the prescribed one), so each gains the dilution term `ρ ∇_Γ·v_Γ`. The bulk's
`motion: none` means its **material is at rest** (decision 5): its domain changes because the
membrane bounding it moves, and the mesh moves under it (`w`, the harmonic extension), but the
ligand does not. So the ligand equation is the lab-frame ALE one, as on the single-mesh path
(decision 3): the conservative time term `(L − r Lⁿ)` with `r = |Kⁿ|/|Kⁿ⁺¹|` per cell, and the
transport `−L (c − w)·∇q` integrated by parts with carrier `c = 0`. Its natural condition is zero
*total* flux relative to the moving boundary, so the interface Neumann (the binding flux) is a total
flux relative to the membrane. Ligand at rest piles up ahead of an expanding membrane (`D ∂L/∂n = v·n
L` with no binding). Total ligand (bulk-free + membrane-bound) is conserved exactly. A bulk with a
prescribed motion of its own beside the moving membrane is refused (decision 6), as are bulk
`source` and `relative_advection` terms, which this path does not assemble.

The meshes are advanced by `_CoupledMeshMotion`: the membrane directly, the bulk's interface
boundary with the same nodal displacement (copied from the membrane's coincident nodes, so the
topological entity map stays valid), the outer boundary fixed, the interior by harmonic extension. A
velocity may read `geom.normal`, the membrane's normal pointing away from the region it encloses
(here, into the bulk), refreshed on the current configuration before each move. The local block
re-assembles each step on the deformed geometry.

Scope: one bulk + one surface-on-its-boundary. >2 subdomains and interface (bulk-bulk)
BCs reuse the same machinery but are not wired here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.boundary_normal import BoundaryNormal
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.geometry import CoupledGeometry, prescribed_velocity
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
    bulk_ctx = CompileContext(bulk_mesh, {"geom.x": ufl.SpatialCoordinate(bulk_mesh), **_const_params(md, bulk_mesh)})
    surf_ctx = CompileContext(surf_mesh, {"geom.x": ufl.SpatialCoordinate(surf_mesh), **_const_params(md, surf_mesh)})
    if any("advection" in eq.terms for eq in md.equations if hasattr(eq, "terms")):
        raise NotImplementedError("the lab-frame 'advection' slot is supported on the single-mesh path only")
    unassembled = sorted(set(bulk_eq.terms) - {"diffusion"})
    if unassembled:
        raise NotImplementedError(
            f"the coupled path assembles a bulk equation's 'diffusion' only; {bulk_var!r} also sets {unassembled}"
        )
    d_l = compile_expression(parse(bulk_eq.terms["diffusion"]), bulk_ctx)
    # A moving membrane: the meshes advance each step (`_CoupledMeshMotion`), the surface species ride the
    # membrane (the mandatory dilution term ρ ∇_Γ·v_Γ), and the bulk's material stays at rest (ADR 013).
    velocity_str = _surface_velocity(md, geometry.surface_subdomain)
    bulk_sub = next(s for s in md.subdomains if s.name == geometry.bulk_subdomain)
    if velocity_str is not None and prescribed_velocity(bulk_sub) is not None:
        raise NotImplementedError(
            f"volume {bulk_sub.name!r} has its own prescribed motion and borders the moving membrane "
            f"{geometry.surface_subdomain!r}; the two motions would have to agree, which is not supported yet "
            "(ADR 013 decision 6)"
        )
    motion = None if velocity_str is None else _CoupledMeshMotion(md, geometry, velocity_str, dt)
    if motion is not None:
        surf_ctx.symbols["__normal__"] = motion.surface_normal
    v_surf = compile_expression(parse(velocity_str), surf_ctx) if velocity_str is not None else None
    if motion is None:
        f_local = ufl.inner(u_l - ligand_prev, w_l) * dx
    else:
        # The ligand at rest in the lab frame on a mesh moving with w (ADR 013 decision 3): the conservative
        # time term carries Lⁿ's mass onto the moved cells, and the transport relative to the mesh,
        # −dt·L (0 − w)·∇q = L (dt w)·∇q, keeps it in place. q = 1 kills the transport term, so mass moves
        # only through the interface flux.
        f_local = ufl.inner(u_l - motion.volume_ratio * ligand_prev, w_l) * dx
        f_local += u_l * ufl.dot(motion.bulk_displacement, ufl.grad(w_l)) * dx
    f_local += dt * d_l * ufl.inner(ufl.grad(u_l), ufl.grad(w_l)) * dx
    f_local += ufl.inner(u_r - surface_prev, w_r) * dx_s

    for k, eq in enumerate(surf_eqs):
        d_k = compile_expression(parse(eq.terms["diffusion"]), surf_ctx)
        f_local += dt * d_k * ufl.inner(ufl.grad(u_r[k]), ufl.grad(w_r[k])) * dx_s
        if v_surf is not None:
            f_local += dt * ufl.div(v_surf) * u_r[k] * w_r[k] * dx_s  # dilution (div on a submesh = ∇_Γ·)
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
        "geom.x": ufl.SpatialCoordinate(bulk_mesh),
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

    # ---- assemble: A_local + A_coupling (per step) ---------------------------
    # A_local is fixed for a static mesh (assemble once); for a moving membrane it
    # re-assembles each step on the deformed geometry, so keep the form.
    a_local_form = fem.form(ufl.extract_blocks(ufl.lhs(f_local)), entity_maps=emaps)
    a_local_static = None
    if motion is None:
        a_local_static = petsc.assemble_matrix(a_local_form)
        a_local_static.assemble()
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
        if motion is not None:
            motion.advance()  # move both meshes before re-assembling on the deformed geometry
            a_local = petsc.assemble_matrix(a_local_form)
            a_local.assemble()
        else:
            assert a_local_static is not None
            a_local = a_local_static
        coupling = petsc.assemble_matrix(coupling_lhs)
        coupling.assemble()
        matrix = a_local.copy()
        matrix.axpy(1.0, coupling, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        if outer_dofs is not None:
            matrix.zeroRows(outer_dofs, diag=1.0)

        rhs = petsc.assemble_vector(rhs_local)
        # Finalise both: add ghost-entry contributions onto their owners (a no-op in serial; lost at
        # partition-boundary dofs under MPI otherwise — `assemble_vector` does not do it).
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
        coupling_b = petsc.assemble_vector(rhs_coupling) if rhs_coupling is not None else None
        if coupling_b is not None:
            coupling_b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
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
        cleanup = [ksp, coupling, coupling_b, matrix, rhs, solution]
        if motion is not None:
            cleanup.append(a_local)  # the static matrix is reused, the moving one is fresh each step
        for obj in cleanup:
            if obj is not None:
                obj.destroy()

    return CoupledProblem(bulk_var, surface_vars, ligand, surface, do_step)


def _surface_symbols(surface_vars: list[str], u_r: UflExpr) -> dict[str, UflExpr]:
    """Bind each surface variable to its component of the surface vector trial."""

    return {name: u_r[k] for k, name in enumerate(surface_vars)}


def _surface_velocity(md: MathDescription, surface_subdomain: str) -> str | None:
    """The surface subdomain's prescribed-velocity expression string, or None when it
    is static (no motion, or a literal-zero velocity)."""

    sub = next((s for s in md.subdomains if s.name == surface_subdomain), None)
    return None if sub is None else prescribed_velocity(sub)


class _CoupledMeshMotion:
    """Advances a coupled bulk + surface-on-its-boundary geometry under a prescribed
    velocity. The surface membrane moves by `dt·v`, and the bulk's interface boundary by
    the same nodal displacement (copied from the coincident membrane nodes, so the two
    meshes stay coincident and the topological entity map keeps the cross-mesh coupling
    valid); the bulk's outer boundary is held fixed and its interior follows by harmonic
    extension. The velocity is evaluated on the membrane, where `geom.normal` is bound to
    the membrane's normal away from the region it encloses (ADR 013 decision 2)."""

    def __init__(self, md: MathDescription, geometry: CoupledGeometry, velocity: str, dt: float) -> None:
        bulk, surf = geometry.bulk_mesh, geometry.surface_mesh
        self._gdim = gdim = bulk.geometry.dim
        self._bulk, self._surf = bulk, surf
        tdim = bulk.topology.dim
        bulk.topology.create_connectivity(tdim - 1, tdim)
        interface_facets = geometry.facet_tags.find(geometry.interface_tag)

        # The membrane's normal: the area-weighted facet normal on the bulk's interface, oriented away from
        # the region the membrane encloses, copied to the coincident membrane nodes.
        self._bulk_normal = BoundaryNormal(bulk, interface_facets)
        space_s = fem.functionspace(surf, ("Lagrange", 1, (gdim,)))
        self._surface_normal = fem.Function(space_s, name="normal")
        bulk_coords = self._bulk_normal.function.function_space.tabulate_dof_coordinates()
        self._surface_from_bulk = cKDTree(bulk_coords).query(space_s.tabulate_dof_coordinates())[1]
        self._refresh_normal()
        v_surf = compile_expression(
            parse(velocity),
            CompileContext(
                surf,
                {
                    "geom.x": ufl.SpatialCoordinate(surf),
                    "__normal__": self._surface_normal,
                    **_const_params(md, surf),
                },
            ),
        )

        # Bulk harmonic extension: ∇²d = 0 with d = the membrane's displacement on the interface boundary
        # and d = 0 on the outer boundary (held fixed). Re-solved each step.
        space_b = fem.functionspace(bulk, ("Lagrange", 1, (gdim,)))
        self._bulk_disp = fem.Function(space_b)
        trial, test = ufl.TrialFunction(space_b), ufl.TestFunction(space_b)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        rhs = ufl.inner(fem.Constant(bulk, np.zeros(gdim, dtype=PETSc.ScalarType)), test) * ufl.dx
        self._interface_disp = fem.Function(space_b)
        interface_dofs = fem.locate_dofs_topological(space_b, tdim - 1, interface_facets)
        outer_dofs = fem.locate_dofs_topological(space_b, tdim - 1, geometry.facet_tags.find(geometry.outer_tag))
        bcs = [
            fem.dirichletbc(self._interface_disp, interface_dofs),
            fem.dirichletbc(fem.Function(space_b), outer_dofs),
        ]
        self._bulk_problem = LinearProblem(
            a,
            rhs,
            u=self._bulk_disp,
            bcs=bcs,
            petsc_options_prefix=f"vcellfenics_coupled_motion_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        self._bulk_perm = cKDTree(space_b.tabulate_dof_coordinates()).query(bulk.geometry.x)[1]
        # each interface dof of the bulk takes the displacement of the coincident membrane node
        surface_coords = space_s.tabulate_dof_coordinates()
        self._interface_dofs = interface_dofs
        interface_coords = space_b.tabulate_dof_coordinates()[interface_dofs]
        self._interface_from_surface = cKDTree(surface_coords).query(interface_coords)[1]

        # Membrane: dt·v directly (every node is on the boundary).
        self._surf_disp = fem.Function(space_s)
        self._surf_expr = fem.Expression(dt * v_surf, space_s.element.interpolation_points)
        self._surf_perm = cKDTree(surface_coords).query(surf.geometry.x)[1]

        # Per-cell |Kⁿ|/|Kⁿ⁺¹| of the bulk (the conservative time term of the ligand at rest), 1 before a move.
        dg0 = fem.functionspace(bulk, ("DG", 0))
        self._cell_volume_form = fem.form(ufl.TestFunction(dg0) * ufl.dx)
        self._volume_ratio = fem.Function(dg0)
        self._volume_ratio.x.array[:] = 1.0

    @property
    def bulk_displacement(self) -> fem.Function:
        """The bulk mesh's nodal displacement this step (= dt·w, w the mesh velocity)."""

        return self._bulk_disp

    @property
    def volume_ratio(self) -> fem.Function:
        """Per-cell `|Kⁿ|/|Kⁿ⁺¹|` of the bulk mesh over the step just taken (1 before the first move)."""

        return self._volume_ratio

    @property
    def surface_normal(self) -> fem.Function:
        """The membrane's normal (away from the region it encloses) at the membrane's nodes."""

        return self._surface_normal

    def _refresh_normal(self) -> None:
        self._bulk_normal.refresh()
        gdim = self._gdim
        bulk_values = self._bulk_normal.function.x.array.reshape((-1, gdim))
        self._surface_normal.x.array[:] = bulk_values[self._surface_from_bulk].ravel()

    def advance(self) -> None:
        gdim = self._gdim
        self._refresh_normal()
        self._surf_disp.interpolate(self._surf_expr)
        surface_disp = self._surf_disp.x.array.reshape((-1, gdim))
        interface = self._interface_disp.x.array.reshape((-1, gdim))
        interface[self._interface_dofs] = surface_disp[self._interface_from_surface]
        self._bulk_problem.solve()
        volumes_before = fem.assemble_vector(self._cell_volume_form).array.copy()
        self._bulk.geometry.x[:, :gdim] += self._bulk_disp.x.array.reshape((-1, gdim))[self._bulk_perm]
        self._surf.geometry.x[:, :gdim] += surface_disp[self._surf_perm]
        self._volume_ratio.x.array[:] = volumes_before / fem.assemble_vector(self._cell_volume_form).array


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

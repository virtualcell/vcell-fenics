"""Assemble a DiscreteProblem from a MathDescription + Geometry.

This is the front of the backend: validate the model (formalism §2.5 plus the
geometry cross-check §1.11.10), then translate the single reaction-diffusion
equation into a `DiscreteProblem` (ADR 004). The supported subset is narrow and
guarded explicitly — anything outside it raises `NotImplementedError` rather than
silently mis-assembling (§3.6.2):

- exactly one equation, template `bulk_radv_diff` (T1) or `surface_pde_with_dilution`
  (T2), `temporality: time_dependent` (coupled multi-equation systems are a later
  increment);
- the `diffusion` and `source` slots (advection is a later increment); a source
  linear in the governed variable lands in the implicit bilinear form;
- a static (`motion: none`) or prescribed-*velocity* subdomain (prescribed
  displacement and unknown motion are later increments);
- constant parameters only.

On a codim-1 submesh `ufl.grad` is the tangential gradient ∇_Γ, so the diffusion
integrand is the same for T1 and T2 — they differ only in the mesh. When the
subdomain moves, a `DILUTION` term `ρ ∇_Γ·v_Γ` is added automatically (the
canonical moving-membrane term, §1.4.2) with `∇_Γ·v_Γ` taken as `div` of the
velocity expression; the per-step mesh motion lives in DiscreteProblem. The
initial condition is applied here by interpolation (§3.2.2).
"""

from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind
from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.core import remap_bulk_function, remap_surface_function
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    MathDescription,
    MotionNone,
    MotionPrescribedVelocity,
    ParameterConstant,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import FormalismValidationError, validate_or_raise

_SUPPORTED_TEMPLATES = {"bulk_radv_diff", "surface_pde_with_dilution"}


def assemble(md: MathDescription, geometry: Geometry, *, dt: float, fe_degree: int = 1) -> DiscreteProblem:
    """Translate `md` (against `geometry`) into a lowered `DiscreteProblem`.

    One equation builds a scalar space; several equations on a shared subdomain
    build one coupled solve over a vector space (component k ↔ equation k), so a
    `source` referencing a sibling variable becomes an off-diagonal coupling that
    the residual lhs/rhs split resolves automatically (ADR 004)."""

    validate_or_raise(md)
    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    equations = _resolve_equations(md)
    mesh = geometry.mesh_of(equations[0].subdomain)
    ctx = _compile_context(md, mesh)
    problem = _build_problem(md, equations, mesh, ctx, dt=dt, fe_degree=fe_degree)
    _apply_initial_conditions(problem, equations, ctx, len(equations))
    return problem


def rebuild_on_mesh(
    problem: DiscreteProblem, md: MathDescription, new_mesh: Mesh, *, conserve: bool = True
) -> DiscreteProblem:
    """Rebuild `problem` on `new_mesh` after a remesh, transferring its state.

    The IR is build-once (ADR 004): its forms, `LinearProblem`, and mesh-motion
    are bound to the *old* mesh, so a remesh is teardown + reassemble, not mutation
    (subtlety 2 of `docs/modeling/ale-remesh-driver.md`). This re-runs the term
    construction on `new_mesh` — same structure (term kinds, parameters, motion),
    with the velocity and dilution coefficient re-derived on the new geometry — and
    then conservatively transfers **both** `unknown` and `previous` via the `core`
    remaps. Transferring `previous` too is load-bearing: backward Euler references
    uⁿ on the new mesh, so leaving it un-transferred corrupts the first post-remesh
    step (subtlety 1).

    No initial condition is applied — the state comes from the transfer, not the
    model's IC. The fresh `_MeshMotion` re-derives its quality budget from the new
    (good) mesh, so the remesh resets the tangling headroom. v1 is P1-only (the
    conservative remaps require P1) and transfers a bulk (2D) or surface (1D)
    subdomain field.
    """

    degree = int(problem.V.ufl_element().degree)
    if degree != 1:
        raise NotImplementedError("the rebuild path is P1-only in v1 (the conservative remaps require P1)")

    equations = _resolve_equations(md)
    ctx = _compile_context(md, new_mesh)
    new_problem = _build_problem(md, equations, new_mesh, ctx, dt=float(problem.dt.value), fe_degree=degree)
    _transfer_state(problem.unknown, new_problem.unknown, conserve=conserve)
    _transfer_state(problem.previous, new_problem.previous, conserve=conserve)
    return new_problem


def _build_problem(
    md: MathDescription,
    equations: list[TemplateEquation],
    mesh: Mesh,
    ctx: CompileContext,
    *,
    dt: float,
    fe_degree: int,
) -> DiscreteProblem:
    """Build a lowered `DiscreteProblem` on `mesh` from already-resolved equations
    and a compile context bound to `mesh`. No validation, no IC — the shared core
    of `assemble` (fresh build) and `rebuild_on_mesh` (post-remesh)."""

    subdomain = equations[0].subdomain
    n = len(equations)
    element = ("Lagrange", fe_degree) if n == 1 else ("Lagrange", fe_degree, (n,))
    V = fem.functionspace(mesh, element)
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    components = [(trial, test)] if n == 1 else [(trial[k], test[k]) for k in range(n)]
    dx = ufl.Measure("dx", domain=mesh)
    velocity = _motion_velocity(md, subdomain, ctx)
    # Each governed variable bound to its component trial, so a source linear in
    # the unknowns (incl. cross-variable terms) lands in the implicit bilinear.
    var_trials = {eq.variable: components[k][0] for k, eq in enumerate(equations)}

    terms: list[Term] = [Term(TermKind.TIME_DERIVATIVE)]
    for (u, w), eq in zip(components, equations, strict=True):
        if "diffusion" in eq.terms:
            diffusion = compile_expression(parse(eq.terms["diffusion"]), ctx)
            terms.append(Term(TermKind.DIFFUSION, diffusion * ufl.dot(ufl.grad(u), ufl.grad(w))))
        if velocity is not None:
            # Auto-dilution ρ ∇_Γ·v_Γ; div on a (sub)mesh is the surface divergence.
            terms.append(Term(TermKind.DILUTION, ufl.div(velocity) * u * w))
        if "source" in eq.terms:
            source = compile_expression(parse(eq.terms["source"]), CompileContext(mesh, {**ctx.symbols, **var_trials}))
            terms.append(Term(TermKind.SOURCE, source * w))

    names = ",".join(eq.variable for eq in equations)
    return DiscreteProblem(
        variable_name=names,
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name=names),
        previous=fem.Function(V, name=f"{names}_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(float(dt))),  # type: ignore[operator]
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=velocity,
    )


def _transfer_state(src: fem.Function, dst: fem.Function, *, conserve: bool) -> None:
    """Conservatively transfer `src` (old mesh) into `dst` (new mesh), component by
    component for a coupled vector space. The remap is chosen by topological
    dimension: a 2D subdomain is a bulk field, a 1D subdomain a surface field."""

    tdim = src.function_space.mesh.topology.dim
    n = src.function_space.num_sub_spaces
    if n == 0:
        dst.x.array[:] = _remap_scalar(src, dst.function_space, tdim, conserve).x.array
        return
    for k in range(n):
        V_src_k, src_dofs = src.function_space.sub(k).collapse()
        src_k = fem.Function(V_src_k, name=src.name)
        src_k.x.array[:] = src.x.array[src_dofs]
        V_dst_k, dst_dofs = dst.function_space.sub(k).collapse()
        dst.x.array[dst_dofs] = _remap_scalar(src_k, V_dst_k, tdim, conserve).x.array


def _remap_scalar(src: fem.Function, V_new: fem.FunctionSpace, tdim: int, conserve: bool) -> fem.Function:
    if tdim == 2:
        return remap_bulk_function(src, V_new, conserve=conserve)
    if tdim == 1:
        return remap_surface_function(src, V_new, conserve=conserve)
    raise NotImplementedError(f"the rebuild path transfers 1D (surface) or 2D (bulk) fields, not tdim {tdim}")


def _apply_initial_conditions(
    problem: DiscreteProblem, equations: list[TemplateEquation], ctx: CompileContext, n: int
) -> None:
    if n == 1:
        eq = equations[0]
        if eq.initial_condition is not None:
            problem.interpolate_initial(compile_expression(parse(eq.initial_condition), ctx))
        return
    for k, eq in enumerate(equations):
        if eq.initial_condition is not None:
            ic = compile_expression(parse(eq.initial_condition), ctx)
            sub = problem.V.sub(k)
            problem.unknown.sub(k).interpolate(fem.Expression(ic, sub.element.interpolation_points))
    problem.previous.x.array[:] = problem.unknown.x.array


def _resolve_equations(md: MathDescription) -> list[TemplateEquation]:
    equations: list[TemplateEquation] = []
    for eq in md.equations:
        if not isinstance(eq, TemplateEquation) or eq.template not in _SUPPORTED_TEMPLATES:
            template = getattr(eq, "template", None)
            raise NotImplementedError(f"backend v1 supports templates {sorted(_SUPPORTED_TEMPLATES)}, not {template!r}")
        if eq.temporality != "time_dependent":
            raise NotImplementedError("backend v1 supports 'time_dependent' equations only")
        unsupported = sorted(set(eq.terms) - {"diffusion", "source"})
        if unsupported:
            raise NotImplementedError(f"backend v1 supports the 'diffusion' and 'source' slots; got {unsupported}")
        equations.append(eq)
    subdomains = {eq.subdomain for eq in equations}
    if len(subdomains) != 1:
        raise NotImplementedError(
            f"backend v1 couples equations on a single shared subdomain; got {sorted(subdomains)} "
            f"(cross-subdomain coupling via trace is a later increment)"
        )
    return equations


def _motion_velocity(md: MathDescription, subdomain: str, ctx: CompileContext) -> UflExpr | None:
    """The compiled substrate velocity for `subdomain`, or None if static.
    Prescribed displacement and unknown motion are later increments."""

    motion = next((s.motion for s in md.subdomains if s.name == subdomain), None)
    if motion is None or isinstance(motion, MotionNone):
        return None
    if isinstance(motion, MotionPrescribedVelocity):
        return compile_expression(parse(motion.velocity), ctx)
    raise NotImplementedError(
        "backend v1 supports static or prescribed-velocity motion; "
        "prescribed displacement and unknown motion are later increments"
    )


def _compile_context(md: MathDescription, mesh: Mesh) -> CompileContext:
    symbols: dict[str, UflExpr] = {"x": ufl.SpatialCoordinate(mesh)}
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("backend v1 supports constant parameters only")
        symbols[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    return CompileContext(mesh=mesh, symbols=symbols)

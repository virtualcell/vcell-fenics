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
- constant parameters, and expression parameters that compile against the
  symbol table (constants, coordinates, time, earlier parameters) — so a VCell
  unit factor like `KFlux = Area/Volume` binds.

On a codim-1 submesh `ufl.grad` is the tangential gradient ∇_Γ, so the diffusion
integrand is the same for T1 and T2 — they differ only in the mesh. When the
subdomain moves, a `DILUTION` term `ρ ∇_Γ·v_Γ` is added automatically (the
canonical moving-membrane term, §1.4.2) with `∇_Γ·v_Γ` taken as `div` of the
velocity expression; the per-step mesh motion lives in DiscreteProblem. The
initial condition is applied here by interpolation (§3.2.2).
"""

from __future__ import annotations

from typing import cast, overload

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.mesh import Mesh
from numpy.typing import NDArray
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.coupled import CoupledProblem, assemble_coupled
from vcell_fenics.backend.discrete import BackwardEuler, BoundaryTerm, DiscreteProblem, Term, TermKind
from vcell_fenics.backend.geometry import CoupledGeometry, Geometry, cross_validate
from vcell_fenics.core import remap_bulk_function, remap_surface_function
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    BCNeumann,
    BCRobin,
    MathDescription,
    MotionNone,
    MotionPrescribedVelocity,
    ParameterConstant,
    ParameterExpression,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import FormalismValidationError, validate_or_raise

_SUPPORTED_TEMPLATES = {"bulk_radv_diff", "surface_pde_with_dilution"}


@overload
def assemble(md: MathDescription, geometry: Geometry, *, dt: float, fe_degree: int = 1) -> DiscreteProblem: ...
@overload
def assemble(md: MathDescription, geometry: CoupledGeometry, *, dt: float, fe_degree: int = 1) -> CoupledProblem: ...
def assemble(
    md: MathDescription, geometry: Geometry | CoupledGeometry, *, dt: float, fe_degree: int = 1
) -> DiscreteProblem | CoupledProblem:
    """Translate `md` (against `geometry`) into a runnable problem.

    A single-mesh `Geometry` builds a `DiscreteProblem`: one equation a scalar space,
    several on a shared subdomain one coupled solve over a vector space (ADR 004). A
    `CoupledGeometry` (a bulk + a surface on its boundary) dispatches to the
    mixed-dimensional cross-mesh block assembly (`assemble_coupled`, the §1.6.6
    class), returning a `CoupledProblem`."""

    if isinstance(geometry, CoupledGeometry):
        return assemble_coupled(md, geometry, dt=dt)

    validate_or_raise(md)
    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    equations = _resolve_equations(md)
    mesh = geometry.mesh_of(equations[0].subdomain)
    ctx = _compile_context(md, mesh)
    problem = _build_problem(md, equations, mesh, ctx, dt=dt, fe_degree=fe_degree, geometry=geometry)
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
    if problem.bcs or problem.boundary_terms:
        raise NotImplementedError(
            "rebuilding a problem with Dirichlet/Neumann/Robin BCs on a remeshed mesh is not supported yet "
            "(the labelled boundary must be re-identified on the new mesh); BCs are for static geometry in v1"
        )

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
    geometry: Geometry | None = None,
) -> DiscreteProblem:
    """Build a lowered `DiscreteProblem` on `mesh` from already-resolved equations
    and a compile context bound to `mesh`. No validation, no IC — the shared core
    of `assemble` (fresh build) and `rebuild_on_mesh` (post-remesh). When `geometry`
    is given, the MathDescription's boundary conditions are translated too (the
    rebuild path passes none — BCs are static-geometry only in v1)."""

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
        if "relative_advection" in eq.terms:
            # Species drift *relative to the substrate/mesh* (§1.4 T1): the convective
            # term w_rel·∇u. On a static mesh this is plain advection (the Eulerian-fluid
            # setup, motion=none + fluid velocity here); on a mesh that moves at v_Ω it is
            # the drift on top of the substrate motion the mesh already carries. `grad` on
            # a (sub)mesh is the surface gradient, so the same term serves bulk and surface.
            drift = compile_expression(parse(eq.terms["relative_advection"]), ctx)
            terms.append(Term(TermKind.ADVECTION, ufl.dot(drift, ufl.grad(u)) * w))
        if velocity is not None:
            # Auto-dilution ρ ∇_Γ·v_Γ; div on a (sub)mesh is the surface divergence.
            terms.append(Term(TermKind.DILUTION, ufl.div(velocity) * u * w))
        if "source" in eq.terms:
            source = compile_expression(parse(eq.terms["source"]), CompileContext(mesh, {**ctx.symbols, **var_trials}))
            terms.append(Term(TermKind.SOURCE, source * w))

    bcs, boundary_terms, dirichlet_refreshers = (
        _build_boundary_conditions(md, equations, geometry, mesh, V, components, ctx, var_trials)
        if geometry is not None
        else ([], [], [])
    )

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
        bcs=bcs,
        boundary_terms=tuple(boundary_terms),
        motion_velocity=velocity,
        time=ctx.symbols["sim.t"],
        dirichlet_refreshers=tuple(dirichlet_refreshers),
    )


def _build_boundary_conditions(
    md: MathDescription,
    equations: list[TemplateEquation],
    geometry: Geometry,
    mesh: Mesh,
    V: fem.FunctionSpace,
    components: list[tuple[UflExpr, UflExpr]],
    ctx: CompileContext,
    var_trials: dict[str, UflExpr],
) -> tuple[list[fem.DirichletBC], list[BoundaryTerm], list[tuple[fem.Function, fem.Expression]]]:
    """Translate the MathDescription's boundary conditions into strong Dirichlet BCs
    and weak Neumann / Robin boundary terms, plus the Dirichlet `(value, expression)`
    refreshers a driver re-interpolates to track a time-dependent `g(t)`.

    Scope: Dirichlet / Neumann / Robin on a labelled boundary of *this* solve's subdomain,
    external *or* an internal interface — a **one-sided** flux where the variable lives only in
    this incident compartment is a Neumann/Robin on the compartment's submesh boundary (the
    facets are re-located onto the submesh). The two `BCInterface*` kinds (genuine cross-compartment
    coupling) and a Dirichlet on an internal interface still raise `NotImplementedError`.

    A **weak** (Neumann / Robin) flux is compiled with the governed variables bound to their trial
    functions (`var_trials`), so a flux that depends on the species — a jump-condition efflux
    `D∇u·n = −k·trace(u)` (§1.6.5) — lands its variable-linear part in the implicit bilinear, exactly
    as a Robin's `αu` term does. A **Dirichlet** value is a prescribed `g` (interpolated), so it is
    compiled against the base `ctx` only; a Dirichlet referencing a variable surfaces as a `CompileError`.
    """

    subdomain = equations[0].subdomain
    var_index = {eq.variable: k for k, eq in enumerate(equations)}
    n = len(equations)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    # Neumann/Robin flux data may reference the species (a u-dependent flux); bind the variables to
    # their trials so the linear-in-u part goes implicit. Dirichlet keeps the base ctx (a prescribed g).
    flux_ctx = CompileContext(mesh=mesh, symbols={**ctx.symbols, **var_trials})

    bcs: list[fem.DirichletBC] = []
    # Weak (Neumann / Robin) boundary terms are built after the loop so every exterior-facet `ds`
    # measure shares ONE subdomain_data `MeshTags` with a distinct tag per boundary — DOLFINx
    # requires all exterior_facet integrals in a form to carry the same subdomain_data object, so a
    # per-boundary `MeshTags` (the natural one-boundary case) breaks the moment a second appears
    # (e.g. all four box faces of a no-flux geometry).
    weak: list[tuple[TermKind, UflExpr, NDArray[np.int32]]] = []
    dirichlet_refreshers: list[tuple[fem.Function, fem.Expression]] = []
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality | BCInterfaceFlux):
            raise NotImplementedError(
                "backend v1 supports external Dirichlet/Neumann/Robin BCs; an interface BC needs an internal "
                "boundary between two subdomains (multi-compartment geometry), a later increment"
            )
        if bc.variable not in var_index:
            raise NotImplementedError(
                f"BC references variable {bc.variable!r}, not governed in this solve's subdomain {subdomain!r}"
            )
        bgeo = geometry.boundary_of(bc.boundary)
        if bgeo is None:  # cross_validate already guards this; belt-and-braces for direct callers
            raise NotImplementedError(f"BC boundary {bc.boundary!r} is not a labelled boundary of the geometry")
        if subdomain not in bgeo.subdomains:
            raise NotImplementedError(
                f"BC boundary {bc.boundary!r} bounds {bgeo.subdomains}, not this solve's subdomain {subdomain!r}"
            )
        if bgeo.is_internal and isinstance(bc, BCDirichlet):
            raise NotImplementedError(
                f"a Dirichlet BC on the internal interface {bc.boundary!r} is a later increment "
                "(a one-sided Neumann/Robin on an interface is supported; cross-compartment coupling is not)"
            )
        # Facets on THIS solve's mesh. For a multi-compartment geometry a labelled boundary's facets
        # index the shared parent mesh, so re-locate them on this subdomain's submesh — for an internal
        # membrane that's the compartment's exterior boundary there, so a one-sided flux (the species
        # lives only in this incident compartment) is a Neumann/Robin on the submesh boundary (the
        # composable bulk-surface pattern §1.6.5). A single-compartment geometry's facets already lie
        # on its mesh.
        facets = (
            _facets_on_submesh(mesh, geometry.parent_mesh, bgeo.facets)
            if geometry.parent_mesh is not None
            else np.asarray(bgeo.facets, dtype=np.int32)
        )
        k = var_index[bc.variable]
        u, w = components[k]
        if isinstance(bc, BCDirichlet):
            dirichlet, refresher = _dirichlet_bc(bc, V, n, k, fdim, facets, ctx)
            bcs.append(dirichlet)
            dirichlet_refreshers.append(refresher)
        elif isinstance(bc, BCNeumann):
            # D∇u·n = h ⇒ the weak boundary term ∫_Γ h·v ds enters the residual as −h·v. `h` may be
            # u-dependent (a jump-condition flux ∝ trace(u)); var-bound, its u-linear part goes implicit.
            h = compile_expression(parse(bc.expression), flux_ctx)
            weak.append((TermKind.NEUMANN, -h * w, facets))
        elif isinstance(bc, BCRobin):
            # αu + βD∇u·n = h ⇒ D∇u·n = (h − αu)/β ⇒ residual gains (α/β)u·v − (h/β)·v.
            alpha = compile_expression(parse(bc.alpha), flux_ctx)
            beta = compile_expression(parse(bc.beta), flux_ctx)
            h = compile_expression(parse(bc.expression), flux_ctx)
            integrand = (alpha / beta) * u * w - (h / beta) * w
            weak.append((TermKind.ROBIN, integrand, facets))

    boundary_terms = _weak_boundary_terms(mesh, fdim, weak)
    return bcs, boundary_terms, dirichlet_refreshers


def _facets_on_submesh(submesh: Mesh, parent: Mesh, parent_facets: NDArray[np.int32]) -> NDArray[np.int32]:
    """Re-locate `parent_facets` (facet indices on the shared `parent` mesh) onto `submesh` by matching
    facet midpoints. `create_submesh` copies the parent geometry, so every boundary facet has an
    identical-midpoint twin among the submesh's exterior facets — including a parent *interior* facet
    on a compartment interface, which becomes *exterior* on that compartment's submesh. Used to apply a
    boundary condition on a compartment's submesh when the geometry labels facets on the parent."""

    pdim = parent.topology.dim - 1
    parent_mid = dmesh.compute_midpoints(parent, pdim, np.asarray(parent_facets, dtype=np.int32))
    sdim = submesh.topology.dim - 1
    submesh.topology.create_connectivity(sdim, submesh.topology.dim)
    exterior = dmesh.exterior_facet_indices(submesh.topology)
    submesh_mid = dmesh.compute_midpoints(submesh, sdim, exterior)
    distances, nearest = cKDTree(submesh_mid).query(parent_mid)
    # Midpoints coincide to round-off (same physical facets); keep only the matched ones.
    matched = exterior[nearest[distances < 1.0e-9]]
    return np.unique(matched).astype(np.int32)


def _weak_boundary_terms(
    mesh: Mesh, fdim: int, weak: list[tuple[TermKind, UflExpr, NDArray[np.int32]]]
) -> list[BoundaryTerm]:
    """Build the Neumann / Robin boundary terms over one shared exterior-facet `MeshTags`: each
    boundary gets a distinct tag, and every term's `ds` measure carries that single subdomain_data
    object (the DOLFINx form-assembly invariant). Returns the terms in input order."""

    if not weak:
        return []
    facets = np.concatenate([f for *_, f in weak]).astype(np.int32)
    values = np.concatenate([np.full(f.size, tag, dtype=np.int32) for tag, (*_, f) in enumerate(weak, start=1)])
    order = np.argsort(facets)
    tags = dmesh.meshtags(mesh, fdim, facets[order], values[order])
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
    return [
        BoundaryTerm(kind, integrand, cast(ufl.Measure, ds(tag)))
        for tag, (kind, integrand, _) in enumerate(weak, start=1)
    ]


def _dirichlet_bc(
    bc: BCDirichlet, V: fem.FunctionSpace, n: int, k: int, fdim: int, facets: np.ndarray, ctx: CompileContext
) -> tuple[fem.DirichletBC, tuple[fem.Function, fem.Expression]]:
    """A strong Dirichlet BC u = g on the labelled boundary. For a coupled vector
    space the constraint is applied on the k-th component subspace.

    Returns the BC and a `(value, expression)` refresher: re-interpolating `value`
    from `expression` re-evaluates `g` against the (mutable) compile context, so a
    `g(t)` tracks the bound time Constant once the driver advances it."""

    g = compile_expression(parse(bc.expression), ctx)
    if n == 1:
        value = fem.Function(V, name=f"{bc.variable}_bc")
        expression = fem.Expression(g, V.element.interpolation_points)
        value.interpolate(expression)
        return fem.dirichletbc(value, fem.locate_dofs_topological(V, fdim, facets)), (value, expression)
    sub = V.sub(k)
    sub_space, _ = sub.collapse()
    value = fem.Function(sub_space, name=f"{bc.variable}_bc")
    expression = fem.Expression(g, sub_space.element.interpolation_points)
    value.interpolate(expression)
    dofs = fem.locate_dofs_topological((sub, sub_space), fdim, facets)
    return fem.dirichletbc(value, dofs, sub), (value, expression)


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
        unsupported = sorted(set(eq.terms) - {"diffusion", "source", "relative_advection"})
        if unsupported:
            raise NotImplementedError(
                f"backend supports the 'diffusion', 'source', and 'relative_advection' slots; got {unsupported}"
            )
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


def _compile_context(md: MathDescription, mesh: Mesh, *, seed: int = 0) -> CompileContext:
    # The namespaced built-ins (ADR 006): `geom.x` is the position field, `sim.t` a mutable time
    # Constant. Expressions outside the IC (which the validator forbids `sim.t` in) may reference
    # the time, and the driver advances it each step — e.g. a time-dependent Dirichlet value
    # g(sim.t). It stays 0 unless a step updates it. `seed` seeds the generator the random IC
    # primitives (`normal`/`uniform`) draw from, so a model's random ICs are reproducible.
    symbols: dict[str, UflExpr] = {
        "geom.x": ufl.SpatialCoordinate(mesh),
        "sim.t": fem.Constant(mesh, PETSc.ScalarType(0.0)),  # type: ignore[operator]
    }
    ctx = CompileContext(mesh=mesh, symbols=symbols, rng=np.random.default_rng(seed))
    # A ParameterExpression compiles against the symbols defined so far (constants, coordinates, time,
    # and earlier parameters) — so a VCell unit factor like KFlux = Area/Volume or
    # UnitFactor = pow(KMOLE, 1) binds as a UFL expression, as do (legitimately) spatial geom.x or
    # time-dependent sim.t parameters. Parameters are processed in declaration order, which the import
    # keeps dependency-ordered (constants before the functions that use them); a forward reference
    # surfaces as a loud CompileError rather than a silent zero.
    for p in md.parameters:
        if isinstance(p, ParameterConstant):
            symbols[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
        elif isinstance(p, ParameterExpression):
            symbols[p.name] = compile_expression(parse(p.expression), ctx)
        else:
            raise NotImplementedError(f"backend v1 cannot bind parameter {p.name!r} of type {type(p).__name__}")
    return ctx

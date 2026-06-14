"""Unknown (mechanics-driven) membrane motion — the §1.10.8 model.

The prescribed-motion path moves a subdomain by a *known* velocity expression. Here
the velocity is **solved**: a `motion: { kind: unknown, variable: v }` subdomain wires
its substrate velocity to a vector unknown governed by a force balance (a weak-form
equation, §1.5, since mechanics templates are v2). An optional surface species on the
same membrane experiences the resulting motion through the standard T2 dilution. This
is the path to genuine cell migration: the membrane moves under force, not by fiat.

The discretisation is **staggered** (quasi-static motion + the species): each step

  1. (if the force balance uses curvature) re-project the mean-curvature vector,
  2. solve the force balance for the velocity `v` on the current membrane,
  3. move the membrane by `dt·v`,
  4. (if present) advance the receptor T2 (mass + diffusion + dilution `ρ ∇_Γ·v`) on
     the moved membrane.

**Curvature forces.** `n(x)` (normal) and `H(x)` (mean curvature) are not pointwise
on a discrete membrane (a polygon's curvature is vertex-concentrated). They are
resolved from a *projected* mean-curvature vector κ = H·n — the weak surface Laplacian
of position, `∫κ·φ ds = ∫∇_Γ X : ∇_Γ φ ds` — so `H(x)` = |κ|, `n(x)` = κ/|κ|, and a
form `σ H(x) inner(n(x), test)` evaluates to the correct weak force `inner(κ, test)`.
κ is re-projected on the deformed membrane each step. A surface-tension force balance
`η v + σ H n = 0` then drives mean-curvature flow (a circle shrinks as `r² = r₀² −
2σt/η`).

**Tangential redistribution (`redistribute=True`).** Pure normal motion crowds nodes
where a non-circular membrane contracts, degrading the mesh until it tangles. For a
*motion-only* curvature membrane this flag switches to the BGN scheme
(`core/bgn_curve_mesh`): each step is one coupled (position, curvature) solve that
follows the same flow *and* slides nodes tangentially to stay equidistributed, so the
flow runs at a usable `dt`. It is a pure discretisation choice (the continuous
solution is unchanged), so it is a backend argument, not a MathDescription field. The
BGN mobility `m = σ/η` is read off the force balance (`_calibrate_mobility`), keeping
it tied to the model's own parameters with no expression parsing. A co-moving receptor
is refused: once the mesh redistributes, `v_mesh ≠ v_material` and the surface PDE
needs the ALE advection term `−(v_mesh − v_material)·∇_Γ ρ`, a later increment.

Scope (v1): one closed membrane (a codim-1 submesh, no bulk), one weak-form motion
equation for a vector velocity, and at most one T2 receptor equation. Vector
*expression* parameters, multiple receptors, redistribution with a receptor, and a
moving membrane coupled to a bulk are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind, _MeshMotion
from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.core.bgn_curve_mesh import bgn_redistribute_membrane
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, UnaryOp, VectorLiteral
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    MathDescription,
    MotionUnknown,
    ParameterConstant,
    TemplateEquation,
    WeakFormEquation,
)
from vcell_fenics.formalism.validator import FormalismValidationError, validate_or_raise


@dataclass
class UnknownMotionProblem:
    """A mechanics-driven membrane: a solved velocity field that moves the membrane,
    and (optionally) a receptor species that dilutes with it. `step()` runs the
    staggered scheme; `velocity` holds the solved motion field and `receptor.unknown`
    the species (when present)."""

    motion_var: str
    velocity: fem.Function
    receptor_var: str | None
    receptor: DiscreteProblem | None
    _force_balance: LinearProblem
    _motion: _MeshMotion
    _curvature: _CurvatureProjection | None
    _bgn: _BGNMotion | None = None

    def step(self) -> None:
        # BGN redistribution (motion-only curvature flow) replaces the force-balance
        # velocity solve and the normal mesh move with one coupled position step that
        # also redistributes nodes tangentially (§ BGN; mobility calibrated at build).
        if self._bgn is not None:
            self._bgn.advance()
            return
        if self._curvature is not None:
            self._curvature.project()  # mean-curvature vector on the current geometry
        self._force_balance.solve()  # force balance → velocity on the current membrane
        self._motion.advance()  # move the membrane by dt·v
        if self.receptor is not None:
            self.receptor.step()  # advance the receptor (dilution from v) on the moved membrane


class _CurvatureProjection:
    """The projected mean-curvature vector κ = H·n on the membrane (the weak surface
    Laplacian of position). `project()` re-solves it on the current geometry; the UFL
    `mean_curvature` (|κ|) and `normal` (κ/|κ|) are what a form's `H(x)`/`n(x)` resolve
    to."""

    def __init__(self, mesh: Mesh) -> None:
        gdim = mesh.geometry.dim
        space = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
        self.kappa = fem.Function(space, name="curvature_vector")
        trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
        dx = ufl.Measure("dx", domain=mesh)
        self._problem = LinearProblem(
            ufl.inner(trial, test) * dx,
            ufl.inner(ufl.grad(ufl.SpatialCoordinate(mesh)), ufl.grad(test)) * dx,
            u=self.kappa,
            petsc_options_prefix=f"vcellfenics_curvature_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        magnitude = ufl.sqrt(ufl.inner(self.kappa, self.kappa) + 1e-14)  # eps guards flat regions
        self.mean_curvature: UflExpr = magnitude
        self.normal: UflExpr = self.kappa / magnitude

    def project(self) -> None:
        self._problem.solve()


class _BGNMotion:
    """Mesh motion by the BGN curvature-flow step — the drop-in replacement for
    `_MeshMotion` when tangential redistribution is requested for a motion-only
    curvature membrane. Each `advance()` moves the membrane by one coupled
    (position, curvature) solve that both follows the flow `V = −m κ` and keeps the
    nodes equidistributed; `mobility` `m = σ/η` is calibrated from the force balance
    at build time (`_calibrate_mobility`)."""

    def __init__(self, mesh: Mesh, *, mobility: float, dt: float) -> None:
        self._mesh = mesh
        self._mobility = mobility
        self._dt = dt

    def advance(self) -> None:
        bgn_redistribute_membrane(self._mesh, mobility=self._mobility, dt=self._dt)


def _calibrate_mobility(curvature: _CurvatureProjection, force_balance: LinearProblem, velocity: fem.Function) -> float:
    """The curvature-flow mobility `m = σ/η` implied by the force balance, read off
    one solve: for surface-tension flow the solved velocity satisfies `v = −m κ⃗`
    nodally (same mass matrix on both sides of `η v = −σ κ⃗`), so `m` is the least-
    squares ratio `−(v·κ⃗)/(κ⃗·κ⃗)`. This keeps BGN's mobility tied to the model's
    own parameters without parsing the force expression. Fails loudly on a flat
    membrane (no curvature to calibrate against)."""

    curvature.project()
    force_balance.solve()
    v = velocity.x.array
    kappa = curvature.kappa.x.array
    denom = float(np.dot(kappa, kappa))
    if denom < 1e-14:
        raise NotImplementedError(
            "cannot calibrate the BGN mobility: the membrane has ~zero curvature, so the force balance "
            "does not pin a curvature-flow speed. Redistribution is defined for mean-curvature flow."
        )
    return -float(np.dot(v, kappa)) / denom


def assemble_unknown_motion(
    md: MathDescription, geometry: Geometry, *, dt: float, redistribute: bool = False
) -> UnknownMotionProblem:
    """Assemble the §1.10.8 unknown-motion model: a weak-form force balance for the
    membrane velocity (optionally curvature-driven) coupled to an optional T2 receptor
    that dilutes with the solved motion.

    `redistribute=True` switches a *motion-only* curvature membrane to the BGN scheme:
    each step advances by curvature flow *and* redistributes nodes tangentially, so
    the mesh stays well-shaped and the flow runs at a usable `dt`. It is a pure
    discretisation choice (the continuous solution is unchanged), hence a backend flag
    rather than a MathDescription field. v1 requires a curvature force balance and no
    receptor — a co-moving species would need the ALE advection term that arises once
    the mesh velocity differs from the material velocity."""

    validate_or_raise(md)
    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    moving = [s for s in md.subdomains if isinstance(s.motion, MotionUnknown)]
    if len(moving) != 1:
        raise NotImplementedError("the unknown-motion path handles exactly one unknown-motion subdomain in v1")
    subdomain = moving[0].name
    motion_var = moving[0].motion.variable  # type: ignore[union-attr]
    mesh = geometry.mesh_of(subdomain)
    gdim = mesh.geometry.dim
    dx = ufl.Measure("dx", domain=mesh)
    dt_const = fem.Constant(mesh, PETSc.ScalarType(float(dt)))  # type: ignore[operator]

    force_eqs = [eq for eq in md.equations if isinstance(eq, WeakFormEquation) and eq.variable == motion_var]
    receptor_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation) and eq.subdomain == subdomain]
    if len(force_eqs) != 1 or len(receptor_eqs) > 1:
        raise NotImplementedError(
            "v1 unknown motion needs one weak-form motion equation and at most one T2 receptor equation; "
            f"got {len(force_eqs)} motion and {len(receptor_eqs)} receptor equations"
        )

    # ---- the velocity solve (a weak-form force balance) ----------------------
    velocity_space = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    velocity = fem.Function(velocity_space, name=motion_var)
    v_trial, v_test = ufl.TrialFunction(velocity_space), ufl.TestFunction(velocity_space)
    motion_symbols: dict[str, UflExpr] = {
        "x": ufl.SpatialCoordinate(mesh),
        motion_var: v_trial,
        f"{motion_var}_test": v_test,
        "dx": dx,
        "dx_Gamma": dx,
        **_const_params(md, mesh),
    }
    force_ast = parse(force_eqs[0].form)
    curvature = None
    if _uses_curvature(force_ast):
        curvature = _CurvatureProjection(mesh)
        motion_symbols["__normal__"] = curvature.normal
        motion_symbols["__mean_curvature__"] = curvature.mean_curvature
    force_form = compile_expression(force_ast, CompileContext(mesh, motion_symbols))
    force_balance = LinearProblem(
        ufl.lhs(force_form),
        ufl.rhs(force_form),
        u=velocity,
        bcs=[],
        petsc_options_prefix=f"vcellfenics_force_{id(velocity):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # ---- optional BGN tangential redistribution (motion-only curvature flow) --
    bgn = None
    if redistribute:
        if curvature is None:
            raise NotImplementedError(
                "redistribute=True requires a curvature (n(x)/H(x)) force balance — BGN tangential "
                "redistribution is defined for mean-curvature flow."
            )
        if receptor_eqs:
            raise NotImplementedError(
                "redistribute=True is motion-only in v1: a co-moving receptor needs the ALE advection term "
                "-(v_mesh - v_material)·∇_Γ ρ that arises when the mesh redistributes; deferred."
            )
        bgn = _BGNMotion(mesh, mobility=_calibrate_mobility(curvature, force_balance, velocity), dt=dt)

    # ---- mesh motion (owned here) + the optional receptor --------------------
    motion = _MeshMotion(mesh, velocity, dt_const)
    receptor = _build_receptor(receptor_eqs[0], md, mesh, dx, velocity, dt_const) if receptor_eqs else None
    receptor_var = receptor_eqs[0].variable if receptor_eqs else None
    return UnknownMotionProblem(motion_var, velocity, receptor_var, receptor, force_balance, motion, curvature, bgn)


def _build_receptor(
    eq: TemplateEquation, md: MathDescription, mesh: Mesh, dx: ufl.Measure, velocity: fem.Function, dt: fem.Constant
) -> DiscreteProblem:
    """A T2 surface PDE whose dilution reads the solved velocity `Function` (so it
    updates as the force balance re-solves). The mesh motion is owned by the
    `UnknownMotionProblem`, so this problem carries no `_MeshMotion` of its own."""

    space = fem.functionspace(mesh, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    ctx = CompileContext(mesh, {"x": ufl.SpatialCoordinate(mesh), **_const_params(md, mesh)})

    diffusion = compile_expression(parse(eq.terms["diffusion"]), ctx)
    terms = [
        Term(TermKind.TIME_DERIVATIVE),
        Term(TermKind.DIFFUSION, diffusion * ufl.dot(ufl.grad(trial), ufl.grad(test))),
        Term(TermKind.DILUTION, ufl.div(velocity) * trial * test),  # ρ ∇_Γ·v with the solved v
    ]
    if "source" in eq.terms:
        source_ctx = CompileContext(mesh, {**ctx.symbols, eq.variable: trial})
        terms.append(Term(TermKind.SOURCE, compile_expression(parse(eq.terms["source"]), source_ctx) * test))

    problem = DiscreteProblem(
        variable_name=eq.variable,
        V=space,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(space, name=eq.variable),
        previous=fem.Function(space, name=f"{eq.variable}_old"),
        dt=dt,
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=None,  # the UnknownMotionProblem advances the mesh
    )
    if eq.initial_condition is not None:
        problem.interpolate_initial(compile_expression(parse(eq.initial_condition), ctx))
    return problem


def _uses_curvature(node: Expr) -> bool:
    """Whether the expression AST references `n(x)` or `H(x)` (so the assembler must
    build and bind the curvature projection)."""

    if isinstance(node, FunctionCall):
        return node.callee in ("n", "H") or any(_uses_curvature(a) for a in node.args)
    if isinstance(node, BinaryOp):
        return _uses_curvature(node.left) or _uses_curvature(node.right)
    if isinstance(node, UnaryOp):
        return _uses_curvature(node.operand)
    if isinstance(node, IndexAccess):
        return _uses_curvature(node.base)
    if isinstance(node, VectorLiteral):
        return any(_uses_curvature(c) for c in node.components)
    return False


def _const_params(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    out: dict[str, UflExpr] = {}
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("the unknown-motion path supports constant parameters only in v1")
        out[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    return out

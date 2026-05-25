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
    """Translate `md` (against `geometry`) into a lowered `DiscreteProblem`."""

    validate_or_raise(md)
    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    eq = _resolve_equation(md)
    mesh = geometry.mesh_of(eq.subdomain)
    V = fem.functionspace(mesh, ("Lagrange", fe_degree))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ctx = _compile_context(md, mesh)

    terms = [Term(TermKind.TIME_DERIVATIVE)]
    if "diffusion" in eq.terms:
        diffusion = compile_expression(parse(eq.terms["diffusion"]), ctx)
        terms.append(Term(TermKind.DIFFUSION, diffusion * ufl.dot(ufl.grad(trial), ufl.grad(test))))

    velocity = _motion_velocity(md, eq, ctx)
    if velocity is not None:
        # Auto-dilution ρ ∇_Γ·v_Γ; div on a (sub)mesh is the surface divergence.
        terms.append(Term(TermKind.DILUTION, ufl.div(velocity) * trial * test))

    if "source" in eq.terms:
        # The source is compiled with the governed variable bound to the trial
        # function, so a source linear in the unknown lands in the implicit
        # (backward-Euler) bilinear form. (A single-equation model can only
        # reference its own variable; cross-variable coupling is increment 3b.)
        source_ctx = CompileContext(mesh=mesh, symbols={**ctx.symbols, eq.variable: trial})
        source = compile_expression(parse(eq.terms["source"]), source_ctx)
        terms.append(Term(TermKind.SOURCE, source * test))

    problem = DiscreteProblem(
        variable_name=eq.variable,
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name=eq.variable),
        previous=fem.Function(V, name=f"{eq.variable}_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(float(dt))),  # type: ignore[operator]
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=velocity,
    )

    if eq.initial_condition is not None:
        problem.interpolate_initial(compile_expression(parse(eq.initial_condition), ctx))
    return problem


def _resolve_equation(md: MathDescription) -> TemplateEquation:
    if len(md.equations) != 1:
        raise NotImplementedError("backend v1 supports a single equation; coupled systems are a later increment")
    eq = md.equations[0]
    if not isinstance(eq, TemplateEquation) or eq.template not in _SUPPORTED_TEMPLATES:
        template = getattr(eq, "template", None)
        raise NotImplementedError(f"backend v1 supports templates {sorted(_SUPPORTED_TEMPLATES)}, not {template!r}")
    if eq.temporality != "time_dependent":
        raise NotImplementedError("backend v1 supports 'time_dependent' equations only")
    unsupported = sorted(set(eq.terms) - {"diffusion", "source"})
    if unsupported:
        raise NotImplementedError(f"backend v1 supports the 'diffusion' and 'source' slots; got {unsupported}")
    return eq


def _motion_velocity(md: MathDescription, eq: TemplateEquation, ctx: CompileContext) -> UflExpr | None:
    """The compiled substrate velocity for `eq`'s subdomain, or None if static.
    Prescribed displacement and unknown motion are later increments."""

    motion = next((s.motion for s in md.subdomains if s.name == eq.subdomain), None)
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

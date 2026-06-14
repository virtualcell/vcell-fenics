"""Unknown (mechanics-driven) membrane motion — the §1.10.8 model.

The prescribed-motion path moves a subdomain by a *known* velocity expression. Here
the velocity is **solved**: a `motion: { kind: unknown, variable: v }` subdomain wires
its substrate velocity to a vector unknown governed by a force balance (a weak-form
equation, §1.5, since mechanics templates are v2), and a surface species on the same
membrane experiences the resulting motion through the standard T2 dilution. This is
the path to genuine cell migration: the membrane moves under force, not by fiat.

The discretisation is **staggered** (quasi-static motion + the species): each step

  1. solve the force balance for the velocity `v` on the current membrane,
  2. move the membrane by `dt·v`,
  3. advance the receptor T2 (mass + diffusion + dilution `ρ ∇_Γ·v`) on the moved
     membrane.

It reuses the existing machinery: the velocity is solved into a `Function`, which is
handed to a `DiscreteProblem` as its `motion_velocity` — so the dilution term and the
per-step `_MeshMotion` both read the freshly-solved field. The force balance and the
receptor share the membrane mesh; both re-assemble on the deformed geometry each step.

Scope (v1): one closed membrane (a codim-1 submesh, no bulk), one weak-form motion
equation for a vector velocity, and one T2 receptor equation. Vector *expression*
parameters (declare `f_active` separately), multiple receptors, the `n(x)`/`H(x)`
curvature helpers, and a moving membrane coupled to a bulk are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind
from vcell_fenics.backend.geometry import Geometry, cross_validate
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
    and a receptor species that dilutes with it. `step()` runs the staggered scheme;
    `velocity` holds the solved motion field and `receptor.unknown` the species."""

    motion_var: str
    velocity: fem.Function
    receptor_var: str
    receptor: DiscreteProblem
    _force_balance: LinearProblem

    def step(self) -> None:
        self._force_balance.solve()  # solve the force balance on the current membrane
        self.receptor.step()  # move the membrane by dt·v, then advance the receptor


def assemble_unknown_motion(md: MathDescription, geometry: Geometry, *, dt: float) -> UnknownMotionProblem:
    """Assemble the §1.10.8 unknown-motion model: a weak-form force balance for the
    membrane velocity coupled to a T2 receptor that dilutes with the solved motion."""

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

    force_eqs = [eq for eq in md.equations if isinstance(eq, WeakFormEquation) and eq.variable == motion_var]
    receptor_eqs = [
        eq for eq in md.equations if isinstance(eq, TemplateEquation) and eq.subdomain == subdomain
    ]
    if len(force_eqs) != 1 or len(receptor_eqs) != 1:
        raise NotImplementedError(
            "v1 unknown motion needs one weak-form motion equation and one T2 receptor equation; "
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
    force_form = compile_expression(parse(force_eqs[0].form), CompileContext(mesh, motion_symbols))
    force_balance = LinearProblem(
        ufl.lhs(force_form),
        ufl.rhs(force_form),
        u=velocity,
        bcs=[],
        petsc_options_prefix=f"vcellfenics_force_{id(velocity):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # ---- the receptor T2 solve, diluting with the solved velocity ------------
    receptor = _build_receptor(receptor_eqs[0], md, mesh, dx, velocity, dt)
    return UnknownMotionProblem(motion_var, velocity, receptor_eqs[0].variable, receptor, force_balance)


def _build_receptor(
    eq: TemplateEquation, md: MathDescription, mesh: Mesh, dx: ufl.Measure, velocity: fem.Function, dt: float
) -> DiscreteProblem:
    """A T2 surface PDE whose dilution and per-step mesh motion read the solved
    velocity `Function` (so they update as the force balance re-solves each step)."""

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
        dt=fem.Constant(mesh, PETSc.ScalarType(float(dt))),  # type: ignore[operator]
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=velocity,
    )
    if eq.initial_condition is not None:
        problem.interpolate_initial(compile_expression(parse(eq.initial_condition), ctx))
    return problem


def _const_params(md: MathDescription, mesh: Mesh) -> dict[str, UflExpr]:
    out: dict[str, UflExpr] = {}
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("the unknown-motion path supports constant parameters only in v1")
        out[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    return out

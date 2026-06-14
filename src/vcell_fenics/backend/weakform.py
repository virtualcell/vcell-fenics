"""Assemble a weak-form escape-hatch equation (§1.5) into a runnable solve.

The operator templates (T1/T2) cover the common cell-biology PDE forms; the
**weak-form escape hatch** is the route for everything else — custom constitutive
laws, mechanics force balances, higher-order operators. The user writes the residual
directly as a UFL-style `form` (the equation is `form = 0` for all admissible test
functions), giving up the template guardrails for full expressiveness (§1.5.1).

This is the route for **membrane mechanics** in v1, since the mechanics templates
(T5–T7) are v2. A surface force balance such as the §1.10.8 viscous membrane is
written as a weak form over the membrane.

What this assembler handles: one weak-form equation governing one scalar or vector
variable on a single subdomain; the form built from the variable, its implicit test
function `<variable>_test`, the calculus / tensor-algebra operators, parameters, `x`,
and the subdomain measure (`dx` / `dx_Gamma`); `partial_t(u)` for a time-dependent
form (lowered to the backward-Euler difference); steady-state forms (solved directly).
The form is split with `ufl.lhs`/`rhs` into a linear system and solved per step.

Deferred (raise / unsupported here): the geometric helpers `n(x)` / `H(x)` (discrete
curvature / normal need a projected formulation), labelled-boundary measures
`ds(·)` / `dS(·)`, BCs on a weak-form variable (the closed-membrane models need none),
coupled multi-equation weak forms, and unknown-motion mesh coupling (the §1.10.8
motion-drives-the-mesh case). These build on this assembler.
"""

from __future__ import annotations

from dataclasses import dataclass

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    MathDescription,
    ParameterConstant,
    Variable,
    WeakFormEquation,
)
from vcell_fenics.formalism.validator import FormalismValidationError, validate_or_raise


@dataclass
class WeakFormProblem:
    """A weak-form solve. `solution` holds the current state; `step()` solves the
    linear system (and, for a time-dependent form, advances the previous step)."""

    variable: str
    V: fem.FunctionSpace
    solution: fem.Function
    previous: fem.Function | None
    _problem: LinearProblem

    def step(self) -> None:
        self._problem.solve()
        if self.previous is not None:
            self.previous.x.array[:] = self.solution.x.array


def assemble_weak_form(md: MathDescription, geometry: Geometry, *, dt: float, fe_degree: int = 1) -> WeakFormProblem:
    """Assemble the MathDescription's single weak-form equation against `geometry`.

    The form's residual is compiled to UFL and split into a backward-Euler linear
    system. v1 handles one `weak_form` equation; a BC on its variable, or extra
    coupled equations, raise `NotImplementedError`.
    """

    validate_or_raise(md)

    weak_eqs = [eq for eq in md.equations if isinstance(eq, WeakFormEquation)]
    if len(weak_eqs) != 1 or len(md.equations) != 1:
        raise NotImplementedError(
            "the weak-form path handles exactly one weak_form equation in v1 "
            "(coupled multi-equation / mixed template+weak-form models are a later increment)"
        )
    eq = weak_eqs[0]
    if any(bc.variable == eq.variable for bc in md.boundary_conditions):
        raise NotImplementedError(
            "boundary conditions on a weak-form variable are not wired yet; encode natural BCs in the form, "
            "and Dirichlet-on-weak-form is a later increment (§1.5.6)"
        )

    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    variable = _variable(md, eq)
    mesh = geometry.mesh_of(eq.subdomain)
    gdim = mesh.geometry.dim
    element = ("Lagrange", fe_degree) if variable.type == "scalar" else ("Lagrange", fe_degree, (gdim,))
    V = fem.functionspace(mesh, element)
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    solution = fem.Function(V, name=eq.variable)
    time_dependent = eq.temporality == "time_dependent"
    previous = fem.Function(V, name=f"{eq.variable}_prev") if time_dependent else None

    dx = ufl.Measure("dx", domain=mesh)
    symbols: dict[str, UflExpr] = {
        "x": ufl.SpatialCoordinate(mesh),
        eq.variable: trial,
        f"{eq.variable}_test": test,
        "dx": dx,
        "dx_Gamma": dx,  # the surface measure on a `surface` subdomain is its dx
    }
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("the weak-form path supports constant parameters only in v1")
        symbols[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    ctx = CompileContext(mesh, symbols)
    if time_dependent:
        assert previous is not None
        ctx.time_derivatives[eq.variable] = (trial - previous) / dt  # backward Euler

    form = compile_expression(parse(eq.form), ctx)
    bilinear = ufl.lhs(form)
    linear = ufl.rhs(form)
    if not (isinstance(linear, ufl.Form) and linear.integrals()):
        linear = ufl.inner(fem.Function(V), test) * dx  # a sourceless form has an empty (zero) RHS
    problem = LinearProblem(
        bilinear,
        linear,
        u=solution,
        bcs=[],
        petsc_options_prefix=f"vcellfenics_weak_{id(eq):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    if time_dependent and eq.initial_condition is not None:
        ic = compile_expression(parse(eq.initial_condition), ctx)
        solution.interpolate(fem.Expression(ic, V.element.interpolation_points))
        assert previous is not None
        previous.x.array[:] = solution.x.array

    return WeakFormProblem(eq.variable, V, solution, previous, problem)


def _variable(md: MathDescription, eq: WeakFormEquation) -> Variable:
    for v in md.variables:
        if isinstance(v, Variable) and v.name == eq.variable and v.subdomain == eq.subdomain:
            return v
    raise NotImplementedError(f"weak-form equation governs {eq.variable!r} which is not a declared variable")

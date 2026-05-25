"""Assemble a DiscreteProblem from a MathDescription + Geometry (T1, inc-0 subset).

This is the front of the backend: validate the model (formalism §2.5 plus the
geometry cross-check §1.11.10), then translate the single bulk reaction-diffusion
equation into a `DiscreteProblem` (ADR 004). The increment-0 subset is narrow and
guarded explicitly — anything outside it raises `NotImplementedError` with a clear
message rather than silently mis-assembling (§3.6.2):

- exactly one equation, template `bulk_radv_diff`, `temporality: time_dependent`;
- the `diffusion` slot only (advection / source / motion / BCs are later increments);
- constant parameters only.

The initial condition is applied here (§3.2.2). Increment-0 expressions are
spatially constant (the compiler's subset has no `x` or helpers yet), so the IC is
evaluated to its constant value; spatial-IC interpolation arrives with increment 1.
"""

from __future__ import annotations

from typing import Any

import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind
from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import MathDescription, ParameterConstant, TemplateEquation
from vcell_fenics.formalism.validator import FormalismValidationError, validate_or_raise


def assemble(md: MathDescription, geometry: Geometry, *, dt: float, fe_degree: int = 1) -> DiscreteProblem:
    """Translate `md` (against `geometry`) into a lowered `DiscreteProblem`."""

    validate_or_raise(md)
    geometry_errors = cross_validate(md, geometry)
    if geometry_errors:
        raise FormalismValidationError(geometry_errors)

    eq = _single_bulk_equation(md)
    mesh = geometry.mesh_of(eq.subdomain)
    V = fem.functionspace(mesh, ("Lagrange", fe_degree))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ctx = _compile_context(md, mesh)

    diffusion = compile_expression(parse(eq.terms["diffusion"]), ctx)
    terms = (
        Term(TermKind.TIME_DERIVATIVE),
        Term(TermKind.DIFFUSION, diffusion * ufl.dot(ufl.grad(trial), ufl.grad(test))),
    )
    problem = DiscreteProblem(
        variable_name=eq.variable,
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name=eq.variable),
        previous=fem.Function(V, name=f"{eq.variable}_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(float(dt))),  # type: ignore[operator]
        terms=terms,
        scheme=BackwardEuler(),
        bcs=[],
    )

    if eq.initial_condition is not None:
        ic = compile_expression(parse(eq.initial_condition), ctx)
        problem.set_initial(_constant_value(ic, mesh, dx))
    return problem


def _single_bulk_equation(md: MathDescription) -> TemplateEquation:
    if len(md.equations) != 1:
        raise NotImplementedError(
            "backend v1 (inc 0) supports a single equation; coupled systems are a later increment"
        )
    eq = md.equations[0]
    if not isinstance(eq, TemplateEquation) or eq.template != "bulk_radv_diff":
        template = getattr(eq, "template", None)
        raise NotImplementedError(f"backend v1 (inc 0) supports only the 'bulk_radv_diff' template, not {template!r}")
    if eq.temporality != "time_dependent":
        raise NotImplementedError("backend v1 (inc 0) supports 'time_dependent' equations only")
    unsupported = sorted(set(eq.terms) - {"diffusion"})
    if unsupported:
        raise NotImplementedError(f"backend v1 (inc 0) supports only the 'diffusion' slot; got {unsupported}")
    if "diffusion" not in eq.terms:
        raise NotImplementedError("backend v1 (inc 0) requires a 'diffusion' slot")
    return eq


def _compile_context(md: MathDescription, mesh: Any) -> CompileContext:
    symbols: dict[str, Any] = {}
    for p in md.parameters:
        if not isinstance(p, ParameterConstant):
            raise NotImplementedError("backend v1 (inc 0) supports constant parameters only")
        symbols[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
    return CompileContext(mesh=mesh, symbols=symbols)


def _constant_value(ufl_expr: Any, mesh: Any, dx: Any) -> float:
    """The constant value of a spatially-uniform expression, as its
    domain average. Valid because the inc-0 compiler subset is x-free."""

    one = fem.Constant(mesh, PETSc.ScalarType(1.0))  # type: ignore[operator]
    area = mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * dx)), op=MPI.SUM)
    integral = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl_expr * dx)), op=MPI.SUM)
    return float(integral / area)

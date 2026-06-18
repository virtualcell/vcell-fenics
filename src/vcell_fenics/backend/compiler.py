"""Compile a formalism expression AST into a UFL expression.

This is the `AST → UFL` stage of the pipeline (ADR 004). It walks the syntactic
AST from `formalism.expr` and emits UFL, resolving each `Name` through a
`CompileContext` whose symbol table maps formalism names to concrete UFL objects
(parameters → `fem.Constant`, the namespaced built-ins `geom.x` → `SpatialCoordinate` and
`sim.t` → time Constant, variables → the solution `Function`, and calculus operators).

UFL is itself the expression IR; there is no second representation here. The
compiler assumes the expression has already passed validation (`formalism.validator`),
so an unresolved name or unsupported construct is an internal error, raised as
`CompileError` rather than returned as a diagnostic.

Supported so far: numeric literals, name lookups (parameters, the namespaced built-ins
`geom.x` → SpatialCoordinate and `sim.t` → time Constant, and — in a weak form — variables,
test functions, and measures the assembler binds), arithmetic (`+ - * / **`, unary `±`),
coordinate indexing (`geom.x[0]`), the standard scalar math functions, the geometry quantities
`geom.radius`/`geom.azimuth` (sugar for functions of position) and `geom.normal`/
`geom.mean_curvature` (bound by a mechanics solve), `trace(·)` (§1.8.2), **vector literals**
`[a, b]`, **tensor-algebra** (`inner`/`dot`/`outer`/`cross`), **first-order calculus**
(`grad`/`div`/`lapl` and the `_surf`/`_beltrami` variants, which on a codim-1 submesh are the
same UFL calls), and **`partial_t(u)`** (resolved to the time-discretised derivative the
assembler bound — the weak-form escape hatch). The `geom.tangent`/`geom.curvature1`/
`geom.curvature2` quantities, multi-argument standard functions, and tensor literals raise
`CompileError` until the increments that add them.

`trace(u)` compiles to `u`'s UFL object unchanged: the cross-dimensional *restriction*
of a bulk variable onto a lower-dimensional evaluation domain is realised by native
mixed-dimensional assembly (entity maps relating the submeshes; §1.8.2 "trace
evaluation is a backend concern"), not by a UFL wrapper. The compiler's only job is
to resolve the name; the assembler that builds a cross-subdomain form provides the
bulk variable's `Function` in the symbol table and the entity map at `fem.form` time.
The validator has already enforced that the argument is a single declared variable on
a strictly higher-dimensional subdomain (the direction rule), so the compiler trusts that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
from petsc4py import PETSc

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Name, Number, UnaryOp, VectorLiteral

# Formalism standard functions that map to a single-argument UFL function
# (§1.8.5). Multi-argument functions (atan2, min, max, pow) and the
# conditionals (if, step, sign) are added when a model needs them.
_UFL_UNARY_FUNCTIONS: dict[str, Any] = {
    "sin": ufl.sin,
    "cos": ufl.cos,
    "tan": ufl.tan,
    "asin": ufl.asin,
    "acos": ufl.acos,
    "atan": ufl.atan,
    "sinh": ufl.sinh,
    "cosh": ufl.cosh,
    "tanh": ufl.tanh,
    "exp": ufl.exp,
    "log": ufl.ln,  # the formalism `log` is the natural log; UFL spells it `ln`
    "log10": lambda a: ufl.ln(a) / ufl.ln(10.0),
    "sqrt": ufl.sqrt,
    "abs": abs,
}
# `floor` / `ceil` are accepted by the validator (they appear in real VCell models) but are
# non-differentiable, so UFL/DOLFINx has no operator for them — compiling one raises below.

# Two-argument tensor-algebra operators (§1.8, TENSOR_ALGEBRA).
_UFL_BINARY: dict[str, Any] = {"inner": ufl.inner, "dot": ufl.dot, "outer": ufl.outer, "cross": ufl.cross}

# First-order calculus operators (§1.8). On a codim-1 submesh `ufl.grad` / `ufl.div`
# are already the tangential (surface) operators, so the `_surf` variants are the
# same UFL calls — the distinction is the mesh, not the operator.
_UFL_CALCULUS: dict[str, Any] = {"grad": ufl.grad, "div": ufl.div, "grad_surf": ufl.grad, "div_surf": ufl.div}


class CompileError(Exception):
    """A MathDescription expression could not be compiled to UFL. Indicates an
    unresolved name or a construct outside the current backend subset — both of
    which a validated MathDescription within the supported subset should avoid."""


@dataclass
class CompileContext:
    """The symbol environment for compilation. `mesh` is needed to build
    `fem.Constant`s; `symbols` maps formalism names to UFL objects (parameters as
    Constants, variables/test functions/measures in a weak-form context).
    `time_derivatives` maps a variable name to the UFL object `partial_t(<var>)`
    compiles to — the backend's time-discretised derivative (e.g. `(uⁿ⁺¹ − uⁿ)/dt`
    for backward Euler) — populated only for a time-dependent weak form."""

    mesh: Mesh
    symbols: dict[str, UflExpr] = field(default_factory=dict)
    time_derivatives: dict[str, UflExpr] = field(default_factory=dict)


def compile_expression(node: Expr, ctx: CompileContext) -> UflExpr:
    """Compile one expression AST node to a UFL expression."""

    if isinstance(node, Number):
        # petsc4py stubs type PETSc.ScalarType as a non-callable dtype; it is a
        # callable scalar alias at runtime.
        return fem.Constant(ctx.mesh, PETSc.ScalarType(node.value))  # type: ignore[operator]
    if isinstance(node, Name):
        if "." in node.name:  # a namespaced built-in: geom.* / sim.* (ADR 006)
            return _resolve_qualified(node.name, ctx)
        try:
            return ctx.symbols[node.name]
        except KeyError:
            raise CompileError(f"unresolved name {node.name!r} (not in the compile context)") from None
    if isinstance(node, UnaryOp):
        operand = compile_expression(node.operand, ctx)
        if node.op == "+":
            return operand
        if node.op == "!":
            return ufl.Not(operand)
        return -operand
    if isinstance(node, BinaryOp):
        left = compile_expression(node.left, ctx)
        right = compile_expression(node.right, ctx)
        match node.op:
            case "+":
                return left + right
            case "-":
                return left - right
            case "*":
                return left * right
            case "/":
                return left / right
            case "**":
                return left**right
            # Relational / logical operators produce a UFL `Condition`, for use inside `if(...)`.
            case "<":
                return ufl.lt(left, right)
            case ">":
                return ufl.gt(left, right)
            case "<=":
                return ufl.le(left, right)
            case ">=":
                return ufl.ge(left, right)
            case "==":
                return ufl.eq(left, right)
            case "!=":
                return ufl.ne(left, right)
            case "&&":
                return ufl.And(left, right)
            case "||":
                return ufl.Or(left, right)
    if isinstance(node, IndexAccess):
        base = compile_expression(node.base, ctx)
        if not isinstance(node.index, Number) or not node.index.value.is_integer():
            raise CompileError("index must be an integer literal, e.g. x[0]")
        return base[int(node.index.value)]
    if isinstance(node, VectorLiteral):
        return ufl.as_vector([compile_expression(c, ctx) for c in node.components])
    if isinstance(node, FunctionCall):
        return _compile_call(node, ctx)
    raise CompileError(f"{type(node).__name__} is not supported by the backend yet")


def _resolve_qualified(name: str, ctx: CompileContext) -> UflExpr:
    """Resolve a namespaced built-in quantity `geom.*` / `sim.*` to its UFL object (ADR 006).

    `geom.x` (position) and `sim.t` (time) are bound directly in the symbol table by the
    assembler; the curvilinear `geom.radius` / `geom.azimuth` are sugar for functions of the
    position field; `geom.normal` / `geom.mean_curvature` resolve to the projected surface normal
    and mean curvature a mechanics assembler binds (on a discrete membrane these are not
    pointwise-defined — they come from a weak projection κ = H·n — so they are unavailable
    outside a force-balance solve)."""

    root, _, member = name.partition(".")
    if root == "geom":
        if member == "x":
            return ctx.symbols["geom.x"]
        if member == "radius":
            x = ctx.symbols["geom.x"]
            return ufl.sqrt(ufl.dot(x, x))
        if member == "azimuth":
            x = ctx.symbols["geom.x"]
            return ufl.atan2(x[1], x[0])
        if member in ("normal", "mean_curvature"):
            bound = ctx.symbols.get("__normal__" if member == "normal" else "__mean_curvature__")
            if bound is None:
                quantity = "normal" if member == "normal" else "mean curvature"
                raise CompileError(
                    f"geom.{member} (surface {quantity}) is only available where the curvature projection is "
                    f"bound — i.e. in a mechanics (force-balance) solve"
                )
            return bound
        raise CompileError(f"geom.{member} is not supported by the backend yet")
    if root == "sim":
        if member == "t":
            return ctx.symbols["sim.t"]
        raise CompileError(f"sim.{member} is not supported by the backend yet")
    raise CompileError(f"unresolved name {name!r} (not in the compile context)")


def _compile_call(node: FunctionCall, ctx: CompileContext) -> UflExpr:
    # `partial_t(<var>)` resolves to the backend's time-discretised derivative, so it
    # is looked up by *name* (not compiled as an expression) — only valid in a
    # time-dependent weak form where the assembler bound it (§1.5.4).
    if node.callee == "partial_t":
        if len(node.args) != 1 or not isinstance(node.args[0], Name):
            raise CompileError("partial_t(...) takes a single variable name")
        derivative = ctx.time_derivatives.get(node.args[0].name)
        if derivative is None:
            raise CompileError(
                f"partial_t({node.args[0].name}) has no time derivative bound — it is only valid for a variable "
                f"governed by a time-dependent weak-form equation"
            )
        return derivative

    args = [compile_expression(arg, ctx) for arg in node.args]
    if node.callee == "if":
        # `if(condition, then, else)` → a UFL conditional. The condition is a relational/
        # logical expression (a UFL `Condition`); the branches are the values.
        if len(args) != 3:
            raise CompileError("if(condition, then, else) takes exactly three arguments")
        return ufl.conditional(args[0], args[1], args[2])
    if node.callee == "trace":
        # The trace of a higher-dimensional variable onto a lower-dimensional
        # evaluation domain (§1.8.2). At the UFL level this is the variable itself;
        # the actual restriction is mixed-dimensional assembly's job (entity maps),
        # so the compiler just returns the resolved argument. The validator has
        # guaranteed a single variable-name argument crossing high → low dimension.
        if len(args) != 1:
            raise CompileError("trace(...) takes exactly one argument")
        return args[0]
    binary = _UFL_BINARY.get(node.callee)
    if binary is not None:
        if len(args) != 2:
            raise CompileError(f"{node.callee}(...) takes two arguments")
        return binary(args[0], args[1])
    calculus = _UFL_CALCULUS.get(node.callee)
    if calculus is not None:
        return calculus(args[0])
    if node.callee in ("lapl", "lapl_beltrami"):
        return ufl.div(ufl.grad(args[0]))  # ∇·∇ — Beltrami on a submesh
    fn = _UFL_UNARY_FUNCTIONS.get(node.callee)
    if fn is not None:
        return fn(*args)
    raise CompileError(f"function {node.callee!r} is not supported by the backend yet")

"""Compile a formalism expression AST into a UFL expression.

This is the `AST → UFL` stage of the pipeline (ADR 004). It walks the syntactic
AST from `formalism.expr` and emits UFL, resolving each `Name` through a
`CompileContext` whose symbol table maps formalism names to concrete UFL objects
(parameters → `fem.Constant`, and — in later increments — variables → the
solution `Function`, `x` → `SpatialCoordinate`, helpers and calculus operators).

UFL is itself the expression IR; there is no second representation here. The
compiler assumes the expression has already passed validation (`formalism.validator`),
so an unresolved name or unsupported construct is an internal error, raised as
`CompileError` rather than returned as a diagnostic.

Supported so far: numeric literals, name lookups (parameters, and `x` when the
assembler binds it to a SpatialCoordinate), arithmetic (`+ - * / **`, unary `±`),
coordinate indexing (`x[0]`), the standard scalar math functions, the geometric
helpers that expand to functions of `x` (`theta`, `r`), and `trace(·)` (the
cross-dimensional reference to a higher-dimensional variable; §1.8.2). Calculus
operators, tensor algebra, multi-argument functions, the `n`/`H`/tangent/curvature
helpers, and vector/tensor literals raise `CompileError` until the increments that
add them.

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
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Name, Number, UnaryOp

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
    "exp": ufl.exp,
    "log": ufl.ln,  # the formalism `log` is the natural log; UFL spells it `ln`
    "sqrt": ufl.sqrt,
    "abs": abs,
}


class CompileError(Exception):
    """A MathDescription expression could not be compiled to UFL. Indicates an
    unresolved name or a construct outside the current backend subset — both of
    which a validated MathDescription within the supported subset should avoid."""


@dataclass
class CompileContext:
    """The symbol environment for compilation. `mesh` is needed to build
    `fem.Constant`s; `symbols` maps formalism names to UFL objects (parameters as
    Constants in increment 0)."""

    mesh: Mesh
    symbols: dict[str, UflExpr] = field(default_factory=dict)


def compile_expression(node: Expr, ctx: CompileContext) -> UflExpr:
    """Compile one expression AST node to a UFL expression."""

    if isinstance(node, Number):
        # petsc4py stubs type PETSc.ScalarType as a non-callable dtype; it is a
        # callable scalar alias at runtime.
        return fem.Constant(ctx.mesh, PETSc.ScalarType(node.value))  # type: ignore[operator]
    if isinstance(node, Name):
        try:
            return ctx.symbols[node.name]
        except KeyError:
            raise CompileError(f"unresolved name {node.name!r} (not in the compile context)") from None
    if isinstance(node, UnaryOp):
        operand = compile_expression(node.operand, ctx)
        return operand if node.op == "+" else -operand
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
    if isinstance(node, IndexAccess):
        base = compile_expression(node.base, ctx)
        if not isinstance(node.index, Number) or not node.index.value.is_integer():
            raise CompileError("index must be an integer literal, e.g. x[0]")
        return base[int(node.index.value)]
    if isinstance(node, FunctionCall):
        return _compile_call(node, ctx)
    raise CompileError(f"{type(node).__name__} is not supported by the backend yet")


def _compile_call(node: FunctionCall, ctx: CompileContext) -> UflExpr:
    args = [compile_expression(arg, ctx) for arg in node.args]
    # Geometric helpers that are sugar for functions of x (§1.8.4).
    if node.callee == "theta":
        x = args[0]
        return ufl.atan2(x[1], x[0])
    if node.callee == "r":
        x = args[0]
        return ufl.sqrt(ufl.dot(x, x))
    if node.callee == "trace":
        # The trace of a higher-dimensional variable onto a lower-dimensional
        # evaluation domain (§1.8.2). At the UFL level this is the variable itself;
        # the actual restriction is mixed-dimensional assembly's job (entity maps),
        # so the compiler just returns the resolved argument. The validator has
        # guaranteed a single variable-name argument crossing high → low dimension.
        if len(args) != 1:
            raise CompileError("trace(...) takes exactly one argument")
        return args[0]
    fn = _UFL_UNARY_FUNCTIONS.get(node.callee)
    if fn is not None:
        return fn(*args)
    raise CompileError(f"function {node.callee!r} is not supported by the backend yet")

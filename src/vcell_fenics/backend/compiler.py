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

Increment-0 scope: numeric literals, name lookups (parameters), and arithmetic
(`+ - * / **`, unary `±`). Coordinates, geometric helpers, calculus operators,
`trace`, tensor algebra, and vector/tensor literals raise `CompileError` until
the increments that add them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.formalism.expr import BinaryOp, Expr, Name, Number, UnaryOp


class CompileError(Exception):
    """A MathDescription expression could not be compiled to UFL. Indicates an
    unresolved name or a construct outside the current backend subset — both of
    which a validated MathDescription within the supported subset should avoid."""


@dataclass
class CompileContext:
    """The symbol environment for compilation. `mesh` is needed to build
    `fem.Constant`s; `symbols` maps formalism names to UFL objects (parameters as
    Constants in increment 0)."""

    mesh: Any
    symbols: dict[str, Any] = field(default_factory=dict)


def compile_expression(node: Expr, ctx: CompileContext) -> Any:
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
    raise CompileError(f"{type(node).__name__} is not supported by the backend yet")

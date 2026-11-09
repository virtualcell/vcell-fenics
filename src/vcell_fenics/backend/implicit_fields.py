"""Numeric evaluation of lowered implicit functions, and the realization error type.

Shared by the realization layer (`realize.py`) and the image label field (`labels.py`), which
rasterizes analytic subvolumes over an image's pixel lattice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp


class RealizationError(ValueError):
    """A :class:`GeometryDescription` is structurally well-formed but cannot be realized into a
    backend :class:`Geometry` (e.g. a non-spatial geometry carrying a spatial subvolume type, or a
    spatial topology v1 does not yet mesh)."""


# a relational / logical node in a numeric position: 0/1 (VCell's arithmetic reading of a boolean)
def _indicator(op: str, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    if op == "<":
        mask = a < b
    elif op == "<=":
        mask = a <= b
    elif op == ">":
        mask = a > b
    elif op == ">=":
        mask = a >= b
    elif op == "&&":
        mask = (a != 0.0) & (b != 0.0)
    else:  # "||"
        mask = (a != 0.0) | (b != 0.0)
    return mask.astype(np.float64)


_INDICATOR = frozenset({"<", "<=", ">", ">=", "&&", "||"})


def eval_field(expr: Expr, coords: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
    """Vectorised numeric evaluation of a (lowered, geom.x-only) implicit-function expression over a
    coordinate grid. `coords` is one array per spatial axis (2 in 2D, 3 in 3D); ``geom.x[i]`` reads
    ``coords[i]``. Handles the arithmetic / min / max / elementary-function subset an implicit function
    contains, plus a relational / logical node inside a comparison's operand, which is a 0/1
    indicator (``formalism.rvachev``); an equality or an unbound node raises."""

    if isinstance(expr, Number):
        return np.full_like(coords[0], expr.value, dtype=np.float64)
    if isinstance(expr, IndexAccess):
        return coords[int(_constant(expr.index))]
    if isinstance(expr, UnaryOp):
        value = eval_field(expr.operand, coords)
        if expr.op == "!":
            negated: NDArray[np.float64] = (value == 0.0).astype(np.float64)
            return negated
        return -value if expr.op == "-" else value
    if isinstance(expr, BinaryOp):
        left, right = eval_field(expr.left, coords), eval_field(expr.right, coords)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "**":
            return left**right
        if expr.op in _INDICATOR:
            return _indicator(expr.op, left, right)
        raise RealizationError(f"operator {expr.op!r} is not valid in an implicit function")
    if isinstance(expr, FunctionCall):
        args = [eval_field(a, coords) for a in expr.args]
        return _call(expr.callee, args)
    raise RealizationError(f"cannot evaluate {type(expr).__name__} in an implicit function (unbound name?)")


def _call(callee: str, args: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    if callee == "min":
        return np.minimum(args[0], args[1])
    if callee == "max":
        return np.maximum(args[0], args[1])
    unary = {
        "sqrt": np.sqrt,
        "abs": np.abs,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
    }
    if callee in unary:
        return unary[callee](args[0])
    if callee == "pow":
        return np.power(args[0], args[1])
    if callee == "atan2":
        return np.arctan2(args[0], args[1])
    raise RealizationError(f"function {callee!r} is not supported in an implicit function")


def _constant(expr: Expr) -> float:
    if isinstance(expr, Number):
        return expr.value
    raise RealizationError("expected a constant index in geom.x[…]")

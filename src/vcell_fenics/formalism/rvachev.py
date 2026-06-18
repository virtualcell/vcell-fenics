"""Lower an analytic-geometry boolean predicate to a Rvachev implicit function (AST → AST).

A VCell ``analytic`` subvolume is a **boolean predicate** over ``geom.x`` — a combination of
spatial inequalities, e.g. ``geom.x[0]**2 + geom.x[1]**2 < 1``. This module lowers such a
predicate into a single real-valued **implicit function** ``φ`` whose *sign* encodes membership:

    φ(x) < 0  inside the region    φ(x) = 0  on the boundary    φ(x) > 0  outside

— the **inside-negative** convention (``geometric-formalism.md`` §1). The output is the natural
pre-mesh intermediate: reinitialise it to a signed distance, optionally smooth (without
shrinkage), then mesh body-fitted (marching) or carry it as a level-set (cut/trace FEM). It is
the same field VCell hands its embedded-boundary fvsolver.

This is a faithful port of VCell's translation (``cbit.vcell.parser.RvachevFunctionUtils`` and
``FiniteVolumeFileWriter.convertAnalyticGeometryToRvachevFunction``). R-functions here are plain
``min`` / ``max`` (Rvachev's foundational system — exact in sign, C⁰ at coincident-zero seams),
not the smooth √-variant:

==========================  ==========================================
predicate node              implicit function
==========================  ==========================================
``a < b`` / ``a <= b``      ``a - b``
``a > b`` / ``a >= b``      ``b - a``
``a && b`` (AND, ∩)         ``max(a, b)``
``a || b`` (OR, ∪)          ``min(a, b)``
``!a`` (NOT, complement)    ``-a``
``a == b`` / ``a != b``     rejected — a measure-zero set has no implicit-function form
==========================  ==========================================

As in VCell, an **all-boolean** ``*`` is read as AND and ``+`` as OR (the boolean-as-arithmetic
idiom common in imported models); a ``*`` / ``+`` mixing boolean and numeric operands is rejected.
"""

from __future__ import annotations

from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse

_RELATIONAL = frozenset({"<", "<=", ">", ">=", "==", "!="})
_LOGICAL = frozenset({"&&", "||"})


class RvachevLoweringError(ValueError):
    """A predicate cannot be lowered to a Rvachev implicit function (equality operator, a
    boolean/numeric-mixed product/sum, or a subvolume with no analytic expression)."""


def is_boolean(expr: Expr) -> bool:
    """Whether ``expr`` evaluates to a boolean, mirroring VCell's ``Node.isBoolean()``:
    relational and logical operators and ``!`` are boolean, and a ``*`` / ``+`` is boolean iff
    *all* its operands are (so ``(x<1)*(y<1)`` reads as ``(x<1) && (y<1)``). Everything else —
    numbers, names, arithmetic, function calls — is numeric."""

    if isinstance(expr, BinaryOp):
        if expr.op in _RELATIONAL or expr.op in _LOGICAL:
            return True
        if expr.op in ("*", "+"):
            return is_boolean(expr.left) and is_boolean(expr.right)
        return False
    if isinstance(expr, UnaryOp):
        return expr.op == "!"
    return False


def lower_predicate(predicate: Expr) -> Expr:
    """Lower a boolean predicate AST to its Rvachev implicit-function AST (inside-negative).
    Raises :class:`RvachevLoweringError` for equality operators or boolean/numeric-mixed
    ``*`` / ``+``."""

    return _lower(predicate)


def subvolume_implicit_functions(geometry: GeometryDescription) -> dict[str, Expr]:
    """Implicit function per subvolume, honouring VCell's priority semantics
    (``convertAnalyticGeometryToRvachevFunction``).

    Subvolumes are priority-ordered by their position in ``geometry.subvolumes`` (index 0 is
    highest priority); each region is its own predicate **minus** every higher-priority region,
    and the **last** subvolume is the background — the complement of the union of all the others.
    A single subvolume fills the whole domain (``φ = -1``). All subvolumes except the last must be
    ``analytic`` (carry an ``expression``); the last's expression, if any, is ignored (it is the
    complement)."""

    subvolumes = geometry.subvolumes
    if not subvolumes:
        raise RvachevLoweringError("geometry has no subvolumes to lower")

    names = [s.name for s in subvolumes]
    if len(subvolumes) == 1:
        # One subvolume owns the whole (non-spatial) domain: inside everywhere.
        return {names[0]: Number(-1.0)}

    # R[i] is subvolume i's own predicate as an implicit function, for i < n-1.
    own: list[Expr] = []
    for s in subvolumes[:-1]:
        if s.expression is None:
            raise RvachevLoweringError(f"subvolume {s.name!r} has no analytic expression to lower")
        own.append(lower_predicate(parse(s.expression)))

    result: dict[str, Expr] = {}
    # φ_i = max(R_i, -R_0, …, -R_{i-1}) — inside R_i and outside every higher-priority region.
    for i in range(len(subvolumes) - 1):
        phi = own[i]
        for j in range(i):
            phi = _max(phi, _negate(own[j]))
        result[names[i]] = phi
    # Background: complement of the union of all the others, max(-R_0, …, -R_{n-2}).
    background = _negate(own[0])
    for j in range(1, len(own)):
        background = _max(background, _negate(own[j]))
    result[names[-1]] = background
    return result


# -- the node-wise transform ----------------------------------------------------


def _lower(expr: Expr) -> Expr:
    if isinstance(expr, BinaryOp):
        op = expr.op
        if op in ("<", "<="):
            return BinaryOp("-", _lower(expr.left), _lower(expr.right))
        if op in (">", ">="):
            return BinaryOp("-", _lower(expr.right), _lower(expr.left))
        if op in ("==", "!="):
            raise RvachevLoweringError(f"{op!r} is not allowed in a Rvachev implicit function (measure-zero set)")
        if op == "&&":
            return _max(_lower(expr.left), _lower(expr.right))
        if op == "||":
            return _min(_lower(expr.left), _lower(expr.right))
        if op in ("*", "+"):
            return _lower_mult_add(expr)
        # Pure arithmetic (-, /, **): a base function — recurse (identity on numeric subtrees).
        return BinaryOp(op, _lower(expr.left), _lower(expr.right))
    if isinstance(expr, UnaryOp):
        if expr.op == "!":
            return _negate(_lower(expr.operand))
        return UnaryOp(expr.op, _lower(expr.operand))
    if isinstance(expr, FunctionCall):
        return FunctionCall(expr.callee, tuple(_lower(a) for a in expr.args))
    if isinstance(expr, IndexAccess):
        return IndexAccess(_lower(expr.base), _lower(expr.index))
    # Number / Name and the literal containers are base functions: returned unchanged.
    return expr


def _lower_mult_add(expr: BinaryOp) -> Expr:
    """A ``*`` / ``+`` is AND / OR iff both operands are boolean (VCell's ``fixMultAdd``);
    mixing boolean and numeric operands is an error; otherwise it is ordinary arithmetic."""

    left_bool, right_bool = is_boolean(expr.left), is_boolean(expr.right)
    if left_bool and right_bool:
        return (
            _max(_lower(expr.left), _lower(expr.right))
            if expr.op == "*"
            else _min(_lower(expr.left), _lower(expr.right))
        )
    if left_bool != right_bool:
        raise RvachevLoweringError(
            f"cannot lower a predicate that mixes boolean and numeric operands under {expr.op!r}"
        )
    return BinaryOp(expr.op, _lower(expr.left), _lower(expr.right))


def _max(left: Expr, right: Expr) -> Expr:
    return FunctionCall("max", (left, right))


def _min(left: Expr, right: Expr) -> Expr:
    return FunctionCall("min", (left, right))


def _negate(expr: Expr) -> Expr:
    return UnaryOp("-", expr)

"""Typed AST for formalism expression strings.

These node types are the canonical internal form of every right-hand-side
expression in a MathDescription — term-slot fillers (§1.4), BC expressions
(§1.6), initial conditions (§1.7), motion-velocity expressions (§1.10), and
the weak-form `form:` residuals (§1.5). The surface syntax and the node-kind
catalogue are specified in docs/modeling/declarative-formalism.md §2.3.

The parser (`vcell_fenics.formalism.parser`) produces these nodes from a
string. The AST it produces is *syntactic only*: bare identifiers all become
`Name`, and `f(...)` always becomes `FunctionCall`. The three resolved
reference kinds the doc's §2.3.3 table names — variable reference, parameter
reference, reserved (`t`, `x`) reference — and per-node types are assigned by
the validator's resolution pass, which has the MathDescription context the
parser lacks. A bare `Name` is therefore an *unresolved* reference; the
validator decides whether it is a local variable, a parameter, a reserved
name, or (for `dx`, `ds`, …) a measure.

Nodes are frozen and slotted; structural equality (the dataclass default) is
intentional so parser output can be compared directly in tests. Nodes carry
no source position — parse errors carry the position instead
(`ExpressionSyntaxError`); positioned validator-time diagnostics are a later
concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

# Operator discriminators. Arithmetic, plus relational / logical operators (used by
# conditionals like `if(a > b, ...)` — common in imported VCell kinetics). Tensor algebra
# (`inner`, `outer`, `cross`) and calculus (`grad`, `div`, …) are spelled as `FunctionCall`s,
# not operators (§2.3.4).
BinOp: TypeAlias = Literal["+", "-", "*", "/", "**", "<", ">", "<=", ">=", "==", "!=", "&&", "||"]
UnOp: TypeAlias = Literal["+", "-", "!"]


@dataclass(frozen=True, slots=True)
class Number:
    """A numeric literal. Always stored as `float`; integer-valued literals
    (e.g. the `0` in `x[0]`) keep `value.is_integer()` True so the validator
    can require integrality where the grammar allows any expression."""

    value: float


@dataclass(frozen=True, slots=True)
class Name:
    """An unresolved bare identifier — a local variable, a parameter, a
    reserved name (`t`, `x`), or a measure (`dx`, `dx_Gamma`, …). The
    validator resolves which (§1.8.6, §2.4.2)."""

    name: str


@dataclass(frozen=True, slots=True)
class IndexAccess:
    """Component access, `base[index]`. v1's only legal use is on the spatial
    coordinate (`x[0]`, `x[1]`, `x[2]`); the parser accepts any base and any
    index expression and leaves that restriction to the validator."""

    base: Expr
    index: Expr


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """A call `callee(arg, ...)`. Covers every callable in the vocabulary
    (§2.3.4): standard functions (`sin`, `if`, …), calculus operators
    (`grad`, `lapl_beltrami`, …), `trace`, tensor algebra (`inner`, `outer`,
    `cross`), `partial_t`, and the parametrised measures (`ds(<boundary>)`, …).
    Geometric quantities are namespaced *values* (`geom.normal`, `geom.azimuth`;
    ADR 006), not calls, so they parse as `Name`, not `FunctionCall`. The callee
    is kept as a raw name; the validator dispatches on it."""

    callee: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: UnOp
    operand: Expr


@dataclass(frozen=True, slots=True)
class BinaryOp:
    op: BinOp
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class VectorLiteral:
    """A bracketed list of scalar-yielding components, `[a, b]` / `[a, b, c]`.
    Shape (component count vs. embedding dimension) is a validator check."""

    components: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class TensorLiteral:
    """A bracketed list of bracketed rows, `[[a, b], [c, d]]`. The parser emits
    this (rather than a `VectorLiteral` of `VectorLiteral`s) when every element
    of a bracket is itself a bracket; rectangularity is a validator check."""

    rows: tuple[VectorLiteral, ...]


Expr: TypeAlias = Number | Name | IndexAccess | FunctionCall | UnaryOp | BinaryOp | VectorLiteral | TensorLiteral

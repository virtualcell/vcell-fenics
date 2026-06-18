"""Hand-written recursive-descent parser for formalism expression strings.

Turns a math-like infix string (§2.3.1) into the typed AST of
`vcell_fenics.formalism.expr`. No third-party dependency: a small tokenizer
feeds a recursive-descent grammar with standard arithmetic precedence.

The parser is *syntactic only* — it resolves no names and checks no types
(those are the validator's job, §1.11 / §2.5). It rejects malformed strings
and reports the offending position via `ExpressionSyntaxError`.

Grammar (precedence low → high), with `**` right-associative and binding
tighter than unary minus on its left (so `-2**2` is `-(2**2)` and `2**-1` is
`2**(-1)`, matching Python):

    expression  := additive
    additive    := multiplicative ( ("+" | "-") multiplicative )*
    multiplicative := unary ( ("*" | "/") unary )*
    unary       := ("+" | "-") unary | power
    power       := postfix ( "**" unary )?
    postfix     := primary ( "[" expression "]" )*
    primary     := NUMBER
                 | NAME ( "(" arglist? ")" )?        # FunctionCall if "(" follows, else Name
                 | "(" expression ")"
                 | "[" listbody? "]"                 # VectorLiteral / TensorLiteral
    arglist     := expression ( "," expression )*
    listbody    := expression ( "," expression )*
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from vcell_fenics.formalism.expr import (
    BinaryOp,
    BinOp,
    Expr,
    FunctionCall,
    IndexAccess,
    Name,
    Number,
    TensorLiteral,
    UnaryOp,
    UnOp,
    VectorLiteral,
)


class ExpressionSyntaxError(Exception):
    """An expression string is not syntactically well-formed.

    `source` is the full offending string, `pos` is the 0-based character
    offset where the problem was detected, and `message` is the human-readable
    explanation. The str() presentation includes the source line and a caret
    pointing at `pos`, so the offending token is visible in tracebacks even for
    multi-line weak-form expressions.
    """

    def __init__(self, source: str, pos: int, message: str) -> None:
        self.source = source
        self.pos = pos
        self.message = message
        super().__init__(_render(source, pos, message))


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def parse(source: str) -> Expr:
    """Parse an expression string into a typed AST.

    Raises `ExpressionSyntaxError` (with a position) on any malformed input,
    including an empty string and trailing characters after a complete
    expression.
    """

    tokens = _tokenize(source)
    parser = _Parser(source, tokens)
    expr = parser.parse_expression()
    parser.expect_end()
    return expr


# ---------------------------------------------------------------------------
# Tokenizer.
# ---------------------------------------------------------------------------

# Order matters: "**" before "*". Numbers accept optional fraction and a
# signed scientific exponent; a leading sign is the parser's unary-minus job,
# not the lexer's, so it is not part of the number pattern.
# A name is an identifier, optionally a dotted *qualified* name (`geom.x`, `sim.t`) — the
# namespaced built-ins of ADR 006. Each dotted segment is a separate identifier, so `geom.x`
# is a single name token (and `geom.x[0]` is that token indexed). The number pattern is tried
# first for a leading digit, so a dot only joins identifiers, never digits.
_NUMBER_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
# Multi-character operators must precede their single-character prefixes (the tokenizer
# takes the first match): `**` before `*`, `<=`/`>=`/`==`/`!=` before `<`/`>`/`!`.
_OPERATORS = (
    "**", "<=", ">=", "==", "!=", "&&", "||",
    "+", "-", "*", "/", "<", ">", "!", "(", ")", "[", "]", ",",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # "number" | "name" | "op" | "end"
    value: str
    pos: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            m = _NUMBER_RE.match(source, i)
            if m is None:  # pragma: no cover - guarded by the isdigit/dot check above
                raise ExpressionSyntaxError(source, i, "malformed number")
            tokens.append(_Token("number", m.group(), i))
            i = m.end()
            continue
        if ch.isalpha() or ch == "_":
            m = _NAME_RE.match(source, i)
            assert m is not None  # guaranteed by the isalpha/underscore check
            tokens.append(_Token("name", m.group(), i))
            i = m.end()
            continue
        matched = next((op for op in _OPERATORS if source.startswith(op, i)), None)
        if matched is not None:
            tokens.append(_Token("op", matched, i))
            i += len(matched)
            continue
        raise ExpressionSyntaxError(source, i, f"unexpected character {ch!r}")
    tokens.append(_Token("end", "", n))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser.
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, source: str, tokens: list[_Token]) -> None:
        self._source = source
        self._tokens = tokens
        self._i = 0

    # -- token cursor helpers ------------------------------------------------

    @property
    def _cur(self) -> _Token:
        return self._tokens[self._i]

    def _advance(self) -> _Token:
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def _at_op(self, *ops: str) -> bool:
        cur = self._cur
        return cur.kind == "op" and cur.value in ops

    def _error(self, message: str) -> ExpressionSyntaxError:
        return ExpressionSyntaxError(self._source, self._cur.pos, message)

    def _expect_op(self, op: str) -> None:
        if not self._at_op(op):
            raise self._error(f"expected {op!r}")
        self._advance()

    def expect_end(self) -> None:
        if self._cur.kind != "end":
            raise self._error(f"unexpected trailing input {self._cur.value!r}")

    # -- grammar -------------------------------------------------------------

    def parse_expression(self) -> Expr:
        return self._parse_logical_or()

    # Precedence (low → high): ||, &&, ==/!=, </>/<=/>=, +/-, */ , unary, **.
    # All left-associative except ** (right). Relational/logical sit below arithmetic
    # so `a + b > c` parses as `(a + b) > c` and `a > b && c > d` as `(a > b) && (c > d)`.
    def _parse_logical_or(self) -> Expr:
        node = self._parse_logical_and()
        while self._at_op("||"):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_logical_and())
        return node

    def _parse_logical_and(self) -> Expr:
        node = self._parse_equality()
        while self._at_op("&&"):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_equality())
        return node

    def _parse_equality(self) -> Expr:
        node = self._parse_relational()
        while self._at_op("==", "!="):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_relational())
        return node

    def _parse_relational(self) -> Expr:
        node = self._parse_additive()
        while self._at_op("<", ">", "<=", ">="):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_additive())
        return node

    def _parse_additive(self) -> Expr:
        node = self._parse_multiplicative()
        while self._at_op("+", "-"):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_multiplicative())
        return node

    def _parse_multiplicative(self) -> Expr:
        node = self._parse_unary()
        while self._at_op("*", "/"):
            op = cast(BinOp, self._advance().value)
            node = BinaryOp(op=op, left=node, right=self._parse_unary())
        return node

    def _parse_unary(self) -> Expr:
        if self._at_op("+", "-", "!"):
            op = cast(UnOp, self._advance().value)
            return UnaryOp(op=op, operand=self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> Expr:
        base = self._parse_postfix()
        if self._at_op("**"):
            self._advance()
            # Right operand is `unary` so `2 ** -1` parses; recursion makes
            # `**` right-associative (`2 ** 3 ** 2` == `2 ** (3 ** 2)`).
            return BinaryOp(op="**", left=base, right=self._parse_unary())
        return base

    def _parse_postfix(self) -> Expr:
        node: Expr = self._parse_primary()
        while self._at_op("["):
            self._advance()
            index = self.parse_expression()
            self._expect_op("]")
            node = IndexAccess(base=node, index=index)
        return node

    def _parse_primary(self) -> Expr:
        tok = self._cur
        if tok.kind == "number":
            self._advance()
            return Number(value=float(tok.value))
        if tok.kind == "name":
            self._advance()
            if self._at_op("("):
                return FunctionCall(callee=tok.value, args=self._parse_call_args())
            return Name(name=tok.value)
        if self._at_op("("):
            self._advance()
            inner = self.parse_expression()
            self._expect_op(")")
            return inner
        if self._at_op("["):
            return self._parse_list_literal()
        if tok.kind == "end":
            raise self._error("unexpected end of expression")
        raise self._error(f"unexpected token {tok.value!r}")

    def _parse_call_args(self) -> tuple[Expr, ...]:
        self._expect_op("(")
        if self._at_op(")"):
            self._advance()
            return ()
        args = [self.parse_expression()]
        while self._at_op(","):
            self._advance()
            args.append(self.parse_expression())
        self._expect_op(")")
        return tuple(args)

    def _parse_list_literal(self) -> VectorLiteral | TensorLiteral:
        self._expect_op("[")
        if self._at_op("]"):
            self._advance()
            return VectorLiteral(components=())
        elements = [self.parse_expression()]
        while self._at_op(","):
            self._advance()
            elements.append(self.parse_expression())
        self._expect_op("]")
        # A bracket whose every element is itself a bracket is a tensor (rows);
        # otherwise it is a vector. Mixed shapes (some rows, some scalars) are a
        # validator concern, so they are left as a VectorLiteral here.
        if all(isinstance(e, VectorLiteral) for e in elements):
            rows = tuple(e for e in elements if isinstance(e, VectorLiteral))
            return TensorLiteral(rows=rows)
        return VectorLiteral(components=tuple(elements))


# ---------------------------------------------------------------------------
# Error rendering.
# ---------------------------------------------------------------------------


def _render(source: str, pos: int, message: str) -> str:
    """Render `message` with the offending source line and a caret at `pos`."""

    pos = max(0, min(pos, len(source)))
    line_start = source.rfind("\n", 0, pos) + 1
    line_end = source.find("\n", pos)
    if line_end == -1:
        line_end = len(source)
    line = source[line_start:line_end]
    column = pos - line_start
    line_no = source.count("\n", 0, pos) + 1
    caret = " " * column + "^"
    return f"{message} (line {line_no}, column {column + 1})\n{line}\n{caret}"

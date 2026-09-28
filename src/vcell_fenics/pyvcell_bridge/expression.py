"""Translate VCell ``MathDescription`` expression strings into the formalism's
expression language (§2.6.2).

VCell and this formalism share a near-identical infix syntax, so the translation is
small and local. Three systematic differences are handled here:

- **Coordinates and time.** VCell exposes the spatial coordinate as the bare names
  ``x`` / ``y`` / ``z`` and time as ``t``. After ADR 006 those are namespaced built-ins
  here: ``geom.x[0]`` / ``geom.x[1]`` / ``geom.x[2]`` and ``sim.t``. The names are matched
  as whole identifiers, so they never rewrite the ``x`` inside ``max``/``exp`` nor a dotted
  ``structure.x``.
- **Power.** VCell writes ``a^b``; the formalism writes ``a**b``.
- **Operator precedence of unary minus vs. power.** VCell's grammar makes a power's base a
  *unary expression* (``PowerTerm = UnaryExpression (POWER UnaryExpression)*``), so the sign
  binds **inside** the power: VCell ``-x^2`` means ``(-x)^2 = x²``. The formalism (like standard
  math / Python) binds ``**`` tighter than unary minus, so ``-x**2`` means ``-(x²)``. VCell's
  power is also **left**-associative (``a^b^c = (a^b)^c``) where ours is right-associative. A
  naïve ``^→**`` swap would silently flip the meaning, so we **parse with VCell's precedence**
  and insert the minimal parentheses needed to preserve it: ``-x^2 → (-geom.x[0])**2`` and
  ``2^3^2 → (2**3)**2``. Expressions without a signed power base or a power chain are unchanged.

The parse is intentionally tolerant: any expression it cannot tokenise/parse falls back to the
old whole-identifier substitution, so no input is rejected — only the precedence-sensitive forms
gain parentheses.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# VCell bare coordinate / time names → the namespaced built-ins (ADR 006).
_COORDINATE_TIME: dict[str, str] = {
    "x": "geom.x[0]",
    "y": "geom.x[1]",
    "z": "geom.x[2]",
    "t": "sim.t",
}

_COORD_RE = re.compile(r"\b(?:x|y|z|t)\b")

_TOKEN_RE = re.compile(
    r"""
      (?P<NUM>\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
    | (?P<NAME>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)
    | (?P<OP><=|>=|==|!=|&&|\|\||\*\*|[-+*/^<>!])
    | (?P<LP>\()
    | (?P<RP>\))
    | (?P<COMMA>,)
    | (?P<WS>\s+)
    """,
    re.VERBOSE,
)

_POWER = ("^", "**")


class _TranslateError(Exception):
    """The expression could not be tokenised/parsed with VCell precedence; fall back to the
    plain substitution."""


@dataclass(frozen=True)
class _Tok:
    kind: str
    text: str
    start: int
    end: int


# A parsed node, reduced to what the re-parenthesisation needs: its source char span and whether
# it is a *signed* unary expression (a leading -/+/! — the case that needs wrapping as a power base).
_Node = tuple[int, int, bool]


def _tokenize(source: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    pos = 0
    while pos < len(source):
        match = _TOKEN_RE.match(source, pos)
        if match is None:
            raise _TranslateError(f"unexpected character at {pos}")
        kind = match.lastgroup
        assert kind is not None
        if kind != "WS":
            tokens.append(_Tok(kind, match.group(), match.start(), match.end()))
        pos = match.end()
    return tokens


class _Parser:
    """Recursive-descent parse over VCell precedence whose only product is ``wraps`` — the source
    spans to parenthesise so the formalism parser reproduces VCell's grouping."""

    def __init__(self, tokens: list[_Tok]) -> None:
        self._tokens = tokens
        self._i = 0
        self.wraps: list[tuple[int, int]] = []

    def parse(self) -> None:
        self._expr()
        if self._i != len(self._tokens):
            raise _TranslateError("trailing tokens")

    def _peek(self) -> _Tok | None:
        return self._tokens[self._i] if self._i < len(self._tokens) else None

    def _is_op(self, *texts: str) -> bool:
        tok = self._peek()
        return tok is not None and tok.kind == "OP" and tok.text in texts

    def _binary(self, sub: Callable[[], _Node], *ops: str) -> _Node:
        node = sub()
        while self._is_op(*ops):
            self._i += 1
            right = sub()
            node = (node[0], right[1], False)
        return node

    def _expr(self) -> _Node:
        return self._binary(self._and, "||")

    def _and(self) -> _Node:
        return self._binary(self._eq, "&&")

    def _eq(self) -> _Node:
        return self._binary(self._rel, "==", "!=")

    def _rel(self) -> _Node:
        return self._binary(self._add, "<", ">", "<=", ">=")

    def _add(self) -> _Node:
        return self._binary(self._mul, "+", "-")

    def _mul(self) -> _Node:
        return self._binary(self._pow, "*", "/")

    def _pow(self) -> _Node:
        base = self._unary()
        if not self._is_op(*_POWER):
            return base
        if base[2]:  # signed base: VCell binds the sign inside the power → wrap it
            self.wraps.append((base[0], base[1]))
        start, end = base[0], base[1]
        while self._is_op(*_POWER):
            self._i += 1
            exponent = self._unary()
            end = exponent[1]
            if self._is_op(*_POWER):  # another power follows → VCell is left-assoc, wrap the left
                self.wraps.append((start, end))
        return (start, end, False)

    def _unary(self) -> _Node:
        tok = self._peek()
        if tok is not None and tok.kind == "OP" and tok.text in ("-", "+", "!"):
            self._i += 1
            operand = self._unary()
            return (tok.start, operand[1], True)
        return self._primary()

    def _primary(self) -> _Node:
        tok = self._peek()
        if tok is None:
            raise _TranslateError("expected a primary")
        if tok.kind == "LP":
            self._i += 1
            self._expr()
            close = self._peek()
            if close is None or close.kind != "RP":
                raise _TranslateError("unbalanced '('")
            self._i += 1
            return (tok.start, close.end, False)
        if tok.kind == "NUM":
            self._i += 1
            return (tok.start, tok.end, False)
        if tok.kind == "NAME":
            self._i += 1
            following = self._peek()
            if following is not None and following.kind == "LP":  # function call
                self._i += 1
                if not (self._peek() is not None and self._peek().kind == "RP"):  # type: ignore[union-attr]
                    self._expr()
                    while self._peek() is not None and self._peek().kind == "COMMA":  # type: ignore[union-attr]
                        self._i += 1
                        self._expr()
                close = self._peek()
                if close is None or close.kind != "RP":
                    raise _TranslateError("unbalanced call '('")
                self._i += 1
                return (tok.start, close.end, False)
            return (tok.start, tok.end, False)
        raise _TranslateError(f"unexpected token {tok.text!r}")


def _apply(source: str, tokens: list[_Tok], wraps: list[tuple[int, int]]) -> str:
    """Emit the translated string: substitute coordinate/time names and ``^→**`` in place, and
    insert the parentheses ``wraps`` requires — all by reconstructing from the original source so
    the formatting is otherwise preserved."""

    replace: dict[int, tuple[int, str]] = {}
    for tok in tokens:
        if tok.kind == "NAME" and tok.text in _COORDINATE_TIME:
            replace[tok.start] = (tok.end, _COORDINATE_TIME[tok.text])
        elif tok.kind == "OP" and tok.text in _POWER:
            replace[tok.start] = (tok.end, "**")

    opens: dict[int, int] = {}
    closes: dict[int, int] = {}
    for a, b in wraps:
        opens[a] = opens.get(a, 0) + 1
        closes[b] = closes.get(b, 0) + 1

    out: list[str] = []
    i = 0
    n = len(source)
    while i <= n:
        out.append(")" * closes.get(i, 0))
        out.append("(" * opens.get(i, 0))
        if i == n:
            break
        if i in replace:
            end, text = replace[i]
            out.append(text)
            i = end
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


# `vcRegionVolume('X')` / `vcRegionArea('X')` (either quote style): a subdomain's measure.
_REGION_SIZE_RE = re.compile(r"""vcRegion(?:Volume|Area)\(\s*['"]([A-Za-z_][\w]*)['"]\s*\)""")


def translate_expression(vcell_expr: str) -> str:
    """Translate one VCell expression string into the formalism's syntax: map the bare
    coordinate/time names to their `geom.*` / `sim.*` built-ins and `^` to `**`, inserting the
    parentheses needed to preserve VCell's unary-minus-vs-power precedence (`-x^2 → (-x)**2`).
    Anything that cannot be parsed falls back to the plain substitution.

    VCell's region-size built-ins, ``vcRegionVolume('X')`` / ``vcRegionArea('X')``, become the formalism's
    ``region_size(X)`` (§1.8.4) first — their quoted argument is a subdomain name, which the formalism writes
    bare."""

    vcell_expr = _REGION_SIZE_RE.sub(lambda m: f"region_size({m.group(1)})", vcell_expr)
    try:
        tokens = _tokenize(vcell_expr)
        parser = _Parser(tokens)
        parser.parse()
    except _TranslateError:
        return _COORD_RE.sub(lambda m: _COORDINATE_TIME[m.group()], vcell_expr).replace("^", "**")
    return _apply(vcell_expr, tokens, parser.wraps)

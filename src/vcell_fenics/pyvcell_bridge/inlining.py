"""Resolve VCell ``MathFunction``s: inline the non-constant ones, keep the rest.

A VCell ``MathFunction`` is a named sub-expression. Some reference state variables —
e.g. ``I = gL*(Voltage_pm - VL)`` — and some reference the spatial coordinates ``x`` / ``y`` /
``z`` or time ``t`` — e.g. an off-centre Gaussian initial condition ``u_init = 10*exp(-(x^2+y^2))``.
The formalism's ``ParameterExpression`` is a position- and time-independent **constant** in the
backend, so neither kind can be a parameter. We therefore:

- **inline** such non-constant functions (recursively, in VCell syntax) into the equation
  expressions that use them, so each equation is self-contained for the form compiler (the form
  compiler binds ``geom.x`` / ``sim.t`` and the state variables directly); and
- expose them as **observables** (derived outputs) on the side, never inside the math
  (`docs/modeling/declarative-formalism.md` §2.6.3: observables live with the output spec).

Functions that reduce to a constant (transitively reference only constants / parameters, no
variable, coordinate, or time) stay as ``ParameterExpression``.

Identifier matching is dotted-name aware: VCell names like ``structure.param`` are single
identifiers, and a ``name(`` is a call (a math/builtin function), not a value reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyvcell.vcml.models_math import MathFunction

# A bare identifier used as a *value* — optionally dotted (VCell `structure.param` names),
# and NOT immediately followed by `(` (that would be a function call, not a name reference).
_IDENT_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?!\s*\()")

# VCell's spatial coordinates and time — bare identifiers that make a function position- or
# time-dependent (hence non-constant), exactly as a state-variable reference does.
_COORDINATE_TIME_NAMES = frozenset({"x", "y", "z", "t"})


class FunctionCycleError(ValueError):
    """A VCell ``MathFunction`` set contains a cyclic reference, so it cannot be inlined."""


@dataclass(frozen=True)
class FunctionResolution:
    """The outcome of resolving a model's functions: the fully-inlined bodies of the non-constant
    (variable- / coordinate- / time-referencing) functions (to substitute into equations and to
    expose as observables), and the names of the constant functions (kept as parameters)."""

    inlined_function_bodies: dict[str, str]
    constant_function_names: frozenset[str]

    @property
    def inlined_function_names(self) -> frozenset[str]:
        return frozenset(self.inlined_function_bodies)

    def inline(self, expr: str | None) -> str | None:
        """Substitute each non-constant function name in ``expr`` with its fully-inlined
        (parenthesised) body. One pass suffices: the bodies themselves contain no
        non-constant-function names."""

        if not expr or not self.inlined_function_bodies:
            return expr

        def repl(match: re.Match[str]) -> str:
            body = self.inlined_function_bodies.get(match.group(1))
            return f"({body})" if body is not None else match.group(1)

        return _IDENT_RE.sub(repl, expr)


def resolve_functions(functions: list[MathFunction], variable_names: set[str]) -> FunctionResolution:
    """Classify ``functions`` (pyvcell ``MathFunction``s) into non-constant vs constant, and compute
    the fully-inlined VCell-syntax body of each non-constant one. A function is non-constant if it
    transitively references a state variable, a spatial coordinate (``x`` / ``y`` / ``z``), or time
    (``t``) — anything the backend cannot fold into a constant parameter."""

    exprs: dict[str, str] = {f.name: (f.exp or "") for f in functions}
    names = set(exprs)
    non_constant_triggers = variable_names | _COORDINATE_TIME_NAMES

    is_non_constant: dict[str, bool] = {}

    def non_constant(name: str, stack: frozenset[str]) -> bool:
        if name in is_non_constant:
            return is_non_constant[name]
        if name in stack:  # cycle — break conservatively (a cyclic function set is rejected below)
            return False
        idents = set(_IDENT_RE.findall(exprs.get(name, "")))
        result = bool(idents & non_constant_triggers) or any(
            g in names and non_constant(g, stack | {name}) for g in idents
        )
        is_non_constant[name] = result
        return result

    inlined_funcs = {n for n in names if non_constant(n, frozenset())}

    inlined: dict[str, str] = {}

    def inline_body(name: str, stack: frozenset[str]) -> str:
        if name in inlined:
            return inlined[name]
        if name in stack:
            raise FunctionCycleError(f"cyclic function reference involving {name!r}")

        def repl(match: re.Match[str]) -> str:
            ref = match.group(1)
            if ref in inlined_funcs:
                return "(" + inline_body(ref, stack | {name}) + ")"
            return ref

        result = _IDENT_RE.sub(repl, exprs.get(name, ""))
        inlined[name] = result
        return result

    inlined_function_bodies = {n: inline_body(n, frozenset()) for n in inlined_funcs}
    return FunctionResolution(
        inlined_function_bodies=inlined_function_bodies,
        constant_function_names=frozenset(names - inlined_funcs),
    )

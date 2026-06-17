"""Resolve VCell ``MathFunction``s: inline the variable-referencing ones, keep the rest.

A VCell ``MathFunction`` is a named sub-expression that may reference state variables —
e.g. ``I = gL*(Voltage_pm - VL)``. The formalism's ``ParameterExpression`` may **not**
reference variables, so a variable-referencing function cannot be a parameter. We therefore:

- **inline** such functions (recursively, in VCell syntax) into the equation expressions that
  use them, so each equation is self-contained for the form compiler; and
- expose them as **observables** (derived outputs) on the side, never inside the math
  (`docs/modeling/declarative-formalism.md` §2.6.3: observables live with the output spec).

Functions that reference only constants / parameters / coordinates (transitively no variable)
stay as ``ParameterExpression``, exactly as before.

Identifier matching is dotted-name aware: VCell names like ``structure.param`` are single
identifiers, and a ``name(`` is a call (a math/builtin function), not a value reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# A bare identifier used as a *value* — optionally dotted (VCell `structure.param` names),
# and NOT immediately followed by `(` (that would be a function call, not a name reference).
_IDENT_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?!\s*\()")


class FunctionCycleError(ValueError):
    """A VCell ``MathFunction`` set contains a cyclic reference, so it cannot be inlined."""


@dataclass(frozen=True)
class FunctionResolution:
    """The outcome of resolving a model's functions: the fully-inlined bodies of the
    variable-referencing functions (to substitute into equations and to expose as
    observables), and the names of the pure functions (kept as parameters)."""

    var_function_bodies: dict[str, str]
    pure_function_names: frozenset[str]

    @property
    def var_function_names(self) -> frozenset[str]:
        return frozenset(self.var_function_bodies)

    def inline(self, expr: str | None) -> str | None:
        """Substitute each variable-referencing function name in ``expr`` with its
        fully-inlined (parenthesised) body. One pass suffices: the bodies themselves contain
        no variable-referencing-function names."""

        if not expr or not self.var_function_bodies:
            return expr

        def repl(match: re.Match[str]) -> str:
            body = self.var_function_bodies.get(match.group(1))
            return f"({body})" if body is not None else match.group(1)

        return _IDENT_RE.sub(repl, expr)


def resolve_functions(functions: list[Any], variable_names: set[str]) -> FunctionResolution:
    """Classify ``functions`` (pyvcell ``MathFunction``s) into variable-referencing vs pure,
    and compute the fully-inlined VCell-syntax body of each variable-referencing one."""

    exprs: dict[str, str] = {f.name: (f.exp or "") for f in functions}
    names = set(exprs)

    references_var: dict[str, bool] = {}

    def refs_var(name: str, stack: frozenset[str]) -> bool:
        if name in references_var:
            return references_var[name]
        if name in stack:  # cycle — break conservatively (a cyclic function set is rejected below)
            return False
        idents = set(_IDENT_RE.findall(exprs.get(name, "")))
        result = bool(idents & variable_names) or any(g in names and refs_var(g, stack | {name}) for g in idents)
        references_var[name] = result
        return result

    var_funcs = {n for n in names if refs_var(n, frozenset())}

    inlined: dict[str, str] = {}

    def inline_body(name: str, stack: frozenset[str]) -> str:
        if name in inlined:
            return inlined[name]
        if name in stack:
            raise FunctionCycleError(f"cyclic function reference involving {name!r}")

        def repl(match: re.Match[str]) -> str:
            ref = match.group(1)
            if ref in var_funcs:
                return "(" + inline_body(ref, stack | {name}) + ")"
            return ref

        result = _IDENT_RE.sub(repl, exprs.get(name, ""))
        inlined[name] = result
        return result

    var_function_bodies = {n: inline_body(n, frozenset()) for n in var_funcs}
    return FunctionResolution(var_function_bodies=var_function_bodies, pure_function_names=frozenset(names - var_funcs))

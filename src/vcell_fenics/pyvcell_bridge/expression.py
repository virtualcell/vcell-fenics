"""Translate VCell ``MathDescription`` expression strings into the formalism's
expression language (§2.6.2).

VCell and this formalism share a near-identical infix syntax, so the translation
is small and local. Two systematic differences are handled here:

- **Coordinates and time.** VCell exposes the spatial coordinate as the bare names
  ``x`` / ``y`` / ``z`` and time as ``t``. After ADR 006 those are namespaced built-ins
  here: ``geom.x[0]`` / ``geom.x[1]`` / ``geom.x[2]`` and ``sim.t``. (The namespacing
  migration is what makes this clean — a VCell species may itself be named ``t`` or
  ``x`` without colliding, because the coordinate/time references are now qualified.)
- **Power.** VCell writes ``a^b``; the formalism writes ``a**b``.

Both replacements are token-aware: the coordinate/time names are matched only as
whole identifiers (a leading/trailing word boundary), so they never rewrite the
``x`` inside ``max`` or ``exp``, nor a parameter named ``Ca_x``. VCell reserves
``x``/``y``/``z``/``t`` as coordinates, so no user symbol can legitimately collide.

Anything else passes through unchanged. A handful of rarer VCell-only constructs
(integer-division semantics, a few function-name aliases) are deliberately *not*
normalised yet; an expression that uses one surfaces downstream as a validation or
compile diagnostic rather than being silently mistranslated.
"""

from __future__ import annotations

import re

# VCell bare coordinate / time names → the namespaced built-ins (ADR 006).
_COORDINATE_TIME: dict[str, str] = {
    "x": "geom.x[0]",
    "y": "geom.x[1]",
    "z": "geom.x[2]",
    "t": "sim.t",
}

# A single whole-identifier match over the four reserved names. `re.sub` scans the
# original string left-to-right and does not re-examine substituted text, so the
# `x` inside an inserted `geom.x[0]` is never re-matched.
_COORD_RE = re.compile(r"\b(?:x|y|z|t)\b")


def translate_expression(vcell_expr: str) -> str:
    """Translate one VCell expression string into the formalism's syntax.

    Maps the bare coordinate/time names to their `geom.*` / `sim.*` built-ins and
    `^` to `**`; everything else is passed through verbatim.
    """

    translated = _COORD_RE.sub(lambda m: _COORDINATE_TIME[m.group()], vcell_expr)
    return translated.replace("^", "**")

"""The one backend type alias that isn't a real class: `UflExpr`.

DOLFINx and UFL ship `py.typed`, and (since we stopped `follow_imports = "skip"`,
ADR 005) mypy checks them concretely — so the backend annotates DOLFINx/UFL
objects with their real classes directly (`fem.Function`, `mesh.Mesh`, `ufl.Form`,
`ufl.Measure`, …). No alias indirection is warranted for those.

The exception is a *UFL expression*. It is heterogeneous — arguments,
coefficients, arithmetic results, wrapped DOLFINx constants — and UFL's operators
(`a + b`, `2 * a`) are themselves typed to return `Any`, so a value flowing
through them cannot be pinned to a concrete type. `UflExpr` is that `Any`, named
so a signature can still say *what* it is.
"""

from __future__ import annotations

from typing import Any, TypeAlias

UflExpr: TypeAlias = Any

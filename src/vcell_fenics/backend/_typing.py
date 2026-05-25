"""Type aliases for the DOLFINx / UFL objects the backend threads around.

DOLFINx and UFL ship no usable type stubs, so the project treats that whole
stack as opaque: `pyproject.toml` sets `follow_imports = "skip"` for `dolfinx.*`,
`ufl.*`, `petsc4py.*`, and friends (see CLAUDE.md, which catalogues the broken
stubs). Every value from those libraries is therefore `Any` to mypy.

Each alias below *is* `Any` — these add no type-checking. What they add is
intent: a backend signature reads `unknown: Function`, `dx: Measure`,
`compose(...) -> tuple[UflForm, UflForm]` instead of a wall of bare `Any`, so a
reader can see what each opaque object actually is. If usable stubs ever land,
repoint the aliases at the real types and the backend gains real checks for free.
"""

from __future__ import annotations

from typing import Any, TypeAlias

# DOLFINx objects (the `Dolfinx` prefix marks the source library).
DolfinxMesh: TypeAlias = Any
DolfinxFunctionSpace: TypeAlias = Any
DolfinxFunction: TypeAlias = Any
DolfinxConstant: TypeAlias = Any
DolfinxDirichletBC: TypeAlias = Any

# UFL objects (the `Ufl` prefix marks the source library).
UflExpr: TypeAlias = Any  # a UFL expression: an integrand, coefficient, or trial/test argument
UflForm: TypeAlias = Any  # an integrated UFL form (bilinear `a` or linear `L`)
UflMeasure: TypeAlias = Any  # a UFL integration measure (dx, ds, ...)

"""DOLFINx backend: translate a validated MathDescription into a runnable solve.

This package is the FEniCSx-specific backend (docs/decisions/004-discreteproblem-ir.md).
The translation pipeline is

    expression AST → UFL (compiler) → DiscreteProblem IR → lowering → dolfinx / PETSc

The `formalism` package (schema / parser / validator) stays pure-Python with no
FEniCSx dependency; this package is where DOLFINx enters.
"""

from vcell_fenics.backend.compiler import CompileContext, CompileError, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, Term, TermKind

__all__ = [
    "BackwardEuler",
    "CompileContext",
    "CompileError",
    "DiscreteProblem",
    "Term",
    "TermKind",
    "compile_expression",
]

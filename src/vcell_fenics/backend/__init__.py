"""DOLFINx backend: translate a validated MathDescription into a runnable solve.

This package is the FEniCSx-specific backend (docs/decisions/004-discreteproblem-ir.md).
The translation pipeline is

    expression AST → UFL (compiler) → DiscreteProblem IR → lowering → dolfinx / PETSc

The `formalism` package (schema / parser / validator) stays pure-Python with no
FEniCSx dependency; this package is where DOLFINx enters.
"""

from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.compiler import CompileContext, CompileError, compile_expression
from vcell_fenics.backend.discrete import BackwardEuler, DiscreteProblem, MeshQualityError, Term, TermKind
from vcell_fenics.backend.geometry import (
    Geometry,
    SubdomainGeometry,
    clear_geometries,
    cross_validate,
    load_geometry,
    make_disk_geometry,
    make_disk_membrane_geometry,
    register_geometry,
)

__all__ = [
    "BackwardEuler",
    "CompileContext",
    "CompileError",
    "DiscreteProblem",
    "Geometry",
    "MeshQualityError",
    "SubdomainGeometry",
    "Term",
    "TermKind",
    "assemble",
    "clear_geometries",
    "compile_expression",
    "cross_validate",
    "load_geometry",
    "make_disk_geometry",
    "make_disk_membrane_geometry",
    "register_geometry",
]

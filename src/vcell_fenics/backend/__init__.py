"""DOLFINx backend: translate a validated MathDescription into a runnable solve.

This package is the FEniCSx-specific backend (docs/decisions/004-discreteproblem-ir.md).
The translation pipeline is

    expression AST → UFL (compiler) → DiscreteProblem IR → lowering → dolfinx / PETSc

The `formalism` package (schema / parser / validator) stays pure-Python with no
FEniCSx dependency; this package is where DOLFINx enters.
"""

from vcell_fenics.backend.ale import ALEState, StepTooLarge, run_with_remeshing, step_with_remeshing
from vcell_fenics.backend.assemble import assemble, rebuild_on_mesh
from vcell_fenics.backend.compiler import CompileContext, CompileError, compile_expression
from vcell_fenics.backend.coupled import CoupledProblem, assemble_coupled
from vcell_fenics.backend.discrete import (
    BackwardEuler,
    BoundaryTerm,
    DiscreteProblem,
    MeshQualityError,
    Term,
    TermKind,
)
from vcell_fenics.backend.geometry import (
    BoundaryGeometry,
    CoupledGeometry,
    Geometry,
    SubdomainGeometry,
    clear_geometries,
    cross_validate,
    load_geometry,
    make_cell_extracellular_geometry,
    make_disk_geometry,
    make_disk_membrane_geometry,
    make_extracellular_annulus_geometry,
    register_geometry,
)
from vcell_fenics.backend.solver import SolverConfiguration, run
from vcell_fenics.backend.unknown_motion import UnknownMotionProblem, assemble_unknown_motion
from vcell_fenics.backend.weakform import WeakFormProblem, assemble_weak_form

__all__ = [
    "ALEState",
    "BackwardEuler",
    "BoundaryGeometry",
    "BoundaryTerm",
    "CompileContext",
    "CompileError",
    "CoupledGeometry",
    "CoupledProblem",
    "DiscreteProblem",
    "Geometry",
    "MeshQualityError",
    "SolverConfiguration",
    "StepTooLarge",
    "SubdomainGeometry",
    "Term",
    "TermKind",
    "UnknownMotionProblem",
    "WeakFormProblem",
    "assemble",
    "assemble_coupled",
    "assemble_unknown_motion",
    "assemble_weak_form",
    "clear_geometries",
    "compile_expression",
    "cross_validate",
    "load_geometry",
    "make_cell_extracellular_geometry",
    "make_disk_geometry",
    "make_disk_membrane_geometry",
    "make_extracellular_annulus_geometry",
    "rebuild_on_mesh",
    "register_geometry",
    "run",
    "run_with_remeshing",
    "step_with_remeshing",
]

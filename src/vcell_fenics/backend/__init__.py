"""DOLFINx backend: translate a validated MathDescription into a runnable solve.

This package is the FEniCSx-specific backend (docs/decisions/004-discreteproblem-ir.md).
The translation pipeline is

    expression AST → UFL (compiler) → DiscreteProblem IR → lowering → dolfinx / PETSc

The `formalism` package (schema / parser / validator) stays pure-Python with no
FEniCSx dependency; this package is where DOLFINx enters.
"""

from vcell_fenics.backend.ale import ALEState, StepTooLarge, run_with_remeshing, step_with_remeshing
from vcell_fenics.backend.assemble import assemble, rebuild_on_mesh
from vcell_fenics.backend.cahn_hilliard import (
    cahn_hilliard_free_energy,
    run_cahn_hilliard,
    solve_cahn_hilliard,
)
from vcell_fenics.backend.compiler import CompileContext, CompileError, compile_expression
from vcell_fenics.backend.coupled import CoupledProblem, assemble_coupled
from vcell_fenics.backend.diagnostics import NonlinearTermError, SolveError
from vcell_fenics.backend.discrete import (
    BackwardEuler,
    BoundaryTerm,
    DiscreteProblem,
    MeshQualityError,
    Term,
    TermKind,
)
from vcell_fenics.backend.fsi import (
    enclosed_volume,
    step_force_balance_fsi,
    step_prescribed_fsi,
    step_two_phase_fsi,
    step_two_phase_fsi_with_nonlinear_species,
    step_two_phase_fsi_with_reacting_species,
    step_two_phase_fsi_with_species,
)
from vcell_fenics.backend.geometry import (
    BoundaryGeometry,
    CoupledGeometry,
    Geometry,
    InterfaceCoupledGeometry,
    SubdomainGeometry,
    clear_geometries,
    cross_validate,
    load_geometry,
    make_cell_extracellular_geometry,
    make_disk_geometry,
    make_disk_membrane_geometry,
    make_extracellular_annulus_geometry,
    make_two_bulk_membrane_geometry,
    membrane_trace,
    register_geometry,
)
from vcell_fenics.backend.interface_coupled import (
    InterfaceCoupledProblem,
    InterfaceCoupledResult,
    assemble_interface_coupled,
    integrate_interface_coupled,
)
from vcell_fenics.backend.multiphase import (
    solve_two_phase_overdamped,
    solve_two_phase_stokes,
    solve_two_phase_stokes_surface_tension,
)
from vcell_fenics.backend.reaction_diffusion import (
    IntegrationResult,
    integrate_discrete_problem,
    integrate_reaction_diffusion,
)
from vcell_fenics.backend.slip import nitsche_normal_slip, solve_overdamped_slip
from vcell_fenics.backend.solver import SolverConfiguration, run
from vcell_fenics.backend.stokes import (
    solve_incompressible_stokes,
    solve_incompressible_stokes_slip,
    solve_incompressible_stokes_surface_tension,
    solve_incompressible_stokes_traction,
)
from vcell_fenics.backend.stokes_hdiv import solve_incompressible_stokes_hdiv_slip
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
    "IntegrationResult",
    "InterfaceCoupledGeometry",
    "InterfaceCoupledProblem",
    "InterfaceCoupledResult",
    "MeshQualityError",
    "NonlinearTermError",
    "SolveError",
    "SolverConfiguration",
    "StepTooLarge",
    "SubdomainGeometry",
    "Term",
    "TermKind",
    "UnknownMotionProblem",
    "WeakFormProblem",
    "assemble",
    "assemble_coupled",
    "assemble_interface_coupled",
    "assemble_unknown_motion",
    "assemble_weak_form",
    "cahn_hilliard_free_energy",
    "clear_geometries",
    "compile_expression",
    "cross_validate",
    "enclosed_volume",
    "integrate_discrete_problem",
    "integrate_interface_coupled",
    "integrate_reaction_diffusion",
    "load_geometry",
    "make_cell_extracellular_geometry",
    "make_disk_geometry",
    "make_disk_membrane_geometry",
    "make_extracellular_annulus_geometry",
    "make_two_bulk_membrane_geometry",
    "membrane_trace",
    "nitsche_normal_slip",
    "rebuild_on_mesh",
    "register_geometry",
    "run",
    "run_cahn_hilliard",
    "run_with_remeshing",
    "solve_cahn_hilliard",
    "solve_incompressible_stokes",
    "solve_incompressible_stokes_hdiv_slip",
    "solve_incompressible_stokes_slip",
    "solve_incompressible_stokes_surface_tension",
    "solve_incompressible_stokes_traction",
    "solve_overdamped_slip",
    "solve_two_phase_overdamped",
    "solve_two_phase_stokes",
    "solve_two_phase_stokes_surface_tension",
    "step_force_balance_fsi",
    "step_prescribed_fsi",
    "step_two_phase_fsi",
    "step_two_phase_fsi_with_nonlinear_species",
    "step_two_phase_fsi_with_reacting_species",
    "step_two_phase_fsi_with_species",
    "step_with_remeshing",
]

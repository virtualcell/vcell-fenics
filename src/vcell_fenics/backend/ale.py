"""ALE remesh driver: step a moving-mesh problem, remeshing before it tangles.

The v1 backend moves mesh nodes by `dt·v` each step and raises `MeshQualityError`
once the mesh distorts past a hard limit — it never remeshes. This module is the
layer above the per-step solve sketched in `docs/modeling/ale-remesh-driver.md`: it
monitors mesh quality and, before a step would tangle, swaps to a fresh mesh and
re-interpolates state conservatively, turning "fail loudly on tangling" into
"remesh at the step boundary and continue."

It composes the three pieces built earlier: the region remesher (`mesh_region`)
makes a fresh mesh of the deformed configuration, `rebuild_on_mesh` reassembles the
build-once IR on it with conservative state transfer, and the per-step
backward-Euler solve is unchanged.

**Scope (v1): the moving *membrane* (a 1D codim-1 submesh in 2D).** The deformed
membrane is remeshed as the boundary of a freshly meshed region; ρ rides along via
the conservative surface remap. The Approach-A case — ρ as a bulk *trace* on a
2D mesh whose interior nodes move by harmonic extension — needs bulk mesh-motion
(not built) and would additionally route the transfer through `correct_surface_trace`
(built, waiting). Until then `_remesh_membrane` guards to 1D meshes.

The remesh trigger is *proactive on accumulated distortion*: each step checks how
far the mesh has deformed since it was last built (`mesh_quality_growth`) and
remeshes on the still-valid geometry once it crosses `quality_limit`, kept well
below the hard `MeshQualityError` threshold so the following step is safe. (This is
simpler than the sketch's tentative look-ahead and, given the margin, equivalent —
it also avoids any rollback.) If a single step tangles the mesh anyway, the step is
genuinely too large and `StepTooLarge` is raised rather than looping remeshes
(subtlety 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from dolfinx import fem
from dolfinx.mesh import Mesh, create_submesh, exterior_facet_indices

from vcell_fenics.backend.assemble import assemble, rebuild_on_mesh
from vcell_fenics.backend.discrete import DiscreteProblem, MeshQualityError
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.solver import SolverConfiguration
from vcell_fenics.core import mesh_region, ordered_membrane_loop
from vcell_fenics.formalism.schema import MathDescription


class StepTooLarge(RuntimeError):
    """A single step tangled the mesh even on a fresh configuration — remeshing
    cannot rescue a step this large, so reduce `dt` or sub-step (subtlety 5 of
    `docs/modeling/ale-remesh-driver.md`). The driver raises this instead of looping
    remeshes."""


@dataclass
class ALEState:
    """The evolving state of an ALE run: the assembled problem (bound to the current
    mesh), the MathDescription needed to reassemble after a remesh, the simulated
    time, and a count of remeshes performed (observable for tests / diagnostics)."""

    problem: DiscreteProblem
    md: MathDescription
    t: float = 0.0
    remesh_count: int = 0

    @classmethod
    def initial(cls, md: MathDescription, geometry: Geometry, config: SolverConfiguration) -> ALEState:
        problem = assemble(md, geometry, dt=config.dt, fe_degree=config.fe_degree)
        return cls(problem=problem, md=md)

    def remesh(self, target_h: float) -> None:
        """Swap to a fresh mesh of the current (deformed) configuration at resolution
        `target_h`, conservatively transferring state. The build-once IR is rebuilt,
        not mutated (ADR 004)."""

        new_mesh = _remesh_membrane(self.problem, target_h)
        self.problem = rebuild_on_mesh(self.problem, self.md, new_mesh)
        self.remesh_count += 1


def step_with_remeshing(state: ALEState, *, quality_limit: float, target_h: float) -> ALEState:
    """Advance `state` by one `dt`, remeshing first if the moving mesh has distorted
    past `quality_limit` (a cell-size-ratio growth factor, e.g. 4.0). Returns the
    same (mutated) state. Raises `StepTooLarge` if a step tangles the mesh anyway."""

    if state.problem.motion_velocity is not None and state.problem.mesh_quality_growth() >= quality_limit:
        state.remesh(target_h)  # swap on the still-valid geometry, before it tangles
    try:
        state.problem.step()
    except MeshQualityError as exc:
        raise StepTooLarge(
            f"a single dt step tangled the mesh at t={state.t:.4g}; remeshing cannot rescue a step this large "
            f"— reduce dt or sub-step (ALE driver subtlety 5)."
        ) from exc
    state.t += float(state.problem.dt.value)
    return state


def run_with_remeshing(
    md: MathDescription,
    geometry: Geometry,
    config: SolverConfiguration,
    *,
    target_h: float,
    quality_limit: float = 4.0,
) -> ALEState:
    """Assemble and advance to `config.t_final`, remeshing whenever the moving mesh
    distorts past `quality_limit` — the remesh-and-continue analogue of `run`.
    Returns the final `ALEState` (its `problem.unknown` holds the final field, its
    `remesh_count` how many remeshes were needed)."""

    state = ALEState.initial(md, geometry, config)
    for _ in range(round(config.t_final / config.dt)):
        step_with_remeshing(state, quality_limit=quality_limit, target_h=target_h)
    return state


def _remesh_membrane(problem: DiscreteProblem, target_h: float) -> Mesh:
    """A fresh 1D membrane mesh of the current (deformed) configuration at resolution
    `target_h`: order the deformed boundary off the current mesh, mesh the region it
    encloses (`mesh_region`), and extract that region's boundary as a new codim-1
    submesh. v1 remeshes a 1D membrane only — bulk (Approach-A) mesh motion is later
    work."""

    mesh = problem.V.mesh
    if mesh.topology.dim != 1:
        raise NotImplementedError(
            "the v1 ALE driver remeshes a 1D membrane; bulk (Approach-A) mesh motion is later work"
        )
    loop, _order = ordered_membrane_loop(fem.functionspace(mesh, ("Lagrange", 1)))
    bulk = mesh_region(loop, target_h)
    tdim = bulk.topology.dim
    bulk.topology.create_connectivity(tdim - 1, tdim)
    facets = exterior_facet_indices(bulk.topology)
    membrane, *_ = create_submesh(bulk, tdim - 1, facets)
    return membrane

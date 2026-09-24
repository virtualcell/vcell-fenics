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

**Scope: a moving 2D region, either a codim-0 bulk or a codim-1 membrane.** A moving
*bulk* (its interior nodes carried by harmonic-extension mesh-motion) is remeshed
directly — the deformed boundary loop becomes the boundary of a fresh `mesh_region`
mesh, and the bulk field is transferred conservatively (`remap_bulk_function`). A
moving *membrane* is remeshed as the boundary of a freshly meshed region, with ρ
carried by the conservative surface remap. `_remesh` dispatches on codimension. The
genuine Approach-A case where ρ is a bulk *trace* (a surface PDE on bulk boundary
facets) is a separate modelling increment; its transfer would route through
`correct_surface_trace` (built, waiting), but the trace-PDE physics does not exist yet.

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
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem_stride
from vcell_fenics.backend.solver import SolverConfiguration
from vcell_fenics.core import BulkBoundaryTrace, ordered_membrane_loop
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
    steps: int = 0  # total inner TS steps (the method-of-lines driver); 0 for the backward-Euler driver

    @classmethod
    def initial(cls, md: MathDescription, geometry: Geometry, config: SolverConfiguration) -> ALEState:
        problem = assemble(md, geometry, dt=config.dt, fe_degree=config.fe_degree)
        return cls(problem=problem, md=md)

    def remesh(self, target_h: float) -> None:
        """Swap to a fresh mesh of the current (deformed) configuration at resolution
        `target_h`, conservatively transferring state. The build-once IR is rebuilt,
        not mutated (ADR 004)."""

        new_mesh = _remesh(self.problem, target_h)
        self.problem = rebuild_on_mesh(self.problem, self.md, new_mesh)
        self.remesh_count += 1


def step_with_remeshing(state: ALEState, *, quality_limit: float, target_h: float) -> ALEState:
    """Advance `state` by one `dt` (to `state.t + dt`, setting the problem's time), remeshing
    first if the moving mesh has distorted past `quality_limit` (a cell-size-ratio growth factor,
    e.g. 4.0). Returns the same (mutated) state. Raises `StepTooLarge` if a step tangles the mesh
    anyway."""

    if state.problem.motion_velocity is not None and state.problem.mesh_quality_growth() >= quality_limit:
        state.remesh(target_h)  # swap on the still-valid geometry, before it tangles
    # The step solves at t + dt: a time-dependent velocity (VCell's `sin(t)` front) or source must see
    # that time. Set it on every step, and so also right after a remesh, whose rebuilt problem starts
    # its `sim.t` at 0.
    state.problem.set_time(state.t + float(state.problem.dt.value))
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


def stride_with_remeshing(
    state: ALEState,
    *,
    h: float,
    t_start: float,
    quality_limit: float,
    target_h: float,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    dt_initial: float | None = None,
) -> ALEState:
    """Advance `state` by one **method-of-lines stride** of length `h`, remeshing first if the moving
    mesh has distorted past `quality_limit`. The MOL analogue of `step_with_remeshing`: remesh on the
    still-valid geometry, move the mesh by `h·v` (`advance_mesh`), then `TS`-integrate the
    reaction–diffusion–dilution over `[t_start, t_start+h]` on the now-fixed mesh — accumulating the TS
    step count into `state.steps`. Returns the same (mutated) state; raises `StepTooLarge` if the move
    tangles the mesh anyway."""

    if state.problem.mesh_quality_growth() >= quality_limit:
        state.remesh(target_h)  # swap on the still-valid geometry, before the move would tangle it
        state.problem.dt.value = h  # rebuild preserves dt; keep the stride magnitude explicit
    try:
        state.problem.advance_mesh()
    except MeshQualityError as exc:
        raise StepTooLarge(
            f"a single MOL stride tangled the mesh at t={state.t:.4g}; remeshing cannot rescue a stride "
            f"this large — reduce h (raise motion_steps) (ALE driver subtlety 5)."
        ) from exc
    state.steps += integrate_discrete_problem_stride(
        state.problem, t_start=t_start, t_final=t_start + h, rtol=rtol, atol=atol, dt_initial=dt_initial
    )
    state.t = t_start + h
    return state


def run_moving_with_remeshing(
    md: MathDescription,
    geometry: Geometry,
    config: SolverConfiguration,
    *,
    motion_steps: int,
    target_h: float,
    quality_limit: float = 4.0,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    dt_initial: float | None = None,
) -> ALEState:
    """Method-of-lines on a **moving** subdomain to `config.t_final`, **remeshing** whenever the moving
    mesh distorts past `quality_limit` — the large-deformation analogue of
    `integrate_discrete_problem_moving` (which fails once the mesh tangles).

    The run is `motion_steps` MOL strides of length `h = config.t_final / motion_steps`
    (`stride_with_remeshing`): each strides the mesh by `h·v` and `TS`-integrates the
    reaction–diffusion–dilution over the stride, swapping to a fresh mesh of the deformed
    configuration (conservative state transfer) before any stride that would move an over-distorted
    mesh. So MOL's adaptive high-order integration *within* a stride is kept while remeshing carries
    the geometry through arbitrarily large deformation. Returns the final `ALEState`: `problem.unknown`
    is the field, `remesh_count` the remeshes, `steps` the total TS steps, `t` the time reached. Raises
    `StepTooLarge` if a single stride tangles even a fresh mesh."""

    state = ALEState.initial(md, geometry, config)
    if state.problem.motion_velocity is None:
        raise NotImplementedError("run_moving_with_remeshing needs a moving subdomain; a static one uses run()")
    h = config.t_final / motion_steps
    state.problem.dt.value = h  # the per-stride mesh-move magnitude (advance_mesh moves by dt·v)
    for i in range(motion_steps):
        stride_with_remeshing(
            state,
            h=h,
            t_start=i * h,
            quality_limit=quality_limit,
            target_h=target_h,
            rtol=rtol,
            atol=atol,
            dt_initial=dt_initial,
        )
    return state


def _remesh(problem: DiscreteProblem, target_h: float) -> Mesh:
    """A fresh mesh of the current (deformed) configuration at resolution `target_h`,
    dispatched on the subdomain's codimension:

    - **codim-0 (a moving bulk region).** The deformed boundary loop is recovered
      from the bulk mesh (`BulkBoundaryTrace.boundary_loop()`) and meshed directly:
      `mesh_region_netgen` returns the new bulk mesh, whose boundary is that same
      polyline, so the old and new meshes triangulate the same deformed polygon and the
      bulk transfer in `rebuild_on_mesh` (`remap_bulk_function`) conserves to round-off.
    - **codim-1 (a moving membrane).** Order the deformed loop off the membrane mesh,
      mesh the region it encloses, and extract that region's boundary as the new
      codim-1 submesh.

    The remesher is the **LGPL Netgen** `mesh_region_netgen` (ADR 008), not the GPL gmsh
    `mesh_region` (now test-only) — so the ALE driver carries no gmsh dependency. The
    default (boundary-resampling) path is all the driver needs; the gmsh-only
    `fix_boundary_nodes` interior fast path is not used here. Imported lazily so merely
    importing `backend` neither loads Netgen nor caps its thread pool.
    """

    from vcell_fenics.core.region_remesh_netgen import mesh_region_netgen

    mesh = problem.V.mesh
    gdim = mesh.geometry.dim
    if gdim == 3:
        raise NotImplementedError("remeshing a moving 3D mesh is not implemented yet; use a coarser h or a shorter run")
    if mesh.topology.dim == gdim:
        loop = BulkBoundaryTrace(problem.V).boundary_loop()
        return mesh_region_netgen(loop, target_h)
    if mesh.topology.dim == gdim - 1:
        loop, _order = ordered_membrane_loop(fem.functionspace(mesh, ("Lagrange", 1)))
        bulk = mesh_region_netgen(loop, target_h)
        tdim = bulk.topology.dim
        bulk.topology.create_connectivity(tdim - 1, tdim)
        facets = exterior_facet_indices(bulk.topology)
        membrane, *_ = create_submesh(bulk, tdim - 1, facets)
        return membrane
    raise NotImplementedError(
        f"the ALE driver remeshes a codim-0 bulk or codim-1 membrane in 2D; got topology dim "
        f"{mesh.topology.dim} in {gdim}D"
    )

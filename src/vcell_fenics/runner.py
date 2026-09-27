"""Run a loaded model and write its results bundle — the solver half of the CLI.

``cli`` turns files and flags into a :class:`ModelInput` and :class:`RunOptions`; this module solves
and writes the [ADR 010](../../docs/decisions/010-results-bundle-vtu-zarr.md) bundle
``<out>/<prefix>.fenics/``: one VTU per VCell domain, zarr ``(T, N)`` fields at every output time,
per-time statistics, the manifest — and, under ``provenance/``, the resolved native formalism
(``math.yaml`` / ``geometry.yaml``) and ``summary.json``.

Three solver paths, chosen by the model rather than by a flag:

- equations on **one** subdomain → ``realize`` → ``assemble`` → backward Euler (dt snapped per output
  interval) or adaptive method of lines (outputs recorded from the integrator's own monitor);
- equations on **two** compartments joined by a membrane → ``realize_interface_coupled`` →
  ``integrate_interface_coupled`` (adaptive method of lines over the blocked two-mesh system); with
  **membrane species** as well (equations on the membrane between them — receptor–ligand binding,
  membrane reactions), ``integrate_membrane_coupled`` over both compartments and the membrane, any number
  of species in each;
- a **moving** subdomain (a VCell moving-boundary front: prescribed-velocity motion of the volume it
  encloses) → ``realize`` → ``assemble`` → backward Euler through the ALE driver, remeshing when the
  moving mesh degrades; the bundle records the mesh's coordinates every row and a new segment per
  remesh (ADR 010 §2–3).

Out of scope, each with its own driver: Stokes/FSI, phase field, membrane species on a moving front, and
a lone surface PDE coupled to a single bulk (§1.6.6) — those raise a message naming the limitation.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import dolfinx
import numpy as np
from mpi4py import MPI

import vcell_fenics
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import ImageGeometryWarning, realize
from vcell_fenics.backend.remesh_3d import RemeshWarning
from vcell_fenics.formalism import GeometryDescription, MathDescription, dump_geometry_yaml, dump_yaml
from vcell_fenics.formalism.schema import REGION_SPACE
from vcell_fenics.results import BundleRecorder, BundleWriter, SolverInfo, SourceInfo
from vcell_fenics.results.schema import DomainKind
from vcell_fenics.status import NullReporter, StatusReporter


class RunError(Exception):
    """The request cannot be run as asked (an unsupported model shape or option): a user error."""


@dataclass(frozen=True)
class ModelInput:
    """A loaded model plus where it came from and what the source suggests for the run.

    `suggested_*` are the source's own settings (a VCell simulation's duration, output interval,
    mesh size) when it carried them — used as defaults so a `.vcml` runs with no further flags, and
    always overridable.
    """

    geometry: GeometryDescription
    math: MathDescription
    source: str  # "simtask" | "vcml" | "vcell-yaml" | "native"
    provenance: dict[str, Any] = field(default_factory=dict)
    suggested_t_final: float | None = None
    suggested_output_dt: float | None = None
    suggested_h: float | None = None
    # A VCell SimulationTask says more (ADR 011 §2): an explicit output schedule, a time step, error
    # tolerances, the FEniCSx options block, and where VCell expects the job's results.
    suggested_output_times: tuple[float, ...] | None = None
    suggested_dt: float | None = None
    suggested_rtol: float | None = None
    suggested_atol: float | None = None
    suggested_fe_degree: int | None = None
    suggested_time_integration: str | None = None
    suggested_out_dir: Path | None = None
    suggested_prefix: str | None = None


@dataclass(frozen=True)
class RunOptions:
    """The resolved discretisation for one run — every field already defaulted.

    `output_times` are the rows the bundle will hold, starting with the initial state at 0 and ending
    at `t_final`; they need not be uniform. `dt` is the backward-Euler step (snapped per output
    interval so every output lands on a step); `rtol`/`atol` drive the adaptive method of lines.
    """

    h: float
    dt: float
    t_final: float
    output_times: tuple[float, ...]
    fe_degree: int
    time_integration: str  # "backward_euler" | "method_of_lines"
    rtol: float = 1.0e-6
    atol: float = 1.0e-8

    def __post_init__(self) -> None:
        times = self.output_times
        if not times or times[0] != 0.0 or times[-1] != self.t_final or any(b <= a for a, b in pairwise(times)):
            raise RunError(f"output times must rise strictly from 0 to t_final={self.t_final}, got {list(times)}")


def uniform_output_times(t_final: float, output_dt: float) -> tuple[float, ...]:
    """``0, Δ, 2Δ, …, t_final`` with ``Δ ≈ output_dt`` adjusted so the last time is exactly ``t_final``."""

    n = max(1, round(t_final / output_dt))
    return (*(t_final * k / n for k in range(n)), t_final)  # the last one exactly, not t_final·n/n


def run_model(
    model: ModelInput,
    options: RunOptions,
    out_dir: Path,
    *,
    prefix: str = "results",
    write_fields: bool = True,
    flag_overrides: dict[str, Any] | None = None,
    status: StatusReporter | None = None,
) -> dict[str, Any]:
    """Realize, integrate, and write the bundle ``out_dir/<prefix>.fenics``. Returns the summary.
    ``flag_overrides`` records which settings an explicit flag took over from the model (manifest
    ``solver.overrides``); ``status`` hears progress during the solve and a data event per written row
    (ADR 011 §4) — starting/completed/failed are the caller's, since they bracket more than the solve."""

    # The model's shape (and any refusal of it) is settled before anything is written.
    moving = _moving_subdomains(model)
    coupling = None if moving else _interface_coupling(model)
    _refuse_region_variables(model.math, coupling if coupling is not None and not coupling.membrane_species else None)
    comm = MPI.COMM_WORLD
    bundle = out_dir / f"{prefix}.fenics"
    if comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    writer = BundleWriter(
        bundle,
        comm=comm,
        planned_times=options.output_times,
        source=_source_info(model),
        solver=SolverInfo(
            version=vcell_fenics.__version__,
            dolfinx=str(getattr(dolfinx, "__version__", "unknown")),
            mpi_ranks=comm.size,
            options=asdict(options),
            overrides=dict(flag_overrides or {}),
        ),
        write_fields=write_fields,
    )
    recorder = BundleRecorder(writer, comm=comm)
    reporter: StatusReporter = status if status is not None else NullReporter()
    recorder.on_row(lambda t, row: reporter.data(t, t / options.t_final))
    run_info: dict[str, Any] = {
        "h": options.h,
        "t_final": options.t_final,
        "fe_degree": options.fe_degree,
        "mpi_ranks": comm.size,
        "backend": "ale"
        if moving
        else ("membrane_coupled" if coupling.membrane_species else "interface_coupled")
        if coupling is not None
        else "single_mesh",
    }
    try:
        if moving:
            cells, steps, dt_used, remeshes = _run_moving(model, moving, options, recorder, reporter)
            run_info |= {
                "dt": dt_used,
                "dt_requested": options.dt,
                "time_integration": "backward_euler",
                "steps": steps,
                "remeshes": remeshes,
            }
        elif coupling is not None and coupling.membrane_species:
            cells, steps = _run_membrane_coupled(model, coupling, options, recorder, reporter)
            run_info |= {"time_integration": "method_of_lines", "steps": steps}
        elif coupling is not None:
            cells, steps = _run_interface_coupled(model, coupling, options, recorder, reporter)
            run_info |= {"time_integration": "method_of_lines", "steps": steps}
        else:
            cells, steps, dt_used = _run_single_mesh(model, options, recorder, reporter)
            run_info |= {
                "dt": dt_used,
                "dt_requested": options.dt,
                "time_integration": options.time_integration,
                "steps": steps,
            }
    except BaseException as error:
        writer.finalize("failed", f"{type(error).__name__}: {error}")
        raise

    records = recorder.records
    summary: dict[str, Any] = {
        "source": model.provenance,
        "bundle": str(bundle),
        "geometry": {
            "name": model.geometry.name,
            "dim": model.geometry.dim,
            "cells": cells,
            "measure": sum(recorder.measure(domain) for domain in recorder.domains),
        },
        "run": run_info,
        "species": list(records[0]["species"]) if records else [],
        "outputs": records,
    }
    writer.write_provenance("geometry.yaml", dump_geometry_yaml(model.geometry))
    writer.write_provenance("math.yaml", dump_yaml(model.math))
    writer.write_provenance("summary.json", json.dumps(summary, indent=2) + "\n")
    writer.finalize("completed")
    return summary


def _refuse_region_variables(math: MathDescription, interface: Coupling | None) -> None:
    """Region variables (§1.4.2 T5: a well-mixed species, the membrane potential) are solved by the
    two-compartment solver — on either compartment or the membrane between them (``interface``). Every
    other path refuses them before any bundle is started (#196)."""
    regions = [v for v in math.variables if v.space == REGION_SPACE]
    homes = {interface.inner, interface.outer, interface.membrane} if interface is not None else set()
    unhosted = [f"{v.name!r} on {v.subdomain!r}" for v in regions if v.subdomain not in homes]
    if unhosted:
        raise RunError(
            f"region variables are not solved yet here: {', '.join(unhosted)} — one value per region (a "
            f"well-mixed species or a membrane potential); only the two-compartment solver hosts them so far "
            f"(on either compartment or the membrane between them); see virtualcell/vcell-fenics#196"
        )


def _source_info(model: ModelInput) -> SourceInfo:
    provenance = model.provenance
    file = provenance.get("file") or provenance.get("math_file")
    job_index, task_id = provenance.get("job_index"), provenance.get("task_id")
    return SourceInfo(
        kind=model.source,
        file=str(file) if file is not None else None,
        sim_key=provenance.get("sim_key"),
        job_index=int(job_index) if job_index is not None else None,
        task_id=int(task_id) if task_id is not None else None,
    )


# -- the single-mesh path -----------------------------------------------------------------------------


def _run_single_mesh(
    model: ModelInput, options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int, float]:
    """realize → assemble → integrate, writing a row at every output time. Returns (cells, steps, dt)."""

    comm = MPI.COMM_WORLD
    log(f"realizing geometry {model.geometry.name!r} (dim {model.geometry.dim}) at h = {options.h:g}")
    with _logged_geometry_warnings():
        geometry: Geometry = realize(model.geometry, h=options.h, comm=comm)
    domain = model.math.equations[0].subdomain
    cells = mesh_cell_count(geometry.mesh_of(domain))
    log(f"assembling (fe_degree {options.fe_degree}, {cells} cells)")
    problem = assemble(model.math, geometry, dt=options.dt, fe_degree=options.fe_degree)
    if not isinstance(problem, DiscreteProblem):  # a CoupledGeometry cannot come out of `realize`
        raise RunError("this model assembles to a coupled bulk-surface problem, which the CLI does not drive yet")

    names = problem.variable_name.split(",")
    kind: DomainKind = "membrane" if geometry.kind_of(domain) == "surface" else "volume"
    recorder.add_domain(
        domain,
        kind,
        problem.V.mesh,
        [(name, source) for name, source in zip(names, _channels(problem.unknown, len(names)), strict=True)],
    )
    recorder.open()

    if options.time_integration == "method_of_lines":
        log(f"integrating to t = {options.t_final:g} (method of lines, adaptive; {len(options.output_times)} outputs)")

        def on_output(t: float, snapshot: Any) -> None:
            recorder.capture(t, {domain: _channels(snapshot, len(names))}, progress=t / options.t_final)

        result = integrate_discrete_problem(
            problem,
            t_final=options.t_final,
            rtol=options.rtol,
            atol=options.atol,
            output_times=options.output_times,
            on_output=on_output,
            on_progress=lambda t: status.progress(t / options.t_final, t),
        )
        return cells, int(result.steps), float("nan")

    recorder.capture(0.0, progress=0.0)
    steps, dt_max = _step_backward_euler(
        problem,
        options,
        lambda t: recorder.capture(t, progress=t / options.t_final),
        on_step=lambda t: status.progress(t / options.t_final, t),
    )
    return cells, steps, dt_max


# -- the moving (ALE) path ----------------------------------------------------------------------------

# A remesh when the moving mesh's worst cell-size ratio has grown this much since it was built.
_REMESH_QUALITY_LIMIT = 4.0


def _moving_subdomains(model: ModelInput) -> list[str]:
    return [s.name for s in model.math.subdomains if s.motion.kind == "prescribed"]


def _run_moving(
    model: ModelInput, moving: list[str], options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int, float, int]:
    """Backward Euler through the ALE driver on the moving volume: the mesh moves with the prescribed
    velocity every step (the conservative ALE time term keeps each species' mass), remeshing when it
    has degraded and continuing on the new mesh (conservatively remapped). Returns (cells, steps,
    largest dt, remeshes)."""

    from vcell_fenics.backend.ale import ALEState, StepTooLarge, step_with_remeshing
    from vcell_fenics.backend.discrete import MeshQualityError

    comm = MPI.COMM_WORLD
    domains = sorted({equation.subdomain for equation in model.math.equations})
    if len(moving) != 1 or domains != moving:
        raise RunError(
            f"the moving-boundary path solves species in the one moving volume; the model moves {moving} "
            f"and has equations on {domains}"
        )
    domain = moving[0]
    if options.fe_degree != 1:
        raise RunError(f"the moving-boundary path is P1 (its remesh transfer is); got fe_degree {options.fe_degree}")
    if options.time_integration != "backward_euler":
        log(f"moving mesh: backward Euler with dt = {options.dt:g} (the method of lines has no moving-output path)")
    log(f"realizing geometry {model.geometry.name!r} (dim {model.geometry.dim}) at h = {options.h:g}")
    with _logged_geometry_warnings():
        geometry: Geometry = realize(model.geometry, h=options.h, comm=comm)
    cells = mesh_cell_count(geometry.mesh_of(domain))
    log(f"assembling a moving problem on {domain!r} (fe_degree {options.fe_degree}, {cells} cells)")
    problem = assemble(model.math, geometry, dt=options.dt, fe_degree=options.fe_degree)
    if not isinstance(problem, DiscreteProblem) or problem.motion_velocity is None:
        raise RunError(f"{domain!r} was given a motion, but it did not assemble to a moving problem")
    state = ALEState(problem=problem, md=model.math)
    names = problem.variable_name.split(",")
    recorder.add_domain(domain, "volume", problem.V.mesh, _named(problem, names), moving=True)
    recorder.open()
    recorder.capture(0.0, progress=0.0)

    steps, t, dt_max = 0, 0.0, 0.0
    try:
        for t_out in options.output_times[1:]:
            n = max(1, round((t_out - t) / options.dt))
            dt = (t_out - t) / n
            state.problem.dt.value = dt
            state.t = t
            for _ in range(n):
                remeshes = state.remesh_count
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RemeshWarning)
                    step_with_remeshing(state, quality_limit=_REMESH_QUALITY_LIMIT, target_h=options.h)
                for warning in caught:
                    if issubclass(warning.category, RemeshWarning):
                        log(f"warning: {warning.message}")
                    else:
                        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
                if state.remesh_count != remeshes:
                    new_cells = mesh_cell_count(state.problem.V.mesh)
                    log(f"remeshed at t = {state.t - dt:g} (mesh quality); continuing on {new_cells} cells")
                    recorder.start_segment(
                        {domain: (state.problem.V.mesh, _channels(state.problem.unknown, len(names)))}
                    )
                steps += 1
                status.progress(state.t / options.t_final, state.t)
            t, dt_max = t_out, max(dt_max, dt)
            recorder.capture(t_out, progress=t_out / options.t_final)
    except (StepTooLarge, MeshQualityError) as error:
        raise RunError(f"the moving mesh failed near t = {state.t:g}: {error}") from error
    except NotImplementedError as error:  # a remesh cannot carry boundary conditions yet
        raise RunError(
            f"the moving mesh needed a remesh near t = {state.t:g}, which is not possible here: {error}"
        ) from error
    return cells, steps, dt_max, state.remesh_count


def _named(problem: DiscreteProblem, names: list[str]) -> list[tuple[str, Any]]:
    return list(zip(names, _channels(problem.unknown, len(names)), strict=True))


def _channels(unknown: Any, n_species: int) -> list[Any]:
    """Several species on one mesh are components of one vector unknown; one is the unknown itself."""

    return [unknown] if n_species == 1 else [unknown.sub(k) for k in range(n_species)]


def _step_backward_euler(
    problem: DiscreteProblem,
    options: RunOptions,
    capture: Callable[[float], object],
    on_step: Callable[[float], object] | None = None,
) -> tuple[int, float]:
    """Step through each output interval with a dt snapped so the interval is a whole number of
    steps, capturing at its end. Returns (steps taken, largest dt used)."""

    step, t, dt_max = 0, 0.0, 0.0
    for t_out in options.output_times[1:]:
        n = max(1, round((t_out - t) / options.dt))
        dt = (t_out - t) / n
        if abs(dt - options.dt) > 1e-12 * max(1.0, options.dt):
            log(f"dt {options.dt:g} → {dt:g} over [{t:g}, {t_out:g}] so the output lands on a step")
        problem.dt.value = dt
        for k in range(1, n + 1):
            step += 1
            problem.set_time(t + k * dt)
            problem.step()
            if on_step is not None:
                on_step(t + k * dt)
        t, dt_max = t_out, max(dt_max, dt)
        capture(t_out)
    return step, dt_max


# -- the two-compartment path -------------------------------------------------------------------------


@dataclass(frozen=True)
class Coupling:
    """The two compartments and the membrane between them, for the interface-coupled path.
    ``membrane_species``: the membrane carries equations of its own (the membrane-coupled path)."""

    inner: str
    outer: str
    membrane: str
    background: str | None
    membrane_species: bool = False


def _run_interface_coupled(
    model: ModelInput, coupling: Coupling, options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int]:
    """Two compartments coupled by the membrane flux, any number of species in each (reactions within a
    compartment may couple them); every output time recorded from the integrator's monitor. Returns
    (cells, steps)."""

    from vcell_fenics.backend.interface_coupled import integrate_interface_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled

    # The blocked two-mesh solve is P1 and adaptive-MOL by construction. Say so instead of accepting
    # a flag and quietly solving something else.
    if options.fe_degree != 1:
        raise RunError("the two-compartment solver is P1 only; drop --fe-degree for this model")
    species: dict[str, list[str]] = {coupling.inner: [], coupling.outer: []}
    for eq in model.math.equations:
        species.setdefault(eq.subdomain, []).append(eq.variable)
    regions = {v.name for v in model.math.variables if v.space == REGION_SPACE}
    if options.time_integration != "method_of_lines":
        log("note: the two-compartment solver is method-of-lines; --time-integration is ignored")

    comm = MPI.COMM_WORLD
    log(
        f"realizing interface-coupled geometry {model.geometry.name!r}: inner {coupling.inner!r}, "
        f"outer {coupling.outer!r}, membrane {coupling.membrane!r} at h = {options.h:g}"
    )
    with _logged_geometry_warnings():
        geometry = realize_interface_coupled(
            model.geometry,
            inner_subdomain=coupling.inner,
            outer_subdomain=coupling.outer,
            membrane_subdomain=coupling.membrane,
            interface=coupling.membrane,
            background_subdomain=coupling.background,
            h=options.h,
            comm=comm,
        )
    for name, mesh in ((coupling.inner, geometry.inner_mesh), (coupling.outer, geometry.outer_mesh)):
        recorder.add_domain(name, "volume", mesh, [(variable, None) for variable in species[name]])
    # A membrane potential (a membrane region variable) is written on the membrane, as a constant field.
    domains = [coupling.inner, coupling.outer]
    if species.get(coupling.membrane):
        recorder.add_domain(
            coupling.membrane, "membrane", geometry.membrane_mesh, [(v, None) for v in species[coupling.membrane]]
        )
        domains.append(coupling.membrane)
    recorder.open()

    cells = mesh_cell_count(geometry.inner_mesh) + mesh_cell_count(geometry.outer_mesh)
    log(f"integrating to t = {options.t_final:g} (interface-coupled method of lines, {cells} cells)")

    def source(variable: str, field: Any) -> Any:
        """A field as is; a region variable (one Real value, held by its owning rank) as a constant."""
        if variable not in regions:
            return field
        n_owned = field.function_space.dofmap.index_map.size_local
        value = comm.allreduce(float(np.sum(field.x.array[:n_owned].real)), op=MPI.SUM)
        return lambda x: np.full(x.shape[1], value)

    def on_output(t: float, fields: dict[str, Any]) -> None:
        recorder.capture(
            t,
            {name: [source(v, fields[v]) for v in species[name]] for name in domains},
            progress=t / options.t_final,
        )

    result = integrate_interface_coupled(
        model.math,
        geometry,
        t_final=options.t_final,
        rtol=options.rtol,
        atol=options.atol,
        output_times=options.output_times,
        on_output_fields=on_output,
        on_progress=lambda t: status.progress(t / options.t_final, t),
    )
    return cells, int(result.steps)


def _run_membrane_coupled(
    model: ModelInput, coupling: Coupling, options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int]:
    """Both compartments and the membrane between them, any number of species in each: bulk species in
    each compartment, membrane species (receptor–ligand binding, membrane reactions) on the membrane,
    coupled by the membrane fluxes. Adaptive method of lines (``integrate_membrane_coupled``); every
    output time recorded from the integrator's monitor. Returns (cells, steps)."""

    from vcell_fenics.backend.interface_coupled import integrate_membrane_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled

    if options.fe_degree != 1:
        raise RunError("the membrane-coupled solver is P1 only; drop --fe-degree for this model")
    if options.time_integration != "method_of_lines":
        log("note: the membrane-coupled solver is method-of-lines; --time-integration is ignored")
    species: dict[str, list[str]] = {coupling.inner: [], coupling.outer: [], coupling.membrane: []}
    for eq in model.math.equations:
        species[eq.subdomain].append(eq.variable)
    empty = [name for name in (coupling.inner, coupling.outer) if not species[name]]
    if empty:
        raise RunError(
            "the membrane-coupled solver needs at least one species in each compartment; "
            f"{', '.join(map(repr, empty))} has none"
        )

    comm = MPI.COMM_WORLD
    log(
        f"realizing membrane-coupled geometry {model.geometry.name!r}: inner {coupling.inner!r}, "
        f"outer {coupling.outer!r}, membrane {coupling.membrane!r} at h = {options.h:g}"
    )
    with _logged_geometry_warnings():
        geometry = realize_interface_coupled(
            model.geometry,
            inner_subdomain=coupling.inner,
            outer_subdomain=coupling.outer,
            membrane_subdomain=coupling.membrane,
            interface=coupling.membrane,
            background_subdomain=coupling.background,
            h=options.h,
            comm=comm,
        )
    regions: tuple[tuple[str, DomainKind, Any], ...] = (
        (coupling.inner, "volume", geometry.inner_mesh),
        (coupling.outer, "volume", geometry.outer_mesh),
        (coupling.membrane, "membrane", geometry.membrane_mesh),
    )
    for name, kind, mesh in regions:
        recorder.add_domain(name, kind, mesh, [(variable, None) for variable in species[name]])
    recorder.open()

    cells = sum(mesh_cell_count(mesh) for _, _, mesh in regions[:2])
    log(
        f"integrating to t = {options.t_final:g} (membrane-coupled method of lines, {cells} cells; species: "
        + "; ".join(f"{name} {', '.join(species[name])}" for name, _, _ in regions)
        + ")"
    )

    scalars: dict[int, Any] = {}

    def components(field: Any, n: int) -> list[Any]:
        """The integrator's fields are vector P1, one component per species. A one-component vector space
        has no sub-spaces in DOLFINx, but its dof layout is scalar P1's, so that one is copied into a
        scalar Function on the same mesh (made once per region)."""
        if n > 1:
            return [field.sub(k) for k in range(n)]
        key = id(field.function_space.mesh)
        if key not in scalars:
            scalars[key] = dolfinx.fem.Function(dolfinx.fem.functionspace(field.function_space.mesh, ("Lagrange", 1)))
        scalars[key].x.array[:] = field.x.array
        return [scalars[key]]

    def on_output(t: float, inner: Any, outer: Any, membrane: Any) -> None:
        recorder.capture(
            t,
            {
                coupling.inner: components(inner, len(species[coupling.inner])),
                coupling.outer: components(outer, len(species[coupling.outer])),
                coupling.membrane: components(membrane, len(species[coupling.membrane])),
            },
            progress=t / options.t_final,
        )

    result = integrate_membrane_coupled(
        model.math,
        geometry,
        t_final=options.t_final,
        rtol=options.rtol,
        atol=options.atol,
        output_times=options.output_times,
        on_output=on_output,
        on_progress=lambda t: status.progress(t / options.t_final, t),
    )
    return cells, int(result.steps)


def _interface_coupling(model: ModelInput) -> Coupling | None:
    """Detect the two-compartment shape, or return None to take the single-mesh path.

    The shape is read off the *geometry*, which is the source of truth for spatial structure
    (ADR 007): a SurfaceClass whose `inside`/`outside` are exactly the two subdomains the math's
    equations live on. A third subvolume that carries no equation is the surrounding background,
    which the realizer drops so the outer compartment's own edge becomes the reservoir wall.
    """

    # A region variable (T5) is one value, not a field: a well-mixed species still makes its compartment a
    # modelled one, but a membrane potential does not make the membrane a third (membrane-species) subdomain.
    surfaces = {sd.name for sd in model.math.subdomains if sd.kind == "surface"}
    membrane_regions = {
        (v.name, v.subdomain) for v in model.math.variables if v.space == REGION_SPACE and v.subdomain in surfaces
    }
    subdomains = {eq.subdomain for eq in model.math.equations if (eq.variable, eq.subdomain) not in membrane_regions}
    if len(subdomains) == 3:
        # two compartments plus the membrane between them, each carrying equations: the membrane-coupled
        # shape — a SurfaceClass named after one of the subdomains, separating the other two
        for surface in model.geometry.surfaces:
            if surface.name in subdomains and {surface.inside, surface.outside} == subdomains - {surface.name}:
                compartments = {surface.inside, surface.outside}
                spare = sorted({sv.name for sv in model.geometry.subvolumes} - compartments)
                if len(spare) > 1:
                    raise RunError(
                        f"geometry {model.geometry.name!r} has more unmodelled subvolumes than the coupled "
                        f"solver can drop ({', '.join(spare)}); it supports two compartments plus one background"
                    )
                return Coupling(
                    inner=surface.inside,
                    outer=surface.outside,
                    membrane=surface.name,
                    background=spare[0] if spare else None,
                    membrane_species=True,
                )
        return None
    if len(subdomains) != 2:
        return None
    for surface in model.geometry.surfaces:
        if surface.name in subdomains and (subdomains - {surface.name}) <= {surface.inside, surface.outside}:
            # membrane species with bulk species on ONE side only (a receptor binding an extracellular ligand,
            # nothing modelled in the cytosol): the membrane-coupled solver needs species in both compartments
            (side,) = subdomains - {surface.name}
            empty = ({surface.inside, surface.outside} - {side}).pop()
            raise RunError(
                f"membrane species on {surface.name!r} with bulk species only in {side!r}: the membrane-coupled "
                f"solver needs species in both compartments it separates ({empty!r} has none) — not supported yet"
            )
    for surface in model.geometry.surfaces:
        if {surface.inside, surface.outside} == subdomains:
            volumes = {sv.name for sv in model.geometry.subvolumes}
            spare = sorted(volumes - subdomains)
            if len(spare) > 1:
                raise RunError(
                    f"geometry {model.geometry.name!r} has more unmodelled subvolumes than the coupled "
                    f"solver can drop ({', '.join(spare)}); it supports two compartments plus one background"
                )
            return Coupling(
                inner=surface.inside,
                outer=surface.outside,
                membrane=surface.name,
                background=spare[0] if spare else None,
            )
    raise RunError(
        f"the model's equations span subdomains {sorted(subdomains)}, but geometry {model.geometry.name!r} "
        f"declares no membrane between them (surfaces: {[s.name for s in model.geometry.surfaces] or 'none'})"
    )


# -- small shared helpers ------------------------------------------------------------------------------


def mesh_cell_count(mesh: Any) -> int:
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return int(mesh.comm.allreduce(index_map.size_local, op=MPI.SUM))


@contextmanager
def _logged_geometry_warnings() -> Iterator[None]:
    """Report an image geometry's realization warnings (topology changed at this mesh size, a touching
    pair without a membrane) on the run log, where a VCell user sees them; other warnings pass through."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ImageGeometryWarning)
        yield
    for warning in caught:
        if issubclass(warning.category, ImageGeometryWarning):
            log(f"warning: {warning.message}")
        else:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)


def log(message: str) -> None:
    """Progress to stderr from rank 0 only."""

    if MPI.COMM_WORLD.rank == 0:
        print(f"[vcell-fenics] {message}", file=sys.stderr, flush=True)


__all__ = [
    "Coupling",
    "ModelInput",
    "RunError",
    "RunOptions",
    "log",
    "mesh_cell_count",
    "run_model",
    "uniform_output_times",
]

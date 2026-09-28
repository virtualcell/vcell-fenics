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
    multi = not moving and _is_multi_compartment(model)
    if not multi:
        _refuse_region_variables(model.math)
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
        "backend": "ale" if moving else "multi_compartment" if multi else "single_mesh",
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
        elif multi:
            cells, steps = _run_multi_compartment(model, options, recorder, reporter)
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


def _refuse_region_variables(math: MathDescription) -> None:
    """Region variables (§1.4.2 T5: a well-mixed species, the membrane potential) are solved by the
    multi-compartment solver. The single-mesh and moving paths refuse them before any bundle is started (#196)."""
    unhosted = [f"{v.name!r} on {v.subdomain!r}" for v in math.variables if v.space == REGION_SPACE]
    if unhosted:
        raise RunError(
            f"region variables are not solved yet here: {', '.join(unhosted)} — one value per region (a "
            f"well-mixed species or a membrane potential); only the multi-compartment solver hosts them so far "
            f"(a model with equations on two or more subdomains); see virtualcell/vcell-fenics#196"
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


def _is_multi_compartment(model: ModelInput) -> bool:
    """Whether the model needs the multi-compartment solver: its equations (field species and region
    variables) span two or more subdomains of a geometry with two or more subvolumes — two compartments, a
    compartment and its membrane, a nucleus in a cytosol in extracellular space. One subdomain is the
    single-mesh path's."""

    homes = {eq.subdomain for eq in model.math.equations}
    return len(model.geometry.subvolumes) >= 2 and len(homes) >= 2


def _run_multi_compartment(
    model: ModelInput, options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int]:
    """Any number of compartments and membranes, any number of species and region variables on each,
    coupled by the membrane fluxes (``integrate_multi_compartment``); every output time recorded from the
    integrator's monitor. Returns (cells, steps)."""

    from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment

    if options.fe_degree != 1:
        raise RunError("the multi-compartment solver is P1 only; drop --fe-degree for this model")
    if options.time_integration != "method_of_lines":
        log("note: the multi-compartment solver is method-of-lines; --time-integration is ignored")
    kinds = {sd.name: sd.kind for sd in model.math.subdomains}
    variables: dict[str, list[str]] = {}
    for eq in model.math.equations:
        variables.setdefault(eq.subdomain, []).append(eq.variable)
    regions = {v.name for v in model.math.variables if v.space == REGION_SPACE}

    comm = MPI.COMM_WORLD
    log(
        f"realizing multi-compartment geometry {model.geometry.name!r} at h = {options.h:g}: "
        + ", ".join(f"{name} ({kinds.get(name, '?')})" for name in variables)
    )
    with _logged_geometry_warnings():
        geometry = realize_multi_compartment(
            model.geometry,
            h=options.h,
            compartments={name for name in variables if kinds.get(name) == "volume"},
            comm=comm,
        )
    missing = sorted(name for name in variables if name not in geometry.compartments and name not in geometry.membranes)
    if missing:
        raise RunError(
            f"subdomain(s) {missing} carry equations but geometry {model.geometry.name!r} realizes no "
            f"cells or membrane facets for them at h = {options.h:g}"
        )
    for name, names in variables.items():
        kind: DomainKind = "volume" if name in geometry.compartments else "membrane"
        recorder.add_domain(name, kind, geometry.mesh_of(name), [(variable, None) for variable in names])
    recorder.open()

    cells = mesh_cell_count(geometry.parent_mesh)
    log(
        f"integrating to t = {options.t_final:g} (multi-compartment method of lines, {cells} cells; "
        + "; ".join(f"{name}: {', '.join(names)}" for name, names in variables.items())
        + ")"
    )

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
            {name: [source(v, fields[v]) for v in names] for name, names in variables.items()},
            progress=t / options.t_final,
        )

    result = integrate_multi_compartment(
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
    "ModelInput",
    "RunError",
    "RunOptions",
    "log",
    "mesh_cell_count",
    "run_model",
    "uniform_output_times",
]

"""Run a loaded model and write its results bundle — the solver half of the CLI.

``cli`` turns files and flags into a :class:`ModelInput` and :class:`RunOptions`; this module solves
and writes the [ADR 010](../../docs/decisions/010-results-bundle-vtu-zarr.md) bundle
``<out>/<prefix>.fenics/``: one VTU per VCell domain, zarr ``(T, N)`` fields at every output time,
per-time statistics, the manifest — and, under ``provenance/``, the resolved native formalism
(``math.yaml`` / ``geometry.yaml``) and ``summary.json``.

Two solver paths, chosen by the model rather than by a flag:

- equations on **one** subdomain → ``realize`` → ``assemble`` → backward Euler (dt snapped per output
  interval) or adaptive method of lines (outputs recorded from the integrator's own monitor);
- equations on **two** compartments joined by a membrane → ``realize_interface_coupled`` →
  ``integrate_interface_coupled`` (adaptive method of lines over the blocked two-mesh system).

Out of scope, each with its own driver: moving membranes / ALE remeshing, Stokes/FSI, phase field, and
surface PDEs coupled to a bulk (§1.6.6) — those raise a message naming the limitation.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import dolfinx
from mpi4py import MPI

import vcell_fenics
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism import GeometryDescription, MathDescription, dump_geometry_yaml, dump_yaml
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
    coupling = _interface_coupling(model)
    run_info: dict[str, Any] = {
        "h": options.h,
        "t_final": options.t_final,
        "fe_degree": options.fe_degree,
        "mpi_ranks": comm.size,
        "backend": "interface_coupled" if coupling is not None else "single_mesh",
    }
    try:
        if coupling is not None:
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
    """The two compartments and the membrane between them, for the interface-coupled path."""

    inner: str
    outer: str
    membrane: str
    background: str | None


def _run_interface_coupled(
    model: ModelInput, coupling: Coupling, options: RunOptions, recorder: BundleRecorder, status: StatusReporter
) -> tuple[int, int]:
    """One species per compartment, coupled by the membrane flux; every output time recorded from
    the integrator's monitor. Returns (cells, steps)."""

    from vcell_fenics.backend.interface_coupled import integrate_interface_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled

    # The blocked two-mesh solve is P1 and adaptive-MOL by construction. Say so instead of accepting
    # a flag and quietly solving something else.
    if options.fe_degree != 1:
        raise RunError("the two-compartment solver is P1 only; drop --fe-degree for this model")
    if options.time_integration != "method_of_lines":
        log("note: the two-compartment solver is method-of-lines; --time-integration is ignored")

    comm = MPI.COMM_WORLD
    log(
        f"realizing interface-coupled geometry {model.geometry.name!r}: inner {coupling.inner!r}, "
        f"outer {coupling.outer!r}, membrane {coupling.membrane!r} at h = {options.h:g}"
    )
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
    variable_of = {eq.subdomain: eq.variable for eq in model.math.equations}
    recorder.add_domain(coupling.inner, "volume", geometry.inner_mesh, [(variable_of[coupling.inner], None)])
    recorder.add_domain(coupling.outer, "volume", geometry.outer_mesh, [(variable_of[coupling.outer], None)])
    recorder.open()

    cells = mesh_cell_count(geometry.inner_mesh) + mesh_cell_count(geometry.outer_mesh)
    log(f"integrating to t = {options.t_final:g} (interface-coupled method of lines, {cells} cells)")

    def on_output(t: float, inner: Any, outer: Any) -> None:
        recorder.capture(t, {coupling.inner: [inner], coupling.outer: [outer]}, progress=t / options.t_final)

    result = integrate_interface_coupled(
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

    subdomains = {eq.subdomain for eq in model.math.equations}
    if len(subdomains) != 2:
        return None
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

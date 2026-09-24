"""Command-line runner: a model file in, a results bundle on disk.

One entry point (``python -m vcell_fenics.cli``, or the ``vcell-fenics`` console script)
for the supported ways to say what to solve:

- **VCell solver task** — the ``SimID_<key>_<job>__<task>.simtask.xml`` document VCell hands every
  solver (ADR 011): math, geometry, and the simulation's own settings (end time, output schedule,
  time step, tolerances, mesh size, parameter-scan point). The bundle defaults to VCell's name for the
  job, ``SimID_<key>_<job>_.fenics``, next to the task file.
- **VCell** — a ``.vcml`` biomodel, or the pyvcell YAML pair a biomodel parses into
  (``*_math.yaml`` + ``*_geom.yaml``, as produced by ``scripts/parse_biomodels_to_yaml.py``).
  Both go through :mod:`~vcell_fenics.pyvcell_bridge` (doc §2.6) and
  :func:`~vcell_fenics.pyvcell_bridge.normalize_to_geometry_frame`.
- **Native** — the declarative formalism itself: a MathDescription YAML
  (``docs/modeling/declarative-formalism.md``) plus a GeometryDescription YAML
  (``docs/modeling/geometric-formalism.md``).

Which one a file is, is detected from its content (``--format`` overrides). The rest of the
run is identical for both, because the VCell path *lands on* the native formalism:

    (geometry, math) → realize (Netgen) → assemble → time-step → results bundle

This module owns loading and argv; :mod:`vcell_fenics.runner` owns the solve. Everything is written
to the bundle ``<--out>/<--output-prefix>.fenics/`` (ADR 010) — what a container bind-mount points at:

    mesh/<domain>.vtu, <domain>/<var>   the fields at every output time (VTU mesh + zarr arrays)
    stats/<domain>/<var>                per-time mean / total / min / max
    .zattrs                             the manifest: domains, variables, times written, status
    provenance/math.yaml, geometry.yaml the resolved formalism actually solved — the thing to diff
                                        when a VCell import surprises you
    provenance/summary.json             run configuration and per-species statistics

``python -m vcell_fenics.results`` summarises a bundle.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from mpi4py import MPI

from vcell_fenics.backend.diagnostics import NonlinearTermError, SolveError
from vcell_fenics.backend.realize import RealizationError
from vcell_fenics.formalism import (
    FormalismLoadError,
    GeometryDescription,
    MathDescription,
    load_dict,
    load_geometry_dict,
)
from vcell_fenics.formalism.validator import FormalismValidationError
from vcell_fenics.pyvcell_bridge.importer import VcellImportError
from vcell_fenics.pyvcell_bridge.overrides import OverrideError
from vcell_fenics.pyvcell_bridge.simtask import (
    SimulationTaskError,
    check_supported,
    front_velocity_dependence,
    read_simtask,
)
from vcell_fenics.runner import ModelInput, RunError, RunOptions, run_model, uniform_output_times
from vcell_fenics.runner import log as _log
from vcell_fenics.status import Fanout, MessagingConfig, RestWorkerEvents, StatusReporter, StdoutMarkers, isolate_stdout

# Errors that mean "the model or the request is wrong", not "the code is broken": reported as a
# one-line `error: …` with exit status 2, no traceback.
_USER_ERRORS = (
    FormalismLoadError,
    FormalismValidationError,
    RealizationError,
    RunError,
    SimulationTaskError,
    OverrideError,
    VcellImportError,  # math the formalism cannot represent (stochastic/particle, …): named, not a traceback
    SolveError,
    NonlinearTermError,  # a nonlinear model under backward Euler: the message names the fix (use MOL)
    NotImplementedError,
    ValueError,
    KeyError,
    FileNotFoundError,
)


class CliError(Exception):
    """A usage/model problem to report as a single `error:` line (no traceback)."""


# ---------------------------------------------------------------------------
# Input: VCML / VCell YAML / native formalism → (GeometryDescription, MathDescription)
# ---------------------------------------------------------------------------


def detect_format(document: object) -> str:
    """Classify a parsed math *or* geometry YAML document as ``"vcell"`` or ``"native"``.

    The two formalisms are structurally distinct at the top level, so this needs no guessing:
    a VCell (pyvcell) math description has `compartment_subdomains` / `membrane_subdomains` /
    `constants`, a native one has `equations` (or the `math_description:` envelope); a VCell
    geometry has `surface_classes` and `subvolume_type`-tagged subvolumes, a native one has
    `surfaces` and `type`/`expression`.
    """

    if not isinstance(document, dict):
        raise CliError("expected a YAML mapping at the top level")
    keys = set(document)
    if keys & {"math_description", "geometry_description", "equations", "subdomains"}:
        return "native"
    if keys & {"compartment_subdomains", "membrane_subdomains", "constants", "functions"}:
        return "vcell"
    if "surface_classes" in keys:
        return "vcell"
    if "surfaces" in keys:
        return "native"
    subvolumes = document.get("subvolumes")
    if isinstance(subvolumes, list) and subvolumes and isinstance(subvolumes[0], dict):
        if {"subvolume_type", "analytic_expr", "handle"} & set(subvolumes[0]):
            return "vcell"
        if {"type", "expression"} & set(subvolumes[0]):
            return "native"
    raise CliError(
        "cannot tell whether this document is a VCell or a native vcell-fenics description; "
        "pass --format vcell|native explicitly"
    )


def load_vcml(path: Path, *, application: str | None = None, simulation: str | None = None) -> ModelInput:
    """Load a VCell ``.vcml`` biomodel and import the chosen application's math + geometry.

    An application is picked by `--application`, else the single one present (ambiguity is an
    error, not a silent choice). Its first simulation — or `--simulation` — supplies the default
    duration, output interval, and mesh size.
    """

    from pyvcell.vcml.vcml_reader import VcmlReader  # the XML reader; only needed on this path

    biomodel = VcmlReader.biomodel_from_file(str(path))
    apps = list(biomodel.applications or [])
    if not apps:
        raise CliError(f"{path}: biomodel {biomodel.name!r} has no applications")
    if application is not None:
        matches = [a for a in apps if a.name == application]
        if not matches:
            raise CliError(f"{path}: no application named {application!r} (have: {', '.join(a.name for a in apps)})")
        app = matches[0]
    elif len(apps) == 1:
        app = apps[0]
    else:
        raise CliError(
            f"{path}: biomodel {biomodel.name!r} has {len(apps)} applications "
            f"({', '.join(a.name for a in apps)}); pick one with --application"
        )
    if app.math_description is None:
        raise CliError(f"{path}: application {app.name!r} carries no generated math description")
    if app.geometry is None:
        raise CliError(f"{path}: application {app.name!r} carries no geometry")

    sims = list(app.simulations or [])
    sim = None
    if simulation is not None:
        sim = next((s for s in sims if s.name == simulation), None)
        if sim is None:
            raise CliError(
                f"{path}: application {app.name!r} has no simulation named {simulation!r} "
                f"(have: {', '.join(s.name for s in sims) or 'none'})"
            )
    elif sims:
        sim = sims[0]

    gd, md = _import_vcell(app.geometry, app.math_description)
    provenance: dict[str, Any] = {
        "kind": "vcml",
        "file": str(path),
        "biomodel": biomodel.name,
        "application": app.name,
        "simulation": sim.name if sim is not None else None,
    }
    return ModelInput(
        geometry=gd,
        math=md,
        source="vcml",
        provenance=provenance,
        suggested_t_final=float(sim.duration) if sim is not None and sim.duration is not None else None,
        suggested_output_dt=(
            float(sim.output_time_step) if sim is not None and sim.output_time_step is not None else None
        ),
        suggested_h=_h_from_mesh_size(gd, sim.mesh_size) if sim is not None else None,
    )


def load_pair(math_path: Path, geometry_path: Path, *, fmt: str = "auto") -> ModelInput:
    """Load a math + geometry YAML pair, in either formalism (auto-detected per file)."""

    math_doc = yaml.safe_load(math_path.read_text())
    geom_doc = yaml.safe_load(geometry_path.read_text())
    math_fmt = detect_format(math_doc) if fmt == "auto" else fmt
    geom_fmt = detect_format(geom_doc) if fmt == "auto" else fmt
    if math_fmt != geom_fmt:
        raise CliError(
            f"{math_path.name} looks like a {math_fmt} description but {geometry_path.name} looks like "
            f"{geom_fmt}; the pair must be in one formalism (or pass --format)"
        )

    if math_fmt == "vcell":
        import pyvcell.vcml.models_geometry as gmod
        import pyvcell.vcml.models_math as mmod

        gd, md = _import_vcell(
            gmod.Geometry.model_validate(geom_doc),
            mmod.MathDescription.model_validate(math_doc),
        )
    else:
        gd = load_geometry_dict(geom_doc)
        md = load_dict(math_doc)
        if md.geometry != gd.name:
            raise CliError(
                f"the math description binds geometry {md.geometry!r} but {geometry_path.name} is named "
                f"{gd.name!r} — the names are the contract between the two documents"
            )

    return ModelInput(
        geometry=gd,
        math=md,
        source="vcell-yaml" if math_fmt == "vcell" else "native",
        provenance={"kind": math_fmt, "math_file": str(math_path), "geometry_file": str(geometry_path)},
    )


def load_simtask(path: Path) -> ModelInput:
    """Load a VCell SimulationTask: this job's math (its parameter-scan point applied) and geometry, and
    the simulation's settings as the run's defaults (ADR 011 §2). Refuses what the solver would
    mis-solve (moving boundaries, field data, steady tasks, a non-zero start time)."""

    task = read_simtask(path)
    for warning in check_supported(task):
        _log(f"warning: {warning}")
    front = task.front_velocities[0] if task.front_velocities else None
    gd, md = _import_vcell(task.geometry, task.math, front_velocity=front)
    times = task.output_times()
    options = task.fenicsx
    if task.num_processors > 1 and MPI.COMM_WORLD.size == 1:
        _log(f"note: the task asks for {task.num_processors} processors; launch under mpiexec to use them")
    provenance: dict[str, Any] = {
        "kind": "simtask",
        "file": str(path),
        "simulation": task.sim_name,
        "sim_key": task.sim_key,
        "job_index": task.job_index,
        "task_id": task.task_id,
        "solver": task.solver,
        "math_overrides": task.resolved_overrides,
        "number_processors": task.num_processors,
    }
    if front is not None:
        provenance["moving_boundary"] = {
            "membrane": front.surface_name,
            "velocity": [str(front.velocity_x), str(front.velocity_y)]
            + ([str(front.velocity_z)] if gd.dim == 3 else []),
            "velocity_dependence": front_velocity_dependence(task),
        }
    max_size = options.max_element_size if options is not None else None
    return ModelInput(
        geometry=gd,
        math=md,
        source="simtask",
        provenance=provenance,
        suggested_t_final=times[-1],
        suggested_h=max_size if max_size is not None else _h_from_mesh_size(gd, task.mesh_size),
        suggested_output_times=times,
        suggested_dt=task.dt_default,
        suggested_rtol=task.rel_tol,
        suggested_atol=task.abs_tol,
        suggested_fe_degree=options.element_degree if options is not None else None,
        # Real VCell kinetics are routinely nonlinear, which backward Euler cannot lower (ADR 011 §2); a
        # moving front runs backward Euler (the ALE driver), the one path that moves the mesh with outputs.
        suggested_time_integration=(options.time_integration if options is not None else None)
        or ("backward_euler" if front is not None else "method_of_lines"),
        suggested_out_dir=path.resolve().parent,
        suggested_prefix=task.job_prefix,
    )


def _import_vcell(
    vcml_geometry: Any, vcml_math: Any, *, front_velocity: Any = None
) -> tuple[GeometryDescription, MathDescription]:
    """VCell geometry + math → the formalism pair, in the geometry's own frame.

    `normalize_to_geometry_frame` is not optional: a VCell math description written for a 3D box
    keeps `z` references that a 2D geometry has no coordinate for, and binding them to the
    geometry's origin is what makes the imported model realizable.
    """

    from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

    gd = import_geometry(vcml_geometry)
    md = import_math_description(vcml_math, geometry=gd.name, dim=gd.dim, front_velocity=front_velocity)
    return normalize_to_geometry_frame(gd, md)


def _h_from_mesh_size(gd: GeometryDescription, mesh_size: Sequence[int] | None) -> float | None:
    """Translate a VCell finite-volume grid (`mesh_size`) into a comparable FEM element size:
    the finest cell spacing over the geometry's *spatial* axes."""

    if not mesh_size:
        return None
    spacings = [
        float(gd.extent[axis]) / float(mesh_size[axis])
        for axis in range(min(gd.dim, len(mesh_size)))
        if mesh_size[axis] > 0
    ]
    return min(spacings) if spacings else None


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


def resolve_options(model: ModelInput, args: argparse.Namespace) -> RunOptions:
    """Merge explicit flags over the source's own suggestions over the built-in defaults."""

    return _resolve(model, args)[0]


def _resolve(model: ModelInput, args: argparse.Namespace) -> tuple[RunOptions, dict[str, Any]]:
    """:func:`resolve_options`, plus which settings an explicit flag took over from the model
    (recorded in the bundle manifest as ``solver.overrides``)."""

    overrides: dict[str, Any] = {}

    def pick(name: str, flag: Any, suggested: Any, default: Any) -> Any:
        if flag is not None:
            if suggested is not None and flag != suggested:
                overrides[name] = {"flag": flag, "model": suggested}
                _log(f"--{name.replace('_', '-')} {flag} overrides the model's {suggested}")
            return flag
        return suggested if suggested is not None else default

    t_final = pick("t_final", args.t_final, model.suggested_t_final, None)
    if t_final is None:
        raise CliError("--t-final is required (the model carries no simulation duration)")
    if t_final <= 0.0:
        raise CliError(f"--t-final must be positive, got {t_final}")

    # The output schedule: an explicit --t-final/--output-dt makes it uniform; otherwise the model's
    # own schedule (a SimulationTask's, possibly non-uniform) is kept as is.
    if model.suggested_output_times is not None and args.t_final is None and args.output_dt is None:
        output_times = model.suggested_output_times
    else:
        output_dt = pick("output_dt", args.output_dt, model.suggested_output_dt, None)
        if output_dt is None or output_dt > t_final:
            output_dt = t_final
        if output_dt <= 0.0:
            raise CliError(f"--output-dt must be positive, got {output_dt}")
        output_times = uniform_output_times(float(t_final), float(output_dt))

    # dt defaults to the output interval — one step per snapshot. That is a *coarse* backward-Euler
    # step for anything stiff, so it is reported in the run header and in summary.json.
    smallest_interval = min(b - a for a, b in pairwise(output_times))
    dt = pick("dt", args.dt, model.suggested_dt, smallest_interval)
    if dt <= 0.0:
        raise CliError(f"--dt must be positive, got {dt}")

    h = pick("h", args.h, model.suggested_h, None)
    if h is None:
        # Scale-aware fallback: ~32 elements across the narrowest spatial axis. An absolute
        # default would be meaningless for a geometry measured in µm vs cm.
        spatial = [model.geometry.extent[axis] for axis in range(max(model.geometry.dim, 1))]
        h = min(spatial) / 32.0 if spatial else 0.05
    if h <= 0.0:
        raise CliError(f"--h must be positive, got {h}")

    fe_degree = pick("fe_degree", args.fe_degree, model.suggested_fe_degree, 1)
    time_integration = pick(
        "time_integration", args.time_integration, model.suggested_time_integration, "backward_euler"
    )
    if time_integration not in ("backward_euler", "method_of_lines"):
        raise CliError(f"--time-integration must be backward_euler or method_of_lines, got {time_integration!r}")

    options = RunOptions(
        h=float(h),
        dt=float(dt),
        t_final=float(output_times[-1]),
        output_times=tuple(float(t) for t in output_times),
        fe_degree=int(fe_degree),
        time_integration=str(time_integration),
        rtol=float(model.suggested_rtol) if model.suggested_rtol is not None else 1.0e-6,
        atol=float(model.suggested_atol) if model.suggested_atol is not None else 1.0e-8,
    )
    return options, overrides


# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcell-fenics",
        description="Solve a VCell or native vcell-fenics model with FEniCSx and write a results bundle (ADR 010).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  vcell-fenics --simtask SimID_123_0__0.simtask.xml\n"
            "  vcell-fenics --vcml model.vcml --out results\n"
            "  vcell-fenics --math m_math.yaml --geometry m_geom.yaml --t-final 1.0 --out results\n"
        ),
    )
    source = parser.add_argument_group("model input")
    source.add_argument(
        "--simtask", type=Path, help="a VCell SimulationTask document (SimID_<key>_<job>__<task>.simtask.xml)"
    )
    source.add_argument("--vcml", type=Path, help="a VCell biomodel (.vcml)")
    source.add_argument("--math", type=Path, help="math description YAML (native formalism or VCell/pyvcell)")
    source.add_argument("--geometry", type=Path, help="geometry description YAML, in the same formalism as --math")
    source.add_argument("--application", help=".vcml only: which application to run (default: the only one)")
    source.add_argument("--simulation", help=".vcml only: which simulation supplies duration/output step/mesh size")
    source.add_argument(
        "--format",
        choices=("auto", "native", "vcell"),
        default="auto",
        help="override content-based detection of the --math/--geometry formalism",
    )

    discretisation = parser.add_argument_group("discretisation (defaults come from the model where it says)")
    discretisation.add_argument("--h", type=float, help="target mesh element size")
    discretisation.add_argument("--dt", type=float, help="time step (backward Euler); default: --output-dt")
    discretisation.add_argument("--t-final", type=float, help="end time")
    discretisation.add_argument("--output-dt", type=float, help="interval between written snapshots")
    discretisation.add_argument("--fe-degree", type=int, help="Lagrange degree (default: the model's, else 1)")
    discretisation.add_argument(
        "--time-integration",
        choices=("backward_euler", "method_of_lines"),
        help=(
            "fixed-step backward Euler, or adaptive PETSc TS (BDF; required for nonlinear kinetics). "
            "Default: method_of_lines for a --simtask, else backward_euler"
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--out", type=Path, help="results directory (default: the --simtask's own directory, else ./out)"
    )
    output.add_argument(
        "--output-prefix",
        help="the bundle is <--out>/<prefix>.fenics (default: SimID_<key>_<job>_ for a --simtask, else results)",
    )
    output.add_argument(
        "--no-fields", action="store_true", help="write the meshes and per-time statistics only, no field arrays"
    )
    output.add_argument("--traceback", action="store_true", help="show the full traceback on a model error")

    vcell = parser.add_argument_group("VCell status reporting (ADR 011 §4)")
    vcell.add_argument(
        "--vc-print-status",
        action="store_true",
        help="write [[[progress:…%%]]] / [[[data:t]]] markers to stdout (everything else goes to stderr)",
    )
    vcell.add_argument(
        "--vc-send-status-config",
        type=Path,
        metavar="FILE",
        help="post REST WorkerEvents to the broker described in FILE (the Langevin properties format)",
    )
    vcell.add_argument("-tid", type=int, help="the task id VCell's batch system appends (checked against the task)")
    return parser


def load_model(args: argparse.Namespace) -> ModelInput:
    if args.simtask is not None:
        if args.vcml is not None or args.math is not None or args.geometry is not None:
            raise CliError("--simtask, --vcml and --math/--geometry are alternatives; pass one of them")
        return load_simtask(args.simtask)
    if args.vcml is not None:
        if args.math is not None or args.geometry is not None:
            raise CliError("--vcml and --math/--geometry are alternatives; pass one or the other")
        return load_vcml(args.vcml, application=args.application, simulation=args.simulation)
    if args.math is None or args.geometry is None:
        raise CliError("give a model: --simtask FILE, --vcml FILE, or --math FILE --geometry FILE")
    return load_pair(args.math, args.geometry, fmt=args.format)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one model, reporting status as ADR 011 §4 describes. Exit status: 0 success, 2 a model or
    usage error, 1 a crash, 143 terminated (SIGTERM — a Slurm cancel or timeout)."""

    args = build_parser().parse_args(argv)
    comm = MPI.COMM_WORLD
    try:
        status = _status_reporter(args, comm)
    except (OSError, ValueError) as error:
        _fail(f"status reporting: {error}", args.traceback)
        return 2
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate)
    status.starting()
    try:
        model = load_model(args)
        _log(f"loaded {model.source} model: {model.provenance}")
        task_id = model.provenance.get("task_id")
        if args.tid is not None and task_id is not None and args.tid != task_id:
            _log(f"warning: -tid {args.tid} differs from the task's TaskId {task_id}; using the document's")
        options, overrides = _resolve(model, args)
        out = args.out or model.suggested_out_dir or Path("out")
        prefix = args.output_prefix or model.suggested_prefix or "results"
        summary = run_model(
            model,
            options,
            out,
            prefix=prefix,
            write_fields=not args.no_fields,
            flag_overrides=overrides,
            status=status,
        )
    except CliError as error:
        return _report_failure(status, str(error), args.traceback, code=2)
    except _USER_ERRORS as error:
        return _report_failure(status, f"{type(error).__name__}: {error}", args.traceback, code=2)
    except _Terminated:
        return _report_failure(status, "terminated by SIGTERM", False, code=143)
    except Exception as error:  # a crash: report it, and never leave the other ranks waiting
        code = _report_failure(status, f"{type(error).__name__}: {error}", True, code=1)
        if comm.size > 1:
            comm.Abort(code)
        return code
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    status.completed(options.t_final)  # only now: the bundle's manifest already says "completed"
    if comm.rank == 0:
        final = summary["outputs"][-1]["species"]
        for name, stats in final.items():
            _log(f"t = {summary['outputs'][-1]['t']:g}  {name}: total {stats['total']:.6g}, max {stats['max']:.6g}")
        _log(f"wrote {summary['bundle']}")
    return 0


class _Terminated(BaseException):
    """Raised from the SIGTERM handler, so the run unwinds (the bundle is marked failed) and exits 143."""


def _terminate(signum: int, frame: object) -> None:
    raise _Terminated("terminated by SIGTERM")


def _status_reporter(args: argparse.Namespace, comm: MPI.Comm) -> Fanout:
    """The status channels the flags ask for. Markers go to a private copy of stdout and everything
    else — on every rank, since ``mpiexec`` merges all ranks' stdout — to stderr (ADR 011 §4)."""

    reporters: list[StatusReporter] = []
    if args.vc_print_status:
        markers = isolate_stdout()
        if comm.rank == 0:
            reporters.append(StdoutMarkers(markers))
    if args.vc_send_status_config is not None and comm.rank == 0:
        reporters.append(RestWorkerEvents(MessagingConfig.from_properties(args.vc_send_status_config)))
    return Fanout(reporters)


def _report_failure(status: Fanout, message: str, show_traceback: bool, *, code: int) -> int:
    _fail(message, show_traceback)
    status.failed(message, status.last_t, status.last_fraction)
    return code


def _fail(message: str, show_traceback: bool) -> None:
    if show_traceback:
        import traceback

        traceback.print_exc()
    if MPI.COMM_WORLD.rank == 0:
        print(f"error: {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

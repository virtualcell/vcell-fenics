"""Command-line runner: a model file in, solved fields on disk.

One entry point (``python -m vcell_fenics.cli``, or the ``vcell-fenics`` console script)
for the two supported ways to say what to solve:

- **VCell** — a ``.vcml`` biomodel, or the pyvcell YAML pair a biomodel parses into
  (``*_math.yaml`` + ``*_geom.yaml``, as produced by ``scripts/parse_biomodels_to_yaml.py``).
  Both go through :mod:`~vcell_fenics.pyvcell_bridge` (doc §2.6) and
  :func:`~vcell_fenics.pyvcell_bridge.normalize_to_geometry_frame`.
- **Native** — the declarative formalism itself: a MathDescription YAML
  (``docs/modeling/declarative-formalism.md``) plus a GeometryDescription YAML
  (``docs/modeling/geometric-formalism.md``).

Which one a file is, is detected from its content (``--format`` overrides). The rest of the
run is identical for both, because the VCell path *lands on* the native formalism:

    (geometry, math) → realize (Netgen) → assemble → time-step → XDMF + summary.json

Everything is written under ``--out`` (default ``./out``), which is what a container bind-mount
points at:

    math.yaml / geometry.yaml   the resolved formalism actually solved — provenance, and the
                                thing to diff when a VCell import surprises you
    fields.xdmf + fields.h5     one ParaView time series, one grid per species (a two-compartment
                                run writes inner/outer.xdmf instead — two meshes, two files)
    summary.json                run configuration, mesh size, and per-species
                                total / mean / min / max at every output time

Two solver paths, chosen by the model rather than by a flag: equations on **one** subdomain go
through ``backend.assemble`` on a single mesh (backward Euler with a full time series, or adaptive
method-of-lines); equations on **two** compartments joined by a membrane go through
``realize_interface_coupled`` + ``integrate_interface_coupled`` (final state only — the blocked
adaptive integrator has no output-time hook).

Out of scope here, each having its own driver: moving membranes / ALE remeshing, the Stokes and
FSI stack, phase field, and surface PDEs coupled to a bulk (§1.6.6). Those raise a message naming
the limitation rather than silently mis-solving.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import ufl
import yaml
from dolfinx import fem
from dolfinx.io import XDMFFile
from mpi4py import MPI

from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.diagnostics import NonlinearTermError, SolveError
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.geometry import Geometry
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import RealizationError, realize
from vcell_fenics.formalism import (
    FormalismLoadError,
    GeometryDescription,
    MathDescription,
    dump_geometry_yaml,
    dump_yaml,
    load_dict,
    load_geometry_dict,
)
from vcell_fenics.formalism.validator import FormalismValidationError

# Errors that mean "the model or the request is wrong", not "the code is broken": reported as a
# one-line `error: …` with exit status 2, no traceback.
_USER_ERRORS = (
    FormalismLoadError,
    FormalismValidationError,
    RealizationError,
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


@dataclass(frozen=True)
class ModelInput:
    """A loaded model plus where it came from and what the source suggests for the run.

    `suggested_*` are the VCell simulation's own settings (duration, output interval, mesh
    size) when the source carried them — used as defaults so a `.vcml` runs with no further
    flags, and always overridable.
    """

    geometry: GeometryDescription
    math: MathDescription
    source: str  # "vcml" | "vcell-yaml" | "native"
    provenance: dict[str, Any] = field(default_factory=dict)
    suggested_t_final: float | None = None
    suggested_output_dt: float | None = None
    suggested_h: float | None = None


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


def _import_vcell(vcml_geometry: Any, vcml_math: Any) -> tuple[GeometryDescription, MathDescription]:
    """VCell geometry + math → the formalism pair, in the geometry's own frame.

    `normalize_to_geometry_frame` is not optional: a VCell math description written for a 3D box
    keeps `z` references that a 2D geometry has no coordinate for, and binding them to the
    geometry's origin is what makes the imported model realizable.
    """

    from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

    gd = import_geometry(vcml_geometry)
    md = import_math_description(vcml_math, geometry=gd.name, dim=gd.dim)
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


@dataclass(frozen=True)
class RunOptions:
    """The resolved discretisation for one run — every field already defaulted."""

    h: float
    dt: float
    t_final: float
    output_dt: float
    fe_degree: int
    time_integration: str


def resolve_options(model: ModelInput, args: argparse.Namespace) -> RunOptions:
    """Merge explicit flags over the source's own suggestions over the built-in defaults."""

    t_final = args.t_final if args.t_final is not None else model.suggested_t_final
    if t_final is None:
        raise CliError("--t-final is required (the model carries no simulation duration)")
    if t_final <= 0.0:
        raise CliError(f"--t-final must be positive, got {t_final}")

    output_dt = args.output_dt if args.output_dt is not None else model.suggested_output_dt
    if output_dt is None or output_dt > t_final:
        output_dt = t_final
    if output_dt <= 0.0:
        raise CliError(f"--output-dt must be positive, got {output_dt}")

    # dt defaults to the output interval — one step per snapshot. That is a *coarse* backward-Euler
    # step for anything stiff, so it is reported in the run header and in summary.json.
    dt = args.dt if args.dt is not None else output_dt
    if dt <= 0.0:
        raise CliError(f"--dt must be positive, got {dt}")

    h = args.h if args.h is not None else model.suggested_h
    if h is None:
        # Scale-aware fallback: ~32 elements across the narrowest spatial axis. An absolute
        # default would be meaningless for a geometry measured in µm vs cm.
        spatial = [model.geometry.extent[axis] for axis in range(max(model.geometry.dim, 1))]
        h = min(spatial) / 32.0 if spatial else 0.05
    if h <= 0.0:
        raise CliError(f"--h must be positive, got {h}")

    if args.time_integration not in ("backward_euler", "method_of_lines"):
        raise CliError(f"--time-integration must be backward_euler or method_of_lines, got {args.time_integration!r}")

    return RunOptions(
        h=float(h),
        dt=float(dt),
        t_final=float(t_final),
        output_dt=float(output_dt),
        fe_degree=int(args.fe_degree),
        time_integration=str(args.time_integration),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class _Recorder:
    """Writes one mesh's fields to an XDMF time series and accumulates their diagnostics.

    `channels` pairs each species name with the *live* Function (or sub-view) holding its state:
    several species coupled on one mesh are components of a single vector `unknown`, so each is
    interpolated into its own P1 scalar Function. That gives ParaView one named grid per species,
    and keeps XDMF writable at `--fe-degree 2` (XDMF carries P1 point data only).

    Every reported number is *reduced* across ranks — the integral ∫u dx and its mesh mean, plus
    min/max over owned dofs — so `mpirun -n N` reports the same values as a serial run.
    """

    def __init__(
        self,
        mesh: Any,
        channels: Sequence[tuple[str, Any]],
        out_dir: Path,
        filename: str,
        *,
        write_fields: bool,
    ) -> None:
        self._comm = mesh.comm
        self._sources = [source for _, source in channels]
        self._names = [name for name, _ in channels]

        space = fem.functionspace(mesh, ("Lagrange", 1))
        self._fields = [fem.Function(space, name=name) for name in self._names]
        dx = ufl.Measure("dx", domain=mesh)
        self._integrals = [fem.form(u * dx) for u in self._fields]
        self._volume = self._reduce(float(fem.assemble_scalar(fem.form(1.0 * dx)).real), MPI.SUM)
        self._records: list[dict[str, Any]] = []

        self._xdmf: XDMFFile | None = None
        if write_fields:
            self._xdmf = XDMFFile(self._comm, str(out_dir / filename), "w")
            self._xdmf.write_mesh(mesh)

    def _reduce(self, value: float, op: MPI.Op) -> float:
        return float(self._comm.allreduce(value, op=op))

    def capture(self, t: float) -> dict[str, Any]:
        """Interpolate the current state, write it, and record its diagnostics at time `t`."""

        stats: dict[str, Any] = {}
        for name, source, out, integral in zip(self._names, self._sources, self._fields, self._integrals, strict=True):
            out.interpolate(source)
            if self._xdmf is not None:
                self._xdmf.write_function(out, t)
            owned = out.x.array[: out.function_space.dofmap.index_map.size_local]
            total = self._reduce(float(fem.assemble_scalar(integral).real), MPI.SUM)
            stats[name] = {
                "total": total,
                "mean": total / self._volume if self._volume > 0.0 else float("nan"),
                "min": self._reduce(float(np.min(owned)) if owned.size else np.inf, MPI.MIN),
                "max": self._reduce(float(np.max(owned)) if owned.size else -np.inf, MPI.MAX),
            }
        self._records.append({"t": t, "species": stats})
        return stats

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    @property
    def volume(self) -> float:
        return self._volume

    def close(self) -> None:
        if self._xdmf is not None:
            self._xdmf.close()
            self._xdmf = None


def _single_mesh_recorder(problem: DiscreteProblem, out_dir: Path, *, write_fields: bool) -> _Recorder:
    names = problem.variable_name.split(",")
    channels = [(name, problem.unknown if len(names) == 1 else problem.unknown.sub(k)) for k, name in enumerate(names)]
    return _Recorder(problem.V.mesh, channels, out_dir, "fields.xdmf", write_fields=write_fields)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_model(model: ModelInput, options: RunOptions, out_dir: Path, *, write_fields: bool = True) -> dict[str, Any]:
    """Realize, integrate, and write everything under `out_dir`. Returns the summary.

    Two backends, chosen by the model itself: a model whose equations live on **one** subdomain
    goes through `assemble` + backward Euler / method-of-lines on a single mesh; one spanning
    **two** compartments joined by a membrane goes through `realize_interface_coupled` +
    `integrate_interface_coupled` (the two-mesh blocked solve, cross-validated against VCell's
    finite-volume solver in `cross_validation/`).
    """

    comm = MPI.COMM_WORLD
    is_root = comm.rank == 0

    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "geometry.yaml").write_text(dump_geometry_yaml(model.geometry))
        (out_dir / "math.yaml").write_text(dump_yaml(model.math))
    comm.barrier()

    coupling = _interface_coupling(model)
    run_info: dict[str, Any] = {
        "h": options.h,
        "t_final": options.t_final,
        "fe_degree": options.fe_degree,
        "mpi_ranks": comm.size,
        "backend": "interface_coupled" if coupling is not None else "single_mesh",
    }
    if coupling is not None:
        recorders, cells, steps = _run_interface_coupled(model, coupling, options, out_dir, write_fields=write_fields)
        run_info |= {"time_integration": "method_of_lines", "steps": steps}
    else:
        recorders, cells, steps, dt_used = _run_single_mesh(model, options, out_dir, write_fields=write_fields)
        run_info |= {
            "dt": dt_used,
            "dt_requested": options.dt,
            "output_dt": options.output_dt,
            "time_integration": options.time_integration,
            "steps": steps,
        }

    summary: dict[str, Any] = {
        "source": model.provenance,
        "geometry": {
            "name": model.geometry.name,
            "dim": model.geometry.dim,
            "cells": cells,
            "measure": sum(r.volume for r in recorders),
        },
        "run": run_info,
        "species": [name for r in recorders for name in r.records[0]["species"]],
        "outputs": _merge_records(recorders),
    }
    if is_root:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    comm.barrier()
    return summary


def _run_single_mesh(
    model: ModelInput, options: RunOptions, out_dir: Path, *, write_fields: bool
) -> tuple[list[_Recorder], int, int, float]:
    """The one-mesh path: realize → assemble → step, capturing every output time."""

    comm = MPI.COMM_WORLD
    _log(f"realizing geometry {model.geometry.name!r} (dim {model.geometry.dim}) at h = {options.h:g}")
    geometry: Geometry = realize(model.geometry, h=options.h, comm=comm)
    cells = _mesh_cell_count(geometry.mesh_of(model.math.equations[0].subdomain))
    _log(f"assembling (fe_degree {options.fe_degree}, {cells} cells)")
    problem = assemble(model.math, geometry, dt=options.dt, fe_degree=options.fe_degree)
    if not isinstance(problem, DiscreteProblem):  # a CoupledGeometry cannot come out of `realize`
        raise CliError("this model assembles to a coupled bulk-surface problem, which the CLI does not drive yet")

    recorder = _single_mesh_recorder(problem, out_dir, write_fields=write_fields)
    try:
        recorder.capture(0.0)
        steps, dt_used = _integrate(problem, options, recorder)
    finally:
        recorder.close()
    return [recorder], cells, steps, dt_used


def _run_interface_coupled(
    model: ModelInput, coupling: _Coupling, options: RunOptions, out_dir: Path, *, write_fields: bool
) -> tuple[list[_Recorder], int, int]:
    """The two-compartment path: one species per compartment, coupled by the membrane flux.

    The integrator is adaptive method-of-lines over the blocked two-mesh system and cannot be
    interrupted at output times, so this writes the **final** state only — one XDMF per
    compartment, since the two live on different submeshes.
    """

    from vcell_fenics.backend.interface_coupled import integrate_interface_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled

    # The blocked two-mesh solve is P1 and adaptive-MOL by construction. Say so instead of
    # accepting a flag and quietly solving something else.
    if options.fe_degree != 1:
        raise CliError("the two-compartment solver is P1 only; drop --fe-degree for this model")
    if options.time_integration != "method_of_lines":
        _log("note: the two-compartment solver is method-of-lines; --time-integration is ignored")

    comm = MPI.COMM_WORLD
    _log(
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
    cells = _mesh_cell_count(geometry.inner_mesh) + _mesh_cell_count(geometry.outer_mesh)
    _log(f"integrating to t = {options.t_final:g} (interface-coupled method of lines, {cells} cells)")
    result = integrate_interface_coupled(model.math, geometry, t_final=options.t_final)

    recorders = [
        _Recorder(
            geometry.inner_mesh, [(result.inner.name, result.inner)], out_dir, "inner.xdmf", write_fields=write_fields
        ),
        _Recorder(
            geometry.outer_mesh, [(result.outer.name, result.outer)], out_dir, "outer.xdmf", write_fields=write_fields
        ),
    ]
    try:
        for recorder in recorders:
            recorder.capture(options.t_final)
    finally:
        for recorder in recorders:
            recorder.close()
    return recorders, cells, int(result.steps)


@dataclass(frozen=True)
class _Coupling:
    """The two compartments and the membrane between them, for the interface-coupled path."""

    inner: str
    outer: str
    membrane: str
    background: str | None


def _interface_coupling(model: ModelInput) -> _Coupling | None:
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
                raise CliError(
                    f"geometry {model.geometry.name!r} has more unmodelled subvolumes than the coupled "
                    f"solver can drop ({', '.join(spare)}); it supports two compartments plus one background"
                )
            return _Coupling(
                inner=surface.inside,
                outer=surface.outside,
                membrane=surface.name,
                background=spare[0] if spare else None,
            )
    raise CliError(
        f"the model's equations span subdomains {sorted(subdomains)}, but geometry {model.geometry.name!r} "
        f"declares no membrane between them (surfaces: {[s.name for s in model.geometry.surfaces] or 'none'})"
    )


def _merge_records(recorders: Sequence[_Recorder]) -> list[dict[str, Any]]:
    """One record per output time, merging the per-mesh species dicts captured at that time."""

    merged: dict[float, dict[str, Any]] = {}
    for recorder in recorders:
        for record in recorder.records:
            merged.setdefault(float(record["t"]), {}).update(record["species"])
    return [{"t": t, "species": merged[t]} for t in sorted(merged)]


def _integrate(problem: DiscreteProblem, options: RunOptions, recorder: _Recorder) -> tuple[int, float]:
    """Advance to `t_final`, capturing at each output time. Returns (steps taken, dt actually used)."""

    if options.time_integration == "method_of_lines":
        # PETSc TS picks (and adapts) its own steps to `t_final` in one shot — there is no
        # supported way to interrupt it at output times, so this path writes t=0 and t=t_final
        # only. Use backward Euler when a time series matters.
        _log(f"integrating to t = {options.t_final:g} (method of lines, adaptive)")
        result = integrate_discrete_problem(problem, t_final=options.t_final)
        recorder.capture(options.t_final)
        return int(result.steps), float("nan")

    n_outputs = max(1, round(options.t_final / options.output_dt))
    steps_per_output = max(1, round(options.output_dt / options.dt))
    dt = options.t_final / (n_outputs * steps_per_output)
    if abs(dt - options.dt) > 1e-12 * max(1.0, options.dt):
        _log(f"dt adjusted {options.dt:g} → {dt:g} so output times land on step boundaries")
    problem.dt.value = dt

    _log(f"stepping to t = {options.t_final:g}: {n_outputs} outputs × {steps_per_output} steps of dt = {dt:g}")
    step = 0
    for out_index in range(n_outputs):
        for _ in range(steps_per_output):
            step += 1
            problem.set_time(step * dt)
            problem.step()
        recorder.capture((out_index + 1) * steps_per_output * dt)
    return step, dt


def _mesh_cell_count(mesh: Any) -> int:
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return int(mesh.comm.allreduce(index_map.size_local, op=MPI.SUM))


def _log(message: str) -> None:
    """Progress to stderr from rank 0 only — stdout stays clean for the summary."""

    if MPI.COMM_WORLD.rank == 0:
        print(f"[vcell-fenics] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcell-fenics",
        description="Solve a VCell or native vcell-fenics model with FEniCSx and write the results to a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  vcell-fenics --vcml model.vcml --out results\n"
            "  vcell-fenics --math m_math.yaml --geometry m_geom.yaml --t-final 1.0 --out results\n"
        ),
    )
    source = parser.add_argument_group("model input")
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
    discretisation.add_argument("--fe-degree", type=int, default=1, help="Lagrange degree (default: 1)")
    discretisation.add_argument(
        "--time-integration",
        choices=("backward_euler", "method_of_lines"),
        default="backward_euler",
        help="fixed-step backward Euler (time series) or adaptive PETSc TS (final state only)",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--out", type=Path, default=Path("out"), help="results directory (default: ./out)")
    output.add_argument("--no-fields", action="store_true", help="write summary.json only, no XDMF fields")
    output.add_argument("--traceback", action="store_true", help="show the full traceback on a model error")
    return parser


def load_model(args: argparse.Namespace) -> ModelInput:
    if args.vcml is not None:
        if args.math is not None or args.geometry is not None:
            raise CliError("--vcml and --math/--geometry are alternatives; pass one or the other")
        return load_vcml(args.vcml, application=args.application, simulation=args.simulation)
    if args.math is None or args.geometry is None:
        raise CliError("give a model: --vcml FILE, or --math FILE --geometry FILE")
    return load_pair(args.math, args.geometry, fmt=args.format)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = load_model(args)
        _log(f"loaded {model.source} model: {model.provenance}")
        options = resolve_options(model, args)
        summary = run_model(model, options, args.out, write_fields=not args.no_fields)
    except CliError as error:
        _fail(str(error), args.traceback)
        return 2
    except _USER_ERRORS as error:
        _fail(f"{type(error).__name__}: {error}", args.traceback)
        return 2

    if MPI.COMM_WORLD.rank == 0:
        final = summary["outputs"][-1]["species"]
        for name, stats in final.items():
            _log(f"t = {summary['outputs'][-1]['t']:g}  {name}: total {stats['total']:.6g}, max {stats['max']:.6g}")
        _log(f"wrote {args.out}")
    return 0


def _fail(message: str, show_traceback: bool) -> None:
    if show_traceback:
        import traceback

        traceback.print_exc()
    if MPI.COMM_WORLD.rank == 0:
        print(f"error: {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

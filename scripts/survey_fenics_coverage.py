#!/usr/bin/env python
"""Characterize FEniCSx coverage of the saved VCell biomodels, by running the FEniCSx CLI on them.

The other survey scripts check the import layer only. This one exercises the solver the way VCell
would: ``python -m vcell_fenics.cli --vcml <file> --application <app> --simulation <sim>`` — load,
option resolution, routing, realization, assembly, a real (short) solve and a results bundle — and
records what happened. Runs in the vcell-fenics **dev env**.

Three steps, each writing into ``--work`` (default ``vcml_biomodels/coverage/``):

``census``
    Parse every ``vcml_biomodels/*.vcml`` (lxml, no solver) into ``census.csv``: one row per
    simulation — biomodel, application, simulation, solver, spatial or not, dimension, geometry kind
    (analytic / csg / image / mixed), moving boundary, stochastic, end time, output step, mesh size.

``run``
    For each spatial, deterministic application served by a VCell finite-volume-family solver
    (``FV_SOLVERS``), run the CLI once — on the application's first such simulation, with its own mesh,
    method of lines (what VCell hands a SimulationTask), no field arrays, and a short horizon of
    ``--steps`` output intervals — under a timeout, several in parallel. Writes ``runs.csv``: exit
    status, the ``error:`` line, the backend it took, cells, wall time. Resumable: rows already in
    ``runs.csv`` are skipped.

``report``
    Summarize ``census.csv`` + ``runs.csv`` into ``report.md``: coverage of FV-served simulations and
    applications, by solver, dimension and geometry kind, and the refusal/failure reasons ranked by how
    many applications (and simulations) each blocks.

Usage::

    .pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py census
    .pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py run --jobs 6 --timeout 600 [--limit N]
    .pixi/envs/dev/bin/python scripts/survey_fenics_coverage.py report
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from lxml import etree

_ROOT = Path(__file__).resolve().parent.parent
_VCML_DIR = _ROOT / "vcml_biomodels"
_WORK = _VCML_DIR / "coverage"

# VCell's deterministic spatial (PDE) solvers — the ones FEniCSx would stand in for. Smoldyn (spatial
# stochastic particles) and the non-spatial ODE/stochastic solvers are out of FEniCSx's scope.
FV_SOLVERS = (
    "Sundials Stiff PDE Solver (Variable Time Step)",
    "Finite Volume, Regular Grid",
    "Finite Volume Standalone, Regular Grid",
    "Chombo Standalone",
    "MovingB",
    "VCellPetsc",
    "Comsol",
)

CENSUS_FIELDS = (
    "biomodel",
    "file",
    "application",
    "simulation",
    "solver",
    "spatial",
    "dim",
    "geometry_kind",
    "moving",
    "stochastic",
    "end_time",
    "output_step",
    "mesh",
    "extent",
)
RUN_FIELDS = (
    "biomodel",
    "file",
    "application",
    "simulation",
    "solver",
    "dim",
    "geometry_kind",
    "moving",
    "status",
    "backend",
    "cells",
    "seconds",
    "error",
    "t_final",
)


def _local(element: Any) -> str:
    return str(etree.QName(element).localname)


def _children(element: Any, name: str) -> Iterator[Any]:
    return (child for child in element if isinstance(child.tag, str) and _local(child) == name)


def _first(element: Any, name: str) -> Any | None:
    return next(_children(element, name), None)


def _geometry_kind(geometry: Any) -> str:
    kinds = set()
    for subvolume in _children(geometry, "SubVolume"):
        kinds.add(
            {"Analytical": "analytic", "CSGObject": "csg", "Image": "image", "Compartmental": "compartmental"}.get(
                subvolume.get("Type", ""), "other"
            )
        )
    if not kinds:
        return "compartmental" if geometry.get("Dimension", "0") == "0" else "none"
    return kinds.pop() if len(kinds) == 1 else "mixed(" + "+".join(sorted(kinds)) + ")"


def census_rows(path: Path) -> Iterator[dict[str, str]]:
    """One row per simulation in the biomodel file (none if it does not parse)."""

    root = etree.parse(str(path), etree.XMLParser(huge_tree=True, recover=False)).getroot()
    biomodel = next((e for e in root.iter() if isinstance(e.tag, str) and _local(e) == "BioModel"), root)
    biomodel_id = path.stem.removeprefix("biomodel_")
    for spec in _children(biomodel, "SimulationSpec"):
        geometry = _first(spec, "Geometry")
        dim = geometry.get("Dimension", "0") if geometry is not None else "0"
        math = _first(spec, "MathDescription")
        moving = math is not None and any(
            _first(m, "Velocity") is not None for m in _children(math, "MembraneSubDomain")
        )
        for simulation in _children(spec, "Simulation"):
            task = _first(simulation, "SolverTaskDescription")
            mesh = _first(simulation, "MeshSpecification")
            size = _first(mesh, "Size") if mesh is not None else None
            bound = _first(task, "TimeBound") if task is not None else None
            extent = _first(geometry, "Extent") if geometry is not None else None
            yield {
                "biomodel": biomodel_id,
                "file": path.name,
                "application": spec.get("Name", ""),
                "simulation": simulation.get("Name", ""),
                "solver": task.get("Solver", "") if task is not None else "",
                "spatial": "1" if mesh is not None else "0",
                "dim": dim,
                "geometry_kind": _geometry_kind(geometry) if geometry is not None else "none",
                "moving": "1" if moving else "0",
                "stochastic": "1" if spec.get("Stochastic") == "true" else "0",
                "end_time": bound.get("EndTime", "") if bound is not None else "",
                "output_step": _output_step(task),
                "mesh": "x".join(size.get(axis, "") for axis in "XYZ" if size.get(axis)) if size is not None else "",
                "extent": "x".join(extent.get(axis, "") for axis in "XYZ" if extent.get(axis))
                if extent is not None
                else "",
            }


def _output_step(task: Any) -> str:
    """The output interval as VCell reads it: ``OutputTimeStep`` (uniform), else ``KeepEvery`` × the default
    time step, else every default time step (VCell's default when a simulation has no OutputOptions)."""

    if task is None:
        return ""
    step = _first(task, "TimeStep")
    default = float(step.get("DefaultTime", "0") or 0) if step is not None else 0.0
    output = _first(task, "OutputOptions")
    if output is not None and output.get("OutputTimeStep"):
        return str(output.get("OutputTimeStep", ""))
    keep = int(output.get("KeepEvery", "1") or 1) if output is not None else 1
    return repr(keep * default) if default > 0 else ""


def census(vcml_dir: Path, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(vcml_dir.glob("*.vcml"))
    failures: list[tuple[str, str]] = []
    count = 0
    with (work / "census.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CENSUS_FIELDS)
        writer.writeheader()
        for path in files:
            try:
                rows = list(census_rows(path))
            except Exception as error:  # a file that does not parse is reported, not fatal
                failures.append((path.name, f"{type(error).__name__}: {error}"[:200]))
                continue
            writer.writerows(rows)
            count += len(rows)
    with (work / "census_failures.csv").open("w", newline="") as out:
        csv.writer(out).writerows([("file", "error"), *failures])
    print(f"census: {len(files)} files, {count} simulations, {len(failures)} files failed to parse -> {work}")


def _candidates(work: Path) -> list[dict[str, str]]:
    """The first FV-served simulation of each spatial, deterministic application."""

    chosen: dict[tuple[str, str], dict[str, str]] = {}
    with (work / "census.csv").open() as source:
        for row in csv.DictReader(source):
            if row["spatial"] == "1" and row["stochastic"] == "0" and row["solver"] in FV_SOLVERS:
                chosen.setdefault((row["file"], row["application"]), row)
    return list(chosen.values())


_ERROR_RE = re.compile(r"^error: (.*)$", re.MULTILINE)
_CELLS_RE = re.compile(r"\((?:[^()]*?, )?(\d+) cells")


def _h(row: dict[str, str]) -> float | None:
    """The CLI's own rule (``cli._h_from_mesh_size``): the finest cell spacing over the spatial axes."""

    try:
        mesh = [int(n) for n in row["mesh"].split("x")]
        extent = [float(e) for e in row["extent"].split("x")]
        dim = int(row["dim"])
    except ValueError:
        return None
    spacings = [extent[a] / mesh[a] for a in range(min(dim, len(mesh), len(extent))) if mesh[a] > 0]
    return min(spacings) if spacings else None


def run_one(
    row: dict[str, str],
    vcml_dir: Path,
    scratch: Path,
    steps: int,
    timeout: float,
    cache: Path,
    max_rss_gb: float,
) -> dict[str, str]:
    out = scratch / f"{row['biomodel']}_{abs(hash((row['application'], row['simulation']))) % 10**8}"
    argv = [
        sys.executable,
        "-m",
        "vcell_fenics.cli",
        "--vcml",
        str(vcml_dir / row["file"]),
        "--application",
        row["application"],
        "--time-integration",
        "method_of_lines",
        "--no-fields",
        "--out",
        str(out),
    ]
    try:
        step = float(row["output_step"]) if row["output_step"] else 0.0
        end = float(row["end_time"]) if row["end_time"] else 0.0
    except ValueError:
        step = end = 0.0
    h = _h(row)
    if h is not None:
        argv += ["--h", repr(h)]
    started = time.monotonic()
    result = {key: row[key] for key in ("biomodel", "file", "application", "simulation", "solver", "dim")}
    result |= {"geometry_kind": row["geometry_kind"], "moving": row["moving"], "t_final": ""}
    # The simulation's settings are passed explicitly rather than via `--simulation`: pyvcell's reader drops a
    # simulation that has no <OutputOptions> (VCell's default: every time step), which is not FEniCSx's gap.
    # A short horizon: `steps` output intervals, never past the simulation's end; a run that times out after
    # reaching its first output time is retried once over that single interval.
    horizons = [min(end, n * step) for n in dict.fromkeys((steps, 1))] if step > 0.0 and end > 0.0 else [None]
    try:
        env = {**os.environ, "XDG_CACHE_HOME": str(cache), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        for k, t_final in enumerate(horizons):
            timed = argv if t_final is None else [*argv, "--t-final", repr(t_final), "--output-dt", repr(step)]
            result["t_final"] = "" if t_final is None else f"{t_final:g}"
            try:
                returncode, text = _run_capped(timed, env, timeout, max_rss_gb)
                break
            except subprocess.TimeoutExpired as expired:
                if k + 1 == len(horizons) or not _PAST_FIRST_OUTPUT.search(str(expired.output or "")):
                    raise
        errors = _ERROR_RE.findall(text)
        backend = re.search(r'"backend": "(\w+)"', _read(out / "results.fenics" / "provenance" / "summary.json"))
        cells = _CELLS_RE.findall(text)
        result |= {
            "status": "ok" if returncode == 0 else f"exit{returncode}",
            "backend": backend.group(1) if backend else "",
            "cells": cells[-1] if cells else "",
            "error": (errors[-1] if errors else ("" if returncode == 0 else text.strip()[-300:])).replace("\n", " "),
        }
    except subprocess.TimeoutExpired:
        result |= {"status": "timeout", "backend": "", "cells": "", "error": f"no result within {timeout:g} s"}
    except MemoryError:
        result |= {"status": "memory", "backend": "", "cells": "", "error": f"over {max_rss_gb:g} GB resident"}
    result["seconds"] = f"{time.monotonic() - started:.1f}"
    return result


# The CLI's per-output-time line (stderr, flushed) at a time after 0: the run reached its first output interval
# within the timeout, so a one-interval rerun (same setup cost) will finish; a run that never got there won't.
_PAST_FIRST_OUTPUT = re.compile(r"^\[vcell-fenics\] t = (?!0 )[0-9.eE+-]+ ", re.MULTILINE)


def _run_capped(argv: list[str], env: dict[str, str], timeout: float, max_rss_gb: float) -> tuple[int, str]:
    """Run ``argv``, killing it past ``timeout`` (TimeoutExpired) or ``max_rss_gb`` resident (MemoryError).

    The memory cap keeps one very large mesh from pushing the whole machine into swap and stalling
    every other worker; macOS does not enforce RLIMIT_AS, so the resident size is polled instead.
    """
    with tempfile.TemporaryFile("w+") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    break
                rss = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True)
                if rss.stdout.strip() and int(rss.stdout.strip()) > max_rss_gb * 1024**2:
                    raise MemoryError
                time.sleep(2.0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        log.seek(0)
        text = log.read()
        if timed_out:
            raise subprocess.TimeoutExpired(argv, timeout, output=text)
        return proc.returncode, text


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def run(
    vcml_dir: Path,
    work: Path,
    *,
    jobs: int,
    timeout: float,
    max_rss_gb: float,
    steps: int,
    limit: int | None,
    sample: int | None,
) -> None:
    runs_csv = work / "runs.csv"
    done: set[tuple[str, str]] = set()
    if runs_csv.exists():
        with runs_csv.open() as source:
            recorded = list(csv.DictReader(source))
        done = {(r["file"], r["application"]) for r in recorded}
        if recorded and set(recorded[0]) != set(RUN_FIELDS):  # an older runs.csv: rewrite it with today's columns
            with runs_csv.open("w", newline="") as rewrite:
                migrate = csv.DictWriter(rewrite, fieldnames=RUN_FIELDS, restval="", extrasaction="ignore")
                migrate.writeheader()
                migrate.writerows(recorded)
    todo = [row for row in _candidates(work) if (row["file"], row["application"]) not in done]
    if sample is not None:  # a seeded random subset, for a pilot (its rows count toward the full run)
        todo = random.Random(20260926).sample(todo, min(sample, len(todo)))
    if limit is not None:
        todo = todo[:limit]
    scratch = work / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"run: {len(todo)} applications to go ({len(done)} already recorded), {jobs} at a time", flush=True)
    new_file = not runs_csv.exists()
    with runs_csv.open("a", newline="") as out, ThreadPoolExecutor(max_workers=jobs) as pool:
        writer = csv.DictWriter(out, fieldnames=RUN_FIELDS)
        if new_file:
            writer.writeheader()
        slots: queue.Queue[Path] = queue.Queue()
        for k in range(jobs):
            slots.put(work / "jit-cache" / f"worker{k}")

        def job(row: dict[str, str]) -> dict[str, str]:
            cache = slots.get()
            try:
                return run_one(row, vcml_dir, scratch, steps, timeout, cache, max_rss_gb)
            finally:
                slots.put(cache)

        futures = [pool.submit(job, row) for row in todo]
        for k, future in enumerate(as_completed(futures), 1):
            result = future.result()
            writer.writerow(result)
            out.flush()
            if k % 10 == 0 or k == len(futures):
                print(f"  {k}/{len(futures)}", flush=True)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "—"


def report(work: Path) -> None:
    with (work / "census.csv").open() as source:
        census_table = list(csv.DictReader(source))
    with (work / "runs.csv").open() as source:
        runs = {(r["file"], r["application"]): r for r in csv.DictReader(source)}
    fv_sims = [r for r in census_table if r["spatial"] == "1" and r["stochastic"] == "0" and r["solver"] in FV_SOLVERS]
    by_app: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fv_sims:
        by_app[(row["file"], row["application"])].append(row)
    run_apps = {key: runs[key] for key in by_app if key in runs}

    def outcome(key: tuple[str, str]) -> str:
        r = run_apps.get(key)
        if r is None:
            return "not run"
        return {"ok": "ran", "memory": "timeout"}.get(r["status"], r["status"])

    lines = ["# FEniCSx coverage of saved VCell biomodels", ""]
    lines.append(
        f"Corpus: {len({r['file'] for r in census_table})} biomodels, {len(census_table)} simulations, of which "
        f"**{len(fv_sims)} simulations in {len(by_app)} applications** are spatial, deterministic and saved with a "
        f"VCell finite-volume-family solver. {len(run_apps)} of those applications were run through the FEniCSx CLI."
    )
    lines.append("")
    ok_apps = [key for key in run_apps if run_apps[key]["status"] == "ok"]
    ok_sims = sum(len(by_app[key]) for key in ok_apps)
    lines.append(
        f"**Ran: {len(ok_apps)} / {len(run_apps)} applications ({_pct(len(ok_apps), len(run_apps))}), covering "
        f"{ok_sims} / {sum(len(by_app[k]) for k in run_apps)} of their simulations.**"
    )
    lines.append("")

    def table(title: str, key_of: Any) -> None:
        groups: dict[str, Counter[str]] = defaultdict(Counter)
        for key in run_apps:
            groups[key_of(by_app[key][0])][outcome(key)] += 1
        lines.extend(
            [
                f"## By {title}",
                "",
                "| " + title + " | applications | ran | timeout / memory | failed |",
                "|---|---|---|---|---|",
            ]
        )
        for name, counts in sorted(groups.items(), key=lambda item: -sum(item[1].values())):
            total = sum(counts.values())
            failed = total - counts["ran"] - counts["timeout"]
            ran = f"{counts['ran']} ({_pct(counts['ran'], total)})"
            lines.append(f"| {name} | {total} | {ran} | {counts['timeout']} | {failed} |")
        lines.append("")

    table("solver", lambda r: r["solver"])
    table("dimension", lambda r: f"{r['dim']}D")
    table("geometry", lambda r: r["geometry_kind"])
    table("moving boundary", lambda r: "moving" if r["moving"] == "1" else "fixed")

    categories: Counter[str] = Counter()
    category_sims: Counter[str] = Counter()
    for key, r in run_apps.items():
        if r["status"] != "ok":
            categories[category(r["error"])] += 1
            category_sims[category(r["error"])] += len(by_app[key])
    meaning = {name: text for name, _, text in CATEGORIES}
    lines.extend(
        [
            "## Why applications don't run (by category, ranked)",
            "",
            "| category | applications | simulations | what it means |",
            "|---|---|---|---|",
        ]
    )
    for name, count in categories.most_common():
        lines.append(f"| {name} | {count} | {category_sims[name]} | {meaning.get(name, '')} |")
    lines.append("")
    offered = [key for key in run_apps if vcell_offers(by_app[key][0])]
    offered_fail = [key for key in offered if run_apps[key]["status"] != "ok"]
    lines.extend(
        [
            "## VCell's gate vs the run",
            "",
            f"VCell's pre-run check would offer FEniCSx for {len(offered)} of {len(run_apps)} applications; "
            f"**{len(offered_fail)} of those then fail in the CLI** (a user finds out from a failed run, not an "
            "up-front refusal). By category:",
            "",
            "| category | offered but fails |",
            "|---|---|",
        ]
    )
    for name, count in Counter(category(run_apps[k]["error"]) for k in offered_fail).most_common():
        lines.append(f"| {name} | {count} |")
    lines.append("")

    reasons: Counter[str] = Counter()
    reason_sims: Counter[str] = Counter()
    example: dict[str, str] = {}
    for key, r in run_apps.items():
        if r["status"] in ("ok", "timeout", "memory"):
            continue
        reason = _normalize(r["error"])
        reasons[reason] += 1
        reason_sims[reason] += len(by_app[key])
        example.setdefault(reason, f"{r['biomodel']} / {r['application']}")
    lines.extend(
        [
            "## Individual messages (normalized, ranked)",
            "",
            "| reason | applications | simulations | example |",
            "|---|---|---|---|",
        ]
    )
    for reason, count in reasons.most_common():
        lines.append(f"| {reason.replace('|', '/')} | {count} | {reason_sims[reason]} | {example[reason]} |")
    lines.append("")
    (work / "report.md").write_text("\n".join(lines) + "\n")
    print(f"report -> {work / 'report.md'}")


# Refusal / failure categories, most specific first: (name, pattern on the error line, what it means).
CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("timeout", r"^no result within", "no result within the timeout at the simulation's own mesh"),
    ("memory", r"^over .* GB resident", "past the survey's per-run memory cap at the simulation's own mesh"),
    (
        "mesh too large",
        r"would need ~.* tetrahedra",
        "the simulation's own mesh size exceeds the FEM mesh limit (may run coarser)",
    ),
    (
        "legacy domain-less variable",
        r"legacy domain-less volume variable",
        "old-style VCell math: one volume variable on both sides of a membrane",
    ),
    (
        "one-sided membrane species",
        r"bulk species only in",
        "membrane species with bulk species in one compartment only (#183)",
    ),
    ("non-diffusing species (lumped_ode)", r"not 'lumped_ode'", "ODE (non-diffusing) species on a spatial domain"),
    (
        "subvolume touches the box",
        r"strictly inside the box",
        "a subvolume boundary touches the domain box (2D/3D realization)",
    ),
    (
        "CSG geometry",
        # pyvcell's reader maps only Type="CSG", so a Type="CSGObject" subvolume arrives as an expression-less
        # 'analytic' one and realization then reports "must be 'analytic' … (got type 'analytic')".
        r"(?i)csg|expression to be realized \(got type 'analytic'\)",
        "constructive solid geometry subvolumes (pyvcell reads them as expression-less analytic ones)",
    ),
    ("1D geometry", r"dim=1|this one is 1D", "one-dimensional geometry"),
    (
        "3+ compartments",
        r"more unmodelled subvolumes|span subdomains",
        "species on three or more compartments, or compartments without a membrane between them",
    ),
    ("math validation", r"FormalismValidationError", "the imported math fails formalism validation"),
    ("unresolved name", r"CompileError: unresolved name", "an expression names something the compiler cannot resolve"),
    ("realization", r"RealizationError", "geometry realization (meshing) failed"),
    ("not implemented (other)", r"NotImplementedError", "another feature not implemented yet"),
    (
        "model loading",
        r"has no simulation named|cannot parse|VcellImportError",
        "the model or application could not be loaded",
    ),
    ("crash", r"^/|Traceback|Error code|error code", "a crash (not a clean refusal)"),
)


def category(error: str) -> str:
    for name, pattern, _ in CATEGORIES:
        if re.search(pattern, error):
            return name
    return "other"


def vcell_offers(row: dict[str, str]) -> bool:
    """Whether VCell's own pre-run check (``FenicsSolver.unsupportedReasons``) would let a user pick FEniCSx:
    a 2D or 3D geometry, no CSG, and a moving boundary only on a non-image geometry."""

    if row["dim"] not in ("2", "3") or "csg" in row["geometry_kind"]:
        return False
    return not (row["moving"] == "1" and "image" in row["geometry_kind"])


_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def _normalize(error: str) -> str:
    """An error message with its model-specific parts (names, numbers) masked, so that the same reason
    across many models counts as one."""

    return _NUMBER.sub("#", _QUOTED.sub("'…'", error))[:160] or "(no error line)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("step", choices=("census", "run", "report"))
    parser.add_argument("--vcml-dir", type=Path, default=_VCML_DIR)
    parser.add_argument("--work", type=Path, default=_WORK)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rss-gb", type=float, default=4.0, help="kill a run past this resident size")
    parser.add_argument("--steps", type=int, default=3, help="output intervals to solve (short horizon)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None, help="run a seeded random subset of this size")
    args = parser.parse_args()
    if args.step == "census":
        census(args.vcml_dir, args.work)
    elif args.step == "run":
        run(
            args.vcml_dir,
            args.work,
            jobs=args.jobs,
            timeout=args.timeout,
            max_rss_gb=args.max_rss_gb,
            steps=args.steps,
            limit=args.limit,
            sample=args.sample,
        )
    else:
        report(args.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

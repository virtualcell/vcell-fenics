"""Stage 2 (dev env): vcell-fenics vs mbsolver on the moving-boundary SWEPT problem — the real fixture.

The model is the real VCell moving-boundary SimulationTask ``tests/fixtures/simtask/SimID_274641196``
run through the CLI path (the same code a VCell cluster or desktop run takes), against the mbsolver
reference ``mb_swept_fv.npz`` (``mb_swept_fv.py``, the same problem authored in pyvcell): a disk swept by a
front moving at ``(sin t, cos t)``, species C = x and Ran = y at rest in the lab frame (D = 10).

At every output time our P1 field is interpolated exactly (the bundle's own triangles, at that row's
coordinates) at mbsolver's inside grid nodes, and compared: relative L2 and L∞ per species. The negative
control runs the same task with the species *carried* by the cell (lab velocity = front velocity — the
other convention), to show the comparison discriminates the two.

    ../pyvcell/.venv/bin/python cross_validation/mb_swept_fv.py [--case …]   # stage 1 (once per case)
    .pixi/envs/dev/bin/python    cross_validation/mb_swept.py [--case translate|deform|remesh] [--dt 0.01] [--h 0.2]

(stage 1 takes the same ``--case``; ``--mesh 61`` refines the mbsolver reference.)
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from numpy.typing import NDArray

from vcell_fenics.cli import load_simtask
from vcell_fenics.formalism.schema import TemplateEquation
from vcell_fenics.results import Bundle
from vcell_fenics.runner import RunOptions, run_model

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent / "tests" / "fixtures" / "simtask" / "SimID_274641196_0__0.simtask.xml"
_SPECIES = {"C": "C_cyt", "Ran": "Ran_cyt"}  # mbsolver name → the fixture's name


def _run(model: Any, options: RunOptions, out: Path) -> Bundle:
    run_model(model, options, out, prefix="run")
    return Bundle.open(out / "run.fenics")


def _carried(model: Any) -> Any:
    """The same model with each species riding the cell: lab-frame advection = the front velocity."""

    front = next(s.motion.velocity for s in model.math.subdomains if s.motion.kind == "prescribed")
    equations = [
        dataclasses.replace(eq, terms={**eq.terms, "advection": front}) if isinstance(eq, TemplateEquation) else eq
        for eq in model.math.equations
    ]
    return dataclasses.replace(model, math=dataclasses.replace(model.math, equations=equations))


def _sample(bundle: Bundle, species: str, row: int, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    coords = bundle.coords("cell", row)
    cells = bundle.mesh("cell", row).cells
    interpolate = LinearTriInterpolator(Triangulation(coords[:, 0], coords[:, 1], cells), bundle.field("cell", species, row))
    values: NDArray[np.float64] = np.ma.filled(interpolate(x, y).astype(float), np.nan)
    return values


def _errors(bundle: Bundle, reference: Any) -> dict[str, list[tuple[float, float, int]]]:
    """Per species, per output time: (relative L2, relative L∞, nodes compared) at mbsolver's inside nodes
    that also lie inside our mesh (the two fronts differ by their own discretization errors)."""

    out: dict[str, list[tuple[float, float, int]]] = {}
    for fv_name, name in _SPECIES.items():
        rows = []
        for row in range(len(reference["times"])):
            x, y = np.asarray(reference["xs"][row], float), np.asarray(reference["ys"][row], float)
            fv = np.asarray(reference[fv_name][row], float)
            ours = _sample(bundle, name, row, x, y)
            keep = np.isfinite(ours)
            diff = ours[keep] - fv[keep]
            rows.append(
                (
                    float(np.linalg.norm(diff) / np.linalg.norm(fv[keep])),
                    float(np.abs(diff).max() / np.abs(fv[keep]).max()),
                    int(keep.sum()),
                )
            )
        out[name] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("translate", "deform", "remesh"), default="translate")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--h", type=float, default=None, help="element size (default: the task's 10/31)")
    args = parser.parse_args()
    reference_path = _HERE / ("mb_swept_fv.npz" if args.case == "translate" else f"mb_swept_{args.case}_fv.npz")
    if not reference_path.exists():
        raise SystemExit(f"missing {reference_path.name} — run mb_swept_fv.py in ../pyvcell/.venv first")
    reference = np.load(reference_path, allow_pickle=True)
    times = tuple(float(t) for t in reference["times"])

    task = _TASK
    if args.case != "translate":  # the fixture with the reference's front velocity (VCell syntax)
        velocity_x, velocity_y = (str(v) for v in reference["velocity"])
        text = _TASK.read_text()
        text = text.replace(">sin(t)</Function>", f">{velocity_x}</Function>").replace(">cos(t)</Function>", f">{velocity_y}</Function>")
        task = Path(tempfile.mkdtemp()) / _TASK.name
        task.write_text(text)
    model = load_simtask(task)
    h = args.h if args.h is not None else float(model.suggested_h or 10.0 / 31)
    options = RunOptions(
        h=h, dt=args.dt, t_final=times[-1], output_times=times, fe_degree=1, time_integration="backward_euler"
    )
    with tempfile.TemporaryDirectory() as tmp:
        swept = _run(model, options, Path(tmp) / "swept")
        carried = _run(_carried(model), options, Path(tmp) / "carried")
        results = {"swept (VCell semantics)": _errors(swept, reference), "carried (control)": _errors(carried, reference)}
        segments = len(swept.manifest.segments)
        last = len(times) - 1
        stats = swept.stats("cell", "C_cyt")
        area = float(stats[last, 1] / stats[last, 0])  # |domain| = total / mean
        xs = swept.coords("cell", last)[:, 0]
        front = np.asarray(reference["fronts"][last], float)
        fv_area = 0.5 * abs(float(np.dot(front[:, 0], np.roll(front[:, 1], -1)) - np.dot(front[:, 1], np.roll(front[:, 0], -1))))

    print(
        f"moving-boundary SWEPT ({args.case}): fixture {_TASK.name} vs mbsolver (mesh {int(reference['mesh'])});"
        f" h = {h:.3g}, dt = {args.dt:g}, bundle segments {segments} ({segments - 1} remesh)"
    )
    print(
        f"  domain at t = {times[-1]:g}: area fenics {area:.4f} / mbsolver {fv_area:.4f};"
        f" x-extent fenics [{xs.min():.3f}, {xs.max():.3f}] / mbsolver [{front[:, 0].min():.3f}, {front[:, 0].max():.3f}]"
    )
    for label, errors in results.items():
        print(f"  {label}:")
        for name, rows in errors.items():
            final = rows[-1]
            worst = max(rows, key=lambda r: r[0])
            print(
                f"    {name:8s} t = {times[-1]:g}: relL2 {final[0]:.3%}, relL∞ {final[1]:.3%} ({final[2]} nodes);"
                f" worst relL2 over time {worst[0]:.3%}"
            )


if __name__ == "__main__":
    main()

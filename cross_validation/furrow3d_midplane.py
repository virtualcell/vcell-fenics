"""The 3D cleavage furrow's mid-plane against the exact front motion and the 2D furrow (dev env only).

The 3D furrow (``tests/fixtures/simtask/furrow3d_*``) pinches a sphere with an axisymmetric ring,
``v = −exp(−y²/0.25)·tanh(ρ/5)·(x, 0, z)/ρ``. Restricted to the ``z = 0`` plane that field is in-plane and
equal to the 2D furrow's, so the 3D domain's mid-plane section follows the 2D domain's boundary exactly — and
at the waist (``y = 0``) the front motion has a closed form, ``dx/dt = −tanh(x/5)``:

    sinh(x(t)/5) = sinh(x₀/5)·exp(−t/5),   x₀ = √30.

There is no mbsolver in 3D, so this closed form (and the 2D run) is the reference. Both SimulationTasks run
through the CLI at the same ``--h``; the 3D boundary is sliced at ``z = 0``; the waist half-width is compared.

    .pixi/envs/dev/bin/python cross_validation/furrow3d_midplane.py [--h 1.0 0.6] [--t-final 5]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vcell_fenics.cli import main as cli
from vcell_fenics.results import Bundle

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "simtask"


def exact_waist(t: float) -> float:
    return float(5.0 * np.arcsinh(np.sinh(np.sqrt(30.0) / 5.0) * np.exp(-t / 5.0)))


def _run(task: str, out: Path, h: float, t_final: float) -> Bundle:
    argv = ["--simtask", str(_FIXTURES / task), "--out", str(out), "--h", str(h), "--t-final", str(t_final)]
    assert cli([*argv, "--output-dt", "1"]) == 0
    return Bundle.open(out / f"{task.split('_', 1)[1].split('__')[0]}_.fenics")


def waist_2d(bundle: Bundle, row: int) -> float:
    """The half-width at y = 0 of the 2D domain: its boundary edges crossing y = 0."""

    x, cells = bundle.coords("Cyt", row)[:, :2], bundle.mesh("Cyt", row).cells
    edges = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return _crossing(x[unique[counts == 1, 0]], x[unique[counts == 1, 1]], axis=1)


def waist_3d(bundle: Bundle, row: int) -> float:
    """The half-width at y = 0 of the 3D domain's z = 0 section: its boundary triangles' edges crossing the
    line y = z = 0 (sliced at z = 0, then read at y = 0)."""

    x, cells = bundle.coords("Cyt", row), bundle.mesh("Cyt", row).cells
    faces = np.sort(
        np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]]), axis=1
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    segments = []  # the section: a segment per boundary triangle crossing z = 0
    for tri in boundary:
        p = x[tri]
        side = p[:, 2] > 0.0
        if side.all() or not side.any():
            continue
        cut = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if side[i] != side[j]:
                s = p[i, 2] / (p[i, 2] - p[j, 2])
                cut.append(p[i, :2] + s * (p[j, :2] - p[i, :2]))
        if len(cut) == 2:
            segments.append(cut)
    ends = np.asarray(segments)
    return _crossing(ends[:, 0], ends[:, 1], axis=1)


def _crossing(p: NDArray[np.float64], q: NDArray[np.float64], *, axis: int) -> float:
    """The smallest |x| where a segment p→q crosses coordinate ``axis`` = 0."""

    crosses = (p[:, axis] * q[:, axis]) <= 0.0
    s = p[crosses, axis] / np.where(p[crosses, axis] == q[crosses, axis], 1.0, p[crosses, axis] - q[crosses, axis])
    xs = p[crosses, 0] + s * (q[crosses, 0] - p[crosses, 0])
    return float(np.abs(xs).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, nargs="+", default=[1.0, 0.6])
    parser.add_argument("--t-final", type=float, default=5.0)
    args = parser.parse_args()
    for h in args.h:
        with tempfile.TemporaryDirectory() as tmp:
            b2 = _run("furrow_SimID_1486629996_0__0.simtask.xml", Path(tmp) / "2d", h, args.t_final)
            b3 = _run("furrow3d_SimID_516481304_0__0.simtask.xml", Path(tmp) / "3d", h, args.t_final)
            print(f"h = {h:g}: waist half-width, exact / 2D / 3D (3D error)")
            for row, t in enumerate(b3.times):
                exact = exact_waist(t)
                w2, w3 = waist_2d(b2, row), waist_3d(b3, row)
                print(f"  t = {t:4.1f}: {exact:.3f} / {w2:.3f} / {w3:.3f}  ({w3 - exact:+.3f})")


if __name__ == "__main__":
    main()

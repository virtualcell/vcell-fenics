"""Stage 1 (../pyvcell/.venv): the mbsolver reference for a RacRho front that reads the normal (ADR 013).

The public RacRho moving-boundary BioModel (``vcml_biomodels/biomodel_104122918.vcml``) moves its membrane
with ``v = (a − 1)|a − 1| n``: the cell protrudes where Rac is high and retracts where it is low, so it
migrates up the initial gradient ``a = 0.1 x + 1``. Its kinetics ``J = (0.067 + a²/(1 + a²)) b − a`` are
nonlinear, which the FEniCSx moving path (backward Euler) cannot assemble, so this comparison uses a
**linear variant**, ``J = 0.5 b − a``, with the document otherwise unchanged (geometry, diffusion, both
species' lab-frame velocity (1, 0), the front). mbsolver runs the document's own SurfaceKinematics.

Writes ``racrho_front_fv.npz`` (front area, centroid, x-extent and the mean of ``a`` per output time) and
the variant ``racrho_front_linear.vcml`` that stage 2 runs. Gitignored and regenerable:

    ../pyvcell/.venv/bin/python -I cross_validation/racrho_front_fv.py [path/to/biomodel_104122918.vcml]
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import libvcell
import numpy as np
from pyvcell._internal.solvers.mbsolver import solve_moving_boundary
from pyvcell.vcml.vcml_mb_simulation import _species_order

_HERE = Path(__file__).resolve().parent
_NONLINEAR = "(((0.067 + ((a ^ 2.0) / (1.0 + (a ^ 2.0)))) * b) - a)"
_LINEAR = "((0.5 * b) - a)"


@contextmanager
def _silence() -> Iterator[None]:
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _area(p: np.ndarray) -> float:
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def _centroid(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    c = x * ys - xs * y
    return np.array([((x + xs) * c).sum(), ((y + ys) * c).sum()]) / (3.0 * c.sum())


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE.parent / "vcml_biomodels" / "biomodel_104122918.vcml"
    text = source.read_text()
    assert text.count(_NONLINEAR) == 2, "the RacRho kinetics (reaction and generated math) were not found"
    variant = _HERE / "racrho_front_linear.vcml"
    variant.write_text(text.replace(_NONLINEAR, _LINEAR))

    out = Path(tempfile.mkdtemp(prefix="racrho_mb_"))
    refined = out / "refined.vcml"
    ok, message = libvcell.vcml_to_vcml(vcml_content=variant.read_text(), vcml_file_path=refined)
    assert ok, message
    ok, message = libvcell.vcml_to_moving_boundary_input(
        vcml_content=refined.read_text(), simulation_name="Simulation4", output_dir_path=out
    )
    assert ok, message
    setup = next(out / f for f in os.listdir(out) if f.endswith("mb.xml"))
    with _silence():
        result = solve_moving_boundary(
            setup, species_names=_species_order(setup), output_times=[round(0.1 * k, 10) for k in range(51)]
        )
    fronts = [np.asarray(f.front, float) for f in result.frames]
    np.savez(
        _HERE / "racrho_front_fv.npz",
        times=np.array(result.times),
        areas=np.array([_area(p) for p in fronts]),
        centroids=np.array([_centroid(p) for p in fronts]),
        xmin=np.array([p[:, 0].min() for p in fronts]),
        xmax=np.array([p[:, 0].max() for p in fronts]),
        a_mean=np.array([float(f.concentrations["a"].mean()) for f in result.frames]),
    )
    print(f"  wrote racrho_front_fv.npz ({len(fronts)} frames) and {variant.name}")


if __name__ == "__main__":
    main()

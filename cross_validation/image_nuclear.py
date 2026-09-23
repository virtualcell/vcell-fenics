"""Stage 2 (dev env): vcell-fenics vs fvsolver on nucleocytoplasmic exchange over a real IMAGE geometry.

The model and the fvsolver reference come from ``image_nuclear_fv.py`` (stage 1): VCell's segmented 3D
tutorial image (ec ⊃ cytosol ⊃ Nucleus), ``c`` in the cytosol and ``n`` in the Nucleus exchanging across
the nuclear membrane (``J = P·(c − n)``). Here the same lowered model (``image_nuclear_math.yaml`` /
``_geom.yaml``, voxels included) runs through vcell-fenics' own pipeline — the image realized body-fitted
and smoothed, the two compartments solved by the interface-coupled method of lines — and is compared with
the FV fields:

- **the mean nuclear and cytosolic concentrations over time**: the nucleus fills at a rate set by its
  area-to-volume ratio, so this measures the realized geometry as much as the solver;
- **conservation**: ``∫c + ∫n`` over time, on each side;
- **the fields**: relative L2 of ``c`` at FV nodes inside the cytosol, interpolated from our P1 field.

FV region membership follows VCell's own sampling (the nearest image pixel to each FV node).

    ../pyvcell/.venv/bin/python cross_validation/image_nuclear_fv.py --mesh 101 101 36   # stage 1
    .pixi/envs/dev/bin/python    cross_validation/image_nuclear.py --fv 101 [--h 1.5] [--dt 0.05]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

from vcell_fenics.cli import load_pair
from vcell_fenics.results import Bundle
from vcell_fenics.runner import RunOptions, run_model

_HERE = Path(__file__).resolve().parent


def _fv_regions(model: Any, x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> Any:
    """The subvolume name at each FV node (Z, Y, X), by VCell's nearest-pixel rule."""

    geometry = model.geometry
    voxels = geometry.image.voxels()  # (nz, ny, nx)
    nz, ny, nx = voxels.shape
    (ox, oy, oz), (ex, ey, ez) = geometry.origin, geometry.extent

    def pixel(coord: NDArray[np.float64], origin: float, extent: float, n: int) -> NDArray[np.int64]:
        return np.clip(((n - 1) * (coord - origin) / extent + 0.5).astype(np.int64), 0, n - 1)

    iz, iy, ix = np.meshgrid(pixel(z, oz, ez, nz), pixel(y, oy, ey, ny), pixel(x, ox, ex, nx), indexing="ij")
    values = voxels[iz, iy, ix]
    names = {s.pixel_value: s.name for s in geometry.subvolumes}
    return np.vectorize(names.get)(values)


def _weights(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trapezoidal (fractional-boundary) FV control volumes on the vertex-centred grid (Z, Y, X)."""

    def axis(coord: NDArray[np.float64]) -> NDArray[np.float64]:
        w = np.full(coord.size, coord[1] - coord[0])
        w[0] = w[-1] = 0.5 * (coord[1] - coord[0])
        return w

    return np.einsum("k,j,i->kji", axis(z), axis(y), axis(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fv", type=int, default=101, help="the stage-1 FV mesh (its x size)")
    parser.add_argument("--h", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.05)
    args = parser.parse_args()
    reference = np.load(_HERE / f"image_nuclear_fv_{args.fv}.npz")
    t, x, y, z = (np.asarray(reference[k], dtype=float) for k in ("t", "x", "y", "z"))
    fv_c, fv_n = np.asarray(reference["c"], dtype=float), np.asarray(reference["n"], dtype=float)

    model = load_pair(_HERE / "image_nuclear_math.yaml", _HERE / "image_nuclear_geom.yaml")
    options = RunOptions(
        h=args.h, dt=args.dt, t_final=float(t[-1]), output_times=tuple(float(v) for v in t), fe_degree=1,
        time_integration="method_of_lines",
    )  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        start = time.time()
        run_model(model, options, Path(tmp), prefix="run")
        elapsed = time.time() - start
        bundle = Bundle.open(Path(tmp) / "run.fenics")
        ours_c, ours_n = bundle.stats("cytosol", "c"), bundle.stats("Nucleus", "n")
        coords = bundle.coords("cytosol", len(t) - 1)[:, :3]
        field_c = bundle.field("cytosol", "c", len(t) - 1)

    regions = _fv_regions(model, x, y, z)
    weights = _weights(x, y, z)
    in_cyt, in_nuc = regions == "cytosol", regions == "Nucleus"
    fv_mean_c = np.array([np.sum(fv_c[k] * weights * in_cyt) / np.sum(weights * in_cyt) for k in range(len(t))])
    fv_mean_n = np.array([np.sum(fv_n[k] * weights * in_nuc) / np.sum(weights * in_nuc) for k in range(len(t))])
    fv_total = np.array([np.sum((fv_c[k] * in_cyt + fv_n[k] * in_nuc) * weights) for k in range(len(t))])
    ours_total = ours_c[:, 1] + ours_n[:, 1]

    # pointwise c at FV cytosol nodes at the final time (interpolated from our P1 nodal field)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    probe = np.stack([xx[in_cyt], yy[in_cyt], zz[in_cyt]], axis=1)
    ours_at = LinearNDInterpolator(coords, field_c)(probe)
    keep = np.isfinite(ours_at)
    rel_l2 = float(np.linalg.norm(ours_at[keep] - fv_c[-1][in_cyt][keep]) / np.linalg.norm(fv_c[-1][in_cyt][keep]))

    volume_fv = float(np.sum(weights * in_cyt))
    volume_ours = float(ours_c[0, 1] / ours_c[0, 0])
    mesh = tuple(int(v) for v in reference["mesh"])
    print(f"nuclear exchange on VCell's tutorial image: FV {mesh} vs FEniCSx h = {args.h:g}, dt = {args.dt:g}")
    print(f"  ({elapsed:.0f} s) cytosol volume: FEniCSx {volume_ours:.1f} µm³, FV {volume_fv:.1f}")
    for label, ours, fv in (("n (nucleus)", ours_n[:, 0], fv_mean_n), ("c (cytosol)", ours_c[:, 0], fv_mean_c)):
        rows = "  ".join(f"t={t[k]:g}: {ours[k]:.4f}/{fv[k]:.4f}" for k in (4, 10, 20))
        print(f"  mean {label}: {rows}")
    worst_n = float(np.max(np.abs(ours_n[1:, 0] - fv_mean_n[1:]) / np.abs(fv_mean_n[1:])))
    print(
        f"  mean n max rel diff over time {worst_n:.2%};"
        f"  c relL2 at t={t[-1]:g}: {rel_l2:.2%} ({int(keep.sum())} nodes)"
    )
    drift_ours = abs(ours_total - ours_total[0]).max() / ours_total[0]
    drift_fv = abs(fv_total - fv_total[0]).max() / fv_total[0]
    print(f"  total substance drift: FEniCSx {drift_ours:.1e}, FV {drift_fv:.1e}")


if __name__ == "__main__":
    main()

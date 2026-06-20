#!/usr/bin/env python
"""Regenerate the membrane time-flux FV reference at several grid resolutions (heavy env).

Stage 1 of the membrane-flux joint-refinement convergence study. Comparing the FEniCSx solve to a
*single* FV grid can't show convergence — the FV solver's own 1st-order membrane error is then the
floor. So we run VCell's FV solver for the time-dependent membrane-flux model (`membrane_timeflux_fv`)
at **64²/128²/256²/512²/1024²** (feasible in 2D), and the dev-env `membrane_convergence.py` refines the
FEniCSx mesh alongside — the FV↔FEniCSx difference should then fall ~first order as *both* grids
refine (a body-fitted polygon membrane and a cut-cell stairstep membrane both → the true circle).

Five levels (not three) on purpose: the per-level error ratio bounces with how each grid's membrane
happens to align with the circle, so a short sweep can read a one-off dip as a plateau.

    ../pyvcell/.venv/bin/python cross_validation/membrane_convergence_fv.py

Writes `membrane_conv_fv_<N>.npz` {field (T,Y,X) cytosol u, t, x, y, radius} (gitignored, regenerable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
from membrane_timeflux_fv import _RADIUS, author_membrane_timeflux
from pyvcell.vcml.utils import to_vcml_str

_CV = Path(__file__).resolve().parent
_RESOLUTIONS = (64, 128, 256, 512, 1024)


def main() -> None:
    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(author_membrane_timeflux(mesh=(n, n, 1)))), "sim")
        try:
            zd = result.zarr_dataset
            index = {c.label: c.index for c in result.channel_data}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, index["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, index["y"], 0, :, 0], dtype=float)
            (u_name,) = (name.split("::")[-1] for name in result.volume_variable_names)
            field = np.asarray(zd[:, index[u_name], 0, :, :], dtype=float)  # (T, Y, X)
        finally:
            result.cleanup()
        np.savez_compressed(_CV / f"membrane_conv_fv_{n}.npz", field=field, t=t, x=x, y=y, radius=_RADIUS)
        print(f"N={n:5d}: field={field.shape}  dx={x[1] - x[0]:.5f}")


if __name__ == "__main__":
    main()

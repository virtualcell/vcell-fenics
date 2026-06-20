#!/usr/bin/env python
"""Generate FV reference fields at several grid resolutions for a joint-refinement convergence study.

Heavy env (pyvcell `[native,solver]`). Reuses the broad-Gaussian pure-diffusion model (the v1b IC,
well resolved even on the coarse grid) and runs VCell's FV solver at 64², 128², 256². The dev-env
`convergence_study.py` then solves the same model with the FEniCSx backend at a matching mesh size and
checks that the FV↔FEniCSx difference shrinks as *both* grids refine — evidence the residual error is
discretization, not a hidden bug.

    ../pyvcell/.venv/bin/python cross_validation/convergence_fv.py

Writes `convergence_fv_<N>.npz` {field (T,Y,X), t, x, y} for N in 64/128/256 (gitignored, regenerable).
The model's lowered math/geom are the v1b YAML already committed (mesh size is a simulation setting,
not part of the MathDescription), so the dev-env side imports those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from author_and_run import _gaussian_ic, author_and_run_diffusion_2d

_CV = Path(__file__).resolve().parent
_RESOLUTIONS = (64, 128, 256)


def main() -> None:
    for n in _RESOLUTIONS:
        out = author_and_run_diffusion_2d(ic_expr=_gaussian_ic(0.05), diffusion=0.1, mesh=(n, n, 1))
        np.savez_compressed(
            _CV / f"convergence_fv_{n}.npz", field=out["field"][:, 0], t=out["t"], x=out["x"], y=out["y"]
        )
        print(f"N={n:4d}: field={out['field'][:, 0].shape}  dx={out['x'][1] - out['x'][0]:.5f}")


if __name__ == "__main__":
    main()

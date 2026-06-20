#!/usr/bin/env python
"""Membrane-flux joint-refinement convergence: FV ↔ FEniCSx (dev env).

Stage 2 of the membrane convergence study. Reads the multi-resolution FV references from
`membrane_convergence_fv.py` and, for each grid `N`, solves the *same* time-dependent membrane-flux
model through the real multi-compartment pipeline (import → normalize-to-geometry-frame → realize →
run, no disk reduction) with the FEniCSx mesh refined alongside (`h ≈ 2/N`; realize resamples the
faceted membrane at the mesh scale). It samples our cytosol field at the FV-N grid and reports the
relative L2.

Because *both* grids refine, the FV↔FEniCSx difference falls ~first order toward 0 (the residual is
the 1st-order membrane discretization — FV's stairstep vs our body-fitted polygon, both → the true
circle), confirming the membrane flux converges to the same solution with no hidden bug. (The
per-level ratio bounces with grid/membrane alignment — read the trend over all five levels.)

    ../pyvcell/.venv/bin/python cross_validation/membrane_convergence_fv.py   # stage 1 (FV, heavy env)
    .pixi/envs/dev/bin/python   cross_validation/membrane_convergence.py      # stage 2 (this, dev env)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import yaml
from compare_fenics_vs_fv import _eval_on_grid

from vcell_fenics.backend import SolverConfiguration, run
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

_CV = Path(__file__).resolve().parent
_RESOLUTIONS = (64, 128, 256, 512, 1024)
_T = 0.5  # compare mid-run, where the flux-driven profile is well developed


def main() -> None:
    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "membrane_timeflux_geom.yaml").read_text()))
    md_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / "membrane_timeflux_math.yaml").read_text()))
    geometry_desc = import_geometry(geo)
    math = import_math_description(md_raw, geometry=geometry_desc.name, dim=2)
    geometry_desc, math = normalize_to_geometry_frame(geometry_desc, math)

    print(f"=== membrane flux: FV ↔ FEniCSx joint refinement, t={_T} ===")
    print(f"{'N':>5} {'h':>9} {'relL2(FEM,FV)':>14} {'ratio':>7}")
    previous = None
    for n in _RESOLUTIONS:
        ref = np.load(_CV / f"membrane_conv_fv_{n}.npz")
        x, y, radius = ref["x"], ref["y"], float(ref["radius"])
        ti = int(np.argmin(np.abs(ref["t"] - _T)))
        fv = ref["field"][ti]
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        inside = (grid_x**2 + grid_y**2) < (0.9 * radius) ** 2  # the cytosol interior (skip the rim)

        geometry = realize(geometry_desc, h=2.0 / n)  # resolution auto-resamples to the mesh scale
        problem = run(math, geometry, SolverConfiguration(dt=0.1, t_final=_T, time_integration="method_of_lines"))
        ours = _eval_on_grid(problem.unknown, geometry.mesh_of("cytosol_dom"), x, y)
        rel = float(np.linalg.norm(ours[inside] - fv[inside]) / np.linalg.norm(fv[inside]))
        print(f"{n:5d} {2.0 / n:9.4f} {rel:14.4%} {'' if previous is None else f'{previous / rel:.2f}x'}")
        previous = rel


if __name__ == "__main__":
    main()

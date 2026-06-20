#!/usr/bin/env python
"""Joint-refinement convergence study: FV ↔ FEniCSx on pure diffusion (dev env).

Answers "is a bug hiding in the ~2% FV/FEniCSx difference, or is it just discretization?" Three parts
on the broad-Gaussian pure-diffusion model (the v1b IC, well resolved on the coarse grid):

1. **Time-error isolation** (fixed FV 128², fixed FEniCSx mesh): the FV solver uses MOL (≈0 time
   error), so our **backward-Euler** step is a first-order time error in the comparison. Shrinking
   `dt` shows the FV↔FEniCSx difference fall ~linearly toward the spatial floor — most of the
   original ~2% was our time step, not a spatial mismatch.
2. **Joint refinement** (small-`dt` backward Euler vs the FV grid AND the free-space analytic): refine
   both grids; the FV↔FEniCSx relL2 stays small and the two solvers agree with **each other** far
   better than either agrees with the *free-space* analytic — that ~2% vs analytic is **wall
   reflection** (bounded domain), physics both capture, not discretization error. A bug would show as
   a large non-decreasing FV↔FEniCSx floor.
3. **MOL startup (found here, now fixed):** the method-of-lines integrator (PETSc `TSBDF`) over-diffused
   by a constant effective-time offset ≈ the *initial* step — a BDF order-1 cold-start error the
   adaptive controller does not catch (tightening `rtol` does nothing). Fixed by a small default
   startup step (and `run()` no longer forwards `config.dt` as the seed); the sweep below shows the
   **default** now lands on `t_final`, while a deliberately large explicit `dt_initial` still
   over-diffuses (why a backward-Euler-sized seed must not be forwarded).

Needs `convergence_fv_<N>.npz` from `convergence_fv.py`; imports the committed v1b lowered math.

    .pixi/envs/dev/bin/python cross_validation/convergence_study.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import yaml
from compare_fenics_vs_fv import _eval_on_grid

from vcell_fenics.backend import SolverConfiguration, assemble, run
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_CV = Path(__file__).resolve().parent
_A, _D, _CX = 10.0, 0.1, 0.3  # the v1b IC amplitude, diffusion, centre (width a=0.05 below)


def _analytic(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Free-space Gaussian-under-diffusion reference (exact until the front reaches a wall)."""
    s = 0.05 + 4.0 * _D * t
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    return _A * (0.05 / s) * np.exp(-((grid_x - _CX) ** 2 + grid_y**2) / s)


def main() -> None:
    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "minimal_diffusion_2d_v1b_geom.yaml").read_text()))
    md_raw = mmod.MathDescription.model_validate(
        yaml.safe_load((_CV / "minimal_diffusion_2d_v1b_math.yaml").read_text())
    )
    gd = import_geometry(geo)
    md = import_math_description(md_raw, geometry=gd.name, dim=2)
    name = gd.subvolumes[0].name
    ref = {n: np.load(_CV / f"convergence_fv_{n}.npz") for n in (64, 128, 256)}
    t = ref[128]["t"]
    t_final = 0.5
    ti = int(np.argmin(np.abs(t - t_final)))

    def be_relL2(h: float, dt: float, fv: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        geometry = realize(gd, h=h)
        problem = run(md, geometry, SolverConfiguration(dt=dt, t_final=t_final))  # backward euler
        ours = _eval_on_grid(problem.unknown, geometry.mesh_of(name), x, y)
        an = _analytic(x, y, t_final)
        return (
            float(np.linalg.norm(ours - fv) / np.linalg.norm(fv)),
            float(np.linalg.norm(ours - an) / np.linalg.norm(an)),
        )

    fv128, x128, y128 = ref[128]["field"][ti], ref[128]["x"], ref[128]["y"]
    print(f"=== 1. time-error isolation (FV 128², our h={2 / 128:.4f}, t={t_final}) ===")
    for dt in (0.02, 0.01, 0.005, 0.0025):
        rel, _ = be_relL2(2 / 128, dt, fv128, x128, y128)
        print(f"  backward Euler dt={dt:<6} relL2(FEM,FV) = {rel:.4%}")

    print(f"\n=== 2. joint refinement (backward Euler dt=0.0025, our h=2/N, t={t_final}) ===")
    print(f"{'N':>5} {'h':>8} {'relL2(FEM,FV)':>14} {'FEM vs analytic':>16} {'FV vs analytic':>15}")
    for n in (64, 128, 256):
        fv, x, y = ref[n]["field"][ti], ref[n]["x"], ref[n]["y"]
        rel_fv, rel_an = be_relL2(2 / n, 0.0025, fv, x, y)
        rel_fv_an = float(np.linalg.norm(fv - _analytic(x, y, t_final)) / np.linalg.norm(_analytic(x, y, t_final)))
        print(f"{n:5d} {2 / n:8.4f} {rel_fv:13.4%} {rel_an:15.4%} {rel_fv_an:14.4%}")

    print(f"\n=== 3. MOL startup: fixed via a small default step (FV 128², our h={2 / 128:.4f}, t={t_final}) ===")
    print("   effective diffusion-time should equal t_final; a large *explicit* startup over-diffuses:")
    for label, dt_initial in (("default", None), ("explicit 0.005", 0.005), ("explicit 0.05 (BE-sized)", 0.05)):
        geometry = realize(gd, h=2 / 128)
        problem = assemble(md, geometry, dt=0.05)
        result = integrate_discrete_problem(problem, t_final=t_final, dt_initial=dt_initial)
        peak = float(np.nanmax(_eval_on_grid(result.solution, geometry.mesh_of(name), x128, y128)))
        eff_t = (_A * 0.05 / peak - 0.05) / (4.0 * _D)  # invert the analytic peak A·a/(a+4Dt)
        print(f"  MOL dt_initial={label:24} effective_t={eff_t:.4f} (want {t_final})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 2 of the **3D reaction-advection-diffusion** cross-validation (vcell-fenics dev env).

`advection_3d_fv.py` (stage 1) ran VCell's FV solver on a diffusing **and advecting** Gaussian (velocity
`v = (0.4, 0, 0)`, D = 0.03) at 32³, and saved the 5D field + lowered math/geometry. The lowered math
carries the species velocity, which our bridge imports into the `bulk_radv_diff` `relative_advection`
slot — so `run()` solves advection-diffusion with no special handling. This stage imports the same model,
**refines the FEniCSx box** `h = 0.1 → 0.05 → 0.025`, solves method-of-lines, samples on the FV grid, and
reports both L2 and L∞ vs FV and vs the free-space advecting-Gaussian analytic (the clean convergence
target at t = 0), plus the centre-of-mass (confirming the field advects at v).

    .pixi/envs/dev/bin/python cross_validation/compare_3d_advection.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import ufl
import yaml
from compare_3d import _eval_on_grid  # sibling stage-2 script (same 3D grid sampler)
from dolfinx import fem
from numpy.typing import NDArray

from vcell_fenics.backend import SolverConfiguration, run
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_CV = Path(__file__).resolve().parent
_STEM = "advection_3d"
_CENTER0 = (-0.5, 0.0, 0.0)
_V = (0.4, 0.0, 0.0)
_SIGMA = 0.12
_D = 0.03
_HS = (0.1, 0.05, 0.025)
_T_SOLVE = 0.5


def _analytic(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Free-space advecting-diffusing Gaussian: centre moves to C0 + v·t, variance σ² + 2 D t."""
    var = _SIGMA**2 + 2.0 * _D * t
    cx, cy, cz = (_CENTER0[k] + _V[k] * t for k in range(3))
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    r2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    return np.transpose((_SIGMA**2 / var) ** 1.5 * np.exp(-r2 / (2.0 * var)), (2, 1, 0))  # (Z, Y, X)


def _com_x(field: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """x centre-of-mass of a (Z, Y, X) field."""
    return float((field.sum(axis=(0, 1)) * x).sum() / field.sum())


def main() -> int:
    ref = np.load(_CV / f"{_STEM}_reference.npz", allow_pickle=True)
    fv = ref["field"][:, 0]  # (T, Z, Y, X)
    t, x, y, z = (ref[k].astype(float) for k in ("t", "x", "y", "z"))
    ti = int(np.argmin(np.abs(t - _T_SOLVE)))
    t_solve = float(t[ti])
    an_ic = _analytic(x, y, z, 0.0)

    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / f"{_STEM}_geom.yaml").read_text()))
    md_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / f"{_STEM}_math.yaml").read_text()))
    gd = import_geometry(geo)
    md = import_math_description(md_raw, geometry=gd.name, dim=3)

    print(f"\n=== {_STEM} h-refinement ===  v={_V}, D={_D}; FV grid={fv.shape[1]}³, FEM↔FV at t={t_solve}")
    print(f"FV centre-of-mass x: {_com_x(fv[0], x):.3f} → {_com_x(fv[ti], x):.3f} (Δ≈v_x·t={_V[0] * t_solve:.3f})")
    header = ("h", "box", "IC relL2(an)", "IC relL∞(an)", "ord∞", "relL2(FV)", "relL∞(FV)", "com_x(FEM)", "massDrift")
    widths = (6, 7, 13, 13, 5, 10, 10, 11, 10)
    print(" ".join(f"{c:>{w}}" for c, w in zip(header, widths, strict=True)))

    prev_linf: float | None = None
    for h in _HS:
        geometry = realize(gd, h=h)
        mesh = geometry.mesh_of(gd.subvolumes[0].name)
        dx_form = ufl.dx(domain=mesh)

        ic = assemble(md, geometry, dt=0.1).unknown
        ic_grid = _eval_on_grid(ic, mesh, x, y, z)
        ic_l2 = np.linalg.norm(ic_grid - an_ic) / (np.linalg.norm(an_ic) or 1.0)
        ic_linf = float(np.abs(ic_grid - an_ic).max() / (np.abs(an_ic).max() or 1.0))
        order = float(np.log2(prev_linf / ic_linf)) if prev_linf else float("nan")
        mass0 = float(fem.assemble_scalar(fem.form(ic * dx_form)).real)

        config = SolverConfiguration(dt=0.1, t_final=t_solve, time_integration="method_of_lines")
        sol = run(md, geometry, config).unknown
        sol_grid = _eval_on_grid(sol, mesh, x, y, z)
        s_l2 = np.linalg.norm(sol_grid - fv[ti]) / (np.linalg.norm(fv[ti]) or 1.0)
        s_linf = np.abs(sol_grid - fv[ti]).max() / (np.abs(fv[ti]).max() or 1.0)
        mass = float(fem.assemble_scalar(fem.form(sol * dx_form)).real)

        print(
            f"{h:6.3f} {f'{round(2.0 / h)}³':>7} {ic_l2:13.3%} {ic_linf:13.3%} {order:5.2f} "
            f"{s_l2:10.3%} {s_linf:10.3%} {_com_x(sol_grid, x):11.3f} {abs(mass - mass0) / mass0:10.1e}"
        )
        prev_linf = ic_linf

    print(f"(FEM com_x should reach ≈ {_CENTER0[0] + _V[0] * t_solve:.3f} = C0 + v_x·t — the field advects at v)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

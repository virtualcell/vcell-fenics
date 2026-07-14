#!/usr/bin/env python
"""Stage 2 of the **3D** single-species diffusion cross-validation (vcell-fenics dev env).

`diffusion_3d_fv.py` (stage 1, pyvcell env) authored a common VCML, ran VCell's finite-volume solver
on a 32³ grid, and saved the 5D field + the lowered math/geometry YAML. This stage imports the *same*
lowered math + geometry, realizes the 3D box, solves through our FEniCSx backend with the
method-of-lines integrator (PETSc TS adaptive BDF — the same strategy as the FV solver's Sundials/CVODE,
so the time error ≈ 0 and the comparison isolates the *spatial* discretisation), samples our solution
at the FV grid points, and reports the agreement (relative L2 per output time vs FV and vs the
free-space analytic, plus total-mass).

    .pixi/envs/dev/bin/python cross_validation/compare_3d.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import ufl
import yaml
from dolfinx import fem
from dolfinx import geometry as dgeom
from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from numpy.typing import NDArray

from vcell_fenics.backend import SolverConfiguration, run
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_CV = Path(__file__).resolve().parent
_STEM = "diffusion_3d"
# Free-space analytic reference (the authored IC in diffusion_3d_fv.py): a Gaussian centred at
# (0.2, 0.1, 0) with σ = 0.15 diffusing at D = 0.1 spreads to σ² + 2 D t (valid until the front
# reaches the no-flux box wall, after which both solvers reflect and diverge from free space).
_CENTER = (0.2, 0.1, 0.0)
_SIGMA = 0.15
_D = 0.1
_HS = (0.1, 0.05, 0.025)  # FEniCSx box mesh sizes — refine to watch convergence (structured box: 20³→40³→80³)
_T_SOLVE = 0.5  # a representative time for cross-solver (FEM↔FV) agreement


def _analytic(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    var = _SIGMA**2 + 2.0 * _D * t
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")  # (X, Y, Z) then transpose to (Z, Y, X)
    r2 = (gx - _CENTER[0]) ** 2 + (gy - _CENTER[1]) ** 2 + (gz - _CENTER[2]) ** 2
    amp = (_SIGMA**2 / var) ** 1.5
    return np.transpose(amp * np.exp(-r2 / (2.0 * var)), (2, 1, 0))  # (Z, Y, X)


def _eval_on_grid(
    u: Function, mesh: Mesh, xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate scalar `u` at every (x[i], y[j], z[k]) — a (Z, Y, X) array matching the FV `[k, j, i]`
    layout. Points located by DOLFINx collision queries; any outside the mesh stays NaN."""
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="xy")  # (Y, X, Z)
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    tree = dgeom.bb_tree(mesh, mesh.topology.dim)
    candidates = dgeom.compute_collisions_points(tree, pts)
    colliding = dgeom.compute_colliding_cells(mesh, candidates, pts)
    cells: list[int] = []
    rows: list[int] = []
    for i in range(pts.shape[0]):
        links = colliding.links(np.int32(i))
        if links.size > 0:
            cells.append(int(links[0]))
            rows.append(i)
    out = np.full(pts.shape[0], np.nan)
    out[rows] = u.eval(pts[rows], np.asarray(cells, dtype=np.int32))[:, 0]
    return np.transpose(out.reshape(gx.shape), (2, 0, 1))  # (Y,X,Z) -> (Z,Y,X)


def main() -> int:
    ref = np.load(_CV / f"{_STEM}_reference.npz", allow_pickle=True)
    fv = ref["field"][:, 0]  # (T, Z, Y, X) — species u
    t, x, y, z = (ref[k].astype(float) for k in ("t", "x", "y", "z"))
    ti = int(np.argmin(np.abs(t - _T_SOLVE)))  # closest FV output time to _T_SOLVE
    t_solve = float(t[ti])
    an_ic = _analytic(x, y, z, 0.0)  # exact IC (the authored Gaussian) — clean spatial-convergence target

    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / f"{_STEM}_geom.yaml").read_text()))
    md_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / f"{_STEM}_math.yaml").read_text()))
    gd = import_geometry(geo)
    md = import_math_description(md_raw, geometry=gd.name, dim=3)

    print(f"\n=== {_STEM} h-refinement ===  FV grid={fv.shape[1]}³, FEM↔FV compared at t={t_solve}")
    print("The box is a structured whole-box mesh (geometry exact at every h); refinement resolves the field.")
    header = ("h", "box", "tets", "IC relL2(an)", "IC relLinf(an)", "ord∞", "relL2(FV)", "relLinf(FV)", "massDrift")
    widths = (6, 7, 9, 13, 15, 5, 11, 13, 10)
    print(" ".join(f"{c:>{w}}" for c, w in zip(header, widths, strict=True)))

    prev_linf: float | None = None
    for h in _HS:
        geometry = realize(gd, h=h)
        mesh = geometry.mesh_of(gd.subvolumes[0].name)
        dx_form = ufl.dx(domain=mesh)
        nx = round(2.0 / h)
        ntet = mesh.topology.index_map(3).size_local

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
            f"{h:6.3f} {f'{nx}³':>7} {ntet:9d} {ic_l2:13.3%} {ic_linf:15.3%} {order:5.2f} "
            f"{s_l2:11.3%} {s_linf:13.3%} {abs(mass - mass0) / mass0:10.1e}"
        )
        prev_linf = ic_linf

    span = [(float(mesh.geometry.x[:, d].min()), float(mesh.geometry.x[:, d].max())) for d in range(3)]
    print(f"(finest box mesh spans x{span[0]} y{span[1]} z{span[2]} — exactly the authored [-1,1]³)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

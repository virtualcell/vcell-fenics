#!/usr/bin/env python
"""Stage 2 of the FV ↔ FEniCSx cross-validation (vcell-fenics **dev env**).

`author_and_run.py` (stage 1, pyvcell `[native,solver]` env) authored a common VCML, ran VCell's
finite-volume solver, and saved the 4D field + the lowered math/geometry YAML next to each model.
This stage imports the *same* lowered math + geometry through the pyvcell bridge, realizes the
geometry, solves it through our FEniCSx backend, samples our solution at the FV solver's own grid
points, and reports the agreement (relative L2 per output time, plus total-mass conservation).

    .pixi/envs/dev/bin/python cross_validation/compare_fenics_vs_fv.py [stem ...]

`field[ti, 0, j, i]` is the FV value of species `u` at `(x[i], y[j])`, time `t[ti]`; we evaluate
our `u_h` at exactly those `(x[i], y[j])` so the comparison is element-wise, no re-interpolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import yaml
from dolfinx import geometry as dgeom
from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from numpy.typing import NDArray

from vcell_fenics.backend import SolverConfiguration
from vcell_fenics.backend.assemble import assemble
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_CV = Path(__file__).resolve().parent
_SOLVER_DT = 0.01  # finer than the FV output interval; we snapshot at each output time


def _eval_on_grid(u: Function, mesh: Mesh, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the scalar field `u` at every (x[i], y[j]) — returns a (Y, X) array matching the FV
    field's `[j, i]` layout. Points are located by DOLFINx collision queries; any point outside the
    mesh (should be none — the grid is the box) stays NaN."""

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")  # (Y, X)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
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
    return out.reshape(grid_y.shape)  # (Y, X)


def _compare(stem: str) -> None:
    ref = np.load(_CV / f"{stem}_reference.npz", allow_pickle=True)
    fv = ref["field"][:, 0]  # (T, Y, X) — species u
    analytic = ref["analytic"][:, 0]
    t, x, y = ref["t"].astype(float), ref["x"].astype(float), ref["y"].astype(float)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / f"{stem}_geom.yaml").read_text()))
    md_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / f"{stem}_math.yaml").read_text()))
    gd = import_geometry(geo)
    md = import_math_description(md_raw, geometry=gd.name, dim=2)
    geometry = realize(gd, h=0.02)
    mesh = geometry.mesh_of(gd.subvolumes[0].name)

    problem = assemble(md, geometry, dt=_SOLVER_DT)
    output_times = t[1:]  # t[0] is the IC, already in the function
    next_out = 0
    ours = np.empty_like(fv)
    ours[0] = _eval_on_grid(problem.unknown, mesh, x, y)  # the interpolated IC at t=0

    nsteps = round(float(t[-1]) / _SOLVER_DT)
    for n in range(nsteps):
        now = (n + 1) * _SOLVER_DT
        problem.set_time(now)
        problem.step()
        if next_out < len(output_times) and np.isclose(now, output_times[next_out], atol=_SOLVER_DT / 2):
            ours[next_out + 1] = _eval_on_grid(problem.unknown, mesh, x, y)
            next_out += 1

    print(f"\n=== {stem} ===  mesh cells={mesh.topology.index_map(2).size_local}  grid={fv.shape[1]}x{fv.shape[2]}")
    print(f"{'t':>6} {'relL2(FEM,FV)':>14} {'relL2(FEM,an)':>14} {'mass_FEM':>10} {'mass_FV':>10} {'peak_FEM':>9} {'peak_FV':>9}")
    for ti in range(len(t)):
        denom = np.linalg.norm(fv[ti]) or 1.0
        rel_fv = np.linalg.norm(ours[ti] - fv[ti]) / denom
        rel_an = np.linalg.norm(ours[ti] - analytic[ti]) / (np.linalg.norm(analytic[ti]) or 1.0)
        mass_fem = ours[ti].sum() * dx * dy
        mass_fv = fv[ti].sum() * dx * dy
        print(
            f"{t[ti]:6.2f} {rel_fv:14.4%} {rel_an:14.4%} {mass_fem:10.5f} {mass_fv:10.5f} "
            f"{ours[ti].max():9.4f} {fv[ti].max():9.4f}"
        )


def main() -> int:
    stems = sys.argv[1:] or [p.name[: -len("_reference.npz")] for p in sorted(_CV.glob("*_reference.npz"))]
    for stem in stems:
        _compare(stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

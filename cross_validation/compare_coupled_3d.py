#!/usr/bin/env python
"""Stage 2 of the **3D interface-coupled permeability** cross-validation (vcell-fenics dev env).

`coupled_3d_fv.py` (stage 1) ran VCell's FV solver on a sphere-in-box two-compartment permeability model
(`s_cyto` inside, `s_ext` outside, coupled by `J = P·(s_ext − s_cyto)`) at 32³ and 48³. This stage runs
the **same model** through the genuine pipeline — import VCell geometry+math → `normalize_to_geometry_frame`
→ `realize_interface_coupled` (the imported 3D geometry, Netgen multi-region) → `integrate_interface_coupled`
(method-of-lines) — with the FEniCSx mesh **refined** alongside, and samples our two compartment fields at
the FV grid points in their own regions (`s_cyto` inside the sphere, `s_ext` outside), reporting L2 and L∞
vs FV plus total-substance conservation.

    ../pyvcell/.venv/bin/python cross_validation/coupled_3d_fv.py         # stage 1 (FV, heavy env)
    .pixi/envs/dev/bin/python   cross_validation/compare_coupled_3d.py    # stage 2 (this, dev env)
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

from vcell_fenics.backend.interface_coupled import integrate_interface_coupled
from vcell_fenics.backend.realize import realize_interface_coupled
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

_CV = Path(__file__).resolve().parent
_REF_N = 48  # the finer FV reference
_HS = (0.1, 0.067, 0.05)  # FEniCSx refinement
_T = 1.0  # mid-transient — the two compartment means are still well apart (sensitive to the coupling)


def _eval_on_grid(
    u: Function, mesh: Mesh, xs: NDArray[np.float64], ys: NDArray[np.float64], zs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate scalar `u` at every (x[i], y[j], z[k]) — a (Z, Y, X) array; points outside the mesh (e.g.
    the other compartment) stay NaN (masked out by the caller)."""
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


def _total_substance(field: Function) -> float:
    mesh = field.function_space.mesh
    return float(fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh))).real)


def main() -> int:
    ref = np.load(_CV / f"coupled_3d_fv_{_REF_N}.npz")
    x, y, z, radius = ref["x"], ref["y"], ref["z"], float(ref["radius"])
    ti = int(np.argmin(np.abs(ref["t"] - _T)))
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    r2 = np.transpose(gx**2 + gy**2 + gz**2, (2, 1, 0))  # (Z,Y,X)
    in_cyto = r2 < (0.9 * radius) ** 2  # skip the membrane band on both sides
    in_ext = r2 > (1.1 * radius) ** 2
    fv_cyto, fv_ext = ref["s_cyto"][ti], ref["s_ext"][ti]

    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "coupled_3d_geom.yaml").read_text()))
    math_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / "coupled_3d_math.yaml").read_text()))
    gd0 = import_geometry(geo)
    md0 = import_math_description(math_raw, geometry=gd0.name, dim=3)

    print(f"\n=== coupled_3d permeability: FV↔FEniCSx refinement, t={_T} (FV {_REF_N}³ reference) ===")
    print(f"FV means: s_cyto={np.nanmean(fv_cyto[in_cyto]):.4f}  s_ext={np.nanmean(fv_ext[in_ext]):.4f}")
    header = ("h", "inner+outer tets", "relL2(FV)", "relL∞(FV)", "ratio", "s_cyto(FEM)", "s_ext(FEM)", "massDrift")
    widths = (6, 17, 10, 10, 6, 12, 11, 10)
    print(" ".join(f"{c:>{w}}" for c, w in zip(header, widths, strict=True)))

    prev: float | None = None
    for h in _HS:
        gd, md = normalize_to_geometry_frame(gd0, md0)
        geometry = realize_interface_coupled(
            gd, inner_subdomain="cyto_dom", outer_subdomain="ext_dom", membrane_subdomain="mem_dom",
            interface="mem_dom", h=h,
        )
        result = integrate_interface_coupled(md, geometry, t_final=_T)
        our_cyto = _eval_on_grid(result.inner, geometry.inner_mesh, x, y, z)
        our_ext = _eval_on_grid(result.outer, geometry.outer_mesh, x, y, z)

        err2 = np.nansum((our_cyto[in_cyto] - fv_cyto[in_cyto]) ** 2)
        err2 += np.nansum((our_ext[in_ext] - fv_ext[in_ext]) ** 2)
        ref2 = np.nansum(fv_cyto[in_cyto] ** 2) + np.nansum(fv_ext[in_ext] ** 2)
        rel_l2 = float(np.sqrt(err2 / ref2))
        linf = max(
            float(np.nanmax(np.abs(our_cyto[in_cyto] - fv_cyto[in_cyto]))),
            float(np.nanmax(np.abs(our_ext[in_ext] - fv_ext[in_ext]))),
        ) / max(float(np.nanmax(np.abs(fv_cyto[in_cyto]))), float(np.nanmax(np.abs(fv_ext[in_ext]))))

        # Solver conservation: total substance ∫s_cyto dV + ∫s_ext dV vs the *realized* initial mass
        # (s_cyto ≡ 1 in cyto, s_ext ≡ 0 → ∫s(0) dV = the realized cyto volume). Using the realized volume
        # (not analytic 4/3πr³) isolates the solver's conservation from the faceted-sphere geometry error.
        mass = _total_substance(result.inner) + _total_substance(result.outer)
        m0 = float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=geometry.inner_mesh))).real)
        ratio = "" if prev is None else f"{prev / rel_l2:.2f}x"
        ntet = geometry.inner_mesh.topology.index_map(3).size_local
        ntet += geometry.outer_mesh.topology.index_map(3).size_local
        print(
            f"{h:6.3f} {ntet:17d} {rel_l2:10.3%} {linf:10.3%} {ratio:>6} "
            f"{np.nanmean(our_cyto[in_cyto]):12.4f} {np.nanmean(our_ext[in_ext]):11.4f} {abs(mass - m0) / m0:10.1e}"
        )
        prev = rel_l2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

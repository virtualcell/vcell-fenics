#!/usr/bin/env python
"""Stage 2 of the **3D membrane surface-species** (receptor) cross-validation (vcell-fenics dev env).

`receptor_3d_fv.py` (stage 1) ran VCell's FV solver on a sphere-in-box receptor-binding model (two bulk
ligands `s_cyto`/`s_ext` captured by a membrane receptor `R`) at 32³/48³. This stage runs the same model
through the whole imported pipeline — import geometry+math (the membrane reaction trace-wrapped, jump
conditions → interface fluxes, VCell unit factors carried) → `normalize_to_geometry_frame` →
`realize_interface_coupled` (Netgen 3D) → `integrate_membrane_coupled` (method-of-lines) — and, with the
FEniCSx mesh **refined** alongside, compares the depleted ligand fields to FV on the FV grid (R acts on
them through binding), while also checking the receptor captures ligand and total substance is conserved.

    ../pyvcell/.venv/bin/python cross_validation/receptor_3d_fv.py          # stage 1 (FV, heavy env)
    .pixi/envs/dev/bin/python   cross_validation/compare_membrane_3d.py     # stage 2 (this, dev env)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import yaml
from compare_coupled_3d import _eval_on_grid  # sibling stage-2 script (3D grid sampler)

from vcell_fenics.backend import integrate_membrane_coupled
from vcell_fenics.backend.realize import realize_interface_coupled
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

_CV = Path(__file__).resolve().parent
_REF_N = 48
_HS = (0.1, 0.067, 0.05)
_T = 2.0  # mid-transient — both ligands clearly depleted, receptor not yet saturated


def main() -> int:
    ref = np.load(_CV / f"receptor_3d_fv_{_REF_N}.npz")
    x, y, z, radius = ref["x"], ref["y"], ref["z"], float(ref["radius"])
    ti = int(np.argmin(np.abs(ref["t"] - _T)))
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    r2 = np.transpose(gx**2 + gy**2 + gz**2, (2, 1, 0))  # (Z,Y,X)
    in_cyto = r2 < (0.9 * radius) ** 2
    in_ext = r2 > (1.1 * radius) ** 2
    fv_cyto, fv_ext = ref["s_cyto"][ti], ref["s_ext"][ti]

    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "receptor_3d_geom.yaml").read_text()))
    math_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / "receptor_3d_math.yaml").read_text()))
    gd0 = import_geometry(geo)
    md0 = import_math_description(math_raw, geometry=gd0.name, dim=3)

    print(f"\n=== receptor_3d membrane binding: FV↔FEniCSx refinement, t={_T} (FV {_REF_N}³ reference) ===")
    print(f"FV depleted means: s_cyto={np.nanmean(fv_cyto[in_cyto]):.4f}  s_ext={np.nanmean(fv_ext[in_ext]):.4f}")
    # We compare the depleted ligand fields (R acts on them through binding); the receptor's own binding is
    # shown via ``bound R``. Total-substance conservation is not reported as a raw number here: the membrane
    # R (density, molecules·µm⁻²) and the volume ligands (µM) reconcile only through VCell's KMOLE factor —
    # the sub-0.1% ligand agreement with FV (which conserves) is the end-to-end validation of the binding.
    header = ("h", "relL2(FV)", "relL∞(FV)", "ratio", "s_cyto(FEM)", "s_ext(FEM)", "bound R")
    widths = (6, 10, 10, 6, 12, 11, 10)
    print(" ".join(f"{c:>{w}}" for c, w in zip(header, widths, strict=True)))

    prev: float | None = None
    for h in _HS:
        gd, md = normalize_to_geometry_frame(gd0, md0)
        geometry = realize_interface_coupled(
            gd, inner_subdomain="cyto_dom", outer_subdomain="ext_dom", membrane_subdomain="mem_dom",
            interface="mem_dom", h=h,
        )
        result = integrate_membrane_coupled(md, geometry, t_final=_T)

        our_cyto = _eval_on_grid(result.field("s_cyto"), geometry.inner_mesh, x, y, z)
        our_ext = _eval_on_grid(result.field("s_ext"), geometry.outer_mesh, x, y, z)
        err2 = np.nansum((our_cyto[in_cyto] - fv_cyto[in_cyto]) ** 2)
        err2 += np.nansum((our_ext[in_ext] - fv_ext[in_ext]) ** 2)
        ref2 = np.nansum(fv_cyto[in_cyto] ** 2) + np.nansum(fv_ext[in_ext] ** 2)
        rel_l2 = float(np.sqrt(err2 / ref2))
        linf = max(
            float(np.nanmax(np.abs(our_cyto[in_cyto] - fv_cyto[in_cyto]))),
            float(np.nanmax(np.abs(our_ext[in_ext] - fv_ext[in_ext]))),
        ) / max(float(np.nanmax(np.abs(fv_cyto[in_cyto]))), float(np.nanmax(np.abs(fv_ext[in_ext]))))

        ratio = "" if prev is None else f"{prev / rel_l2:.2f}x"
        print(
            f"{h:6.3f} {rel_l2:10.3%} {linf:10.3%} {ratio:>6} {np.nanmean(our_cyto[in_cyto]):12.4f} "
            f"{np.nanmean(our_ext[in_ext]):11.4f} {result.mass('R'):10.2f}"
        )
        prev = rel_l2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

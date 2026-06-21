#!/usr/bin/env python
"""Cross-compartment permeability coupling: FV ↔ FEniCSx joint-refinement convergence (dev env).

Stage 2 of the definitive cross-solver check for `integrate_interface_coupled`. Reads the
multi-resolution FV references from `coupled_perm_fv.py` and, for each grid N, runs the **same model**
through the genuine pipeline — import VCell's geometry → `normalize_to_geometry_frame` →
`realize_interface_coupled` (the imported geometry, *not* a hand-built mesh) → `integrate_interface_coupled`
(method-of-lines) — with the FEniCSx mesh refined alongside (h ≈ 2/N). It samples our two
compartment fields at the FV grid points in their own regions (s_cyto inside the disk, s_ext outside)
and reports the relative L2.

The math is a hand-written flux-balance model matching VCell's permeability flux (P=0.5, D=1, unit
factor 1 — verified equal): the importer does not yet route a coupled jump condition to the flux-balance
path, so the *geometry* comes from the import pipeline (the point of the test) while the *physics* is
matched by hand. Both solvers use MOL (≈0 time error); refining both grids drives the FV↔FEniCSx
difference down ~first order (the membrane is 1st-order on each side), confirming the coupled solver
converges to the FV solution.

    ../pyvcell/.venv/bin/python cross_validation/coupled_perm_fv.py            # stage 1 (FV, heavy env)
    .pixi/envs/dev/bin/python   cross_validation/coupled_perm_convergence.py   # stage 2 (this, dev env)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import yaml
from compare_fenics_vs_fv import _eval_on_grid

from vcell_fenics.backend.interface_coupled import integrate_interface_coupled
from vcell_fenics.backend.realize import realize_interface_coupled
from vcell_fenics.formalism.schema import (
    BCInterfaceFluxBalance,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.pyvcell_bridge import import_geometry, normalize_to_geometry_frame

_CV = Path(__file__).resolve().parent
# FEniCSx caps lower than FV: the body-fitted realize + the coupled MOL block solve (GMRES+ILU) make
# 256² (~170k cells, ~5 min) the practical ceiling here, where 512² is cheap for the FV grid. 64/128/256
# is enough to show the joint-refinement trend.
_RESOLUTIONS = (64, 128, 256)
_T = 1.0  # compare mid-transient (the means are still well apart, sensitive to the coupling)


def _model(geometry_name: str, *, permeability: float, diffusion: float) -> MathDescription:
    return MathDescription(
        geometry=geometry_name,
        subdomains=[Subdomain(name="cyto_dom", kind="volume"), Subdomain(name="ext_dom", kind="volume")],
        variables=[Variable(name="s_cyto", subdomain="cyto_dom"), Variable(name="s_ext", subdomain="ext_dom")],
        parameters=[ParameterConstant(name="P", value=permeability)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="s_cyto",
                subdomain="cyto_dom",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion)},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="s_ext",
                subdomain="ext_dom",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion)},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFluxBalance(
                variable="s_cyto", partner_variable="s_ext", boundary="mem_dom", expression="P * (s_ext - s_cyto)"
            )
        ],
    )


def main() -> None:
    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "coupled_perm_geom.yaml").read_text()))
    geometry_desc = import_geometry(geo)

    print(f"=== permeability coupling: FV ↔ FEniCSx joint refinement, t={_T} ===")
    print(f"{'N':>5} {'h':>9} {'relL2(FEM,FV)':>14} {'ratio':>7}  (s_cyto in disk, s_ext outside)")
    previous = None
    for n in _RESOLUTIONS:
        ref = np.load(_CV / f"coupled_perm_fv_{n}.npz")
        x, y, radius, p, d = ref["x"], ref["y"], float(ref["radius"]), float(ref["P"]), float(ref["D"])
        ti = int(np.argmin(np.abs(ref["t"] - _T)))
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        r2 = grid_x**2 + grid_y**2
        in_disk = r2 < (0.9 * radius) ** 2  # interior of the cytosol (skip the membrane band)
        in_ext = r2 > (1.1 * radius) ** 2  # interior of the extracellular

        gd, md = normalize_to_geometry_frame(geometry_desc, _model(geometry_desc.name, permeability=p, diffusion=d))
        geometry = realize_interface_coupled(
            gd,
            inner_subdomain="cyto_dom",
            outer_subdomain="ext_dom",
            membrane_subdomain="mem_dom",
            interface="mem_dom",
            h=2.0 / n,
        )
        result = integrate_interface_coupled(md, geometry, t_final=_T)
        ours_cyto = _eval_on_grid(result.inner, geometry.inner_mesh, x, y)
        ours_ext = _eval_on_grid(result.outer, geometry.outer_mesh, x, y)

        # Combine the two regions: compare s_cyto on the disk and s_ext outside, against the FV fields.
        err2 = np.nansum((ours_cyto[in_disk] - ref["s_cyto"][ti][in_disk]) ** 2)
        err2 += np.nansum((ours_ext[in_ext] - ref["s_ext"][ti][in_ext]) ** 2)
        ref2 = np.nansum(ref["s_cyto"][ti][in_disk] ** 2) + np.nansum(ref["s_ext"][ti][in_ext] ** 2)
        rel = float(np.sqrt(err2 / ref2))
        print(f"{n:5d} {2.0 / n:9.4f} {rel:14.4%} {'' if previous is None else f'{previous / rel:.2f}x'}")
        previous = rel


if __name__ == "__main__":
    main()

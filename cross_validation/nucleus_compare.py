#!/usr/bin/env python
"""Stage 2 of the three-compartment cross-solver check: FV ↔ FEniCSx joint refinement for the
multi-compartment solver (`integrate_multi_compartment`).

Reads the lowered geometry+math (`nucleus_{geom,math}.yaml`) and the FV fields from `nucleus_fv.py`, runs the
imported pipeline (`import_geometry` + `import_math_description` → `realize_multi_compartment` →
`integrate_multi_compartment`), and compares each ligand to FV on the FV grid, away from the membranes
(a band of 0.1·r around each), under joint refinement; and the substance total, KMOLE-reconciled
(∫s_nuc + ∫s_cyto + ∫s_ext + KMOLE·∫R), over time.

    ../pyvcell/.venv/bin/python cross_validation/nucleus_fv.py        # stage 1 (FV, pyvcell env)
    .pixi/envs/dev/bin/python   cross_validation/nucleus_compare.py   # stage 2 (this, dev env)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml.models_geometry as gmod
import pyvcell.vcml.models_math as mmod
import yaml
from compare_fenics_vs_fv import _eval_on_grid

from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment, species_mass
from vcell_fenics.formalism.schema import ParameterConstant
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description, normalize_to_geometry_frame

_CV = Path(__file__).resolve().parent
_RESOLUTIONS = (64, 128, 256)
_T = 1.0
_NUC, _R_NUC, _R_CELL = (0.1, 0.0), 0.25, 0.6
_HOMES = {"s_nuc": "nuc_dom", "s_cyto": "cyto_dom", "s_ext": "ext_dom"}


def _interiors(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    gx, gy = np.meshgrid(x, y, indexing="xy")
    r_nuc = np.hypot(gx - _NUC[0], gy - _NUC[1])
    r_cell = np.hypot(gx, gy)
    return {
        "s_nuc": r_nuc < 0.9 * _R_NUC,
        "s_cyto": (r_nuc > 1.1 * _R_NUC) & (r_cell < 0.9 * _R_CELL),
        "s_ext": r_cell > 1.1 * _R_CELL,
    }


def main() -> None:
    geo = gmod.Geometry.model_validate(yaml.safe_load((_CV / "nucleus_geom.yaml").read_text()))
    raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / "nucleus_math.yaml").read_text()))
    gd, md = normalize_to_geometry_frame(
        import_geometry(geo), import_math_description(raw, geometry=import_geometry(geo).name, dim=2)
    )
    kmole = next(p.value for p in md.parameters if isinstance(p, ParameterConstant) and p.name == "KMOLE")

    print(f"=== nucleus | cytosol | outside: FV ↔ FEniCSx joint refinement, t = {_T} ===")
    print(f"{'N':>5} {'h':>8} " + " ".join(f"{name + ' L2':>11} {name + ' L∞':>11}" for name in _HOMES) + "   drift")
    for n in _RESOLUTIONS:
        ref = np.load(_CV / f"nucleus_fv_{n}.npz")
        x, y = ref["x"], ref["y"]
        ti = int(np.argmin(np.abs(ref["t"] - _T)))
        geometry = realize_multi_compartment(gd, h=2.0 / n)
        start = integrate_multi_compartment(md, geometry, t_final=1e-9)
        result = integrate_multi_compartment(md, geometry, t_final=_T, rtol=1e-8, atol=1e-11)
        inside = _interiors(x, y)
        cells = []
        for name, home in _HOMES.items():
            ours = _eval_on_grid(result.fields[name], geometry.mesh_of(home), x, y)[inside[name]]
            theirs = ref[name][ti][inside[name]]
            l2 = np.sqrt(np.nanmean((ours - theirs) ** 2) / np.nanmean(theirs**2))
            linf = np.nanmax(np.abs(ours - theirs)) / np.nanmax(np.abs(theirs))
            cells.append(f"{l2:11.3%} {linf:11.3%}")

        def total(r: object) -> float:
            return sum(species_mass(r, v) for v in _HOMES) + kmole * species_mass(r, "R")  # type: ignore[arg-type]

        drift = total(result) / total(start) - 1.0
        print(f"{n:5d} {2.0 / n:8.4f} " + " ".join(cells) + f"   {drift:+.1e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 1 (../pyvcell/.venv): the fvsolver reference for the **3D reaction-advection-diffusion**
cross-validation — the 3D diffusion case (`diffusion_3d_fv.py`) plus a prescribed advection velocity.

A single species ``u`` diffuses (D) **and advects** at a constant velocity ``v = (vx, 0, 0)`` on the box
``[-1, 1]³`` with no-flux boundaries, from an off-centre Gaussian started near the ``-x`` wall so the
bump advects across the interior without reaching the far wall over the run (free-space ≈ bounded). The
velocity is set on the species mapping (``SpeciesMapping.velocity_x`` → the lowered PDE ``velocity``
slot → our ``relative_advection``).

Writes ``advection_3d.vcml`` + ``advection_3d_math.yaml`` / ``advection_3d_geom.yaml`` +
``advection_3d_reference.npz`` (``field (T,C,Z,Y,X)`` + coords, gitignored).

    ../pyvcell/.venv/bin/python cross_validation/advection_3d_fv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
import yaml
from diffusion_3d_fv import result_to_5d  # sibling stage-1 script (same 5D extractor)
from pyvcell.vcml.models import Biomodel, Model
from pyvcell.vcml.utils import to_vcml_str
from pyvcell.vcml.vcml_reader import VcmlReader

_CV = Path(__file__).resolve().parent
_STEM = "advection_3d"
_IC = "exp(-((x+0.5)^2 + y^2 + z^2) / (2 * 0.12^2))"  # Gaussian centred at (-0.5, 0, 0), σ = 0.12
_D = 0.03
_V = (0.4, 0.0, 0.0)  # advection velocity (+x)
_EXTENT = (2.0, 2.0, 2.0)
_ORIGIN = (-1.0, -1.0, -1.0)
_MESH = (32, 32, 32)
_END = 1.0
_OUT_DT = 0.1


def _author() -> Biomodel:
    geo = vc.Geometry(name="box3d", dim=3, extent=_EXTENT, origin=_ORIGIN)
    geo.add_background("domain")
    model = Model(name="advection3d")
    model.add_compartment("cell", dim=3)
    model.add_species("u", "cell")
    biomodel = Biomodel(name="Advection3D", model=model)
    app = biomodel.add_application("app", geometry=geo)
    app.map_compartment("cell", "domain")
    sm = app.map_species("u", init_conc=_IC, diff_coef=_D)
    sm.velocity_x, sm.velocity_y, sm.velocity_z = _V
    app.add_sim(name="sim", duration=_END, output_time_step=_OUT_DT, mesh_size=_MESH)
    return biomodel


def main() -> None:
    biomodel = _author()
    vcml = to_vcml_str(biomodel)
    (_CV / f"{_STEM}.vcml").write_text(vcml)

    app = VcmlReader.biomodel_from_str(vcml).applications[0]
    (_CV / f"{_STEM}_geom.yaml").write_text(
        yaml.safe_dump(app.geometry.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False)
    )
    (_CV / f"{_STEM}_math.yaml").write_text(
        yaml.safe_dump(
            app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False
        )
    )

    result = vc.simulate(vc.load_vcml_str(vcml), "sim")
    try:
        out = result_to_5d(result)
    finally:
        result.cleanup()
    np.savez_compressed(
        _CV / f"{_STEM}_reference.npz",
        field=out["field"],
        channels=np.asarray(out["channels"]),
        t=out["t"],
        x=out["x"],
        y=out["y"],
        z=out["z"],
    )
    f = out["field"]
    # centre of mass in x per output time — should advect at v_x.
    xg = out["x"]
    com_x = [float((f[ti, 0].sum(axis=(0, 1)) * xg).sum() / f[ti, 0].sum()) for ti in range(f.shape[0])]
    print(f"wrote {_STEM}.vcml / _math.yaml / _geom.yaml / _reference.npz")
    print(f"field {f.shape} (T,C,Z,Y,X); times {out['t']}; u range {f.min():.4g}..{f.max():.4g}")
    print(f"centre-of-mass x(t): {[round(c, 4) for c in com_x]}  (expect drift ≈ v_x·t = {_V[0]})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 1 (../pyvcell/.venv): the fvsolver reference for the **3D** single-species diffusion
cross-validation — the 3D analog of `author_and_run.py` (2D).

A single species ``u`` diffuses (D constant) on the box ``[-1, 1]³`` with no-flux boundaries (mass
conserved), from an off-centre Gaussian initial condition. VCell's finite-volume solver runs it on a
structured ``nx × ny × nz`` grid; we save the 5D field ``u(t, z, y, x)`` on the solver's own grid plus
the lowered MathDescription / Geometry the dev side imports.

Writes next to this file:
- ``diffusion_3d.vcml`` — the exact VCML the FV solver consumed;
- ``diffusion_3d_math.yaml`` / ``diffusion_3d_geom.yaml`` — the lowered model our bridge imports;
- ``diffusion_3d_reference.npz`` — ``field (T,C,Z,Y,X)`` + coords ``t,x,y,z`` + ``channels`` (gitignored).

    ../pyvcell/.venv/bin/python cross_validation/diffusion_3d_fv.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml as vc
import yaml
from pyvcell.vcml.models import Biomodel, Model
from pyvcell.vcml.utils import to_vcml_str
from pyvcell.vcml.vcml_reader import VcmlReader

_CV = Path(__file__).resolve().parent

_STEM = "diffusion_3d"
_IC = "exp(-((x-0.2)^2 + (y-0.1)^2 + z^2) / (2 * 0.15^2))"  # off-centre Gaussian, well resolved on 32³
_D = 0.1
_EXTENT = (2.0, 2.0, 2.0)
_ORIGIN = (-1.0, -1.0, -1.0)
_MESH = (32, 32, 32)
_END = 1.0
_OUT_DT = 0.1


def _author() -> Biomodel:
    geo = vc.Geometry(name="box3d", dim=3, extent=_EXTENT, origin=_ORIGIN)
    geo.add_background("domain")  # analytic '1.0' → the whole box
    model = Model(name="diffusion3d")
    model.add_compartment("cell", dim=3)
    model.add_species("u", "cell")
    biomodel = Biomodel(name="Diffusion3D", model=model)
    app = biomodel.add_application("app", geometry=geo)
    app.map_compartment("cell", "domain")
    app.map_species("u", init_conc=_IC, diff_coef=_D)
    app.add_sim(name="sim", duration=_END, output_time_step=_OUT_DT, mesh_size=_MESH)
    return biomodel


def result_to_5d(result: Any) -> dict[str, Any]:
    """Extract the 5D solution field from a pyvcell ``Result``: ``field (T,C,Z,Y,X)`` plus 1D coord
    arrays ``t,x,y,z`` (the FV solver's own cell coordinates), and species ``channels``. So
    ``field[ti, ci, k, j, i]`` is species ``channels[ci]`` at ``(x[i], y[j], z[k])``, time ``t[ti]``."""
    zd = result.zarr_dataset  # (T, C_all, Z, Y, X)
    idx = {c.label: c.index for c in result.channel_data}
    t = np.asarray(result.time_points, dtype=float)
    x = np.asarray(zd[0, idx["x"], 0, 0, :], dtype=float)
    y = np.asarray(zd[0, idx["y"], 0, :, 0], dtype=float)
    z = np.asarray(zd[0, idx["z"], :, 0, 0], dtype=float)
    channels = [name.split("::")[-1] for name in result.volume_variable_names]
    field = np.stack([np.asarray(zd[:, idx[name], :, :, :], dtype=float) for name in channels], axis=1)
    return {"field": field, "channels": channels, "t": t, "x": x, "y": y, "z": z}


def main() -> None:
    biomodel = _author()
    vcml = to_vcml_str(biomodel)
    (_CV / f"{_STEM}.vcml").write_text(vcml)

    # Lowered geometry + math (what the dev side imports); mesh size is a sim setting, not part of these.
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
    dv = (out["x"][1] - out["x"][0]) ** 3
    mass = [round(float(f[ti, 0].sum()) * dv, 5) for ti in range(f.shape[0])]
    print(f"wrote {_STEM}.vcml, {_STEM}_math.yaml, {_STEM}_geom.yaml, {_STEM}_reference.npz")
    print(f"field {f.shape} (T,C,Z,Y,X); times {out['t']}; u range {f.min():.4g}..{f.max():.4g}")
    print(f"total mass ∫u dV over time: {mass}")


if __name__ == "__main__":
    main()

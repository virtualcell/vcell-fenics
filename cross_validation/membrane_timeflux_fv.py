#!/usr/bin/env python
"""Author a **time-dependent membrane flux** model, run VCell's FV solver, save the 4D cytosol field.

Stage 1 (pyvcell `[native,solver]` env) of the time-dependent membrane-flux cross-validation — the
case that exercises the backend's `sim.t` path end-to-end (a `g(t)` Neumann on a membrane). A
two-compartment cell (inner-disk `cytosol` + `background` extracellular, `membrane` between) with one
cytosol species `u` and a **general-kinetics** membrane influx whose net rate is an explicit function
of time, `J = A·exp(−k·t)` (general kinetics is the only way to author a time-dependent reaction term;
see the authoring memo). The flux depends only on `t`, so the cytosol problem is decoupled from the
extracellular — the membrane is the whole cytosol-disk boundary, a `g(t)` Neumann.

    ../pyvcell/.venv/bin/python cross_validation/membrane_timeflux_fv.py

Writes next to this file:
  - `membrane_timeflux.vcml` — the exact VCML the FV solver ran;
  - `membrane_timeflux_math.yaml` / `_geom.yaml` — the lowered MathDescription / Geometry (imported by
    the dev-env comparison `compare_membrane_timeflux.py`);
  - `membrane_timeflux_reference.npz` — the FV cytosol `u(t, y, x)` field on the solver grid (+ `t`,
    `x`, `y`, the cytosol `radius`, and the membrane `in_flux` expression for reference).
    Gitignored (large); regenerate with this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml as vc
import yaml
from pyvcell.vcml.models import (
    Biomodel,
    Kinetics,
    KineticsParameter,
    Model,
    Reaction,
    SpeciesReference,
    SpeciesRefType,
)
from pyvcell.vcml.utils import to_vcml_str
from pyvcell.vcml.vcml_reader import VcmlReader

_CV = Path(__file__).resolve().parent
_RADIUS = 0.5
_RATE = "100.0 * exp( - 2.0 * t)"  # net membrane influx J(t): explicit, monotone-decaying time signature


def author_membrane_timeflux(*, diffusion: float = 0.1, mesh: tuple[int, int, int] = (128, 128, 1)) -> Biomodel:
    geo = vc.Geometry(name="cell", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0))
    geo.add_sphere("cytosol_dom", radius=_RADIUS, center=(0.0, 0.0, 0.0))
    geo.add_background("extra_dom")
    geo.add_surface("membrane_dom", "cytosol_dom", "extra_dom")

    model = Model(name="timeflux")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("u", "cyto")
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(name="J", value=_RATE, role="reaction rate", unit="", reaction_name="influx")
        ],
    )
    reaction = Reaction(name="influx", compartment_name="mem", reversible=False, is_flux=False, kinetics=kinetics)
    reaction.products.append(SpeciesReference(name="u", stoichiometry=1, species_ref_type=SpeciesRefType.product))
    model.reactions.append(reaction)

    biomodel = Biomodel(name="MembraneTimeFlux", model=model)
    app = biomodel.add_application("cellApp", geometry=geo)
    app.map_compartment("cyto", "cytosol_dom")
    app.map_compartment("ext", "extra_dom")
    app.map_compartment("mem", "membrane_dom")
    app.map_species("u", init_conc="0.0", diff_coef=diffusion)
    app.map_reaction("influx", True)
    app.add_sim("sim", duration=1.0, output_time_step=0.1, mesh_size=mesh)
    return biomodel


def _dump(model: Any, path: Path, exclude: Any = None) -> None:
    path.write_text(
        yaml.safe_dump(
            model.model_dump(mode="json", exclude_none=True, exclude_defaults=True, exclude=exclude), sort_keys=False
        )
    )


def main() -> None:
    biomodel = author_membrane_timeflux()
    vcml = to_vcml_str(biomodel)
    md = VcmlReader.biomodel_from_str(vcml).applications[0].math_description
    (_CV / "membrane_timeflux.vcml").write_text(vcml)
    _dump(md, _CV / "membrane_timeflux_math.yaml")
    _dump(
        biomodel.applications[0].geometry,
        _CV / "membrane_timeflux_geom.yaml",
        exclude={"image": {"compressed_content"}},
    )
    (in_flux,) = [jc.in_flux for m in md.membrane_subdomains for jc in m.jump_conditions if jc.name == "u"]

    result = vc.simulate(vc.load_vcml_str(vcml), "sim")
    try:
        zd = result.zarr_dataset
        index = {c.label: c.index for c in result.channel_data}
        t = np.asarray(result.time_points, dtype=float)
        x = np.asarray(zd[0, index["x"], 0, 0, :], dtype=float)
        y = np.asarray(zd[0, index["y"], 0, :, 0], dtype=float)
        (u_name,) = (n.split("::")[-1] for n in result.volume_variable_names)  # channel_data labels are bare names
        field = np.asarray(zd[:, index[u_name], 0, :, :], dtype=float)  # (T, Y, X)
    finally:
        result.cleanup()

    np.savez_compressed(
        _CV / "membrane_timeflux_reference.npz", field=field, t=t, x=x, y=y, radius=_RADIUS, in_flux=in_flux
    )
    total = field.reshape(len(t), -1).sum(1) * (x[1] - x[0]) * (y[1] - y[0])
    print(f"in_flux(u) = {in_flux}")
    print(f"field {field.shape}; cytosol total u {total[0]:.4f} -> {total[-1]:.4f} (rises then plateaus as J decays)")


if __name__ == "__main__":
    main()

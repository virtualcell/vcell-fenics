#!/usr/bin/env python
"""Author a three-compartment cell — nucleus in cytosol in extracellular space — dump its lowered geometry+math,
and run FV at several grids: stage 1 of the multi-compartment cross-solver check (`integrate_multi_compartment`).

- A ligand in each compartment: `s_nuc`, `s_cyto`, `s_ext`.
- Nuclear transport across the nuclear envelope `ne`: a membrane reaction `s_cyto → s_nuc`, rate
  `kt·(s_cyto − s_nuc)` (passive, both ways).
- A receptor `R` on the plasma membrane `pm` capturing both the cytosolic and the extracellular ligand,
  `kon·s·(Rmax − R)` each (as `receptor_fv.py`).

VCell lowers these to jump conditions on both membranes (with its `KFlux·KMOLE` unit factors) and a
membrane PDE for R; the dev side imports that math verbatim, so both solvers solve the identical problem.

    ../pyvcell/.venv/bin/python cross_validation/nucleus_fv.py

Writes `nucleus_geom.yaml`, `nucleus_math.yaml`, and `nucleus_fv_<N>.npz` {s_nuc, s_cyto, s_ext (T,Y,X), t, x, y}
per N (npz gitignored, regenerable).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
import yaml
from pyvcell.vcml.models import Biomodel, Kinetics, KineticsParameter, Model, Reaction, SpeciesReference, SpeciesRefType
from pyvcell.vcml.utils import to_vcml_str
from pyvcell.vcml.vcml_reader import VcmlReader

_CV = Path(__file__).resolve().parent
_R_NUC, _R_CELL = 0.25, 0.6
_D, _DS = 1.0, 0.05
_KT, _KON, _RMAX, _DURATION = 30.0, 0.01, 2000.0, 2.0
_RESOLUTIONS = (64, 128, 256)


def _reaction(name: str, compartment: str, reactant: str, product: str, rate: str) -> Reaction:
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(name="J", value=rate, role="reaction rate", unit="", reaction_name=name)
        ],
    )
    reaction = Reaction(name=name, compartment_name=compartment, reversible=True, is_flux=False, kinetics=kinetics)
    reaction.reactants.append(
        SpeciesReference(name=reactant, stoichiometry=1, species_ref_type=SpeciesRefType.reactant)
    )
    reaction.products.append(SpeciesReference(name=product, stoichiometry=1, species_ref_type=SpeciesRefType.product))
    return reaction


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geo = vc.Geometry(name="nucleus", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0))
    geo.add_sphere("nuc_dom", radius=_R_NUC, center=(0.1, 0.0, 0.0))  # off-centre: no symmetry to hide behind
    geo.add_sphere("cyto_dom", radius=_R_CELL, center=(0.0, 0.0, 0.0))
    geo.add_background("ext_dom")
    geo.add_surface("ne_dom", "nuc_dom", "cyto_dom")
    geo.add_surface("pm_dom", "cyto_dom", "ext_dom")

    model = Model(name="nucleus")
    for name, dim in (("nuc", 3), ("cyto", 3), ("ext", 3), ("ne", 2), ("pm", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("s_nuc", "nuc")
    model.add_species("s_cyto", "cyto")
    model.add_species("s_ext", "ext")
    model.add_species("R", "pm")
    model.reactions.append(_reaction("transport", "ne", "s_cyto", "s_nuc", f"{_KT} * (s_cyto - s_nuc)"))
    for name, ligand in (("bind_in", "s_cyto"), ("bind_out", "s_ext")):
        model.reactions.append(_reaction(name, "pm", ligand, "R", f"{_KON} * {ligand} * ({_RMAX} - R)"))

    biomodel = Biomodel(name="Nucleus", model=model)
    app = biomodel.add_application("app", geometry=geo)
    for c, d in (("nuc", "nuc_dom"), ("cyto", "cyto_dom"), ("ext", "ext_dom"), ("ne", "ne_dom"), ("pm", "pm_dom")):
        app.map_compartment(c, d)
    app.map_species("s_nuc", init_conc="0.0", diff_coef=_D)
    app.map_species("s_cyto", init_conc="2.0", diff_coef=_D)
    app.map_species("s_ext", init_conc="1.0", diff_coef=_D)
    app.map_species("R", init_conc="0.0", diff_coef=_DS)
    for name in ("transport", "bind_in", "bind_out"):
        app.map_reaction(name, True)
    app.add_sim("sim", duration=_DURATION, output_time_step=0.5, mesh_size=mesh)
    return biomodel


def main() -> None:
    app = VcmlReader.biomodel_from_str(to_vcml_str(_author((64, 64, 1)))).applications[0]
    (_CV / "nucleus_geom.yaml").write_text(
        yaml.safe_dump(
            app.geometry.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"image": {"compressed_content"}}
            ),
            sort_keys=False,
        )
    )
    (_CV / "nucleus_math.yaml").write_text(
        yaml.safe_dump(
            app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False
        )
    )
    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author((n, n, 1)))), "sim")
        try:
            zd = result.zarr_dataset
            idx = {c.label: c.index for c in result.channel_data}
            fields = {name: np.asarray(zd[:, idx[name], 0, :, :], dtype=float) for name in ("s_nuc", "s_cyto", "s_ext")}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, idx["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, idx["y"], 0, :, 0], dtype=float)
        finally:
            result.cleanup()
        np.savez_compressed(_CV / f"nucleus_fv_{n}.npz", t=t, x=x, y=y, **fields)
        print(f"N={n:4d}: " + ", ".join(f"{k} max {np.nanmax(v[-1]):.4f}" for k, v in fields.items()))


if __name__ == "__main__":
    main()

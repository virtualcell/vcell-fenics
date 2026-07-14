#!/usr/bin/env python
"""Stage 1 (../pyvcell/.venv): the fvsolver reference for the **3D membrane surface-species** (receptor)
cross-validation — the 3D analog of `receptor_fv.py`, exercising `integrate_membrane_coupled`.

A spherical `cyto` in an `ext` background on `[-1, 1]³`, two bulk ligands `s_cyto` / `s_ext` (both init 1)
captured by a **membrane receptor `R`** (a surface species, bound density) via two saturating binding
reactions `kon·ligand·(Rmax − R)` (VCell membrane reactions: volume reactant → membrane product). VCell
lowers these to a membrane PDE for `R` plus the jump conditions that deplete the ligands, carrying the
volume↔membrane unit factors. The bulk ligands are compared on the FV grid (R acts on them through
binding); R's effect + total-substance conservation are checked dev-side.

Writes `receptor_3d_geom.yaml` / `receptor_3d_math.yaml` + `receptor_3d_fv_<N>.npz`
(`s_cyto`/`s_ext` `(T,Z,Y,X)` + coords, gitignored).

    ../pyvcell/.venv/bin/python cross_validation/receptor_3d_fv.py
"""

from __future__ import annotations

from pathlib import Path

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
_RADIUS, _D, _DS = 0.5, 1.0, 0.05
_KON, _RMAX, _DURATION = 0.01, 2000.0, 4.0
_RESOLUTIONS = (32, 48)


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geo = vc.Geometry(name="cell", dim=3, extent=(2.0, 2.0, 2.0), origin=(-1.0, -1.0, -1.0))
    geo.add_sphere("cyto_dom", radius=_RADIUS, center=(0.0, 0.0, 0.0))
    geo.add_background("ext_dom")
    geo.add_surface("mem_dom", "cyto_dom", "ext_dom")

    model = Model(name="receptor")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("s_cyto", "cyto")
    model.add_species("s_ext", "ext")
    model.add_species("R", "mem")  # membrane receptor (bound density)

    for rxn_name, ligand in (("bind_in", "s_cyto"), ("bind_out", "s_ext")):
        kin = Kinetics(
            kinetics_type="GeneralKinetics",
            kinetics_parameters=[
                KineticsParameter(
                    name="J",
                    value=f"{_KON} * {ligand} * ({_RMAX} - R)",
                    role="reaction rate",
                    unit="",
                    reaction_name=rxn_name,
                )
            ],
        )
        rxn = Reaction(name=rxn_name, compartment_name="mem", reversible=False, is_flux=False, kinetics=kin)
        rxn.reactants.append(SpeciesReference(name=ligand, stoichiometry=1, species_ref_type=SpeciesRefType.reactant))
        rxn.products.append(SpeciesReference(name="R", stoichiometry=1, species_ref_type=SpeciesRefType.product))
        model.reactions.append(rxn)

    biomodel = Biomodel(name="Receptor3D", model=model)
    app = biomodel.add_application("app", geometry=geo)
    for c, d in (("cyto", "cyto_dom"), ("ext", "ext_dom"), ("mem", "mem_dom")):
        app.map_compartment(c, d)
    app.map_species("s_cyto", init_conc="1.0", diff_coef=_D)
    app.map_species("s_ext", init_conc="1.0", diff_coef=_D)
    app.map_species("R", init_conc="0.0", diff_coef=_DS)
    app.map_reaction("bind_in", True)
    app.map_reaction("bind_out", True)
    app.add_sim("sim", duration=_DURATION, output_time_step=0.5, mesh_size=mesh)
    return biomodel


def main() -> None:
    app = VcmlReader.biomodel_from_str(to_vcml_str(_author((32, 32, 32)))).applications[0]
    (_CV / "receptor_3d_geom.yaml").write_text(
        yaml.safe_dump(
            app.geometry.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"image": {"compressed_content"}}
            ),
            sort_keys=False,
        )
    )
    (_CV / "receptor_3d_math.yaml").write_text(
        yaml.safe_dump(
            app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False
        )
    )

    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author((n, n, n)))), "sim")
        try:
            zd = result.zarr_dataset
            idx = {c.label: c.index for c in result.channel_data}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, idx["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, idx["y"], 0, :, 0], dtype=float)
            z = np.asarray(zd[0, idx["z"], :, 0, 0], dtype=float)
            s_cyto = np.asarray(zd[:, idx["s_cyto"], :, :, :], dtype=float)
            s_ext = np.asarray(zd[:, idx["s_ext"], :, :, :], dtype=float)
        finally:
            result.cleanup()
        np.savez_compressed(
            _CV / f"receptor_3d_fv_{n}.npz", s_cyto=s_cyto, s_ext=s_ext, t=t, x=x, y=y, z=z, radius=_RADIUS
        )
        print(f"N={n:3d}³: s_cyto={s_cyto.shape}  s_cyto(t_end) mean-in-sphere depletes below 1; times={t}")


if __name__ == "__main__":
    main()

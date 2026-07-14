#!/usr/bin/env python
"""Stage 1 (../pyvcell/.venv): the fvsolver reference for the **3D interface-coupled permeability**
cross-validation — the 3D analog of `coupled_perm_fv.py`.

A spherical `cyto` (radius 0.5) inside an `ext` background on the box `[-1, 1]³`, two bulk species
`s_cyto` (init 1) and `s_ext` (init 0) coupled by a **membrane permeability flux** `J = P·(s_ext −
s_cyto)` (a VCell flux reaction). VCell's generated jump conditions become the equal-and-opposite pair
of single-sided interface fluxes our `integrate_interface_coupled` solves. Runs FV at a couple of grids
so the dev side can refine alongside.

Writes `coupled_3d_geom.yaml` / `coupled_3d_math.yaml` (imported by the dev side) and
`coupled_3d_fv_<N>.npz` (`s_cyto`/`s_ext` `(T,Z,Y,X)` + coords, gitignored).

    ../pyvcell/.venv/bin/python cross_validation/coupled_3d_fv.py
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
_RADIUS, _P, _D = 0.5, 0.5, 1.0
_RESOLUTIONS = (32, 48)
_END, _OUT_DT = 2.0, 0.25


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geo = vc.Geometry(name="cell", dim=3, extent=(2.0, 2.0, 2.0), origin=(-1.0, -1.0, -1.0))
    geo.add_sphere("cyto_dom", radius=_RADIUS, center=(0.0, 0.0, 0.0))
    geo.add_background("ext_dom")
    geo.add_surface("mem_dom", "cyto_dom", "ext_dom")

    model = Model(name="perm")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("s_cyto", "cyto")
    model.add_species("s_ext", "ext")
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(
                name="J", value=f"{_P} * (s_ext - s_cyto)", role="reaction rate", unit="", reaction_name="perm"
            )
        ],
    )
    reaction = Reaction(name="perm", compartment_name="mem", reversible=True, is_flux=True, kinetics=kinetics)
    reaction.reactants.append(SpeciesReference(name="s_ext", stoichiometry=1, species_ref_type=SpeciesRefType.reactant))
    reaction.products.append(SpeciesReference(name="s_cyto", stoichiometry=1, species_ref_type=SpeciesRefType.product))
    model.reactions.append(reaction)

    biomodel = Biomodel(name="Permeability3D", model=model)
    app = biomodel.add_application("app", geometry=geo)
    app.map_compartment("cyto", "cyto_dom")
    app.map_compartment("ext", "ext_dom")
    app.map_compartment("mem", "mem_dom")
    app.map_species("s_cyto", init_conc="1.0", diff_coef=_D)
    app.map_species("s_ext", init_conc="0.0", diff_coef=_D)
    app.map_reaction("perm", True)
    app.add_sim("sim", duration=_END, output_time_step=_OUT_DT, mesh_size=mesh)
    return biomodel


def main() -> None:
    app = VcmlReader.biomodel_from_str(to_vcml_str(_author((32, 32, 32)))).applications[0]
    (_CV / "coupled_3d_geom.yaml").write_text(
        yaml.safe_dump(
            app.geometry.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"image": {"compressed_content"}}
            ),
            sort_keys=False,
        )
    )
    (_CV / "coupled_3d_math.yaml").write_text(
        yaml.safe_dump(
            app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False
        )
    )

    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author((n, n, n)))), "sim")
        try:
            zd = result.zarr_dataset  # (T, C, Z, Y, X)
            index = {c.label: c.index for c in result.channel_data}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, index["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, index["y"], 0, :, 0], dtype=float)
            z = np.asarray(zd[0, index["z"], :, 0, 0], dtype=float)
            bare = {name.split("::")[-1] for name in result.volume_variable_names}
            assert {"s_cyto", "s_ext"} <= bare, bare
            s_cyto = np.asarray(zd[:, index["s_cyto"], :, :, :], dtype=float)
            s_ext = np.asarray(zd[:, index["s_ext"], :, :, :], dtype=float)
        finally:
            result.cleanup()
        np.savez_compressed(
            _CV / f"coupled_3d_fv_{n}.npz",
            s_cyto=s_cyto,
            s_ext=s_ext,
            t=t,
            x=x,
            y=y,
            z=z,
            radius=_RADIUS,
            P=_P,
            D=_D,
        )
        print(f"N={n:3d}³: s_cyto={s_cyto.shape}  dx={x[1] - x[0]:.5f}  times={t}")


if __name__ == "__main__":
    main()

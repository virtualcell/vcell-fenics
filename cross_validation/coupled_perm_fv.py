#!/usr/bin/env python
"""Author the two-compartment permeability model, dump its lowered geometry, run FV at several grids.

Stage 1 of the cross-compartment-coupling convergence study (the definitive cross-solver check for
`integrate_interface_coupled`). A disk-in-box cell: species `s_cyto` in the inner disk and `s_ext` in
the surrounding box, coupled by a **membrane permeability flux** `J = P·(s_ext − s_cyto)` (a VCell
flux reaction). VCell's generated jump conditions are exactly the flux-balance the FEniCSx coupled
integrator solves (s_cyto gains +J, s_ext loses it), unit factor 1 (both volume species in µM), so P
maps straight across.

The FEniCSx side imports **this same geometry** (`coupled_perm_geom.yaml`) and realizes it via
`realize_interface_coupled` — the genuine pipeline, no hand-built parallel mesh. The FV solver uses MOL
(Sundials/CVODE); comparing the FEniCSx coupled MOL solve to it under joint mesh refinement
(`coupled_perm_convergence.py`) tests both at ≈0 time error. A single FV grid can't show convergence
(the FV membrane's 1st-order error is the floor), so this runs several grids; FEniCSx refines alongside.

    ../pyvcell/.venv/bin/python cross_validation/coupled_perm_fv.py

Writes `coupled_perm_geom.yaml` (the lowered geometry the dev side imports) and `coupled_perm_fv_<N>.npz`
{s_cyto (T,Y,X), s_ext (T,Y,X), t, x, y, radius, P, D} for each N (npz gitignored, regenerable).
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
# FV could run 512² cheaply, but the FEniCSx body-fitted realize + coupled MOL solve cap the joint
# refinement at 256², so we generate the matching set.
_RESOLUTIONS = (64, 128, 256)


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geo = vc.Geometry(name="cell", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0))
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

    biomodel = Biomodel(name="Permeability", model=model)
    app = biomodel.add_application("app", geometry=geo)
    app.map_compartment("cyto", "cyto_dom")
    app.map_compartment("ext", "ext_dom")
    app.map_compartment("mem", "mem_dom")
    app.map_species("s_cyto", init_conc="1.0", diff_coef=_D)
    app.map_species("s_ext", init_conc="0.0", diff_coef=_D)
    app.map_reaction("perm", True)
    app.add_sim("sim", duration=2.0, output_time_step=0.25, mesh_size=mesh)
    return biomodel


def main() -> None:
    # Dump the lowered geometry AND math once (mesh size is a simulation setting, not part of either).
    # The dev side imports both — the coupled jump conditions route to a BCInterfaceFluxBalance.
    app = VcmlReader.biomodel_from_str(to_vcml_str(_author((64, 64, 1)))).applications[0]
    (_CV / "coupled_perm_geom.yaml").write_text(
        yaml.safe_dump(
            app.geometry.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"image": {"compressed_content"}}
            ),
            sort_keys=False,
        )
    )
    (_CV / "coupled_perm_math.yaml").write_text(
        yaml.safe_dump(
            app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False
        )
    )

    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author((n, n, 1)))), "sim")
        try:
            zd = result.zarr_dataset
            index = {c.label: c.index for c in result.channel_data}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, index["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, index["y"], 0, :, 0], dtype=float)
            bare = {name.split("::")[-1] for name in result.volume_variable_names}
            assert {"s_cyto", "s_ext"} <= bare, bare
            s_cyto = np.asarray(zd[:, index["s_cyto"], 0, :, :], dtype=float)
            s_ext = np.asarray(zd[:, index["s_ext"], 0, :, :], dtype=float)
        finally:
            result.cleanup()
        np.savez_compressed(
            _CV / f"coupled_perm_fv_{n}.npz", s_cyto=s_cyto, s_ext=s_ext, t=t, x=x, y=y, radius=_RADIUS, P=_P, D=_D
        )
        print(f"N={n:5d}: s_cyto={s_cyto.shape}  dx={x[1] - x[0]:.5f}")


if __name__ == "__main__":
    main()

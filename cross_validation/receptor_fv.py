#!/usr/bin/env python
"""Author the two-bulk + membrane-receptor binding model, dump its lowered geometry+math, run FV at
several grids — stage 1 of the receptor cross-solver convergence study for `integrate_membrane_coupled`.

A disk-in-box cell: ligand `s_cyto` in the inner disk and `s_ext` in the surrounding box, both captured
by a membrane receptor `R` (bound density) via two saturating binding reactions `kon·s·(Rmax−R)` (a
VCell membrane reaction: volume reactant → membrane product). VCell lowers these to a membrane PDE for R
plus the jump conditions that consume the ligands, carrying the volume↔membrane unit factors
(`KFlux·KMOLE`); the dev side imports that math verbatim and solves it with the three-region coupled MOL,
so both solvers solve the identical problem. `Rmax`/`kon` are tuned so the ligands deplete visibly under
the `KMOLE≈1/602` conversion (else the bulk fields barely move and the comparison is uninformative).

    ../pyvcell/.venv/bin/python cross_validation/receptor_fv.py

Writes `receptor_geom.yaml`, `receptor_math.yaml`, and `receptor_fv_<N>.npz` {s_cyto, s_ext (T,Y,X), t,
x, y, radius} per N (npz gitignored, regenerable).
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
_RADIUS, _D, _DS = 0.5, 1.0, 0.05
# Tuned for a strong signal under the KMOLE≈1/602 volume↔membrane conversion: high Rmax so the membrane
# never saturates, kon set so both ligands deplete substantially by the comparison time.
_KON, _RMAX, _DURATION = 0.01, 2000.0, 4.0
_RESOLUTIONS = (64, 128, 256)


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geo = vc.Geometry(name="cell", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0))
    geo.add_sphere("cyto_dom", radius=_RADIUS, center=(0.0, 0.0, 0.0))
    geo.add_background("ext_dom")
    geo.add_surface("mem_dom", "cyto_dom", "ext_dom")

    model = Model(name="receptor")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("s_cyto", "cyto")
    model.add_species("s_ext", "ext")
    model.add_species("R", "mem")  # membrane receptor (bound density, molecules/µm²)

    for rxn_name, ligand in (("bind_in", "s_cyto"), ("bind_out", "s_ext")):
        kin = Kinetics(
            kinetics_type="GeneralKinetics",
            kinetics_parameters=[
                KineticsParameter(name="J", value=f"{_KON} * {ligand} * ({_RMAX} - R)", role="reaction rate",
                                  unit="", reaction_name=rxn_name)
            ],
        )
        rxn = Reaction(name=rxn_name, compartment_name="mem", reversible=False, is_flux=False, kinetics=kin)
        rxn.reactants.append(SpeciesReference(name=ligand, stoichiometry=1, species_ref_type=SpeciesRefType.reactant))
        rxn.products.append(SpeciesReference(name="R", stoichiometry=1, species_ref_type=SpeciesRefType.product))
        model.reactions.append(rxn)

    biomodel = Biomodel(name="Receptor", model=model)
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
    app = VcmlReader.biomodel_from_str(to_vcml_str(_author((64, 64, 1)))).applications[0]
    (_CV / "receptor_geom.yaml").write_text(
        yaml.safe_dump(
            app.geometry.model_dump(mode="json", exclude_none=True, exclude_defaults=True,
                                    exclude={"image": {"compressed_content"}}),
            sort_keys=False,
        )
    )
    (_CV / "receptor_math.yaml").write_text(
        yaml.safe_dump(app.math_description.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
                       sort_keys=False)
    )

    for n in _RESOLUTIONS:
        result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author((n, n, 1)))), "sim")
        try:
            zd = result.zarr_dataset
            idx = {c.label: c.index for c in result.channel_data}
            t = np.asarray(result.time_points, dtype=float)
            x = np.asarray(zd[0, idx["x"], 0, 0, :], dtype=float)
            y = np.asarray(zd[0, idx["y"], 0, :, 0], dtype=float)
            s_cyto = np.asarray(zd[:, idx["s_cyto"], 0, :, :], dtype=float)
            s_ext = np.asarray(zd[:, idx["s_ext"], 0, :, :], dtype=float)
        finally:
            result.cleanup()
        np.savez_compressed(_CV / f"receptor_fv_{n}.npz", s_cyto=s_cyto, s_ext=s_ext, t=t, x=x, y=y, radius=_RADIUS)
        in_disk = np.nanmean(s_cyto[-1][(x[None, :] ** 2 + y[:, None] ** 2) < (0.9 * _RADIUS) ** 2])
        print(f"N={n:5d}: s_cyto in-disk mean at tEnd = {in_disk:.4f} (started 1.0)")


if __name__ == "__main__":
    main()

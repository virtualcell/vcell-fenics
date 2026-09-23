#!/usr/bin/env python
"""Stage 1 (../pyvcell/.venv): fvsolver reference for nucleocytoplasmic exchange on a real IMAGE geometry.

The geometry is VCell's own segmented 3D image from the pyvcell tutorial model
(``examples/models/Tutorial_MultiApp_PDE.vcml``: 256×256×34 pixels, 74.24 × 74.24 × 26 µm, ec ⊃ cytosol ⊃
Nucleus — the same image as the SimulationTask fixture ``image3d_SimID_274630052``). On it: species ``c``
in the cytosol (initially 1 µM) and ``n`` in the Nucleus (initially 0), exchanging across the nuclear
membrane by a permeability flux ``J = P·(c − n)`` (a VCell flux reaction); ``ec`` carries nothing. Both
diffuse (D). The nucleus fills at a rate set by its **area-to-volume ratio**, so the mean nuclear
concentration over time is a sharp test of the realized geometry, not just the solver.

VCell's FV solver meshes the image as voxels (its membranes are voxel faces with smoothed areas); vcell-fenics
realizes it body-fitted and smoothed (`backend/labels.py`, `label_surfaces.py`). Stage 2
(``image_nuclear.py``) compares the two.

    ../pyvcell/.venv/bin/python cross_validation/image_nuclear_fv.py [--mesh 101 101 36]

Writes ``image_nuclear_math.yaml`` / ``image_nuclear_geom.yaml`` (the lowered model, voxels included — the
dev side imports them) and ``image_nuclear_fv_<nx>.npz`` {t, x, y, z, c (T,Z,Y,X), n, region (Z,Y,X), P, D}
(npz gitignored, regenerable).
"""

from __future__ import annotations

import argparse
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
_TUTORIAL = _CV.parent.parent / "pyvcell" / "examples" / "models" / "Tutorial_MultiApp_PDE.vcml"
_P, _D, _DURATION, _DT_OUT = 0.5, 5.0, 5.0, 0.25


def _author(mesh: tuple[int, int, int]) -> Biomodel:
    geometry = vc.load_vcml_file(str(_TUTORIAL)).applications[0].geometry  # the image geometry, unchanged

    model = Model(name="nuclear")
    for name, dim in (("ec", 3), ("cyt", 3), ("nuc", 3), ("pm", 2), ("nm", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("c", "cyt")
    model.add_species("n", "nuc")
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(name="J", value=f"{_P} * (c - n)", role="reaction rate", unit="", reaction_name="imp")
        ],
    )
    reaction = Reaction(name="imp", compartment_name="nm", reversible=True, is_flux=True, kinetics=kinetics)
    reaction.reactants.append(SpeciesReference(name="c", stoichiometry=1, species_ref_type=SpeciesRefType.reactant))
    reaction.products.append(SpeciesReference(name="n", stoichiometry=1, species_ref_type=SpeciesRefType.product))
    model.reactions.append(reaction)

    biomodel = Biomodel(name="NuclearExchange", model=model)
    app = biomodel.add_application("app", geometry=geometry)
    for compartment, domain in (
        ("ec", "ec"),
        ("cyt", "cytosol"),
        ("nuc", "Nucleus"),
        ("pm", "cytosol_ec_membrane"),
        ("nm", "Nucleus_cytosol_membrane"),
    ):
        app.map_compartment(compartment, domain)
    app.map_species("c", init_conc="1.0", diff_coef=_D)
    app.map_species("n", init_conc="0.0", diff_coef=_D)
    app.map_reaction("imp", True)
    app.add_sim("sim", duration=_DURATION, output_time_step=_DT_OUT, mesh_size=mesh)
    return biomodel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, nargs=3, default=(101, 101, 36))
    args = parser.parse_args()
    mesh = (int(args.mesh[0]), int(args.mesh[1]), int(args.mesh[2]))

    app = VcmlReader.biomodel_from_str(to_vcml_str(_author(mesh))).applications[0]
    for stem, node in (("geom", app.geometry), ("math", app.math_description)):
        text = yaml.safe_dump(node.model_dump(mode="json", exclude_none=True, exclude_defaults=True), sort_keys=False)
        (_CV / f"image_nuclear_{stem}.yaml").write_text(text)

    result = vc.simulate(vc.load_vcml_str(to_vcml_str(_author(mesh))), "sim")
    try:
        zd = result.zarr_dataset
        index = {c.label: c.index for c in result.channel_data}
        t = np.asarray(result.time_points, dtype=float)
        x = np.asarray(zd[0, index["x"], 0, 0, :], dtype=float)
        y = np.asarray(zd[0, index["y"], 0, :, 0], dtype=float)
        z = np.asarray(zd[0, index["z"], :, 0, 0], dtype=float)
        c = np.asarray(zd[:, index["c"], :, :, :], dtype=float)
        n = np.asarray(zd[:, index["n"], :, :, :], dtype=float)
        region = np.asarray(zd[0, index["region_mask"], :, :, :], dtype=float) if "region_mask" in index else None
    finally:
        result.cleanup()
    np.savez_compressed(
        _CV / f"image_nuclear_fv_{mesh[0]}.npz",
        t=t, x=x, y=y, z=z, c=c, n=n, P=_P, D=_D, mesh=np.array(mesh),
        **({"region": region} if region is not None else {}),
    )  # fmt: skip
    print(f"mesh {mesh}: {len(t)} times, c {c.shape}; channels {sorted(index)}")


if __name__ == "__main__":
    main()

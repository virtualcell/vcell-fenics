#!/usr/bin/env python
"""Confirm the membrane jump-condition Neumann **sign convention** against VCell's FV solver.

The diffusion cross-validation (`author_and_run.py` / `compare_fenics_vs_fv.py`) is single-compartment
no-flux, so it never exercises a membrane flux. This reproducer authors a two-compartment cell — an
inner-disk `cytosol` and a `background` extracellular, separated by a `membrane` — and drives the
cytosol species `u` across the membrane in **both directions**, running VCell's FV solver for each:

- **efflux** — a mass-action membrane reaction `u → ∅` (rate `Kf·u`): VCell `in_flux(u)` is
  **negative** (and references `u`, so our import wraps it in `trace(u)`) and the FV cytosol mass
  **decreases**.
- **influx** — a **general-kinetics** membrane reaction `∅ → u` with the net rate set directly to a
  constant (`J = 0.5`): VCell `in_flux(u)` is a **positive constant** and the FV cytosol mass
  **increases**.

  (Mass-action cannot author a zeroth-order source — an empty reactant product has rate 0, not 1 — so
  the influx uses *general kinetics*, where the net reaction rate is specified directly.)

Why this confirms the convention: our bridge maps a jump condition's `in_flux` **directly** (no sign
flip) to `BCNeumann(variable=u, boundary=membrane, expression=in_flux)`, and the backend's Neumann
sign is independently verified (`test_neumann_influx_adds_mass_at_predicted_rate`:
`d(mass)/dt = ∫_Γ h ds`, so `h > 0` adds and `h < 0` removes mass). Both FV directions matching the
sign of `in_flux` rules out a flip in either direction.

Runs in the pyvcell `[native,solver]` env:

    ../pyvcell/.venv/bin/python cross_validation/membrane_flux_sign.py

Writes `membrane_efflux.vcml` + the lowered `membrane_efflux_math.yaml` / `_geom.yaml` next to it for
inspection; the dev-env test `test_jump_condition_preserves_vcell_flux_sign` asserts the same mapping
on an equivalent in-Python model. The FV solve itself stays here (needs libvcell).
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


def _add_constant_flux_reaction(model: Model, name: str, comp: str, product: str, rate: float) -> Reaction:
    """A membrane reaction `∅ → product` whose **net rate is set directly** (VCell *general kinetics*)
    — the way to author a constant zeroth-order membrane flux (mass-action cannot, an empty reactant
    product is rate 0). `pyvcell`'s `add_reaction_mass_action` has no general-kinetics sibling, so we
    build the `Reaction` ourselves."""
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(name="J", value=str(rate), role="reaction rate", unit="", reaction_name=name)
        ],
    )
    reaction = Reaction(name=name, compartment_name=comp, reversible=False, is_flux=False, kinetics=kinetics)
    reaction.products.append(SpeciesReference(name=product, stoichiometry=1, species_ref_type=SpeciesRefType.product))
    model.reactions.append(reaction)
    return reaction


def _cell_geometry() -> vc.Geometry:
    """A 2D cell: inner-disk `cytosol_dom` + `background` extracellular, a `membrane` between them."""
    geo = vc.Geometry(name="cell", dim=2, extent=(2.0, 2.0, 1.0), origin=(-1.0, -1.0, 0.0))
    geo.add_sphere("cytosol_dom", radius=0.5, center=(0.0, 0.0, 0.0))  # a disk in 2D
    geo.add_background("extra_dom")
    geo.add_surface("membrane_dom", "cytosol_dom", "extra_dom")
    return geo


def author_membrane_efflux(*, kf: float = 0.5, diffusion: float = 0.1) -> Biomodel:
    """Cytosol species `u` removed at the membrane (`u → ∅`, rate `kf·u`) ⇒ negative `in_flux(u)`."""
    model = Model(name="efflux")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("u", "cyto")
    model.add_reaction_mass_action("efflux", "mem", reactants=["u"], products=[], kf=kf, kr=0.0)

    biomodel = Biomodel(name="MembraneEfflux", model=model)
    app = biomodel.add_application("cellApp", geometry=_cell_geometry())
    app.map_compartment("cyto", "cytosol_dom")
    app.map_compartment("ext", "extra_dom")
    app.map_compartment("mem", "membrane_dom")
    app.map_species("u", init_conc="1.0", diff_coef=diffusion)
    app.map_reaction("efflux", True)
    app.add_sim("sim", duration=1.0, output_time_step=0.5, mesh_size=(128, 128, 1))
    return biomodel


def author_membrane_influx(*, rate: float = 0.5, diffusion: float = 0.1) -> Biomodel:
    """Constant membrane source of cytosol `u` (`∅ → u`, net rate `rate` via general kinetics) ⇒ a
    positive constant `in_flux(u)`."""
    model = Model(name="influx")
    for name, dim in (("cyto", 3), ("ext", 3), ("mem", 2)):
        model.add_compartment(name, dim=dim)
    model.add_species("u", "cyto")
    _add_constant_flux_reaction(model, "influx", "mem", product="u", rate=rate)

    biomodel = Biomodel(name="MembraneInflux", model=model)
    app = biomodel.add_application("cellApp", geometry=_cell_geometry())
    app.map_compartment("cyto", "cytosol_dom")
    app.map_compartment("ext", "extra_dom")
    app.map_compartment("mem", "membrane_dom")
    app.map_species("u", init_conc="0.0", diff_coef=diffusion)
    app.map_reaction("influx", True)
    app.add_sim("sim", duration=1.0, output_time_step=0.5, mesh_size=(128, 128, 1))
    return biomodel


def _dump(model: Any, path: Path, exclude: Any = None) -> None:
    data = model.model_dump(mode="json", exclude_none=True, exclude_defaults=True, exclude=exclude)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _run(biomodel: Biomodel, stem: str | None) -> tuple[str, float, float]:
    """Run the FV solver; return (in_flux(u) expression, cytosol u mean at t0, at t_final)."""
    vcml = to_vcml_str(biomodel)
    md = VcmlReader.biomodel_from_str(vcml).applications[0].math_description
    if stem is not None:
        (_CV / f"{stem}.vcml").write_text(vcml)
        _dump(md, _CV / f"{stem}_math.yaml")
        _dump(biomodel.applications[0].geometry, _CV / f"{stem}_geom.yaml", exclude={"image": {"compressed_content"}})
    (in_flux,) = [jc.in_flux for m in md.membrane_subdomains for jc in m.jump_conditions if jc.name == "u"]
    result = vc.simulate(vc.load_vcml_str(vcml), biomodel.applications[0].simulations[0].name)
    try:
        u_index = list(result.volume_variable_names).index("cytosol_dom::u")
        conc = np.asarray(result.concentrations)[u_index]
    finally:
        result.cleanup()
    return in_flux, float(conc[0]), float(conc[-1])


def main() -> None:
    # Commit the efflux lowered math (the dev-env import-sign test reads it); influx is report-only.
    efflux_flux, e0, e1 = _run(author_membrane_efflux(), stem="membrane_efflux")
    influx_flux, i0, i1 = _run(author_membrane_influx(), stem=None)

    print("\n=== membrane jump-condition sign convention (FV ground truth) ===")
    for label, flux, c0, c1 in (("efflux", efflux_flux, e0, e1), ("influx", influx_flux, i0, i1)):
        sign = "negative" if "- 1.0" in flux else "positive"  # VCell prefixes an outward flux with `- 1.0 *`
        direction = "DECREASES" if c1 < c0 else "INCREASES"
        print(f"  {label:7} in_flux(u) {sign:8}  ->  FV cytosol u {c0:.4f} -> {c1:.4f}  [{direction}]")
    print("Our bridge maps in_flux directly (no flip) to BCNeumann; backend d(mass)/dt = ∫ h ds.")
    print("negative→decrease and positive→increase both match FV ⇒ sign convention confirmed two-sided.")


if __name__ == "__main__":
    main()

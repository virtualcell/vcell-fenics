"""Method-of-Manufactured-Solutions runner for pyvcell's **fvsolver** (fixed-grid finite volume).

Run in the pyvcell env — libvcell + the native solver are NOT in the DOLFINx env:

    ../pyvcell/.venv/bin/python mms/runner_fv.py mms/cases/<case>.yaml
    ../pyvcell/.venv/bin/python mms/runner_fv.py --all

Consumes the same persistent MMS case files as the vcell-fenics runner (`runner.py`), for any case whose
`applicable_solvers` includes `fvsolver`. It authors a VCell biomodel from the case's `fv:` block — a box
domain, one diffusing species, and the manufactured forcing as a **general-kinetics** volume source
`∅ → u` with rate `J = f` (mass-action cannot author a spatial/signed zeroth-order source; general kinetics
sets the net rate directly) — runs VCell's FV solver at a sequence of mesh sizes, samples the field on the
solver's own grid, and reports the **true L2/L∞ error vs the recorded exact solution `u*`** and the
h-convergence order. One manufactured model, checked against the same `u*` as vcell-fenics and mbsolver.

Case fields used (see `box_static_diffusion.yaml`):
  geometry: {kind: box, extent: [Lx, Ly], origin: [ox, oy]}   # the FV box
  exact:    {u: "np.cos(x)*np.cos(y)"}                          # for the error norm (numpy; x,y,z,t)
  fv: {ic: "<VCell init expr>", source_j: "<VCell rate expr>", diffusion: 0.1}  # VCell syntax: x, y, z, t
  fv_mesh_sizes: [41, 81, 161]                                  # FV element counts per axis
"""

from __future__ import annotations

import math
import sys
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

_CASES = Path(__file__).parent / "cases"
_NS = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi, "sqrt": np.sqrt, "tanh": np.tanh}


def _author(case: dict[str, Any], mesh: int) -> Biomodel:
    """A box biomodel: one diffusing species `u`, IC `u*(·,0)`, and the manufactured forcing as a
    general-kinetics volume source `∅ → u` (net rate J = f)."""
    ext = case["geometry"]["extent"]
    org = case["geometry"].get("origin", [0.0, 0.0])
    fv = case["fv"]
    geo = vc.Geometry(name="square", dim=2, extent=(ext[0], ext[1], 1.0), origin=(org[0], org[1], 0.0))
    geo.add_background("domain")
    model = Model(name="mms")
    model.add_compartment("cell", dim=3)
    model.add_species("u", "cell")
    kinetics = Kinetics(
        kinetics_type="GeneralKinetics",
        kinetics_parameters=[
            KineticsParameter(name="J", value=str(fv["source_j"]), role="reaction rate", unit="", reaction_name="src")
        ],
    )
    reaction = Reaction(name="src", compartment_name="cell", reversible=False, is_flux=False, kinetics=kinetics)
    reaction.products.append(SpeciesReference(name="u", stoichiometry=1, species_ref_type=SpeciesRefType.product))
    model.reactions.append(reaction)
    biomodel = Biomodel(name="MMS", model=model)
    app = biomodel.add_application("app", geometry=geo)
    app.map_compartment("cell", "domain")
    app.map_species("u", init_conc=str(fv["ic"]), diff_coef=fv["diffusion"])
    app.map_reaction("src", True)
    app.add_sim(name="s", duration=case["t_final"], output_time_step=case["t_final"], mesh_size=(mesh, mesh, 1))
    return biomodel


def _eval_exact(expr: str, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(eval(expr, {"__builtins__": {}}, {**_NS, "x": x, "y": y, "z": np.zeros_like(x), "t": t}))


def _run_fv(case: dict[str, Any], mesh: int) -> tuple[float, float]:
    biomodel = _author(case, mesh)
    result = vc.simulate(vc.load_vcml_str(to_vcml_str(biomodel)), "s")
    try:
        zd = result.zarr_dataset
        idx = {c.label: c.index for c in result.channel_data}
        x = np.asarray(zd[0, idx["x"], 0, 0, :], dtype=float)
        y = np.asarray(zd[0, idx["y"], 0, :, 0], dtype=float)
        name = result.volume_variable_names[0].split("::")[-1]
        u = np.asarray(zd[-1, idx[name], 0, :, :], dtype=float)  # (Y, X) at t_final
    finally:
        result.cleanup()
    xg, yg = np.meshgrid(x, y)
    u_star = _eval_exact(case["exact"]["u"], xg, yg, float(case["t_final"]))
    e = np.abs(u - u_star)
    return float(e.max()), float(np.sqrt(np.mean(e**2)))


def _order(errs: list[float], hs: list[float]) -> float | None:
    fine = [(e, h) for e, h in zip(errs, hs, strict=True) if e > 1e-13]
    return None if len(fine) < 2 else math.log(fine[0][0] / fine[-1][0]) / math.log(fine[0][1] / fine[-1][1])


def run_case(path: Path) -> None:
    case = yaml.safe_load(path.read_text())
    if "fvsolver" not in case.get("applicable_solvers", []) or "fv" not in case:
        print(f"  (skip {case['name']}: not an fvsolver case)")
        return
    meshes = case.get("fv_mesh_sizes", [41, 81, 161])
    ext = case["geometry"]["extent"]
    hs = [ext[0] / (m - 1) for m in meshes]
    print(f"\n=== {case['name']} (fvsolver) ===\n{case['description'].strip()}\n")
    linfs, l2s = [], []
    for m in meshes:
        linf, l2 = _run_fv(case, m)
        linfs.append(linf)
        l2s.append(l2)
    order = _order(linfs, hs)
    expected = case["expected_order_h"]
    ok = order is not None and order >= expected - 0.5
    ordstr = "n/a" if order is None else round(order, 2)
    print(f"  fvsolver  mesh={meshes}  h={[round(h, 4) for h in hs]}")
    print(f"    L∞ = {[f'{e:.2e}' for e in linfs]}   L2 = {[f'{e:.2e}' for e in l2s]}")
    print(f"    order(L∞) = {ordstr}  (expected ≥ {expected})  → {'OK' if ok else 'PROBLEM'}")


def main() -> None:
    args = sys.argv[1:]
    paths = sorted(_CASES.glob("*.yaml")) if (not args or args == ["--all"]) else [Path(a) for a in args]
    for p in paths:
        run_case(p)


if __name__ == "__main__":
    main()

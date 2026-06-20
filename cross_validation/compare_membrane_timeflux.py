#!/usr/bin/env python
"""Stage 2 of the time-dependent membrane-flux cross-validation (vcell-fenics **dev env**).

`membrane_timeflux_fv.py` (stage 1, pyvcell env) authored a two-compartment cell with a
**time-dependent** general-kinetics membrane influx `J(t) = A·exp(−k·t)`, ran VCell's FV solver, and
saved the cytosol `u(t, y, x)` field. This stage imports the same lowered math, solves the cytosol
through our backend, and compares — the case that exercises the backend's `sim.t` path end-to-end (a
`g(t)` Neumann that the driver re-evaluates each step).

The membrane flux depends only on `t`, so the cytosol PDE is decoupled from the extracellular and the
membrane is the whole cytosol-disk boundary. We import the real lowered math, then **reduce** it to
the cytosol (drop the species-free extracellular and the membrane *surface* subdomain; the membrane
survives as the disk's external boundary carrying the imported `BCNeumann(u, membrane_dom, g(t))`),
and solve on a single-compartment disk of the cytosol's radius. The unit factors VCell bakes into the
flux (`KFlux`, `UnitFactor = KMOLE`, reconciling membrane molecules·µm⁻² ↔ volume µM) are carried as
parameters, so our applied Neumann matches FV's flux in magnitude as well as sign.

    .pixi/envs/dev/bin/python cross_validation/compare_membrane_timeflux.py

(A full multi-compartment realize→run of a membrane flux — one-sided flux on an *internal* interface —
is a later backend increment; here the cytosol-disk reduction is exact because the flux is `g(t)`.)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pyvcell.vcml.models_math as mmod
import yaml
from compare_fenics_vs_fv import _eval_on_grid

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.formalism.schema import MathDescription
from vcell_fenics.pyvcell_bridge import import_math_description

_CV = Path(__file__).resolve().parent
_SOLVER_DT = 0.01


def _reduce_to_cytosol(
    md: MathDescription, *, compartment: str, membrane_boundary: str, geometry: str
) -> MathDescription:
    """Reduce an imported two-compartment membrane-flux model to the single flux-bearing compartment:
    keep that subdomain, its variables/equations, and only the membrane boundary condition; drop the
    other (species-free) compartment and the membrane surface subdomain, and re-scope any parameter
    that was homed on a dropped subdomain to global (we no longer carry those subdomains)."""
    keep = {compartment}
    allowed = {None} | keep
    params = tuple(
        replace(p, subdomain=None) if getattr(p, "subdomain", None) not in allowed else p for p in md.parameters
    )
    return replace(
        md,
        geometry=geometry,
        subdomains=tuple(s for s in md.subdomains if s.name in keep),
        variables=tuple(v for v in md.variables if v.subdomain in keep),
        equations=tuple(e for e in md.equations if e.subdomain in keep),
        parameters=params,
        boundary_conditions=tuple(b for b in md.boundary_conditions if b.boundary == membrane_boundary),
    )


def main() -> None:
    ref = np.load(_CV / "membrane_timeflux_reference.npz", allow_pickle=True)
    fv = ref["field"]  # (T, Y, X) cytosol u
    t, x, y = ref["t"].astype(float), ref["x"].astype(float), ref["y"].astype(float)
    radius = float(ref["radius"])
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    md_raw = mmod.MathDescription.model_validate(yaml.safe_load((_CV / "membrane_timeflux_math.yaml").read_text()))
    md = import_math_description(md_raw, geometry="cell", dim=2)
    reduced = _reduce_to_cytosol(md, compartment="cytosol_dom", membrane_boundary="membrane_dom", geometry="cyto_disk")
    geometry = make_disk_geometry(
        "cyto_disk", volume_subdomain="cytosol_dom", boundary="membrane_dom", radius=radius, h=0.02
    )
    mesh = geometry.mesh_of("cytosol_dom")

    problem = assemble(reduced, geometry, dt=_SOLVER_DT)
    # Sample only strictly inside the disk — the FV cytosol field is defined on the cytosol region, and
    # our disk mesh covers r < radius (a thin rim is dropped to avoid boundary-interpolation noise).
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    inside = (grid_x**2 + grid_y**2) < (0.95 * radius) ** 2

    ours = np.empty_like(fv)
    ours[0] = _eval_on_grid(problem.unknown, mesh, x, y)
    out_i = 0
    for n in range(round(float(t[-1]) / _SOLVER_DT)):
        now = (n + 1) * _SOLVER_DT
        problem.set_time(now)  # advances sim.t -> the membrane Neumann g(t) is re-evaluated this step
        problem.step()
        if out_i + 1 < len(t) and np.isclose(now, t[out_i + 1], atol=_SOLVER_DT / 2):
            out_i += 1
            ours[out_i] = _eval_on_grid(problem.unknown, mesh, x, y)

    print(
        f"\n=== membrane time-flux: FV vs FEniCSx (cytosol disk r={radius}, {int(inside.sum())} interior grid pts) ==="
    )
    print(f"in_flux = {ref['in_flux']}")
    print(f"{'t':>6} {'relL2(FEM,FV)':>14} {'mass_FEM':>10} {'mass_FV':>10}")
    for ti in range(len(t)):
        a, b = ours[ti][inside], fv[ti][inside]
        rel = np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0)
        mass_fem = ours[ti][inside].sum() * dx * dy
        mass_fv = fv[ti][inside].sum() * dx * dy
        print(f"{t[ti]:6.2f} {rel:14.4%} {mass_fem:10.5f} {mass_fv:10.5f}")


if __name__ == "__main__":
    main()

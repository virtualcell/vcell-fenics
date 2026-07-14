"""Stage 1 (../pyvcell/.venv): the mbsolver reference for the moving-boundary EXPANSION cross-validation.

A disk of cytoplasm (radius R0, centred at (C,C)) in an extracellular background, carrying an interior
species u (uniform initial condition u=1, no reaction). The front and the species both move with the
Lagrangian radial velocity `v = k (x − c)` — so `∇·v = 2k > 0` and the cell dilates. Unlike the
translation case (rigid, `∇·v = 0`), this exercises the mandatory `ρ ∇·v` **dilution** term: the exact
solution is `R(t) = R0 e^{kt}`, `u(t) = e^{−2k t}` (spatially uniform), total substance conserved.

With `v = v_b` (species carried with the cell), the moving-frame PDE is diffusion + dilution and the
Rankine–Hugoniot BC reduces to no-flux — the convention our ALE `MotionPrescribedVelocity` matches.

Writes `mb_expansion_fv.npz` (output times, front radius, u mean/spread). Gitignored + regenerable:

    ../pyvcell/.venv/bin/python cross_validation/mb_expansion_fv.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
from pyvcell.vcml.models import Biomodel, Model

R0, D, K, DURATION, OUT_STEP = 3.0, 1.0, 0.25, 2.0, 0.5
CX = 8.0  # disk centre; box is 16×16 so the dilating front (R(2)=R0 e^{0.5}=4.95) stays interior
EXTENT = 16.0


@contextmanager
def _silence():  # type: ignore[no-untyped-def]
    """Mute the native solver's per-substep chatter (it writes to fd 1 from C++)."""
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def main() -> None:
    geo = vc.Geometry(name="square", dim=2, extent=(EXTENT, EXTENT, EXTENT), origin=(0.0, 0.0, 0.0))
    geo.add_sphere("cell", radius=R0, center=(CX, CX, 0.0))
    geo.add_background("ec")
    geo.add_surface("cell_ec_membrane", "cell", "ec")

    model = Model(name="m")
    for compartment, dim in (("cyt", 3), ("ec", 3), ("pm", 2)):
        model.add_compartment(compartment, dim=dim)
    model.add_species("u", "cyt")

    biomodel = Biomodel(name="MB", model=model)
    app = biomodel.add_application("a", geometry=geo)
    app.map_compartment("cyt", "cell")
    app.map_compartment("ec", "ec")
    app.map_compartment("pm", "cell_ec_membrane")
    sm = app.map_species("u", init_conc="1", diff_coef=D)
    # v = v_b: radial dilating velocity k (x − c), species carried with the cell (Lagrangian).
    vx, vy = f"{K}*(x - {CX})", f"{K}*(y - {CX})"
    sm.velocity_x, sm.velocity_y = vx, vy
    app.set_moving_boundary_front(velocity_x=vx, velocity_y=vy)
    app.add_moving_boundary_sim(name="s", duration=DURATION, output_time_step=OUT_STEP, mesh_size=(41, 41, 1))

    with _silence():
        result = vc.simulate_moving_boundary(biomodel, "s")

    times = np.array(result.times, dtype=float)
    # front radius = mean distance of the front polygon vertices from the (fixed) centre
    front_r = np.array([float(np.hypot(f.front[:, 0] - CX, f.front[:, 1] - CX).mean()) for f in result.frames])
    u_mean = np.array([float(f.concentrations["u"].mean()) for f in result.frames])
    u_spread = np.array([float(f.concentrations["u"].max() - f.concentrations["u"].min()) for f in result.frames])

    out = Path(__file__).parent / "mb_expansion_fv.npz"
    np.savez(out, times=times, front_r=front_r, u_mean=u_mean, u_spread=u_spread, R0=R0, D=D, K=K, CX=CX)
    print(f"  wrote {out.name}")
    print(f"    front radius {front_r[0]:.3f} → {front_r[-1]:.3f}  (exact {R0} → {R0 * np.exp(K * DURATION):.3f})")
    print(f"    u mean       {u_mean[0]:.3f} → {u_mean[-1]:.3f}  (exact 1.0 → {np.exp(-2 * K * DURATION):.3f})")


if __name__ == "__main__":
    main()

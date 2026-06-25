"""Stage 1 (../pyvcell/.venv): the mbsolver reference for the moving-boundary TRANSLATION cross-validation.

A disk of cytoplasm (radius 3) in an extracellular background, carrying an interior species C
(diffusion 10, initial condition C = x). Both the **front** and the species **advection** move at
`v = v_b = (0.5, 0)` — the Lagrangian "cell carries its cytoplasm" convention. This is what matches our
ALE `MotionPrescribedVelocity`: per Novak & Slepchenko (2014), with `v = v_b` the parabolic equation is
pure diffusion in the moving frame and the Rankine–Hugoniot boundary condition `(J − v_b u)·n = 0`
reduces to plain no-flux. (Setting only the front velocity, `v = 0`, instead sweeps a fixed lab-frame
field — a traveling exponential — a *different* physical problem; that is the discrepancy this comparison
resolves.)

Writes `mb_translation_fv.npz` (output times, front-centroid x, C spread, C mean) for the comparison
report, and `mb_translation_fv_fields.npz` (per-frame front polygon + inside scatter) for the picture.
Both are gitignored and regenerable; run before the fenics-side stage 2:

    ../pyvcell/.venv/bin/python cross_validation/mb_translation_fv.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
from pyvcell.vcml.models import Biomodel, Model

R, D, V, DURATION, OUT_STEP = 3.0, 10.0, 0.5, 2.0, 0.5


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
    geo = vc.Geometry(name="square", dim=2, extent=(10.0, 10.0, 10.0), origin=(0.0, 0.0, 0.0))
    geo.add_sphere("cell", radius=R, center=(5.0, 5.0, 0.0))
    geo.add_background("ec")
    geo.add_surface("cell_ec_membrane", "cell", "ec")

    model = Model(name="m")
    for compartment, dim in (("cyt", 3), ("ec", 3), ("pm", 2)):
        model.add_compartment(compartment, dim=dim)
    model.add_species("C", "cyt")

    biomodel = Biomodel(name="MB", model=model)
    app = biomodel.add_application("a", geometry=geo)
    app.map_compartment("cyt", "cell")
    app.map_compartment("ec", "ec")
    app.map_compartment("pm", "cell_ec_membrane")
    sm = app.map_species("C", init_conc="x", diff_coef=D)
    # v = v_b: advect C with the cell. "0" (not "0.0") dodges the writer's zero-component drop on older
    # pyvcell; the fix is virtualcell/pyvcell#54.
    sm.velocity_x, sm.velocity_y = str(V), "0"
    app.set_moving_boundary_front(velocity_x=str(V), velocity_y="0")
    app.add_moving_boundary_sim(name="s", duration=DURATION, output_time_step=OUT_STEP, mesh_size=(31, 31, 1))

    with _silence():
        result = vc.simulate_moving_boundary(biomodel, "s")

    times = np.array(result.times, dtype=float)
    front_cx = np.array([float(f.front.mean(axis=0)[0]) for f in result.frames])
    c_spread = np.array([float(f.concentrations["C"].max() - f.concentrations["C"].min()) for f in result.frames])
    c_mean = np.array([float(f.concentrations["C"].mean()) for f in result.frames])

    out = Path(__file__).parent / "mb_translation_fv.npz"
    np.savez(out, times=times, front_cx=front_cx, c_spread=c_spread, c_mean=c_mean, R=R, D=D, V=V)

    # Per-frame spatial fields for the picture (mb_translation_plot.py): the moving front polygon and
    # the inside scatter. NB this mbsolver build's per-node `x`/`y` accessors return the front centroid
    # for every node (all collapse to the cell centre), so we save the reliable integer `grid_i`/`grid_j`
    # instead and reconstruct positions from the background grid in the plotter. Ragged across frames →
    # object arrays (loaded with allow_pickle).
    fields = Path(__file__).parent / "mb_translation_fv_fields.npz"
    np.savez(
        fields,
        times=times,
        fronts=np.array([f.front for f in result.frames], dtype=object),
        grid_i=np.array([f.grid_i for f in result.frames], dtype=object),
        grid_j=np.array([f.grid_j for f in result.frames], dtype=object),
        cs=np.array([f.concentrations["C"] for f in result.frames], dtype=object),
        extent=10.0,
        mesh_n=31,
        R=R,
        V=V,
    )
    print(f"  wrote {out.name}, {fields.name}")
    print(f"    front centroid Δx over t={DURATION}: {front_cx[-1] - front_cx[0]:+.3f}  (ideal {V * DURATION:+.3f})")
    print(f"    C spread {c_spread[0]:.3f} -> {c_spread[-1]:.4f}; mean ~ {c_mean.mean():.3f} (conserved)")


if __name__ == "__main__":
    main()

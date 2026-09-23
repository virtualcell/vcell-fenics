"""Stage 1 (../pyvcell/.venv): the mbsolver reference for the moving-boundary SWEPT cross-validation.

VCell's own moving-boundary semantics, as the real fixture ``tests/fixtures/simtask/SimID_274641196`` has
them: a disk ``cell`` (radius 3, centre (5, 5)) in a 10 × 10 box, species C (initial x) and Ran (initial y)
diffusing (D = 10) with **no velocity of their own**, and a front moving at ``(sin t, cos t)``. The species
stay in the lab frame and the front sweeps them (Rankine–Hugoniot ``(−D∇u − v_b u)·n = 0`` at the front):
in the cell's frame they drift backwards and pile against the trailing membrane. This is the problem the
vcell-fenics SimulationTask path solves for that fixture (tracker M3: the lab-frame ``advection`` slot);
``mb_translation_fv.py`` is the other convention (species velocity = front velocity, carried).

Writes ``mb_swept_fv.npz`` (per output time: the inside nodes' positions and C/Ran values, and the front
polygon) — gitignored and regenerable. Run before stage 2 (``mb_swept.py``):

    ../pyvcell/.venv/bin/python cross_validation/mb_swept_fv.py [--case translate|deform|remesh] [--mesh 31] [--dt-out 0.1]
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyvcell.vcml as vc
from pyvcell.vcml.models import Biomodel, Model

R, D, EXTENT, DURATION = 3.0, 10.0, 10.0, 1.0
# The front velocity per case (VCell syntax). `translate` is the fixture's own. `deform` stretches the cell's
# right side and squeezes its left, staying inside the box to t = 1 (on the x-axis the front's extremes follow
# du/dt = a u² − b, u = x − 5, which is exact to compare against). `remesh` deforms enough that vcell-fenics
# remeshes once; this mbsolver build overflows (Voronoi32) above mesh 31 on it, so its reference is coarse.
CASES = {
    "translate": ("sin(t)", "cos(t)"),
    "deform": ("0.2 * (x - 5.0)^2 - 1.2", "0.0"),
    "remesh": ("0.4 * (x - 5.0)^2 - 3.4", "0.0"),
}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), default="translate")
    parser.add_argument("--mesh", type=int, default=31)
    parser.add_argument("--dt-out", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(__file__).parent / ("mb_swept_fv.npz" if args.case == "translate" else f"mb_swept_{args.case}_fv.npz")

    geo = vc.Geometry(name="square", dim=2, extent=(EXTENT, EXTENT, EXTENT), origin=(0.0, 0.0, 0.0))
    geo.add_sphere("cell", radius=R, center=(5.0, 5.0, 0.0))
    geo.add_background("ec")
    geo.add_surface("cell_ec_membrane", "cell", "ec")

    model = Model(name="m")
    for compartment, dim in (("cyt", 3), ("ec", 3), ("pm", 2)):
        model.add_compartment(compartment, dim=dim)
    model.add_species("C", "cyt")
    model.add_species("Ran", "cyt")

    biomodel = Biomodel(name="MBswept", model=model)
    app = biomodel.add_application("a", geometry=geo)
    app.map_compartment("cyt", "cell")
    app.map_compartment("ec", "ec")
    app.map_compartment("pm", "cell_ec_membrane")
    app.map_species("C", init_conc="x", diff_coef=D)  # no species velocity: the lab frame
    app.map_species("Ran", init_conc="y", diff_coef=D)
    velocity_x, velocity_y = CASES[args.case]
    app.set_moving_boundary_front(velocity_x=velocity_x, velocity_y=velocity_y)
    app.add_moving_boundary_sim(
        name="s", duration=DURATION, output_time_step=args.dt_out, mesh_size=(args.mesh, args.mesh, 1)
    )

    with _silence():
        result = vc.simulate_moving_boundary(biomodel, "s")

    # This mbsolver build's per-node x/y accessors return the front centroid for every node, so the
    # positions come from the integer grid indices on the background grid (as in mb_translation_fv.py).
    h = EXTENT / (args.mesh - 1)
    times = np.array(result.times, dtype=float)
    np.savez(
        out,
        times=times,
        xs=np.array([np.asarray(f.grid_i) * h for f in result.frames], dtype=object),
        ys=np.array([np.asarray(f.grid_j) * h for f in result.frames], dtype=object),
        C=np.array([np.asarray(f.concentrations["C"]) for f in result.frames], dtype=object),
        Ran=np.array([np.asarray(f.concentrations["Ran"]) for f in result.frames], dtype=object),
        fronts=np.array([f.front for f in result.frames], dtype=object),
        R=R,
        D=D,
        mesh=args.mesh,
        velocity=np.array(CASES[args.case]),
    )
    centroid = [f.front.mean(axis=0) for f in result.frames]
    print(f"  wrote {out.name}: {len(times)} frames, mesh {args.mesh}, front velocity {CASES[args.case]}")
    print(f"    front centroid shift over t={DURATION}: {centroid[-1] - centroid[0]}  (exact (1 - cos 1, sin 1))")


if __name__ == "__main__":
    main()

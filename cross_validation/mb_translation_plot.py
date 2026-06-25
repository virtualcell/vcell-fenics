"""Picture for the moving-boundary TRANSLATION cross-validation: mbsolver (FronTier FV) vs fenics (ALE).

A tiled figure, one row per output time, two columns:

* left  — the mbsolver reference: the moving front polygon + the inside scatter, coloured by C;
* right — the fenics ALE solution on its moving mesh, coloured by C.

Both show a disk translating right at v = v_b = (0.5, 0) while its interior gradient (C = x at t = 0)
homogenises. To make them directly comparable the positions are **recentred** to a common start (the FV
domain puts the cell at (5, 5); the fenics disk is at the origin) and the colour is **C − mean(C)** — the
decaying gradient — since diffusion conserves the mean and the two setups differ only by that constant
offset. The point is qualitative: same translation, same homogenisation.

Stage 2 of the comparison (the report is `mb_translation.py`). Run stage 1 first to make the fields file:

    ../pyvcell/.venv/bin/python cross_validation/mb_translation_fv.py
    .pixi/envs/dev/bin/python   cross_validation/mb_translation_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt
from mb_translation import _DT, _model  # the same fenics model the report uses

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml

_HERE = Path(__file__).parent


def main() -> None:
    fields_path = _HERE / "mb_translation_fv_fields.npz"
    summary_path = _HERE / "mb_translation_fv.npz"
    if not (fields_path.exists() and summary_path.exists()):
        raise SystemExit("missing FV reference — run mb_translation_fv.py in ../pyvcell/.venv first")
    fv = np.load(fields_path, allow_pickle=True)
    D = float(np.load(summary_path)["D"])
    R, V = float(fv["R"]), float(fv["V"])
    times = fv["times"]
    fronts, grid_i, grid_j, cs = fv["fronts"], fv["grid_i"], fv["grid_j"], fv["cs"]

    # Reconstruct inside-node positions from the background grid (this mbsolver build's per-node x/y
    # accessors are unusable — they return the cell centroid; the integer grid indices are reliable).
    # Calibrated against the trusted front: grid index k → k · extent/(mesh_n − 1) (origin at 0).
    h = float(fv["extent"]) / (int(fv["mesh_n"]) - 1)
    fv_xs = [gi * h for gi in grid_i]
    fv_ys = [gj * h for gj in grid_j]

    # Recentre the FV fields on the cell's start (its front centroid at t=0), so both panels begin at 0.
    fv_x0, fv_y0 = fronts[0][:, 0].mean(), fronts[0][:, 1].mean()

    # Run the fenics ALE, snapshotting (dof coords, C) at each FV output time.
    geom = make_disk_geometry("disk", volume_subdomain="cyto", radius=R, h=0.15)
    dp = assemble(load_yaml(_model(D, V)), geom, dt=_DT)
    steps_per_out = round(float(times[1] - times[0]) / _DT)

    def snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = dp.V.tabulate_dof_coordinates()
        return coords[:, 0].copy(), coords[:, 1].copy(), dp.unknown.x.array.copy()

    fe = [snapshot()]
    for _ in range(len(times) - 1):
        for _ in range(steps_per_out):
            dp.step()
        fe.append(snapshot())

    # C − mean is the comparable gradient; it starts at ±R and decays to ~0. Shared diverging scale.
    vlim = R
    cmap = "coolwarm"
    nrows = len(times)
    fig, axes = plt.subplots(nrows, 2, figsize=(7.2, 2.7 * nrows), squeeze=False)

    for i, t in enumerate(times):
        fx = fv_xs[i] - fv_x0
        fy = fv_ys[i] - fv_y0
        fc = cs[i] - cs[i].mean()
        ax_fv = axes[i][0]
        ax_fv.scatter(fx, fy, c=fc, cmap=cmap, vmin=-vlim, vmax=vlim, s=70, marker="s", edgecolors="none")
        front = fronts[i]
        ring = np.vstack([front, front[:1]])
        ax_fv.plot(ring[:, 0] - fv_x0, ring[:, 1] - fv_y0, color="k", lw=1.0)

        ex, ey, ec = fe[i]
        ec = ec - ec.mean()
        ax_fe = axes[i][1]
        mappable = ax_fe.tripcolor(mtri.Triangulation(ex, ey), ec, cmap=cmap, vmin=-vlim, vmax=vlim, shading="gouraud")

        for ax in (ax_fv, ax_fe):
            ax.set_aspect("equal")
            ax.set_xlim(-4.5, 5.0)
            ax.set_ylim(-4.5, 4.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axvline(0.0, color="0.7", lw=0.6, ls=":")  # the start; the cell drifts right of it
        ax_fv.set_ylabel(f"t = {t:.1f}", fontsize=11)

    axes[0][0].set_title("mbsolver  (FronTier FV)", fontsize=11)
    axes[0][1].set_title("fenics  (ALE)", fontsize=11)
    fig.suptitle(
        f"moving-boundary translation  ·  v = v_b = ({V}, 0),  D = {D}  ·  colour = C − mean (decaying gradient)",
        fontsize=11,
    )
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.6, location="right", pad=0.02)
    cbar.set_label("C − mean")

    out = _HERE / "mb_translation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

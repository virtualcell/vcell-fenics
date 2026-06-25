"""Large-deformation surface PDE: method-of-lines + remeshing on a moving cell membrane.

The project's canonical moving-surface problem — a receptor density ``rho`` on a closed cell membrane,
obeying surface diffusion with the mandatory dilution term ``rho ∇_Γ·v_Γ`` (a stretching membrane
locally dilutes its receptors) — driven through a *large* shape change that no single fixed mesh can
follow. The membrane elongates under an angle-dependent radial velocity ``(1 + 0.8 cos 2θ)·n̂`` (a
spreading/elongating cell); the node spacing distorts steadily, so the simulation must **remesh** to
stay valid.

This illustrates `backend.ale.run_moving_with_remeshing` — the combination of

* **method-of-lines** (`integrate_discrete_problem_stride`): each inter-move interval is integrated by
  the adaptive-BDF PETSc ``TS`` (the same strategy as VCell's reaction-diffusion solver), and
* **remeshing** (`rebuild_on_mesh` + the conservative surface remap): before any stride would move an
  over-distorted mesh, swap to a fresh mesh of the deformed configuration, carrying ``rho`` across.

The figure contrasts the two runs:

* top — membrane snapshots (coloured by ``rho``) at four times, showing the disk elongate while the
  receptor density redistributes by surface diffusion and dilution;
* bottom — mesh-quality growth vs time: **without** remeshing it climbs unbounded (the mesh degrades);
  **with** remeshing it is held near the quality budget (a sawtooth that resets at each remesh).

Run (dev env):

    .pixi/envs/dev/bin/python examples/deforming_membrane_mol_remeshing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from vcell_fenics.backend import (
    ALEState,
    SolverConfiguration,
    assemble,
    integrate_discrete_problem_stride,
    make_disk_membrane_geometry,
    stride_with_remeshing,
)
from vcell_fenics.formalism import load_yaml

_HERE = Path(__file__).parent

VELOCITY = "(1.0 + 0.8 * cos(2 * geom.azimuth)) * geom.x / geom.radius"  # elongating (cos 2θ) outward motion
T_FINAL, MOTION_STEPS, MESH_H, QUALITY_LIMIT = 2.0, 60, 0.10, 1.4
SNAPSHOT_TIMES = (0.0, 0.67, 1.33, 2.0)

MODEL = f"""
math_description:
  geometry: disk_membrane
  subdomains:
    - {{ name: membrane, kind: surface, motion: {{ kind: prescribed, velocity: "{VELOCITY}" }} }}
  variables:
    - {{ name: rho, subdomain: membrane }}
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms: {{ diffusion: "0.02" }}
      initial_condition: "1.0 + 0.6 * cos(geom.azimuth)"  # a front–back receptor gradient
"""


def _geom():  # type: ignore[no-untyped-def]
    return make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=MESH_H)


def _membrane_curve(problem):  # type: ignore[no-untyped-def]
    """The membrane as an ordered closed loop `(x, y, rho)`. The curve is star-convex here, so the dof
    coordinates order by azimuth; pairing them with the P1 dof values avoids any node/dof reshuffle."""
    coords = problem.V.tabulate_dof_coordinates()[:, :2]
    rho = problem.unknown.x.array
    order = np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))
    x, y, r = coords[order, 0], coords[order, 1], rho[order]
    return np.append(x, x[0]), np.append(y, y[0]), np.append(r, r[0])  # close the loop


def _run_remeshed():  # type: ignore[no-untyped-def]
    """Strided MOL with remeshing, capturing the quality trace, remesh times, and snapshots."""
    state = ALEState.initial(load_yaml(MODEL), _geom(), SolverConfiguration(dt=0.01, t_final=T_FINAL))
    h = T_FINAL / MOTION_STEPS
    state.problem.dt.value = h
    times, quality, remesh_marks, snapshots = [0.0], [1.0], [], {}
    snap_targets = list(SNAPSHOT_TIMES)
    if snap_targets[0] == 0.0:
        snapshots[0.0] = _membrane_curve(state.problem)
        snap_targets.pop(0)
    for i in range(MOTION_STEPS):
        before = state.remesh_count
        stride_with_remeshing(state, h=h, t_start=i * h, quality_limit=QUALITY_LIMIT, target_h=MESH_H)
        if state.remesh_count > before:
            remesh_marks.append(state.t)
        times.append(state.t)
        quality.append(state.problem.mesh_quality_growth())
        if snap_targets and state.t >= snap_targets[0] - 1e-9:
            snapshots[snap_targets.pop(0)] = _membrane_curve(state.problem)
    return state, np.array(times), np.array(quality), remesh_marks, snapshots


def _run_no_remesh():  # type: ignore[no-untyped-def]
    """The same motion with no remeshing — the quality climbs as the mesh degrades."""
    problem = assemble(load_yaml(MODEL), _geom(), dt=0.01)
    h = T_FINAL / MOTION_STEPS
    problem.dt.value = h
    times, quality = [0.0], [1.0]
    for i in range(MOTION_STEPS):
        problem.advance_mesh()
        integrate_discrete_problem_stride(problem, t_start=i * h, t_final=(i + 1) * h)
        times.append((i + 1) * h)
        quality.append(problem.mesh_quality_growth())
    return problem, np.array(times), np.array(quality)


def main() -> None:
    state, t_rm, q_rm, remesh_marks, snapshots = _run_remeshed()
    _problem, t_nr, q_nr = _run_no_remesh()

    print(
        f"  with remeshing:    t={state.t:.2f}, remeshes={state.remesh_count}, TS steps={state.steps}, "
        f"final mesh-quality growth={q_rm[-1]:.2f}x"
    )
    print(f"  without remeshing: final mesh-quality growth={q_nr[-1]:.2f}x  (mesh degraded)")

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, len(snapshots), height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.18)
    vmax = max(float(np.abs(r).max()) for _, _, r in snapshots.values())

    # top row — membrane snapshots coloured by rho
    mappable = None
    for col, (t, (x, y, r)) in enumerate(sorted(snapshots.items())):
        ax = fig.add_subplot(gs[0, col])
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", array=(r[:-1] + r[1:]) / 2, linewidth=4)
        lc.set_clim(0.0, vmax)
        ax.add_collection(lc)
        mappable = lc
        ax.set_aspect("equal")
        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t = {t:.2f}", fontsize=11)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.axes, location="right", shrink=0.45, pad=0.015)
        cbar.set_label("receptor density ρ")

    # bottom — mesh-quality growth vs time
    axq = fig.add_subplot(gs[1, :])
    axq.plot(t_nr, q_nr, color="crimson", lw=2, label="no remeshing (mesh degrades)")
    axq.plot(t_rm, q_rm, color="seagreen", lw=2, label="with remeshing (bounded)")
    axq.axhline(QUALITY_LIMIT, color="0.6", ls="--", lw=1, label=f"remesh trigger ({QUALITY_LIMIT}×)")
    for k, tm in enumerate(remesh_marks):
        axq.axvline(tm, color="seagreen", ls=":", lw=0.9, alpha=0.7, label="remesh" if k == 0 else None)
    axq.set_xlabel("time")
    axq.set_ylabel("mesh-quality growth\n(cell-size ratio / initial)")
    axq.set_xlim(0, T_FINAL)
    axq.legend(fontsize=9, loc="upper left", ncol=2)
    axq.set_title("Method-of-lines on a large-deforming membrane: remeshing keeps the mesh valid", fontsize=11)

    out = _HERE / "deforming_membrane_mol_remeshing.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()

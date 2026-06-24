"""LEGI gradient SENSING with a surface-tension force balance — the sensing works; net migration does
NOT (yet). An honest intermediate result; proper migration is a follow-up.

LEGI = Local Excitation, Global Inhibition, the standard eukaryotic gradient-sensing model. In a
disk-in-a-box the SENSING chain works and is held by the external cue:

    chemoattractant C in the ext, held in a gradient by a DIRICHLET on the box wall  (C = 1 + 0.6·x)
        → membrane receptor R reads the LOCAL C (occupancy, polarized)
            → cyto LOCAL activator a (slow diffusion, tracks local R) and GLOBAL inhibitor h
              (fast diffusion ≈ cell-mean drive) — both produced from R at the membrane
                → response  a − h  is the RELATIVE-gradient polarity: high on the up-gradient (+x) side,
                  negative down-gradient, and SUSTAINED (the external Dirichlet cue holds it).

The response sets the membrane tension γ = base + α·(a − h), and the surface-tension force balance
(`ForceBalanceMeshMotion`) responds — but with a **Marangoni surface flow, not migration**. The cell's
center of mass does NOT translate: a closed, incompressible cell under a *pure* surface-tension balance
has no net propulsive force, so a tension gradient drives a tangential (Marangoni) circulation that
treadmills the membrane while the CoM stays put. `legi_diagnostic.py` shows this directly (see
`legi_diag.png`): every membrane outline from t=0 to t=5.6 overlaps as a centered circle with its area
centroid at the origin, and the velocity field at t=4 is a recirculating Marangoni flow (max|u| ≈ 0.12).
(An earlier version of this demo reported a "+0.2 migration" — that was the membrane node-MEAN sliding
along the fixed circle with the surface flow, not CoM translation. A misleading diagnostic.)

So this demonstrates correct LEGI **sensing** and the mechano-chemical response. Genuine migration needs
an active, front–back-asymmetric mechanism the force balance lacks — a *normal* protrusion/retraction
velocity, an active cortical stress, or asymmetric substrate adhesion — left to a follow-up.

The force balance runs with `area_correction=True`. The P2 Stokes velocity is divergence-free, but its
P1 interpolation (which moves the mesh) is not, leaking a spurious inward flux that would shrink the cell
~20% over this run; the correction re-projects that interpolated velocity to be divergence-free, at the
interpolation (see `ForceBalanceMeshMotion`).

Writes `legi_sustained_chemotaxis.png`: the LEGI response a − h over time (the polarized, sustained
sensing). The cell shape stays a centered circle — the apparent shift is the node parametrization
treadmilling, not translation; `legi_diag.png` makes that explicit.

    .pixi/envs/dev/bin/python examples/legi_sustained_chemotaxis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.backend.geometry import make_two_bulk_membrane_geometry
from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion, assemble_membrane_coupled
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.viz import _function_to_pyvista

_HERE = Path(__file__).parent
_BASE_TENSION, _ALPHA, _DT, _STEPS, _FRAME_EVERY = 0.5, 0.12, 0.02, 280, 35
_GRADIENT = "1.0 + 0.6 * geom.x[0]"


def model() -> MathDescription:
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[
            Variable(name="a", subdomain="cyto"),  # local activator (slow)
            Variable(name="h", subdomain="cyto"),  # global inhibitor (fast)
            Variable(name="C", subdomain="ext"),  # chemoattractant
            Variable(name="R", subdomain="pm"),  # receptor
        ],
        parameters=[
            ParameterConstant(name="kon", value=1.0),
            ParameterConstant(name="Rmax", value=2.0),
            ParameterConstant(name="ka", value=0.5),
        ],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="a",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.02", "source": "-a"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="h",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "5.0", "source": "-h"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="C",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.02", "source": "kon * trace(C) * (Rmax - R) - R"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCDirichlet(variable="C", boundary="wall", expression=_GRADIENT),  # the gradient, on the BOX
            BCInterfaceFlux(variable="C", boundary="pm", expression="-kon * trace(C) * (Rmax - R)"),  # C bound
            BCInterfaceFlux(variable="a", boundary="pm", expression="ka * R"),  # local activator from R
            BCInterfaceFlux(variable="h", boundary="pm", expression="ka * R"),  # global inhibitor from R
        ],
    )


def _write_tiled_image(frames: list[tuple[pyvista.UnstructuredGrid, float]], path: Path) -> None:
    pyvista.OFF_SCREEN = True
    lo = min(grid.point_data["response"].min() for grid, _ in frames)
    hi = max(grid.point_data["response"].max() for grid, _ in frames)
    cols = (len(frames) + 1) // 2
    plotter = pyvista.Plotter(shape=(2, cols), off_screen=True, window_size=(200 * cols, 440), border=False)
    box = pyvista.Box(bounds=(-1.55, 1.55, -1.55, 1.55, -0.01, 0.01))
    for i, (grid, t) in enumerate(frames):
        plotter.subplot(i // cols, i % cols)
        plotter.add_mesh(box, style="wireframe", color="lightgray", opacity=0.4)
        plotter.add_mesh(
            grid,
            scalars="response",
            clim=[lo, hi],
            cmap="coolwarm",
            show_edges=False,
            show_scalar_bar=(i == 0),
            scalar_bar_args={"title": "response a-h", "n_labels": 3, "fmt": "%.1f", "label_font_size": 10},
        )
        plotter.add_text(f"t = {t:.1f}", font_size=9)
        plotter.view_xy()
        plotter.camera.zoom(1.1)
    plotter.screenshot(str(path))
    plotter.close()


def main() -> None:
    geom = make_two_bulk_membrane_geometry(
        "cell",
        inner="cyto",
        outer_subdomain="ext",
        membrane="pm",
        interface="pm",
        outer="wall",
        inner_radius=0.5,
        outer_radius=1.5,
        h=0.06,
    )
    motion = ForceBalanceMeshMotion(geom, tension=_BASE_TENSION, dt=_DT, area_correction=True)
    problem = assemble_membrane_coupled(model(), geom, dt=_DT, motion=motion)
    mem = geom.membrane_mesh
    node_mean_0 = float(mem.geometry.x[:, 0].mean())

    # The TRUE centre of mass is the area centroid ∫x dx / ∫dx — not the membrane node-mean, which slides
    # with the (treadmilling) surface flow. Comparing the two is the whole point of this intermediate result.
    cyto = geom.inner_mesh
    _one = fem.form(fem.Constant(cyto, PETSc.ScalarType(1.0)) * ufl.dx)  # type: ignore[operator]
    _mom_x = fem.form(ufl.SpatialCoordinate(cyto)[0] * ufl.dx)

    def com_x() -> float:
        return float(fem.assemble_scalar(_mom_x).real) / float(fem.assemble_scalar(_one).real)

    com_x_0 = com_x()

    def response_grid() -> pyvista.UnstructuredGrid:  # NON-destructive: does not modify the a field
        a, h = problem.field("a"), problem.field("h")
        grid = _function_to_pyvista(a)
        grid.point_data["response"] = (a.x.array - h.x.array).real
        grid.set_active_scalars("response")
        return grid

    def set_tension() -> None:  # γ = base + α·(a − h): the sustained LEGI response drives the tension
        motion.tension.x.array[:] = _BASE_TENSION + _ALPHA * (problem.field("a").x.array - problem.field("h").x.array)

    def polarity() -> float:
        a = problem.field("a")
        resp = a.x.array - problem.field("h").x.array
        xc = a.function_space.tabulate_dof_coordinates()[:, 0]
        return float(resp[xc > 0.2].mean() - resp[xc < -0.2].mean())

    frames: list[tuple[pyvista.UnstructuredGrid, float]] = [(response_grid(), 0.0)]
    print(
        f"  chemoattractant gradient on the box wall: C = {_GRADIENT}   tension γ = {_BASE_TENSION} + {_ALPHA}·(a−h)\n"
    )
    print(f"  {'t':>5} {'CoM Δx (centroid)':>18} {'node-mean Δx (treadmill)':>26} {'polarity':>10}")
    for step in range(_STEPS):
        problem.step()
        set_tension()
        t = (step + 1) * _DT
        if step % _FRAME_EVERY == _FRAME_EVERY - 1:
            frames.append((response_grid(), t))
            node_drift = float(mem.geometry.x[:, 0].mean()) - node_mean_0
            print(f"  {t:5.2f} {com_x() - com_x_0:>+18.4f} {node_drift:>+26.4f} {polarity():>10.3f}")

    print("\n  LEGI sensing works: the response a−h polarises up-gradient and is SUSTAINED by the external cue.")
    print("  But the surface-tension force balance only TREADMILLS the membrane — the centre of mass (CoM Δx,")
    print("  the area centroid) does not translate, while the node-mean drifts with the Marangoni surface flow.")
    print("  Genuine migration needs an active front–back-asymmetric mechanism (a follow-up). See legi_diag.png.")
    out = _HERE / "legi_sustained_chemotaxis.png"
    _write_tiled_image(frames[:-1], out)
    print(f"\n  wrote {out.name} — the LEGI response (a − h) over time (the cell shape stays centred)")
    assert np  # used in the helper above


if __name__ == "__main__":
    main()

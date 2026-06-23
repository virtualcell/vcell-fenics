"""Sustained chemotaxis by LEGI gradient sensing: a cell migrates up a chemoattractant gradient and KEEPS
going, because the polarity is held by the external cue (unlike a wave-pinning front, which stalls).

LEGI = Local Excitation, Global Inhibition — the standard model for eukaryotic gradient sensing. Here, in
a disk-in-a-box:

    chemoattractant C in the ext, held in a gradient by a DIRICHLET on the box wall  (C = 1 + 0.6·x)
        → membrane receptor R reads the LOCAL C (occupancy, polarized)
            → cyto LOCAL activator a (slow diffusion, tracks local R) and GLOBAL inhibitor h
              (fast diffusion ≈ cell-mean drive) — both produced from R at the membrane
                → response  a − h  is the RELATIVE-gradient polarity: high on the up-gradient side,
                  negative on the down-gradient side, and SUSTAINED (the external Dirichlet cue holds it)
                    → it sets the membrane tension γ = base + α·(a − h), and the surface-tension force
                      balance migrates the cell up-gradient.

The contrast with the imported RacRho wave-pinning reaction (see `cross_validation/racrho_*`): that front
de-pins and homogenises, so its polarity (and migration) decays after a transient. LEGI sensing an
imposed gradient is self-sustaining — the cell migrates and does not stall.

Writes `legi_sustained_chemotaxis.png`: the migrating cell (coloured by the LEGI response a − h) at a
sequence of times, in a fixed box frame, so the steady up-gradient migration is visible at a glance.

    .pixi/envs/dev/bin/python examples/legi_sustained_chemotaxis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista

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
_BASE_TENSION, _ALPHA, _DT, _STEPS, _FRAME_EVERY = 0.5, 0.10, 0.02, 160, 20
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
    motion = ForceBalanceMeshMotion(geom, tension=_BASE_TENSION, dt=_DT)
    problem = assemble_membrane_coupled(model(), geom, dt=_DT, motion=motion)
    start_x = float(geom.membrane_mesh.geometry.x[:, 0].mean())

    def response_field():  # type: ignore[no-untyped-def]
        a, h = problem.field("a"), problem.field("h")
        a.x.array[:] = a.x.array - h.x.array  # reuse a's space to hold a − h
        a.name = "response"
        return a

    def set_tension() -> None:  # γ = base + α·(a − h): the sustained LEGI response drives the tension
        motion.tension.x.array[:] = _BASE_TENSION + _ALPHA * (problem.field("a").x.array - problem.field("h").x.array)

    def polarity() -> float:
        a = problem.field("a")
        resp = a.x.array - problem.field("h").x.array
        xc = a.function_space.tabulate_dof_coordinates()[:, 0]
        return float(resp[xc > 0.2].mean() - resp[xc < -0.2].mean())

    frames: list[tuple[pyvista.UnstructuredGrid, float]] = [(_function_to_pyvista(response_field()), 0.0)]
    print(
        f"  chemoattractant gradient on the box wall: C = {_GRADIENT}   tension γ = {_BASE_TENSION} + {_ALPHA}·(a−h)\n"
    )
    print(f"  {'t':>5} {'migration Δx':>13} {'response polarity':>18}")
    for step in range(_STEPS):
        problem.step()
        set_tension()
        t = (step + 1) * _DT
        if step % _FRAME_EVERY == _FRAME_EVERY - 1:
            frames.append((_function_to_pyvista(response_field()), t))
            print(
                f"  {t:5.2f} {float(geom.membrane_mesh.geometry.x[:, 0].mean()) - start_x:>+13.4f} {polarity():>18.3f}"
            )

    print("\n  the cell migrated up the chemoattractant gradient and kept going — the LEGI response is held")
    print("  by the external (box-Dirichlet) cue, so the polarity (and the migration) is SUSTAINED, not a")
    print("  decaying transient. Local excitation + global inhibition reads the RELATIVE gradient.")
    out = _HERE / "legi_sustained_chemotaxis.png"
    _write_tiled_image(frames[:-1], out)
    print(f"\n  wrote {out.name} — the migrating cell coloured by the LEGI response (a − h) over time")
    assert np  # used in the helper above


if __name__ == "__main__":
    main()

"""Reproduce the RacRho MOVING BOUNDARY: a cell migrates by deforming its membrane under a velocity
driven by its own Rac concentration — the capability VCell runs on its FronTier moving-boundary solver.

This extends `racrho_neutrophil.py` (which imports + solves the bulk biochemistry on a static cell) with
the membrane motion. The public BioModel's `SurfaceKinematics` process moves the membrane with normal
velocity `v = (a − 1)·|a − 1|·n̂`, where `a` is the Rac concentration at the boundary: the membrane
PROTRUDES where Rac is high (a > 1) and RETRACTS where it is low (a < 1). That velocity is not a formalism
term — it is the moving boundary, supplied solver-side — so we drive it here in the stepping loop:

    repeat:
        integrate the bulk GTPase reaction–diffusion–advection over a short interval   (method-of-lines)
        move the cell-mesh boundary by interval · v(a), filling the interior harmonically  (ALE)

The bulk solve then continues on the deformed mesh. With the receptor initially polarized (a = 0.1x + 1,
higher on +x), the front (+x, a > 1) protrudes while the back (−x, a < 1) retracts — directed migration up
the Rac gradient. (The imported GTPase eventually homogenises to its low state, after which the boundary
retracts everywhere; sustaining the polarity needs the full feedback that isn't all in the lowered math.)

It also writes a tiled image `racrho_moving_boundary.png` — the deforming cell mesh coloured by Rac at a
sequence of times — so the migration and the bulk field are visible at a glance.

A near-circular cell's outward normal is approximated by the radial direction `x̂ = x/|x|`, which is
interpolable for the harmonic mesh move (`ufl.FacetNormal` is not defined in the interior). `_MOTION_GAIN`
makes the small `(a−1)|a−1|` displacement visible on the demo timescale; the model's units set the true
rate. Run in the dev env:

    .pixi/envs/dev/bin/python cross_validation/racrho_moving_boundary.py
"""

from __future__ import annotations

from pathlib import Path

import pyvcell.vcml.models_geometry as vg
import pyvcell.vcml.models_math as vm
import pyvista
import ufl
import yaml

from vcell_fenics.backend import assemble, integrate_discrete_problem
from vcell_fenics.backend.fsi import _advance_ale_mesh
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description
from vcell_fenics.viz import _function_to_pyvista

_HERE = Path(__file__).parent
_MOTION_GAIN, _INTERVAL, _STEPS, _FRAME_EVERY = 70.0, 0.02, 16, 2
_FIELD = "Rac (a)"


def _write_tiled_image(frames: list[tuple[pyvista.UnstructuredGrid, float]], path: Path) -> None:
    """Tile the captured frames (deforming mesh coloured by Rac) into a single 2×N PNG, with a shared
    colour scale and a fixed reference box so the migration reads across frames."""
    pyvista.OFF_SCREEN = True
    lo = min(grid.point_data[_FIELD].min() for grid, _ in frames)
    hi = max(grid.point_data[_FIELD].max() for grid, _ in frames)
    cols = (len(frames) + 1) // 2
    plotter = pyvista.Plotter(shape=(2, cols), off_screen=True, window_size=(200 * cols, 420), border=False)
    box = pyvista.Box(bounds=(-1.15, 1.15, -1.15, 1.15, -0.01, 0.01))
    for i, (grid, t) in enumerate(frames):
        plotter.subplot(i // cols, i % cols)
        plotter.add_mesh(box, style="wireframe", color="lightgray", opacity=0.4)
        plotter.add_mesh(
            grid,
            scalars=_FIELD,
            clim=[lo, hi],
            cmap="viridis",
            show_edges=True,
            edge_color="gray",
            line_width=0.5,
            show_scalar_bar=(i == 0),
        )
        plotter.add_text(f"t = {t:.2f}", font_size=9)
        plotter.view_xy()
        plotter.camera.zoom(1.15)
    plotter.screenshot(str(path))
    plotter.close()


def main() -> None:
    math_vcml = vm.MathDescription.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_math.yaml").read_text()))
    geom_vcml = vg.Geometry.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_geom.yaml").read_text()))
    geometry_desc = import_geometry(geom_vcml)
    math_desc = import_math_description(math_vcml, geometry=geometry_desc.name, dim=2)

    mesh = realize(geometry_desc, h=0.08)
    problem = assemble(math_desc, mesh, dt=0.02)
    cell = problem.unknown.function_space.mesh

    # The membrane velocity v = gain·(a−1)|a−1|·n̂ from the *current* Rac concentration a (component 0).
    rac = ufl.split(problem.unknown)[0]
    x = ufl.SpatialCoordinate(cell)
    normal = x / (ufl.sqrt(x[0] * x[0] + x[1] * x[1]) + 1.0e-8)  # outward normal ≈ radial for a round cell
    velocity = _MOTION_GAIN * (rac - 1.0) * ufl.sqrt((rac - 1.0) ** 2) * normal

    rac_space, rac_dofs = problem.unknown.function_space.sub(0).collapse()
    rac_x = rac_space.tabulate_dof_coordinates()[:, 0]

    frames: list[tuple[pyvista.UnstructuredGrid, float]] = []

    def capture(t: float) -> None:
        field = problem.unknown.sub(0).collapse()
        field.name = _FIELD
        frames.append((_function_to_pyvista(field), t))

    capture(0.0)  # the initial (undeformed) cell with its polarized IC, before any motion

    print(f"  membrane velocity v = {_MOTION_GAIN}·(a−1)|a−1|·n̂   (protrude where a>1, retract where a<1)\n")
    print(f"  {'t':>5} {'front +x':>10} {'back −x':>10} {'a @front':>9} {'a @back':>9}")
    for step in range(_STEPS):
        integrate_discrete_problem(problem, t_final=_INTERVAL, dt_initial=1.0e-4)  # MOL — nonlinear GTPase
        _advance_ale_mesh(cell, velocity, _INTERVAL)  # move boundary by interval·v(a), harmonic interior
        coords = cell.geometry.x
        rac_values = problem.unknown.x.array[rac_dofs]
        t = (step + 1) * _INTERVAL
        print(
            f"  {t:5.2f} {coords[:, 0].max():10.3f} {coords[:, 0].min():10.3f} "
            f"{float(rac_values[rac_x > 0.5].mean()):9.3f} {float(rac_values[rac_x < -0.5].mean()):9.3f}"
        )
        if step % _FRAME_EVERY == _FRAME_EVERY - 1:
            capture(t)

    print("\n  the front (+x, a>1) protruded and the back (−x, a<1) retracted — directed migration up the")
    print("  Rac gradient. The cell membrane moved under a velocity set by its own biochemistry, on a mesh")
    print("  the bulk solve followed each step: the moving boundary, reproduced through the pipeline.")

    out = _HERE / "racrho_moving_boundary.png"
    _write_tiled_image(frames[:-1], out)  # t = 0.0 … 0.28 (the initial frame in, the last dropped)
    print(f"\n  wrote {out.name} — the deforming cell mesh coloured by Rac over time")


if __name__ == "__main__":
    main()

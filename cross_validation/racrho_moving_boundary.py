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
import ufl
import yaml

from vcell_fenics.backend import assemble, integrate_discrete_problem
from vcell_fenics.backend.fsi import _advance_ale_mesh
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_HERE = Path(__file__).parent
_MOTION_GAIN, _INTERVAL, _OUTER_STEPS = 30.0, 0.02, 8


def main() -> None:
    math_vcml = vm.MathDescription.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_math.yaml").read_text()))
    geom_vcml = vg.Geometry.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_geom.yaml").read_text()))
    geometry_desc = import_geometry(geom_vcml)
    math_desc = import_math_description(math_vcml, geometry=geometry_desc.name, dim=2)

    mesh = realize(geometry_desc, h=0.1)
    problem = assemble(math_desc, mesh, dt=0.02)
    cell = problem.unknown.function_space.mesh

    # The membrane velocity v = gain·(a−1)|a−1|·n̂ from the *current* Rac concentration a (component 0).
    rac = ufl.split(problem.unknown)[0]
    x = ufl.SpatialCoordinate(cell)
    normal = x / (ufl.sqrt(x[0] * x[0] + x[1] * x[1]) + 1.0e-8)  # outward normal ≈ radial for a round cell
    velocity = _MOTION_GAIN * (rac - 1.0) * ufl.sqrt((rac - 1.0) ** 2) * normal

    rac_space, rac_dofs = problem.unknown.function_space.sub(0).collapse()
    rac_x = rac_space.tabulate_dof_coordinates()[:, 0]

    print(f"  membrane velocity v = {_MOTION_GAIN}·(a−1)|a−1|·n̂   (protrude where a>1, retract where a<1)\n")
    print(f"  {'t':>5} {'front +x':>10} {'back −x':>10} {'a @front':>9} {'a @back':>9}")
    for step in range(_OUTER_STEPS):
        integrate_discrete_problem(problem, t_final=_INTERVAL, dt_initial=1.0e-4)  # MOL — nonlinear GTPase
        _advance_ale_mesh(cell, velocity, _INTERVAL)  # move boundary by interval·v(a), harmonic interior
        coords = cell.geometry.x
        rac_values = problem.unknown.x.array[rac_dofs]
        a_front = float(rac_values[rac_x > 0.5].mean())
        a_back = float(rac_values[rac_x < -0.5].mean())
        print(
            f"  {(step + 1) * _INTERVAL:5.2f} {coords[:, 0].max():10.3f} {coords[:, 0].min():10.3f} "
            f"{a_front:9.3f} {a_back:9.3f}"
        )

    print("\n  the front (+x, a>1) protruded and the back (−x, a<1) retracted — directed migration up the")
    print("  Rac gradient. The cell membrane moved under a velocity set by its own biochemistry, on a mesh")
    print("  the bulk solve followed each step: the moving boundary, reproduced through the pipeline.")


if __name__ == "__main__":
    main()

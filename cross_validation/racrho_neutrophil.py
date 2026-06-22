"""Import and solve a REAL public VCell moving-boundary BioModel through the formalism pipeline.

Model: `RhoRac_BioModel_MovingBoundary`, application "ChasingNeutrophil" (public BioModel 165035197) — a
2D Rac/Rho GTPase cell-polarity model that VCell runs on its FronTier moving-boundary solver (`MovingB`).
It is one of 18 public models with kinematics on BOTH a bulk subdomain and a membrane (a prescribed bulk
`<Velocity>` plus a membrane `SurfaceKinematics` process). Its bulk biochemistry is

    ∂a/∂t = D_a ∇²a + (0.067 + a²/(1+a²))·b − a   +   v_chase·∇a
    ∂b/∂t = D_b ∇²b − (...)                        +   v_chase·∇b

a bistable GTPase switch (the `a²/(1+a²)` Hill term) advected by `v_chase`, a velocity field pointing at a
chemoattractant point that moves at 0.6 µm/s — the cell "chasing the neutrophil".

What this demonstrates and what it does NOT:
  - The bulk reaction–diffusion–ADVECTION imports cleanly into the formalism (the `<Velocity>` lowers to
    the `relative_advection` slot; `^` → `**`; the moving chase-point becomes a time-dependent velocity).
  - It SOLVES end-to-end via the method-of-lines integrator. Two backend changes made this possible: the
    bulk `relative_advection` term, and the LAZY backward-Euler composition (the rational `a²/(1+a²)`
    source makes `ufl.lhs` raise, so eagerly composing the BE split used to crash assembly even for MOL).
  - The MEMBRANE motion (the `SurfaceKinematics` velocity `(a−1)|a−1|·n̂`) is NOT a formalism term — it is
    the moving boundary, supplied solver-side. Run here on a STATIC cell, the GTPase settles to its
    steady state without the migration; reproducing the full "chase" means driving the cell mesh with that
    velocity via the ALE / force-balance motion drivers (the moving-membrane work in `interface_coupled`).

The model also exercises exactly the two axes VCell's mbsolver cannot: it leaves the external bulk (`extr`)
empty and the membrane stateless — both of which this platform supports as a superset.

The parsed math/geometry YAML (`racrho_neutrophil_{math,geom}.yaml`) is committed so this runs in the dev
env; regenerate from the raw VCML with `scripts/parse_biomodels_to_yaml.py` in the full pyvcell env.

    .pixi/envs/dev/bin/python cross_validation/racrho_neutrophil.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from mpi4py import MPI

import pyvcell.vcml.models_geometry as vg
import pyvcell.vcml.models_math as vm
from vcell_fenics.backend import assemble, integrate_discrete_problem
from vcell_fenics.backend.realize import realize
from vcell_fenics.pyvcell_bridge import import_geometry, import_math_description

_HERE = Path(__file__).parent


def main() -> None:
    math_vcml = vm.MathDescription.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_math.yaml").read_text()))
    geom_vcml = vg.Geometry.model_validate(yaml.safe_load((_HERE / "racrho_neutrophil_geom.yaml").read_text()))

    geometry_desc = import_geometry(geom_vcml)
    math_desc = import_math_description(math_vcml, geometry=geometry_desc.name, dim=2)
    print(f"  imported public BioModel 165035197 'ChasingNeutrophil'  ({geometry_desc.dim}D)")
    print(
        f"  subvolumes: {[s.name for s in geometry_desc.subvolumes]}   species: {[v.name for v in math_desc.variables]}"
    )
    for eq in math_desc.equations:
        print(f"    {eq.variable} on {eq.subdomain}: terms = {list(eq.terms)}")

    mesh = realize(geometry_desc, h=0.12)
    problem = assemble(math_desc, mesh, dt=0.05)
    result = integrate_discrete_problem(problem, t_final=2.0, dt_initial=1.0e-4)
    print(f"\n  solved the GTPase reaction–diffusion–advection to t = {result.time:.1f} in {result.steps} steps")

    unknown = problem.unknown
    cell_mesh = unknown.function_space.mesh
    for k, name in enumerate(v.name for v in math_desc.variables):
        _, dofs = unknown.function_space.sub(k).collapse()
        values = unknown.x.array[dofs]
        lo = float(cell_mesh.comm.allreduce(values.min(), op=MPI.MIN))
        hi = float(cell_mesh.comm.allreduce(values.max(), op=MPI.MAX))
        print(f"    {name}: range [{lo:.3f}, {hi:.3f}]")

    print("\n  the real public moving-boundary model's biochemistry imports and solves through the pipeline.")
    print("  the membrane SurfaceKinematics velocity (the moving boundary) is supplied solver-side — drive")
    print("  the cell mesh with it via the ALE / force-balance motion to reproduce the full chase.")


if __name__ == "__main__":
    main()

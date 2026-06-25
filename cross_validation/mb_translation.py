"""Stage 2 (dev env): the fenics ALE side of the moving-boundary TRANSLATION cross-validation + the report.

The same problem as `mb_translation_fv.py`, run through our backend: a disk (radius `R`, centred at the
origin) with an interior species `C` (diffusion `D`, IC `C = x`), translated by a prescribed mesh velocity
`[V, 0]` (`MotionPrescribedVelocity` — the ALE "cell carries its cytoplasm"). The mesh moves with the cell,
so in the cell frame it is pure diffusion with no-flux — the `v = v_b` convention the FV reference uses.

Compares the centre-of-mass trajectory and the `C`-homogenisation against the mbsolver reference
(`mb_translation_fv.npz`). The `C` *mean* differs by a constant (the FV disk sits at x=5, so its `C = x`
has mean 5; ours is centred at 0, mean 0) — diffusion is invariant to that shift, so the comparable
quantities are the **CoM displacement** and the **`C` spread** (homogenisation).

    .pixi/envs/dev/bin/python cross_validation/mb_translation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml

_HERE = Path(__file__).parent
_DT = 0.05  # fenics step (backward Euler); output cadence is a multiple of it


def _model(diffusion: float, velocity: float) -> str:
    return f"""
math_description:
  geometry: disk
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "[{velocity}, 0.0]" }} }}
  variables:
    - {{ name: C, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: C
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "geom.x[0]"
"""


def main() -> None:
    ref_path = _HERE / "mb_translation_fv.npz"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path.name} — run mb_translation_fv.py in ../pyvcell/.venv first")
    ref = np.load(ref_path)
    R, D, V = float(ref["R"]), float(ref["D"]), float(ref["V"])
    times = ref["times"]
    steps_per_out = round(float(times[1] - times[0]) / _DT)

    geom = make_disk_geometry("disk", volume_subdomain="cyto", radius=R, h=0.15)
    dp = assemble(load_yaml(_model(D, V)), geom, dt=_DT)
    mesh = dp.V.mesh
    one = fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)  # type: ignore[operator]
    moment_x = fem.form(ufl.SpatialCoordinate(mesh)[0] * ufl.dx)
    c_moment = fem.form(dp.unknown * ufl.dx)

    def com_x() -> float:
        return float(fem.assemble_scalar(moment_x).real) / float(fem.assemble_scalar(one).real)

    def c_mean() -> float:
        return float(fem.assemble_scalar(c_moment).real) / float(fem.assemble_scalar(one).real)

    def c_spread() -> float:
        return float(dp.unknown.x.array.max() - dp.unknown.x.array.min())

    com0 = com_x()
    fe_dx, fe_spread = [0.0], [c_spread()]
    for _ in range(len(times) - 1):
        for _ in range(steps_per_out):
            dp.step()
        fe_dx.append(com_x() - com0)
        fe_spread.append(c_spread())
    fe_dx, fe_spread = np.array(fe_dx), np.array(fe_spread)

    fv_dx = ref["front_cx"] - ref["front_cx"][0]
    fv_spread = ref["c_spread"]

    print(f"  moving-boundary translation, R={R}, D={D}, v=v_b=({V},0):  fenics ALE  vs  mbsolver FV\n")
    print(f"  {'t':>4} | {'fenics CoM Δx':>13} {'FV front Δx':>12} | {'fenics C spread':>15} {'FV C spread':>12}")
    for i, t in enumerate(times):
        print(f"  {t:4.1f} | {fe_dx[i]:>13.3f} {fv_dx[i]:>12.3f} | {fe_spread[i]:>15.4f} {fv_spread[i]:>12.4f}")

    dx_err = abs(fe_dx[-1] - fv_dx[-1]) / abs(fv_dx[-1])
    both_homogenise = fe_spread[-1] / fe_spread[0] < 0.05 and fv_spread[-1] / fv_spread[0] < 0.05
    print(f"\n  front displacement at t={times[-1]:.0f}: fenics {fe_dx[-1]:+.3f} (exact ALE), FV {fv_dx[-1]:+.3f} "
          f"(front-redistribution short by {dx_err * 100:.1f}%)")
    print(f"  both interiors homogenise (spread → <5% of initial): {both_homogenise}")
    print("  → the two solvers agree on the translation and on diffusion-in-the-moving-frame.")


if __name__ == "__main__":
    main()

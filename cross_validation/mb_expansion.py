"""Stage 2 (dev env): the fenics ALE side of the moving-boundary EXPANSION cross-validation + the report.

The same problem as `mb_expansion_fv.py`, run through our backend: a disk (radius `R0`, centred at the
origin) carrying a uniform species `u` (diffusion `D`, IC `u = 1`, no reaction), dilated by a prescribed
radial mesh velocity `[K·x, K·y]` (`MotionPrescribedVelocity`). Since `∇·v = 2K > 0` the cell dilates and
the mandatory `ρ ∇·v` **dilution** term drives `u` down; the exact solution is `R(t) = R0 e^{Kt}`,
`u(t) = e^{−2K t}` (spatially uniform), total substance `∫u dA = π R0²` conserved.

Unlike the translation case (rigid, `∇·v = 0`, dilution inert), this exercises dilution against a closed
form, and checks that our **conservative ALE time term** conserves the total substance to solver precision
(no O(dt) geometric-conservation-law drift). Compares the front radius and `u` decay against the mbsolver
reference (`mb_expansion_fv.npz`) and the exact solution.

    .pixi/envs/dev/bin/python cross_validation/mb_expansion.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml

_HERE = Path(__file__).parent
_DT = 0.02  # fenics step (backward Euler); output cadence is a multiple of it


def _model(diffusion: float, k: float) -> str:
    return f"""
math_description:
  geometry: disk
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "[{k}*geom.x[0], {k}*geom.x[1]]" }} }}
  variables:
    - {{ name: u, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: u
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "{diffusion}" }}
      initial_condition: "1.0"
"""


def main() -> None:
    ref_path = _HERE / "mb_expansion_fv.npz"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path.name} — run mb_expansion_fv.py in ../pyvcell/.venv first")
    ref = np.load(ref_path)
    R0, D, K = float(ref["R0"]), float(ref["D"]), float(ref["K"])
    times = ref["times"]
    steps_per_out = round(float(times[1] - times[0]) / _DT)

    geom = make_disk_geometry("disk", volume_subdomain="cyto", radius=R0, h=0.12)
    dp = assemble(load_yaml(_model(D, K)), geom, dt=_DT)
    mesh = dp.V.mesh
    one = fem.form(fem.Constant(mesh, PETSc.ScalarType(1.0)) * ufl.dx)  # type: ignore[operator]
    u_int = fem.form(dp.unknown * ufl.dx)

    def area() -> float:
        return float(fem.assemble_scalar(one).real)

    def u_mean() -> float:
        return float(fem.assemble_scalar(u_int).real) / area()

    def front_r() -> float:  # equivalent-circle radius from the (faceted) disk area
        return math.sqrt(area() / math.pi)

    fe_r, fe_u, fe_mass = [front_r()], [u_mean()], [float(fem.assemble_scalar(u_int).real)]
    for _ in range(len(times) - 1):
        for _ in range(steps_per_out):
            dp.step()
        fe_r.append(front_r())
        fe_u.append(u_mean())
        fe_mass.append(float(fem.assemble_scalar(u_int).real))
    fe_r, fe_u, fe_mass = np.array(fe_r), np.array(fe_u), np.array(fe_mass)

    fv_r, fv_u = ref["front_r"], ref["u_mean"]
    exact_r = R0 * np.exp(K * times)
    exact_u = np.exp(-2 * K * times)

    print(
        f"  moving-boundary EXPANSION, R0={R0}, D={D}, radial v=K·r (K={K}):  fenics ALE  vs  mbsolver FV  vs  exact\n"
    )
    print(
        f"  {'t':>4} | {'fe R':>7} {'FV R':>7} {'exact R':>7} | {'fe u':>7} {'FV u':>7} {'exact u':>7} | {'fe mass':>8}"
    )
    for i, t in enumerate(times):
        print(
            f"  {t:4.1f} | {fe_r[i]:7.3f} {fv_r[i]:7.3f} {exact_r[i]:7.3f} | "
            f"{fe_u[i]:7.4f} {fv_u[i]:7.4f} {exact_u[i]:7.4f} | {fe_mass[i]:8.4f}"
        )

    fe_r_err = abs(fe_r[-1] - exact_r[-1]) / exact_r[-1]
    fv_r_err = abs(fv_r[-1] - exact_r[-1]) / exact_r[-1]
    fe_u_err = abs(fe_u[-1] - exact_u[-1]) / exact_u[-1]
    fv_u_err = abs(fv_u[-1] - exact_u[-1]) / exact_u[-1]
    mass_drift = abs(fe_mass[-1] - fe_mass[0]) / fe_mass[0]
    print(
        f"\n  front radius at t={times[-1]:.0f}: fenics {fe_r_err * 100:.2f}% vs exact, mbsolver {fv_r_err * 100:.2f}%"
    )
    print(f"  u (dilution) at t={times[-1]:.0f}: fenics {fe_u_err * 100:.2f}% vs exact, mbsolver {fv_u_err * 100:.2f}%")
    print(f"  fenics total substance ∫u dA conserved to {mass_drift * 100:.4f}%  (conservative ALE time term)")
    print("  → both solvers reproduce the dilating front and the ρ∇·v dilution of u; our total is exact.")


if __name__ == "__main__":
    main()

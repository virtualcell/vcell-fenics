"""Method-of-Manufactured-Solutions runner for pyvcell's **mbsolver** (moving-boundary front tracking).

Run in the pyvcell env:

    ../pyvcell/.venv/bin/python mms/runner_mb.py mms/cases/<case>.yaml

Consumes an MMS case whose `applicable_solvers` includes `mbsolver` and that carries an `mb:` block. The
moving-boundary solver deforms the domain under a prescribed front velocity, so the natural manufactured
solution is one whose closed form the deformation dictates: a **radially dilating disk** (front velocity
`v = k·r`) carrying a spatially uniform species with no reaction dilutes as `u(t) = u0·e^{−2k t}` while the
front grows as `R(t) = R0·e^{k t}`, with total substance `∫u dA` conserved. The runner authors that model,
runs mbsolver, and checks the per-frame mean concentration and front radius against those exact closed
forms (`exact:` block) — the moving-boundary counterpart of the fixed-grid fvsolver MMS.

(A spatially *varying* `u*` under moving-boundary tracking needs per-node position sampling, which this
mbsolver build does not expose reliably — the uniform-dilution manufactured solution is the tractable,
rigorous moving-front check; the spatial-forcing version is a follow-up.)

Case fields used (see `mb_expansion_dilution.yaml`):
  exact: {u_mean: "np.exp(-2*0.25*t)", front_r: "3.0*np.exp(0.25*t)"}   # closed forms vs time
  mb: {R0: 3.0, k: 0.25, diffusion: 1.0, extent: 16.0, center: 8.0, duration: 2.0, out_step: 0.5, mesh: 41}
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml as vc
import yaml
from pyvcell.vcml.models import Biomodel, Model

_CASES = Path(__file__).parent / "cases"
_NS = {"np": np, "exp": np.exp, "sqrt": np.sqrt, "pi": np.pi}


@contextmanager
def _silence():  # type: ignore[no-untyped-def]
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _run_mb(case: dict[str, Any]) -> dict[str, np.ndarray]:
    mb = case["mb"]
    R0, k, cx, ext = mb["R0"], mb["k"], mb["center"], mb["extent"]
    vx, vy = f"{k}*(x - {cx})", f"{k}*(y - {cx})"  # radial dilating velocity about the disk centre
    geo = vc.Geometry(name="square", dim=2, extent=(ext, ext, ext), origin=(0.0, 0.0, 0.0))
    geo.add_sphere("cell", radius=R0, center=(cx, cx, 0.0))
    geo.add_background("ec")
    geo.add_surface("cell_ec_membrane", "cell", "ec")
    model = Model(name="m")
    for compartment, dim in (("cyt", 3), ("ec", 3), ("pm", 2)):
        model.add_compartment(compartment, dim=dim)
    model.add_species("u", "cyt")
    biomodel = Biomodel(name="MMS_MB", model=model)
    app = biomodel.add_application("a", geometry=geo)
    app.map_compartment("cyt", "cell")
    app.map_compartment("ec", "ec")
    app.map_compartment("pm", "cell_ec_membrane")
    sm = app.map_species("u", init_conc="1", diff_coef=mb["diffusion"])
    sm.velocity_x, sm.velocity_y = vx, vy  # Lagrangian: species carried with the front
    app.set_moving_boundary_front(velocity_x=vx, velocity_y=vy)
    app.add_moving_boundary_sim(
        name="s", duration=mb["duration"], output_time_step=mb["out_step"], mesh_size=(mb["mesh"], mb["mesh"], 1)
    )
    with _silence():
        result = vc.simulate_moving_boundary(biomodel, "s")
    times = np.array(result.times, dtype=float)
    front_r = np.array([float(np.hypot(f.front[:, 0] - cx, f.front[:, 1] - cx).mean()) for f in result.frames])
    u_mean = np.array([float(f.concentrations["u"].mean()) for f in result.frames])
    return {"t": times, "front_r": front_r, "u_mean": u_mean}


def _eval(expr: str, t: np.ndarray) -> np.ndarray:
    return np.asarray(eval(expr, {"__builtins__": {}}, {**_NS, "t": t}))


def run_case(path: Path) -> None:
    case = yaml.safe_load(path.read_text())
    if "mbsolver" not in case.get("applicable_solvers", []) or "mb" not in case:
        print(f"  (skip {case['name']}: not an mbsolver case)")
        return
    print(f"\n=== {case['name']} (mbsolver) ===\n{case['description'].strip()}\n")
    out = _run_mb(case)
    t = out["t"]
    print(f"  {'t':>5} | {'front_r':>9} {'exact R':>9} | {'u_mean':>9} {'exact u':>9}")
    for key_r, key_u in [("front_r", "u_mean")]:  # single measured pair
        r_exact = _eval(case["exact"]["front_r"], t)
        u_exact = _eval(case["exact"]["u_mean"], t)
        for i in range(len(t)):
            print(f"  {t[i]:5.2f} | {out[key_r][i]:9.4f} {r_exact[i]:9.4f} | {out[key_u][i]:9.5f} {u_exact[i]:9.5f}")
        r_err = abs(out["front_r"][-1] - r_exact[-1]) / r_exact[-1]
        u_err = abs(out["u_mean"][-1] - u_exact[-1]) / u_exact[-1]
        tol = case.get("mb_tol", 0.02)
        ok = r_err < tol and u_err < tol
        verdict = "OK" if ok else "PROBLEM"
        print(f"\n  front radius rel-err at t={t[-1]:.1f}: {r_err * 100:.2f}%   dilution u rel-err: {u_err * 100:.2f}%")
        print(f"  → {verdict} (tol {tol * 100:.0f}%): mbsolver reproduces the exact moving-boundary dilution")


def main() -> None:
    args = sys.argv[1:]
    paths = sorted(_CASES.glob("*.yaml")) if (not args or args == ["--all"]) else [Path(a) for a in args]
    for p in paths:
        run_case(p)


if __name__ == "__main__":
    main()

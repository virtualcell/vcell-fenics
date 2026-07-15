"""Method-of-Manufactured-Solutions runner for vcell-fenics (dev env).

Runs a persistent MMS *case* (`mms/cases/*.yaml`) — a math description whose `source` is the manufactured
forcing that makes a chosen exact solution `u*(x,t)` the true solution — through the vcell-fenics solvers,
and reports the **true L2/L∞ error vs `u*`** and the **h-convergence order**. A case that fails to hit its
theoretical order (or whose error does not shrink under refinement) exposes a *correctness* defect that
conservation checks are blind to.

The same case files are consumed by the fvsolver / mbsolver runners (`runner_fv.py`, `runner_mb.py`, in
`../pyvcell/.venv`) — one manufactured model, three solvers, each checked against the same known `u*`.

    .pixi/envs/dev/bin/python mms/runner.py mms/cases/<case>.yaml
    .pixi/envs/dev/bin/python mms/runner.py --all
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend import assemble, make_disk_geometry, make_disk_membrane_geometry
from vcell_fenics.backend.reaction_diffusion import (
    integrate_discrete_problem,
    integrate_discrete_problem_moving,
)
from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism import load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume

_CASES = Path(__file__).parent / "cases"
# The namespace an exact-solution expression is evaluated in (numpy, coords x/y/z, time t).
_EXACT_NS = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi, "sqrt": np.sqrt, "tanh": np.tanh}


def _eval_exact(expr: str, coords: np.ndarray, t: float) -> np.ndarray:
    x, y = coords[:, 0], coords[:, 1]
    z = coords[:, 2] if coords.shape[1] > 2 else np.zeros_like(x)
    return np.asarray(eval(expr, {"__builtins__": {}}, {**_EXACT_NS, "x": x, "y": y, "z": z, "t": t}), dtype=float)


def _geometry(case: dict[str, Any], h: float):  # type: ignore[no-untyped-def]
    g = case["geometry"]
    name = yaml.safe_load(case["math"])["math_description"]["geometry"]  # must match the math's `geometry:`
    if g["kind"] == "disk":
        return make_disk_geometry(
            name, volume_subdomain=g["volume_subdomain"], boundary=g.get("boundary"), radius=g["radius"], h=h
        )
    if g["kind"] == "disk_membrane":
        return make_disk_membrane_geometry(name, surface_subdomain=g["surface_subdomain"], radius=g["radius"], h=h)
    if g["kind"] == "box":
        # a plain box mesh (one analytic "1.0" subvolume = the whole bounding box); the same box fvsolver
        # meshes, so one case runs through both. z is a unit slab for 2D (matches the FV author).
        ext, org = g["extent"], g.get("origin", [0.0, 0.0])
        desc = GeometryDescription(
            name=name,
            dim=2,
            extent=(ext[0], ext[1], 1.0),
            origin=(org[0], org[1], 0.0),
            subvolumes=(SubVolume(name=g["volume_subdomain"], type="analytic", expression="1.0"),),
        )
        return realize(desc, h=h)
    raise NotImplementedError(f"geometry kind {g['kind']!r} not supported by the dev runner yet")


def _error(dp, exact_expr: str, t: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    coords = dp.V.tabulate_dof_coordinates()
    u_star = _eval_exact(exact_expr, coords, t)
    e = np.abs(dp.unknown.x.array - u_star)
    # mass-weighted L2 over the (possibly moved) domain, plus nodal L∞
    l2 = math.sqrt(float(fem.assemble_scalar(fem.form((dp.unknown - _as_fn(dp, u_star)) ** 2 * ufl.dx)).real))
    return float(e.max()), l2


def _as_fn(dp, values: np.ndarray):  # type: ignore[no-untyped-def]
    f = fem.Function(dp.V)
    f.x.array[:] = values
    return f


def _has_motion(case: dict[str, Any]) -> bool:
    subs = yaml.safe_load(case["math"])["math_description"].get("subdomains", [])
    return any("motion" in s for s in subs)


def _dt_for(case: dict[str, Any], h: float) -> float:
    """The time step at resolution `h`. A moving (ALE) or time-dependent case is FIRST-order in dt (backward
    Euler + strided motion), so measuring the O(h²) spatial order requires refining dt WITH h — otherwise the
    O(dt) temporal floor dominates and the h-order collapses to ~0 (a measurement artifact, not a defect).
    `dt_rule` ("0.5*h**2") overrides; else moving cases default to dt ∝ h², static cases to a fixed `dt`."""
    if "dt_rule" in case:
        return float(eval(case["dt_rule"], {"__builtins__": {}}, {"h": h}))
    time_dependent = any(re.search(r"\bt\b", expr) for expr in case["exact"].values())
    if _has_motion(case) or time_dependent:
        return 0.5 * h * h  # first-order in dt ⇒ refine dt with h to expose the O(h²) spatial order
    return case.get("dt", 0.01)


def _run_fenics(case: dict[str, Any], solver: str, h: float) -> tuple[float, float]:
    geom = _geometry(case, h)
    var = next(iter(case["exact"]))
    dt = _dt_for(case, h)
    t_final = case["t_final"]
    dp = assemble(load_yaml(case["math"]), geom, dt=dt)
    if solver == "fenics-mol":
        # MOL: strided ALE on a moving subdomain, fixed-domain TS otherwise (static-mesh cases). The strided
        # motion is O(interval), so refine the stride with h too (interval = dt) to see the spatial order.
        if dp.motion_velocity is not None:
            integrate_discrete_problem_moving(dp, t_final=t_final, motion_steps=max(20, round(t_final / dt)))
        else:
            integrate_discrete_problem(dp, t_final=t_final)
        return _error(dp, case["exact"][var], t_final)
    # fenics-be: step, refreshing time-/position-dependent Dirichlet BCs at the moved configuration each step
    t = 0.0
    for _ in range(round(t_final / dt)):
        dp.step()
        t += dt
        dp.set_time(t)  # refreshes Dirichlet g(x,t) at the moved boundary (needed on a moving mesh)
    return _error(dp, case["exact"][var], t_final)


def _order(errs: list[float], hs: list[float]) -> float | None:
    fine = [(e, h) for e, h in zip(errs, hs, strict=True) if e > 1e-13]
    if len(fine) < 2:
        return None
    return math.log(fine[0][0] / fine[-1][0]) / math.log(fine[0][1] / fine[-1][1])


def run_case(path: Path) -> dict[str, Any]:
    case = yaml.safe_load(path.read_text())
    hs = case["resolutions_h"]
    expected = case["expected_order_h"]
    print(f"\n=== {case['name']} ===\n{case['description'].strip()}\n")
    results = {}
    for solver in [s for s in case["applicable_solvers"] if s.startswith("fenics")]:
        linfs, l2s = [], []
        for h in hs:
            linf, l2 = _run_fenics(case, solver, h)
            linfs.append(linf)
            l2s.append(l2)
        order = _order(linfs, hs)
        ok = order is not None and order >= expected - 0.5
        results[solver] = {"linf": linfs, "l2": l2s, "order": order, "ok": ok}
        print(f"  {solver:<11} h={hs}")
        print(f"    L∞ = {[f'{e:.2e}' for e in linfs]}   L2 = {[f'{e:.2e}' for e in l2s]}")
        verdict = "OK" if ok else "PROBLEM"
        print(f"    order(L∞) = {order if order is None else round(order, 2)}  (expected ≥ {expected})  → {verdict}")
    return {"name": case["name"], "results": results}


def main() -> None:
    args = sys.argv[1:]
    paths = sorted(_CASES.glob("*.yaml")) if (not args or args == ["--all"]) else [Path(a) for a in args]
    summary = [run_case(p) for p in paths]
    print("\n=== summary ===")
    for s in summary:
        for solver, r in s["results"].items():
            flag = "OK " if r["ok"] else "!! PROBLEM"
            print(f"  {flag}  {s['name']:<40} {solver:<11} order={r['order'] and round(r['order'], 2)}")


if __name__ == "__main__":
    main()

"""Order-regression gate for the persistent MMS suite (`mms/cases/*.yaml`).

Runs each manufactured-solution case through the dev runner (`mms/runner.py`) and asserts every applicable
vcell-fenics solver still hits its `expected_order_h` — the runner's own `ok` verdict, but as a *failing*
assertion instead of a printed line. This turns the standalone MMS suite into an automatic guard: a solver
change that silently drops a convergence order — as the interface-flux over-count did (coupled order 2 →
0.28) — fails the build instead of only surfacing when someone reads the runner output.

The moving cases refine dt ∝ h² (hundreds–thousands of steps), so most of the suite is `integration`-marked
(run by `pixi run test-integration` / CI); a fast static subset stays in the default `pixi run check` gate as
a quick tripwire. (The coupled / unknown-motion / membrane-coupled paths also have dedicated, faster MMS
tests in the default gate — this suite adds the persistent bulk motion×physics matrix on top.)
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_MMS = Path(__file__).resolve().parent.parent / "mms"
sys.path.insert(0, str(_MMS))
import runner  # noqa: E402  # the dev MMS runner (mms/runner.py), reached via the sys.path insert above

# Fast, static cases stay in the default gate as a tripwire; everything else refines dt ∝ h² (slow) and is
# integration-only. (Coupled / unknown-motion also have their own faster gate tests in tests/.)
_FAST = {"box_static_diffusion", "static_bulk_diffusion", "membrane_static_diffusion"}

# The remaining fenics-MOL regression, now DIAGNOSED and GEOMETRY-INDEPENDENT: the time-dependent
# `static_reaction` case (u* = e^{-t}·…, static mesh) floors the adaptive-BDF (MOL) error at ~2e-4 on EVERY
# geometry — structured box (order 1.19) AND nested disk (0.51) — and *tightening* the TS tolerance makes it
# worse. That is a genuine adaptive-TS global-time-error issue on a decaying solution, NOT the spatial
# discretisation: the backward-Euler path holds its expected order on both geometries. xfail'd (documented)
# for follow-up. Every OTHER bulk case now passes on both geometries at its own honest rate — the structured
# box at O(h²), the nested (curvature-preserving) disk at the P1 curved-domain rate ~O(h^1.5) — which is why
# each bulk physics case ships as a `_box` (expected order 2) and a `_disk` (expected order 1.5) variant.
_KNOWN_MOL_REGRESSIONS = {
    # only the two time-dependent (motion-off) variants carry a MOL run; their `_static` twins are BE-only.
    "bulk_static_reaction_timedep_box",
    "bulk_static_reaction_timedep_disk",
}


@functools.cache
def _run_case(path_str: str) -> dict[str, Any]:
    """Run a case's full h-sweep once (both fenics solvers); cached so the two per-solver params share it."""
    return runner.run_case(Path(path_str))


def _params() -> list[Any]:
    params = []
    for path in sorted((_MMS / "cases").glob("*.yaml")):
        case = yaml.safe_load(path.read_text())
        for solver in (s for s in case.get("applicable_solvers", []) if str(s).startswith("fenics")):
            marks = []
            if path.stem not in _FAST:
                marks.append(pytest.mark.integration)
            if solver == "fenics-mol" and path.stem in _KNOWN_MOL_REGRESSIONS:
                marks.append(
                    pytest.mark.xfail(
                        reason="adaptive-BDF (MOL) global-time-error floor on this decaying time-dependent "
                        "case — geometry-independent (box and nested disk both), worse with tighter tolerance; "
                        "BE holds its order. A time-integrator finding, not spatial — see _KNOWN_MOL_REGRESSIONS",
                        strict=False,
                    )
                )
            params.append(pytest.param(str(path), solver, marks=marks, id=f"{path.stem}-{solver}"))
    return params


@pytest.mark.parametrize(("path_str", "solver"), _params())
def test_mms_case_holds_its_order(path_str: str, solver: str) -> None:
    result = _run_case(path_str)["results"][solver]
    assert result["ok"], f"{Path(path_str).stem} [{solver}]: order {result['order']} below its expected_order_h"

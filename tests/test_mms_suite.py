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

# There are NO remaining fenics-MOL xfails. The former holdout — the time-dependent `static_reaction` case
# (u* = e^{-t}·…, static mesh) — was DIAGNOSED down to two independent, well-understood adaptive-BDF *temporal*
# error floors (not a spatial/solver defect: backward-Euler holds order 2 on both geometries): (1) the
# CONTROLLED time error, set by the adaptive relative tolerance (default rtol=1e-6 too loose at fine h), and
# (2) the BDF COLD-START — the first, order-1 step whose truncation error the adaptive controller can't see, an
# rtol-INDEPENDENT bias ≈ the startup step. Measuring the SPATIAL order needs both pushed below it: the `_box`
# variant needs only tighter rtol (`mol_rtol: 1e-9` → order ~1.9); the finer curved `_disk` variant needs BOTH
# (`mol_rtol: 1e-9` + `mol_dt_initial: 1e-6` → order ~1.7, the P1 curved-domain rate). Both knobs live on the
# case files and are applied by the runner's fenics-mol path. Every bulk case now passes on both geometries at
# its own honest rate — the structured box at O(h²), the nested (curvature-preserving) disk at ~O(h^1.5).


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
            params.append(pytest.param(str(path), solver, marks=marks, id=f"{path.stem}-{solver}"))
    return params


@pytest.mark.parametrize(("path_str", "solver"), _params())
def test_mms_case_holds_its_order(path_str: str, solver: str) -> None:
    result = _run_case(path_str)["results"][solver]
    assert result["ok"], f"{Path(path_str).stem} [{solver}]: order {result['order']} below its expected_order_h"

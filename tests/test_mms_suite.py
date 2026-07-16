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

# fenics-MOL "regressions" on the disk cases — DIAGNOSED (not a solver defect): all are `kind: disk`, whose
# realize mesh is rebuilt independently at each h, so its P1 spatial error does NOT converge at O(h²) — it
# floors on the mesh-topology noise (~4e-4 here). The backward-Euler runs pass only because the harness
# refines dt ∝ h² for these time-dependent/moving cases, so BE's O(dt) *time* error is O(h²) and dominates
# (masks) that spatial floor; the time-error-free MOL has nothing masking it and reads the floor directly.
# Verified: BE with a tiny fixed dt (pure spatial) floors even harder (order ~0.27) on the same disks. It is
# the same non-nestable-geometry effect as the coupled disk-in-annulus (~0.65) — the clean spatial-order
# checks live in the pytest MMS tests on nested split-box fixtures. A future runner increment could add a
# nestable structured bulk geometry to lift these; until then they are xfail'd (the BE order-2 check stands).
_KNOWN_MOL_REGRESSIONS = {
    "bulk_anisotropic_stretch_diffusion",
    "bulk_shear_diffusion",
    "bulk_shrinkage_diffusion",
    "bulk_static_reaction_timedep",
    "bulk_translation_diffusion",
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
                        reason="disk-geometry remeshing-noise floor on the time-error-free MOL path (the "
                        "disk is rebuilt per h, so its spatial error is not O(h²)); BE passes only because "
                        "dt∝h² makes its time error dominate. Not a solver defect — see _KNOWN_MOL_REGRESSIONS",
                        strict=False,
                    )
                )
            params.append(pytest.param(str(path), solver, marks=marks, id=f"{path.stem}-{solver}"))
    return params


@pytest.mark.parametrize(("path_str", "solver"), _params())
def test_mms_case_holds_its_order(path_str: str, solver: str) -> None:
    result = _run_case(path_str)["results"][solver]
    assert result["ok"], f"{Path(path_str).stem} [{solver}]: order {result['order']} below its expected_order_h"

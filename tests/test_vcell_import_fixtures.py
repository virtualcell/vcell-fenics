"""Golden-fixture corpus for the VCell import layer (`vcell_fenics.pyvcell_bridge`).

A data-driven complement to the hand-written unit tests in `test_pyvcell_bridge.py`:
the unit tests assert specific facts with precise messages; this corpus gives *breadth*
across math-description structure and expression variety, and guards against regression.
Three tiers, under `tests/fixtures/vcell_import/`:

- **accept/<case>/** — `input.yaml` (a pyvcell `MathDescription`) + `expected.yaml`
  (the golden formalism `MathDescription`) + optional `meta.yaml` (`geometry:` for the
  import's geometry arg; `dim:` its dimension, needed to translate per-face boundary conditions;
  `run:` to also solve it end-to-end; `note:`). Both YAML, for
  readability. Each accepted golden must also *validate* (no formalism errors).
- **reject/<case>/** — `input.yaml` + `error.txt` (`<ExceptionType>: <message substring>`):
  constructs that must be rejected loudly (§2.6.3 out-of-scope / §2.6.2 not-yet).
- **expressions.yaml** — a dense `vcell` → `formalism` expression-translation table.

**Adding a case.** Write a VCell `input.yaml` (a pyvcell `MathDescription`; export from
pyvcell with `yaml.safe_dump(md.model_dump(exclude_none=True, exclude_defaults=True))`)
into `accept/<case>/`, then regenerate the golden with `UPDATE_GOLDENS=1 pytest -k vcell_import`
and **review the produced `expected.yaml` in the PR diff** — that review is the real check;
the golden then guards against regressions, with `validate` and the runnable subset as
backstops against blessing a plausible-but-wrong translation.

The accept/reject tiers reconstruct pyvcell `MathDescription`s (pyvcell is a normal
dependency); the expression table is pure.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import pytest
import pyvcell.vcml.models_math as vm
import yaml

from vcell_fenics.formalism import dump_yaml, load_yaml, validate
from vcell_fenics.pyvcell_bridge import VcellImportError, import_math_description, translate_expression

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "vcell_import"
_UPDATE_GOLDENS = os.environ.get("UPDATE_GOLDENS") == "1"

_REJECT_EXCEPTIONS: dict[str, type[Exception]] = {
    "VcellImportError": VcellImportError,
    "NotImplementedError": NotImplementedError,
}


def _case_dirs(tier: str) -> list[str]:
    base = _FIXTURES / tier
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


def _or_skip(items: list[Any], reason: str) -> list[Any]:
    """pytest errors on an empty parametrize set; fall back to one skipped placeholder."""
    return items or [pytest.param("(none)", marks=pytest.mark.skip(reason=reason))]


def _load_input(case_dir: pathlib.Path) -> Any:
    return vm.MathDescription.model_validate(yaml.safe_load((case_dir / "input.yaml").read_text()))


def _load_meta(case_dir: pathlib.Path) -> dict[str, Any]:
    meta_path = case_dir / "meta.yaml"
    return yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}


# --- accept: VCell math → golden formalism math --------------------------------


@pytest.mark.parametrize("case", _or_skip(_case_dirs("accept"), "no accept fixtures"))
def test_accept_fixture_matches_golden(case: str) -> None:
    case_dir = _FIXTURES / "accept" / case
    meta = _load_meta(case_dir)
    imported = import_math_description(_load_input(case_dir), geometry=meta.get("geometry"), dim=meta.get("dim"))

    expected_path = case_dir / "expected.yaml"
    if _UPDATE_GOLDENS:
        expected_path.write_text(dump_yaml(imported))

    assert imported == load_yaml(expected_path.read_text())  # structural (format-independent) match
    # Every accepted golden must be a valid formalism model — catches "translates but ill-posed".
    assert [d for d in validate(imported) if d.severity == "error"] == []


@pytest.mark.parametrize(
    "case",
    _or_skip(
        [c for c in _case_dirs("accept") if "run" in _load_meta(_FIXTURES / "accept" / c)],
        "no runnable accept fixtures",
    ),
)
def test_accept_fixture_runs_end_to_end(case: str) -> None:
    # The runnable subset (meta `run:`) actually solves through the FEniCSx backend on a
    # disk geometry — proving the imported model is not just valid but solvable.
    import numpy as np

    from vcell_fenics.backend import SolverConfiguration, make_disk_geometry, run

    case_dir = _FIXTURES / "accept" / case
    meta = _load_meta(case_dir)
    run_cfg = meta["run"]
    md = import_math_description(_load_input(case_dir), geometry=meta["geometry"])
    geometry = make_disk_geometry(
        meta["geometry"], volume_subdomain=run_cfg["volume_subdomain"], h=run_cfg.get("h", 0.2)
    )
    problem = run(md, geometry, SolverConfiguration(dt=run_cfg["dt"], t_final=run_cfg["t_final"]))
    assert np.all(np.isfinite(problem.unknown.x.array))


# --- reject: out-of-scope / not-yet constructs must raise ----------------------


@pytest.mark.parametrize("case", _or_skip(_case_dirs("reject"), "no reject fixtures"))
def test_reject_fixture_raises(case: str) -> None:
    case_dir = _FIXTURES / "reject" / case
    # error.txt: optional `# comment` lines, then `<ExceptionType>: <message substring>`.
    spec = next(line for line in (case_dir / "error.txt").read_text().splitlines() if line and not line.startswith("#"))
    typename, _, substring = spec.partition(":")
    with pytest.raises(_REJECT_EXCEPTIONS[typename.strip()]) as excinfo:
        import_math_description(_load_input(case_dir))
    assert substring.strip() in str(excinfo.value)


# --- expressions: VCell syntax → formalism syntax ------------------------------


def _expression_cases() -> list[Any]:
    path = _FIXTURES / "expressions.yaml"
    if not path.exists():
        return _or_skip([], "no expressions.yaml")
    rows = yaml.safe_load(path.read_text()) or []
    return [pytest.param(r["vcell"], r["formalism"], id=r["vcell"]) for r in rows]


@pytest.mark.parametrize("vcell,expected", _expression_cases())
def test_expression_translation(vcell: str, expected: str) -> None:
    assert translate_expression(vcell) == expected

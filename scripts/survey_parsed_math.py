#!/usr/bin/env python
"""Survey the parsed VCell math descriptions for import-layer coverage.

Runs in the vcell-fenics **dev env** (no full pyvcell needed): `pyvcell.vcml.models_math` is
pydantic-only and lazily importable, and the bridge + validator are local. For each
`vcml_biomodels/parsed/*_math.yaml` it reconstructs a pyvcell `MathDescription`, attempts
`import_math_description`, validates the result, and records structure + expression features.

Outputs (to the parsed dir):
- a printed summary (import buckets, structure distribution, expression/function frequency),
- `survey.csv` — one row per app (structure, import status, validation, expression flags),
- `fixture_candidates.txt` — a small, diverse shortlist of clean, importable models to promote
  into the `tests/fixtures/vcell_import/accept/` corpus.

Usage: `.pixi/envs/dev/bin/python scripts/survey_parsed_math.py [--limit N]`
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

# VCell expressions can nest deeply (long parenthesised kinetic laws); the recursive-descent
# parser needs headroom. Pathological cases are still caught per-model in _classify.
sys.setrecursionlimit(20000)

from vcell_fenics.formalism import validate
from vcell_fenics.pyvcell_bridge import VcellImportError, import_math_description

_ROOT = Path(__file__).resolve().parent.parent
_PARSED = _ROOT / "vcml_biomodels" / "parsed"


def _load_models_math() -> Any:
    """Import `pyvcell.vcml.models_math` (pydantic-only) bypassing the heavy `pyvcell.vcml`
    package __init__, which (depending on the pyvcell working-tree state) may eagerly import
    numexpr/libvcell/etc. that the vcell-fenics dev env omits."""
    if "pyvcell.vcml.models_math" in sys.modules:
        return sys.modules["pyvcell.vcml.models_math"]
    import pyvcell  # noqa: F401 — empty top-level package init

    vcml_dir = Path(pyvcell.__file__).parent / "vcml"
    if "pyvcell.vcml" not in sys.modules:
        pkg = types.ModuleType("pyvcell.vcml")
        pkg.__path__ = [str(vcml_dir)]  # type: ignore[attr-defined]
        sys.modules["pyvcell.vcml"] = pkg
    spec = importlib.util.spec_from_file_location("pyvcell.vcml.models_math", vcml_dir / "models_math.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pyvcell.vcml.models_math"] = module
    spec.loader.exec_module(module)
    return module


vm = _load_models_math()

_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_COORD_RE = re.compile(r"\b(?:x|y|z)\b")
_TIME_RE = re.compile(r"\bt\b")


def _expr_strings(md: Any) -> list[str]:
    out: list[str] = []
    for c in md.constants:
        out.append(c.exp or "")
    for fn in md.functions:
        out.append(fn.exp or "")
    for sub in [*md.compartment_subdomains, *md.membrane_subdomains]:
        for pde in sub.pde_equations:
            out += [pde.rate or "", pde.diffusion or "", pde.initial or ""]
            if pde.velocity is not None:
                out += [pde.velocity.x or "", pde.velocity.y or "", pde.velocity.z or ""]
        for ode in sub.ode_equations:
            out += [ode.rate or "", ode.initial or ""]
    return [s for s in out if s]


def _structure(md: Any) -> dict[str, Any]:
    comps, mems = md.compartment_subdomains, md.membrane_subdomains
    pde = sum(len(s.pde_equations) for s in [*comps, *mems])
    ode = sum(len(s.ode_equations) for s in [*comps, *mems])
    stochastic = any(
        getattr(s, a, None)
        for s in [*comps, *mems]
        for a in ("jump_processes", "particle_jump_processes", "particle_properties", "variable_initial_counts")
    )
    has_boundaries = any(
        (p.boundaries is not None or any(bt.type.lower() != "flux" for bt in getattr(s, "boundary_types", [])))
        for s in [*comps, *mems]
        for p in s.pde_equations
    ) or any(getattr(s, "boundary_types", []) for s in [*comps, *mems])
    has_velocity = any(p.velocity is not None for s in [*comps, *mems] for p in s.pde_equations)
    has_jump = any(getattr(s, "jump_conditions", []) for s in mems)
    steady = any(p.steady for s in [*comps, *mems] for p in s.pde_equations)
    return {
        "compartments": len(comps),
        "membranes": len(mems),
        "pde": pde,
        "ode": ode,
        "equations": pde + ode,
        "stochastic": stochastic,
        "has_boundaries": has_boundaries,
        "has_velocity": has_velocity,
        "has_jump_condition": has_jump,
        "steady": steady,
    }


def _classify(md: Any) -> tuple[str, str]:
    """(bucket, detail) for import + validation of one model."""
    try:
        imported = import_math_description(md)
    except VcellImportError as exc:
        return "reject_stochastic", str(exc).splitlines()[0][:120]
    except NotImplementedError as exc:
        return "reject_not_implemented", str(exc).splitlines()[0][:120]
    except Exception as exc:  # unexpected importer error
        return "import_error", f"{type(exc).__name__}: {exc}"[:120]
    try:
        errors = [d for d in validate(imported) if d.severity == "error"]
    except Exception as exc:  # e.g. RecursionError on a pathologically nested expression
        return "validate_crash", f"{type(exc).__name__}"[:120]
    if errors:
        return "validate_fail", errors[0].message[:120]
    return "ok", ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey parsed VCell math descriptions")
    parser.add_argument("--parsed", type=Path, default=_PARSED)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    files = sorted(args.parsed.glob("*_math.yaml"))
    if args.limit is not None:
        files = files[: args.limit]
    total = len(files)
    print(f"Surveying {total} math descriptions in {args.parsed}\n")

    buckets: Counter[str] = Counter()
    functions: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    structure_hist: Counter[str] = Counter()
    detail_by_bucket: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, dict[str, Any]]] = []

    for i, f in enumerate(files, 1):
        try:
            data = yaml.safe_load(f.read_text())
            md = vm.MathDescription.model_validate(data)
        except Exception as exc:
            buckets["reconstruct_fail"] += 1
            rows.append({"file": f.name, "bucket": "reconstruct_fail", "detail": f"{type(exc).__name__}: {exc}"[:120]})
            continue

        st = _structure(md)
        bucket, detail = _classify(md)
        buckets[bucket] += 1
        if detail:
            detail_by_bucket.setdefault(bucket, Counter())[detail] += 1

        exprs = _expr_strings(md)
        for e in exprs:
            for name in _CALL_RE.findall(e):
                functions[name] += 1
        used_features = set()
        if any(_COORD_RE.search(e) for e in exprs):
            used_features.add("coordinates")
        if any(_TIME_RE.search(e) for e in exprs):
            used_features.add("time")
        if any("^" in e for e in exprs):
            used_features.add("power_caret")
        if any("/" in e for e in exprs):
            used_features.add("division")
        for feat in used_features:
            feature_counts[feat] += 1

        kind = (
            "ode_only"
            if st["pde"] == 0 and st["ode"] > 0
            else "compartment+membrane"
            if st["membranes"] and st["compartments"]
            else "membrane_only"
            if st["membranes"]
            else "multi_compartment"
            if st["compartments"] > 1
            else "single_compartment"
        )
        structure_hist[kind] += 1

        rows.append({"file": f.name, "bucket": bucket, "detail": detail, "kind": kind, **st})

        # Shortlist: clean + small + with an IC, score by simplicity & diversity value.
        if bucket == "ok" and 1 <= st["equations"] <= 6 and st["compartments"] + st["membranes"] <= 3:
            score = st["equations"] + 0.5 * st["compartments"] + (0.0 if used_features else 1.0)
            candidates.append((score, f.name, {**st, "kind": kind, "features": sorted(used_features)}))

        if i % 500 == 0 or i == total:
            print(f"  [{i}/{total}] ...")

    # ---- summary ----
    print("\n=== import buckets ===")
    for name, n in buckets.most_common():
        print(f"  {n:5d}  {name}  ({100 * n / total:.1f}%)")
    print("\n=== structure (importable + not) ===")
    for name, n in structure_hist.most_common():
        print(f"  {n:5d}  {name}")
    print("\n=== expression features (models using each) ===")
    for name, n in feature_counts.most_common():
        print(f"  {n:5d}  {name}")
    print("\n=== top function names across all expressions ===")
    for name, n in functions.most_common(30):
        print(f"  {n:6d}  {name}")
    for bucket in ("reject_not_implemented", "validate_fail", "reject_stochastic", "import_error"):
        if bucket in detail_by_bucket:
            print(f"\n=== top '{bucket}' reasons ===")
            for msg, n in detail_by_bucket[bucket].most_common(12):
                print(f"  {n:5d}  {msg}")

    # ---- csv ----
    csv_path = args.parsed / "survey.csv"
    cols = ["file", "bucket", "kind", "compartments", "membranes", "pde", "ode", "equations",
            "stochastic", "has_boundaries", "has_velocity", "has_jump_condition", "steady", "detail"]  # fmt: skip
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # ---- fixture shortlist: diverse spread of clean, small models ----
    candidates.sort(key=lambda c: c[0])
    by_kind: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
    for c in candidates:
        by_kind.setdefault(c[2]["kind"], []).append(c)
    shortlist: list[tuple[float, str, dict[str, Any]]] = []
    for kind, items in by_kind.items():
        shortlist += items[:6]  # up to 6 simplest per structural kind
    shortlist.sort(key=lambda c: (c[2]["kind"], c[0]))
    cand_path = args.parsed / "fixture_candidates.txt"
    with cand_path.open("w") as handle:
        handle.write(f"# {len(candidates)} clean+small importable models; shortlist below (<=6 per kind)\n")
        for score, name, info in shortlist:
            handle.write(f"{name}\t{info['kind']}\teq={info['equations']} comp={info['compartments']} "
                         f"mem={info['membranes']} feats={','.join(info['features']) or '-'}\n")  # fmt: skip
    print(f"Wrote {cand_path}  ({len(candidates)} clean+small candidates, {len(shortlist)} shortlisted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Stage 2 of the VCell ↔ vcell-fenics geometry cross-check (vcell-fenics **dev env**).

Reads the region-mask `.npz` files produced by `export_vcell_region_masks.py` (stage 1, pyvcell
env) and verifies — voxel by voxel — that vcell-fenics classifies each point into the **same
subvolume VCell does**. For each voxel centre, our priority-resolved Rvachev implicit fields
(`subvolume_implicit_functions`) pick the subvolume whose `φ < 0`; that is compared to VCell's own
`domain_name` label for the voxel.

This is the external cross-implementation oracle (VCell's authoritative rasterization), distinct from
the internal predicate↔R-function↔mesh self-consistency check in `check_realize_membership.py`. It
covers the full 2D-analytic corpus (no meshing needed). Voxels our partition leaves ambiguous (none
or several fields negative — the near-membrane band) are skipped.

Usage: `.pixi/envs/dev/bin/python scripts/check_vcell_membership.py [--masks DIR]`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml.models_geometry as g
import yaml

from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.rvachev import subvolume_implicit_functions
from vcell_fenics.pyvcell_bridge import import_geometry

_ROOT = Path(__file__).resolve().parent.parent
_PARSED = _ROOT / "vcml_biomodels" / "parsed"


def _reconstruct(data: dict[str, Any]) -> Any:
    image = data.get("image")
    if isinstance(image, dict) and "compressed_content" not in image:
        image["compressed_content"] = ""
    return g.Geometry.model_validate(data)


def _eval(expr: Expr, x: float, y: float) -> float:
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, IndexAccess):
        assert isinstance(expr.index, Number)
        return (x, y)[int(expr.index.value)]
    if isinstance(expr, UnaryOp):
        v = _eval(expr.operand, x, y)
        return -v if expr.op == "-" else v
    if isinstance(expr, BinaryOp):
        a, b = _eval(expr.left, x, y), _eval(expr.right, x, y)
        ops = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b else float("nan"))}
        if expr.op in ops:
            return ops[expr.op]
        if expr.op == "**":
            return float(a**b) if not isinstance(a**b, complex) else float("nan")
        raise ValueError(f"operator {expr.op!r} not in an implicit function")
    if isinstance(expr, FunctionCall):
        args = [_eval(a, x, y) for a in expr.args]
        if expr.callee == "min":
            return min(args)
        if expr.callee == "max":
            return max(args)
        raise ValueError(f"function {expr.callee!r} unsupported")
    raise ValueError(f"unbound/unhandled {type(expr).__name__}")


def _check_mask(npz_path: Path) -> tuple[int, int, int] | None:
    """Returns (agree, disagree, skipped) over the voxels, or None if out of scope."""
    data = np.load(npz_path, allow_pickle=True)
    coords = data["coords"]
    domains = data["domains"].astype(str)
    geom_yaml = _PARSED / f"{data['source']}.yaml"
    gd = import_geometry(_reconstruct(yaml.safe_load(geom_yaml.read_text())))
    fields = subvolume_implicit_functions(gd)

    if not (set(domains) & set(fields)):
        raise ValueError(f"no overlap between VCell domains {sorted(set(domains))} and ours {sorted(fields)}")

    agree = disagree = skipped = 0
    for (x, y, _z), vcell_domain in zip(coords, domains):
        inside = [name for name, field in fields.items() if _eval(field, float(x), float(y)) < 0]
        if len(inside) != 1:  # ambiguous: near a membrane
            skipped += 1
        elif inside[0] == vcell_domain:
            agree += 1
        else:
            disagree += 1
    return agree, disagree, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, default=_ROOT / "vcml_biomodels" / "region_masks")
    args = parser.parse_args()

    total_agree = total_disagree = total_skip = 0
    flagged: list[str] = []
    checked = 0
    for npz_path in sorted(args.masks.glob("*.npz")):
        try:
            result = _check_mask(npz_path)
        except Exception as exc:  # noqa: BLE001 — param-bearing expr, name mismatch, etc.
            flagged.append(f"  skip {npz_path.stem}: {type(exc).__name__}: {exc}"[:130])
            continue
        if result is None:
            continue
        agree, disagree, skipped = result
        checked += 1
        total_agree += agree
        total_disagree += disagree
        total_skip += skipped
        rate = disagree / (agree + disagree) if (agree + disagree) else 0.0
        status = "OK   " if rate < 0.01 else "DISAGREE"
        if rate >= 0.01:
            flagged.append(f"  {status} {npz_path.stem}: {disagree}/{agree + disagree} voxels ({rate:.1%})")

    checked_voxels = total_agree + total_disagree
    print(f"=== VCell vs vcell-fenics subvolume membership ({checked} geometries) ===")
    print(f"voxels: {total_agree} agree, {total_disagree} disagree, {total_skip} skipped (near-membrane)")
    if checked_voxels:
        print(f"agreement: {total_agree / checked_voxels:.4%}")
    if flagged:
        print("\n=== flagged ===")
        for line in flagged[:40]:
            print(line)
    return 1 if total_disagree > 0.01 * max(checked_voxels, 1) else 0


if __name__ == "__main__":
    raise SystemExit(main())

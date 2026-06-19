#!/usr/bin/env python
"""Three-way point-membership check over 2D analytic corpus geometries (vcell-fenics dev env).

For each realizable 2D `analytic` VCell geometry, sample random points in the bounding box and
verify that three independent representations agree on which region each point is in:

1. the **original boolean predicate** (the subvolume's `expression`),
2. its **Rvachev implicit-function** sign (`formalism/rvachev.py`), and
3. the **realized gmsh mesh** (`backend/realize.py`).

(1) vs (2) is exact (a measure-zero disagreement only on `φ = 0`). (3) is the discretized one, so a
point is checked against the mesh only where the mesh classifies it *unambiguously* — in exactly one
of the two submeshes; the ambiguous band is precisely the near-membrane region (the scale-free analogue
of "ignore points where the R-function is ~0").

Kept as a script, not a unit test: it needs the corpus and meshes many geometries. v1 `realize` only
covers the *one interior subvolume + background* topology, so most corpus geometries fall outside scope
and are reported as such — this measures correctness on the realizable subset.

Usage: `.pixi/envs/dev/bin/python scripts/check_realize_membership.py [--limit N] [--points N] [--seed S]`
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml.models_geometry as g
import yaml
from dolfinx import geometry as dgeom
from dolfinx.mesh import Mesh
from numpy.typing import NDArray

from vcell_fenics.backend.realize import realize
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import subvolume_implicit_functions
from vcell_fenics.pyvcell_bridge import import_geometry

_ROOT = Path(__file__).resolve().parent.parent
_PARSED = _ROOT / "vcml_biomodels" / "parsed"


def _reconstruct(data: dict[str, Any]) -> Any:
    image = data.get("image")
    if isinstance(image, dict) and "compressed_content" not in image:
        image["compressed_content"] = ""
    return g.Geometry.model_validate(data)


def _eval(expr: Expr, x: float, y: float) -> float | bool:
    """Scalar evaluation of a geom.x-only predicate / implicit function. Raises on an unbound name
    (a parameter-bearing expression), which the caller treats as out of scope."""
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, IndexAccess):
        assert isinstance(expr.index, Number)
        return (x, y)[int(expr.index.value)]
    if isinstance(expr, UnaryOp):
        v = _eval(expr.operand, x, y)
        return -float(v) if expr.op == "-" else (float(v) if expr.op == "+" else not v)
    if isinstance(expr, BinaryOp):
        a, b = float(_eval(expr.left, x, y)), float(_eval(expr.right, x, y))
        if expr.op == "+":
            return a + b
        if expr.op == "-":
            return a - b
        if expr.op == "*":
            return a * b
        if expr.op == "/":
            return a / b
        if expr.op == "**":
            return float(a**b) if not isinstance(a**b, complex) else float("nan")
        if expr.op == "<":
            return a < b
        if expr.op == "<=":
            return a <= b
        if expr.op == ">":
            return a > b
        if expr.op == ">=":
            return a >= b
        if expr.op == "&&":
            return bool(a) and bool(b)
        if expr.op == "||":
            return bool(a) or bool(b)
        raise ValueError(f"unhandled operator {expr.op!r}")
    if isinstance(expr, FunctionCall):
        args = [float(_eval(a, x, y)) for a in expr.args]
        if expr.callee == "min":
            return min(args)
        if expr.callee == "max":
            return max(args)
        raise ValueError(f"unsupported function {expr.callee!r}")
    raise ValueError(f"unbound or unhandled node {type(expr).__name__}")


def _in_mesh(mesh: Mesh, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    tree = dgeom.bb_tree(mesh, mesh.topology.dim)
    candidates = dgeom.compute_collisions_points(tree, points)
    colliding = dgeom.compute_colliding_cells(mesh, candidates, points)
    return np.array([colliding.links(np.int32(i)).size > 0 for i in range(len(points))])


def _check(gd: GeometryDescription, rng: np.random.Generator, n_points: int) -> tuple[int, int, int]:
    """Realize and check one geometry. Returns (predicate-vs-Rfunction mismatches, mesh-vs-predicate
    mismatches, points checked against the mesh)."""
    geometry = realize(gd, h=0.04, resolution=201)
    interior, background = gd.subvolumes[0], gd.subvolumes[-1]
    assert interior.expression is not None
    predicate = parse(interior.expression)
    field = subvolume_implicit_functions(gd)[interior.name]

    ox, oy = gd.origin[0], gd.origin[1]
    lx, ly = gd.extent[0], gd.extent[1]
    pts2d = np.column_stack(
        [rng.uniform(ox + 0.02 * lx, ox + 0.98 * lx, n_points), rng.uniform(oy + 0.02 * ly, oy + 0.98 * ly, n_points)]
    )
    points = np.column_stack([pts2d, np.zeros(n_points)]).astype(np.float64)
    in_interior = _in_mesh(geometry.mesh_of(interior.name), points)
    in_background = _in_mesh(geometry.mesh_of(background.name), points)

    rfunc_mismatch = mesh_mismatch = mesh_checked = 0
    for i, (x, y, _z) in enumerate(points):
        predicate_inside = bool(_eval(predicate, float(x), float(y)))
        if (float(_eval(field, float(x), float(y))) < 0) != predicate_inside:
            rfunc_mismatch += 1
        if bool(in_interior[i]) != bool(in_background[i]):  # unambiguous mesh classification
            mesh_checked += 1
            if bool(in_interior[i]) != predicate_inside:
                mesh_mismatch += 1
    return rfunc_mismatch, mesh_mismatch, mesh_checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, default=_PARSED)
    parser.add_argument("--limit", type=int, default=40, help="max realizable geometries to check")
    parser.add_argument("--points", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    buckets: Counter[str] = Counter()
    total_mesh_checked = total_rfunc_mismatch = total_mesh_mismatch = 0
    flagged: list[str] = []
    realized = failures = 0

    for f in sorted(args.parsed.glob("*_geom.yaml")):
        if realized >= args.limit:
            break
        try:
            gd = import_geometry(_reconstruct(yaml.safe_load(f.read_text())))
        except Exception:
            continue
        if gd.dim != 2 or not any(sv.type == "analytic" for sv in gd.subvolumes):
            continue  # focus: 2D analytic geometries

        try:
            rfunc_mismatch, mesh_mismatch, mesh_checked = _check(gd, rng, args.points)
        except NotImplementedError:
            buckets["not_in_scope_v1"] += 1
            continue
        except Exception as exc:  # noqa: BLE001 — parameter-bearing expr, gmsh failure, etc.
            buckets["realize_error"] += 1
            flagged.append(f"  ERROR  {f.name}: {type(exc).__name__}: {exc}"[:140])
            continue

        realized += 1
        total_mesh_checked += mesh_checked
        total_rfunc_mismatch += rfunc_mismatch
        total_mesh_mismatch += mesh_mismatch
        # A systematic disagreement (any predicate-vs-Rfunction, or >5% of mesh points) is a real
        # failure; a handful of mesh points is the near-membrane discretization band (boundary noise).
        systematic = rfunc_mismatch > 0 or (mesh_checked > 0 and mesh_mismatch > 0.05 * mesh_checked)
        if systematic:
            buckets["SYSTEMATIC_mismatch"] += 1
            failures += 1
            flagged.append(f"  SYSTEMATIC  {f.name}: rfunc={rfunc_mismatch} mesh={mesh_mismatch}/{mesh_checked}")
        elif mesh_mismatch:
            buckets["boundary_noise"] += 1
            flagged.append(f"  boundary    {f.name}: {mesh_mismatch}/{mesh_checked} near-membrane points")
        else:
            buckets["realized_clean"] += 1

    print(f"=== 2D analytic membership check (limit={args.limit}, {args.points} pts/geom, seed={args.seed}) ===")
    for name, n in buckets.most_common():
        print(f"  {n:4d}  {name}")
    print(
        f"\nrealized & checked: {realized} geometries, {total_mesh_checked} unambiguous mesh points; "
        f"predicate-vs-Rfunction mismatches={total_rfunc_mismatch}, mesh-vs-predicate mismatches={total_mesh_mismatch}"
    )
    if flagged:
        print("\n=== flagged ===")
        for line in flagged[:40]:
            print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

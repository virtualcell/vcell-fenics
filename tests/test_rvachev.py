"""Rvachev lowering: analytic boolean predicate → implicit function (inside-negative).

Two layers of coverage:

1. **Structure** — the node-wise rules (relational → signed margin, `&&`→max, `||`→min,
   `!`→negate, all-boolean `*`/`+` → AND/OR, `==`/`!=` and mixed operands rejected).
2. **Sign correctness** — the predicates are the ground truth: on a sample grid, the lowered
   `φ` is negative exactly where its predicate is true, and the priority assembly reproduces the
   painter's-algorithm partition (a point belongs to the lowest-index subvolume whose predicate
   holds, else the background).
"""

from __future__ import annotations

import pytest

from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Name, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import (
    RvachevLoweringError,
    is_boolean,
    lower_predicate,
    subvolume_implicit_functions,
)

# -- a tiny evaluator over the subset these tests use ---------------------------


def _eval(expr: Expr, point: tuple[float, ...], params: dict[str, float]) -> float | bool:
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, Name):
        if expr.name in params:
            return params[expr.name]
        raise AssertionError(f"unbound name {expr.name!r} (geom.x must be indexed)")
    if isinstance(expr, IndexAccess):
        assert isinstance(expr.base, Name) and expr.base.name == "geom.x"
        idx = _eval(expr.index, point, params)
        return point[int(idx)]
    if isinstance(expr, UnaryOp):
        v = _eval(expr.operand, point, params)
        if expr.op == "-":
            return -float(v)
        if expr.op == "+":
            return float(v)
        return not v  # "!"
    if isinstance(expr, BinaryOp):
        left = _eval(expr.left, point, params)
        right = _eval(expr.right, point, params)
        match expr.op:
            case "+":
                return float(left) + float(right)
            case "-":
                return float(left) - float(right)
            case "*":
                return float(left) * float(right)
            case "/":
                return float(left) / float(right)
            case "**":
                return float(float(left) ** float(right))
            case "<":
                return float(left) < float(right)
            case "<=":
                return float(left) <= float(right)
            case ">":
                return float(left) > float(right)
            case ">=":
                return float(left) >= float(right)
            case "&&":
                return bool(left) and bool(right)
            case "||":
                return bool(left) or bool(right)
            case _:
                raise AssertionError(f"unexpected op {expr.op!r}")
    if isinstance(expr, FunctionCall):
        vals = [float(_eval(a, point, params)) for a in expr.args]
        if expr.callee == "max":
            return max(vals)
        if expr.callee == "min":
            return min(vals)
        raise AssertionError(f"unexpected call {expr.callee!r}")
    raise AssertionError(f"unhandled node {type(expr).__name__}")


def _grid(n: int = 21, lo: float = -2.0, hi: float = 2.0) -> list[tuple[float, float]]:
    step = (hi - lo) / (n - 1)
    return [(lo + i * step, lo + j * step) for i in range(n) for j in range(n)]


# On the zero-set (a membrane) the sign of φ is undefined and the point belongs to no open
# region, so the grid-based sign checks skip points within this tolerance of any boundary.
_BOUNDARY_TOL = 1e-9


# -- structure ------------------------------------------------------------------


def test_less_than_is_left_minus_right() -> None:
    assert lower_predicate(parse("geom.x[0] < geom.x[1]")) == BinaryOp(
        "-", IndexAccess(Name("geom.x"), Number(0.0)), IndexAccess(Name("geom.x"), Number(1.0))
    )


def test_greater_than_is_right_minus_left() -> None:
    # a > b  ->  b - a
    assert lower_predicate(parse("geom.x[0] > 5")) == BinaryOp(
        "-", Number(5.0), IndexAccess(Name("geom.x"), Number(0.0))
    )


def test_and_is_max_or_is_min_not_is_negate() -> None:
    lo = IndexAccess(Name("geom.x"), Number(0.0))
    assert lower_predicate(parse("geom.x[0] < 1 && geom.x[0] > 0")) == FunctionCall(
        "max", (BinaryOp("-", lo, Number(1.0)), BinaryOp("-", Number(0.0), lo))
    )
    assert lower_predicate(parse("geom.x[0] < 1 || geom.x[0] > 0")) == FunctionCall(
        "min", (BinaryOp("-", lo, Number(1.0)), BinaryOp("-", Number(0.0), lo))
    )
    assert lower_predicate(parse("!(geom.x[0] < 1)")) == UnaryOp("-", BinaryOp("-", lo, Number(1.0)))


def test_all_boolean_mult_is_and_add_is_or() -> None:
    # (x<1)*(y<1) lowers identically to (x<1) && (y<1); (x<1)+(y<1) to ||.
    assert lower_predicate(parse("(geom.x[0] < 1) * (geom.x[1] < 1)")) == lower_predicate(
        parse("(geom.x[0] < 1) && (geom.x[1] < 1)")
    )
    assert lower_predicate(parse("(geom.x[0] < 1) + (geom.x[1] < 1)")) == lower_predicate(
        parse("(geom.x[0] < 1) || (geom.x[1] < 1)")
    )


def test_is_boolean_classifies_nodes() -> None:
    assert is_boolean(parse("geom.x[0] < 1"))
    assert is_boolean(parse("(geom.x[0] < 1) * (geom.x[1] < 1)"))  # all-boolean product
    assert not is_boolean(parse("geom.x[0] * 2"))
    assert not is_boolean(parse("geom.x[0] + geom.x[1]"))


def test_equality_operators_rejected() -> None:
    for src in ("geom.x[0] == 0", "geom.x[0] != 0"):
        with pytest.raises(RvachevLoweringError, match="measure-zero"):
            lower_predicate(parse(src))


def test_mixed_boolean_numeric_rejected() -> None:
    with pytest.raises(RvachevLoweringError, match="mixes boolean and numeric"):
        lower_predicate(parse("(geom.x[0] < 1) * 2"))


# -- sign correctness (predicate is the ground truth) ---------------------------


def test_disk_sign_matches_predicate() -> None:
    pred = parse("geom.x[0] ** 2 + geom.x[1] ** 2 < 1")
    phi = lower_predicate(pred)
    for p in _grid():
        value = float(_eval(phi, p, {}))
        if abs(value) < _BOUNDARY_TOL:
            continue
        assert (value < 0) == _eval(pred, p, {})


def test_disk_implicit_is_x2_plus_y2_minus_r2() -> None:
    # Oracle: the lowered field equals VCell's emitted IF, x^2 + y^2 - R^2.
    phi = lower_predicate(parse("geom.x[0] ** 2 + geom.x[1] ** 2 < 4"))
    for p in _grid():
        assert _eval(phi, p, {}) == pytest.approx(p[0] ** 2 + p[1] ** 2 - 4.0)


def test_intersection_box_sign() -> None:
    pred = parse("geom.x[0] > -1 && geom.x[0] < 1 && geom.x[1] > -1 && geom.x[1] < 1")
    phi = lower_predicate(pred)
    for p in _grid():
        value = float(_eval(phi, p, {}))
        if abs(value) < _BOUNDARY_TOL:
            continue
        assert (value < 0) == _eval(pred, p, {})


# -- priority assembly ----------------------------------------------------------


def test_single_subvolume_is_constant_negative() -> None:
    geom = GeometryDescription(name="cell", dim=0, subvolumes=(SubVolume(name="cytosol", type="compartmental"),))
    assert subvolume_implicit_functions(geom) == {"cytosol": Number(-1.0)}


def test_two_region_cell_in_extracellular_partition() -> None:
    # cytosol = disk (priority 0); extracellular = background (last).
    geom = GeometryDescription(
        name="cell",
        dim=2,
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 1"),
            SubVolume(name="extracellular", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 > 1"),
        ),
    )
    fields = subvolume_implicit_functions(geom)
    for p in _grid():
        cyto = float(_eval(fields["cytosol"], p, {}))
        if abs(cyto) < _BOUNDARY_TOL:
            continue  # on the membrane
        in_disk = p[0] ** 2 + p[1] ** 2 < 1
        assert (cyto < 0) == in_disk
        # Background is the exact complement of the disk.
        assert (float(_eval(fields["extracellular"], p, {})) < 0) == (not in_disk)


def test_priority_subtraction_three_overlapping_regions() -> None:
    # Two overlapping disks + background. Painter's algorithm: a point in the overlap belongs to
    # the higher-priority (lower-index) subvolume, so the lower-priority one must exclude it.
    geom = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="a", type="analytic", expression="(geom.x[0] + 0.5) ** 2 + geom.x[1] ** 2 < 1"),
            SubVolume(name="b", type="analytic", expression="(geom.x[0] - 0.5) ** 2 + geom.x[1] ** 2 < 1"),
            SubVolume(name="bg", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 > 100"),
        ),
    )
    fields = subvolume_implicit_functions(geom)

    def owner(p: tuple[float, float]) -> str:
        in_a = (p[0] + 0.5) ** 2 + p[1] ** 2 < 1
        in_b = (p[0] - 0.5) ** 2 + p[1] ** 2 < 1
        return "a" if in_a else "b" if in_b else "bg"

    for p in _grid():
        values = {name: float(_eval(phi, p, {})) for name, phi in fields.items()}
        if any(abs(v) < _BOUNDARY_TOL for v in values.values()):
            continue  # on a membrane between two regions
        want = owner(p)
        for name, value in values.items():
            assert (value < 0) == (name == want), f"{name} at {p}"


def test_non_analytic_non_last_rejected() -> None:
    geom = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="img", type="image", pixel_value=1),  # no expression, not last
            SubVolume(name="bg", type="analytic", expression="geom.x[0] < 1"),
        ),
    )
    with pytest.raises(RvachevLoweringError, match="no analytic expression"):
        subvolume_implicit_functions(geom)

"""Tests for the expression → UFL compiler (ADR 004, inc-0 subset).

Two checks: (1) compiled expressions evaluate to the right number — established
by integrating the compiled coefficient over a mesh and dividing by its area;
(2) a compiled coefficient assembles into the same matrix as a hand-written UFL
form (the compiler ↔ UFL equivalence layer). Plus the error paths for
unsupported constructs and unresolved names.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.approaches.static import create_disk
from vcell_fenics.backend import CompileContext, CompileError, compile_expression
from vcell_fenics.formalism import parse


def _ctx(params: dict[str, float] | None = None) -> CompileContext:
    mesh = create_disk(radius=1.0, h=0.3).mesh
    symbols: dict[str, Any] = {
        name: fem.Constant(mesh, PETSc.ScalarType(value))  # type: ignore[operator]
        for name, value in (params or {}).items()
    }
    # The assembler binds `x` to the spatial coordinate; mirror that here.
    symbols["x"] = ufl.SpatialCoordinate(mesh)
    return CompileContext(mesh=mesh, symbols=symbols)


def _mean(expr_str: str, params: dict[str, float] | None = None) -> float:
    """Compile `expr_str` and return its spatially-averaged value over the disk
    (= the constant value, for spatially-constant expressions)."""

    ctx = _ctx(params)
    compiled = compile_expression(parse(expr_str), ctx)
    dx = ufl.Measure("dx", domain=ctx.mesh)
    one = fem.Constant(ctx.mesh, PETSc.ScalarType(1.0))  # type: ignore[operator]
    area = fem.assemble_scalar(fem.form(one * dx))
    value = fem.assemble_scalar(fem.form(compiled * dx))
    # assemble_scalar is typed float | complex (PETSc may be complex-valued);
    # this is a real build, so take the real part.
    return (value / area).real


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("0.5", 0.5),
        ("0.5 * 2 + 1", 2.0),
        ("-3 + 1", -2.0),
        ("2 ** 3", 8.0),
        ("10 / 4", 2.5),
        ("1.0e-3", 1.0e-3),
        ("-(2 + 3)", -5.0),
    ],
)
def test_arithmetic_compiles_to_correct_value(expr: str, expected: float) -> None:
    assert _mean(expr) == pytest.approx(expected)


def test_parameter_resolves_from_context() -> None:
    assert _mean("k_on * 2 - 1", {"k_on": 3.0}) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Coordinates, indexing, and functions of x (inc 1).
# ---------------------------------------------------------------------------


def test_coordinate_component_averages_to_zero_on_centred_disk() -> None:
    # ∫_disk x[0] dA = 0 for a disk centred at the origin.
    assert _mean("x[0]") == pytest.approx(0.0, abs=1e-9)


def test_radius_helper_averages_to_two_thirds() -> None:
    # ∫_disk r dA / area = (2/3)R = 2/3 for R = 1; coarse mesh, loose tol.
    assert _mean("r(x)") == pytest.approx(2.0 / 3.0, abs=2e-2)


def test_cos_of_theta_mode_averages_to_constant() -> None:
    # ∫_disk cos(2θ) dA = 0 analytically, so the mean of 1 + 0.5·cos(2θ) is 1.
    # θ is singular at the origin, so a coarse mesh leaves a small residue; loose tol.
    assert _mean("1.0 + 0.5 * cos(2 * theta(x))") == pytest.approx(1.0, abs=2e-3)


def test_compiled_coefficient_matches_handwritten_ufl() -> None:
    # The compiler ↔ UFL equivalence layer: a compiled coefficient assembles
    # into the same matrix as the hand-written form with the same coefficient.
    ctx = _ctx()
    mesh = ctx.mesh
    V = fem.functionspace(mesh, ("Lagrange", 1))
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    compiled = compile_expression(parse("0.25 * 4"), ctx)
    hand = fem.Constant(mesh, PETSc.ScalarType(1.0))  # type: ignore[operator]

    a_compiled = cast(npt.NDArray[Any], fem.assemble_matrix(fem.form(compiled * u * w * dx)).to_dense())
    a_hand = cast(npt.NDArray[Any], fem.assemble_matrix(fem.form(hand * u * w * dx)).to_dense())
    assert np.allclose(a_compiled, a_hand)


def test_unresolved_name_raises() -> None:
    with pytest.raises(CompileError, match="unresolved name 'mystery'"):
        compile_expression(parse("mystery + 1"), _ctx())


def test_unsupported_construct_raises() -> None:
    # Vector literals are not in the supported subset yet.
    with pytest.raises(CompileError, match="not supported"):
        compile_expression(parse("[1.0, 2.0]"), _ctx())


def test_unsupported_function_raises() -> None:
    with pytest.raises(CompileError, match="not supported"):
        compile_expression(parse("sinh(x[0])"), _ctx())

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
    symbols["geom.x"] = ufl.SpatialCoordinate(mesh)
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
    assert _mean("geom.x[0]") == pytest.approx(0.0, abs=1e-9)


def test_radius_helper_averages_to_two_thirds() -> None:
    # ∫_disk r dA / area = (2/3)R = 2/3 for R = 1; coarse mesh, loose tol.
    assert _mean("geom.radius") == pytest.approx(2.0 / 3.0, abs=2e-2)


def test_cos_of_theta_mode_averages_to_constant() -> None:
    # ∫_disk cos(2θ) dA = 0 analytically, so the mean of 1 + 0.5·cos(2θ) is 1.
    # θ is singular at the origin, so a coarse mesh leaves a small residue; loose tol.
    assert _mean("1.0 + 0.5 * cos(2 * geom.azimuth)") == pytest.approx(1.0, abs=2e-3)


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
    # Tensor literals are not in the supported subset yet (vector literals now are).
    with pytest.raises(CompileError, match="not supported"):
        compile_expression(parse("[[1.0, 2.0], [3.0, 4.0]]"), _ctx())


def test_hyperbolic_and_log10_functions_compile() -> None:
    # Added for the VCell import layer (real models use these); UFL supports them directly.
    for src in ("sinh(geom.x[0])", "cosh(geom.x[0])", "tanh(geom.x[0])", "log10(geom.radius)"):
        compile_expression(parse(src), _ctx())  # must not raise


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("pow(2.0, 3.0)", 8.0),  # VCell emits pow(a, b); the importer keeps it as a call
        ("pow(9.0, 0.5)", 3.0),
        ("min(2.0, 5.0)", 2.0),
        ("max(2.0, 5.0)", 5.0),
        ("atan2(0.0, 1.0)", 0.0),
    ],
)
def test_two_argument_math_functions_compile(expr: str, expected: float) -> None:
    # Added for the VCell import layer: pow / min / max / atan2 are two-argument scalar functions.
    assert _mean(expr) == pytest.approx(expected)


def test_pow_takes_two_arguments() -> None:
    with pytest.raises(CompileError, match="takes two arguments"):
        compile_expression(parse("pow(2.0)"), _ctx())


def test_floor_validates_but_does_not_compile() -> None:
    # floor / ceil are accepted by the vocabulary (real VCell models use them) but are
    # non-differentiable, so UFL/DOLFINx has no operator — compiling one raises.
    with pytest.raises(CompileError, match="not supported"):
        compile_expression(parse("floor(geom.x[0])"), _ctx())


def test_conditional_and_relational_operators_compile() -> None:
    # Relational/logical operators + if(...) — common in imported VCell kinetics.
    for src in (
        "geom.x[0] >= 0.0",
        "geom.x[0] > 0.0 && geom.x[1] < 1.0",
        "if(geom.x[0] > 0.5, 1.0, 2.0)",
        "if(geom.x[0] > 0.0 || geom.x[1] != 0.0, 1.0, 0.0)",
    ):
        compile_expression(parse(src), _ctx())  # must not raise


def test_conditional_selects_branch_pointwise() -> None:
    # if(x[0] > 0, +1, -1) over a disk centred at the origin integrates to ~0 (antisymmetric).
    assert _mean("if(geom.x[0] > 0.0, 1.0, -1.0)") == pytest.approx(0.0, abs=2e-2)


def test_relational_in_arithmetic_equals_explicit_if() -> None:
    # VCell boolean-as-number semantics: a comparison used in arithmetic is 1 if true, 0 if
    # false, so `10*(x<0)` must be identical to `if(x<0, 10, 0)` (the user's question).
    assert _mean("10.0*(geom.x[0]<0.0)") == pytest.approx(_mean("if(geom.x[0]<0.0, 10.0, 0.0)"))


def test_logical_and_band_equals_explicit_if() -> None:
    # `(x>-0.5) && (x<0.5)` is the 0/1 indicator of a band — equal to its if(...) form. The
    # VCell pulse idiom `A*((t>t0) && (t<t1))` relies on this.
    band = "(geom.x[0] > -0.5) && (geom.x[0] < 0.5)"
    assert _mean(band) == pytest.approx(_mean(f"if({band}, 1.0, 0.0)"))


# ---------------------------------------------------------------------------
# trace(·) — the cross-dimensional reference (§1.8.2).
# ---------------------------------------------------------------------------


def _ctx_with_variable(name: str, value: float, params: dict[str, float] | None = None) -> CompileContext:
    """A context with a P1 `Function` named `name` (a stand-in for a bulk variable's
    solution Function), interpolated to the constant `value`."""

    ctx = _ctx(params)
    field = fem.Function(fem.functionspace(ctx.mesh, ("Lagrange", 1)), name=name)
    field.x.array[:] = value
    ctx.symbols[name] = field
    return ctx


def test_trace_resolves_to_the_variable_object() -> None:
    # The cross-dimensional restriction is mixed-dimensional assembly's job, so the
    # compiler returns the resolved variable unchanged (passthrough).
    ctx = _ctx_with_variable("L", 2.0)
    assert compile_expression(parse("trace(L)"), ctx) is ctx.symbols["L"]


def test_trace_passes_through_arithmetic() -> None:
    # k_on * trace(L) with L ≡ 2 and k_on = 3 averages to 6 — the traced value flows
    # through the surrounding arithmetic.
    ctx = _ctx_with_variable("L", 2.0, {"k_on": 3.0})
    compiled = compile_expression(parse("k_on * trace(L)"), ctx)
    dx = ufl.Measure("dx", domain=ctx.mesh)
    one = fem.Constant(ctx.mesh, PETSc.ScalarType(1.0))  # type: ignore[operator]
    area = fem.assemble_scalar(fem.form(one * dx))
    value = fem.assemble_scalar(fem.form(compiled * dx))
    assert (value / area).real == pytest.approx(6.0)


def test_trace_requires_one_argument() -> None:
    ctx = _ctx_with_variable("L", 1.0)
    with pytest.raises(CompileError, match="exactly one argument"):
        compile_expression(parse("trace(L, L)"), ctx)


def test_trace_of_unresolved_variable_raises() -> None:
    # The bulk variable must be in the symbol table (the cross-subdomain assembler
    # provides it); otherwise it is an unresolved name like any other.
    with pytest.raises(CompileError, match="unresolved name 'L'"):
        compile_expression(parse("trace(L)"), _ctx())

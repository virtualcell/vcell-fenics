"""Tests for the expression-string parser (vcell_fenics.formalism.parser).

The parser is purely syntactic: it turns a math-like infix string into the
typed AST of vcell_fenics.formalism.expr and resolves no names / checks no
types. Coverage is in four layers:

1. **Literals and primaries** — numbers (int / decimal / scientific), names,
   index access, vector and tensor literals, parenthesised grouping.
2. **Operator grammar** — arithmetic precedence, left-associativity of
   +-*/, right-associativity of **, and unary-minus interaction with **.
3. **Real expressions** — every expression string in the worked-example
   fixtures (tests/fixtures/*.yaml) parses without error.
4. **Error reporting** — malformed strings raise ExpressionSyntaxError with a
   position pointing at the offending character.
"""

from __future__ import annotations

import pytest

from vcell_fenics.formalism import (
    BinaryOp,
    ExpressionSyntaxError,
    FunctionCall,
    IndexAccess,
    Name,
    Number,
    TensorLiteral,
    UnaryOp,
    VectorLiteral,
    parse,
)

# ---------------------------------------------------------------------------
# Literals and primaries.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("0", 0.0),
        ("1.0", 1.0),
        ("0.5", 0.5),
        ("1.0e-3", 1.0e-3),
        ("1e5", 1e5),
        (".5", 0.5),
        ("42.", 42.0),
        ("6.022E23", 6.022e23),
    ],
)
def test_number_literals(source: str, value: float) -> None:
    assert parse(source) == Number(value=value)


def test_bare_name() -> None:
    assert parse("rho_active") == Name(name="rho_active")


def test_whitespace_is_insignificant() -> None:
    assert parse("  a  +\tb\n") == parse("a+b")


def test_index_access_on_coordinate() -> None:
    # `geom.x` is a single qualified name token (ADR 006), indexed by component.
    assert parse("geom.x[0]") == IndexAccess(base=Name(name="geom.x"), index=Number(value=0.0))


def test_vector_literal() -> None:
    # `geom.azimuth` parses as a qualified Name (a value), not a `theta(x)` call.
    assert parse("[f0 * cos(geom.azimuth), 0]") == VectorLiteral(
        components=(
            BinaryOp(
                op="*",
                left=Name(name="f0"),
                right=FunctionCall(callee="cos", args=(Name(name="geom.azimuth"),)),
            ),
            Number(value=0.0),
        )
    )


def test_tensor_literal_is_rows_of_vectors() -> None:
    node = parse("[[a, b], [c, d]]")
    assert node == TensorLiteral(
        rows=(
            VectorLiteral(components=(Name(name="a"), Name(name="b"))),
            VectorLiteral(components=(Name(name="c"), Name(name="d"))),
        )
    )


def test_empty_bracket_is_empty_vector() -> None:
    # Shape (a zero-component vector is meaningless) is a validator concern;
    # the parser accepts it structurally.
    assert parse("[]") == VectorLiteral(components=())


def test_parentheses_group() -> None:
    assert parse("(a + b) * c") == BinaryOp(
        op="*",
        left=BinaryOp(op="+", left=Name(name="a"), right=Name(name="b")),
        right=Name(name="c"),
    )


# ---------------------------------------------------------------------------
# Function calls.
# ---------------------------------------------------------------------------


def test_function_call_zero_args() -> None:
    assert parse("f()") == FunctionCall(callee="f", args=())


def test_function_call_multiple_args() -> None:
    assert parse("if(a, b, c)") == FunctionCall(callee="if", args=(Name(name="a"), Name(name="b"), Name(name="c")))


def test_calculus_operator_is_a_function_call() -> None:
    # `grad`, `trace`, `n`, `partial_t`, measures — all parse as FunctionCall;
    # the validator dispatches on the callee name, not the parser.
    assert parse("trace(L)") == FunctionCall(callee="trace", args=(Name(name="L"),))


def test_indexing_a_call_result() -> None:
    assert parse("grad(u)[1]") == IndexAccess(
        base=FunctionCall(callee="grad", args=(Name(name="u"),)),
        index=Number(value=1.0),
    )


# ---------------------------------------------------------------------------
# Operator grammar: precedence and associativity.
# ---------------------------------------------------------------------------


def test_multiplication_binds_tighter_than_addition() -> None:
    assert parse("a + b * c") == BinaryOp(
        op="+",
        left=Name(name="a"),
        right=BinaryOp(op="*", left=Name(name="b"), right=Name(name="c")),
    )


def test_additive_is_left_associative() -> None:
    assert parse("a - b - c") == BinaryOp(
        op="-",
        left=BinaryOp(op="-", left=Name(name="a"), right=Name(name="b")),
        right=Name(name="c"),
    )


def test_power_is_right_associative() -> None:
    assert parse("2 ** 3 ** 2") == BinaryOp(
        op="**",
        left=Number(value=2.0),
        right=BinaryOp(op="**", left=Number(value=3.0), right=Number(value=2.0)),
    )


def test_power_binds_tighter_than_unary_minus() -> None:
    # -2 ** 2 == -(2 ** 2), matching Python.
    assert parse("-2 ** 2") == UnaryOp(
        op="-", operand=BinaryOp(op="**", left=Number(value=2.0), right=Number(value=2.0))
    )


def test_unary_minus_allowed_in_exponent() -> None:
    assert parse("2 ** -1") == BinaryOp(
        op="**", left=Number(value=2.0), right=UnaryOp(op="-", operand=Number(value=1.0))
    )


def test_unary_minus_on_parenthesised_group() -> None:
    assert parse("-(a - b)") == UnaryOp(op="-", operand=BinaryOp(op="-", left=Name(name="a"), right=Name(name="b")))


def test_leading_unary_minus_on_term() -> None:
    assert parse("-k_on * rho + k_off") == BinaryOp(
        op="+",
        left=BinaryOp(op="*", left=UnaryOp(op="-", operand=Name(name="k_on")), right=Name(name="rho")),
        right=Name(name="k_off"),
    )


# ---------------------------------------------------------------------------
# Every expression string in the worked-example fixtures parses.
# ---------------------------------------------------------------------------


FIXTURE_EXPRESSIONS = [
    "r_dot * geom.x / geom.radius",
    "k_on * rho_inactive - k_off * rho_active",
    "1.0 + 0.5 * cos(2 * geom.azimuth)",
    "-k_on * rho_inactive + k_off * rho_active",
    "0.5",
    "-(k_on * trace(L) * rho_f - k_off * rho_b)",
    "k_on * trace(L) * rho_f - k_off * rho_b",
    "L_reservoir",
    "[f0 * cos(geom.azimuth), 0]",
    "1.0 + 0.3 * cos(2 * geom.azimuth)",
    # The §2.7 weak-form residual: multi-line, _test functions, dx_Gamma measure.
    """
    ( eta * inner(v_membrane, v_membrane_test)
      + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)
      - inner(f_active, v_membrane_test)
    ) * dx_Gamma
    """,
]


@pytest.mark.parametrize("source", FIXTURE_EXPRESSIONS)
def test_fixture_expressions_parse(source: str) -> None:
    parse(source)  # must not raise


def test_weak_form_measure_is_a_bare_name() -> None:
    # `dx_Gamma` has no parser-special status; it parses as a Name and the
    # validator recognises it as a measure in weak-form context.
    node = parse("rho * dx_Gamma")
    assert node == BinaryOp(op="*", left=Name(name="rho"), right=Name(name="dx_Gamma"))


# ---------------------------------------------------------------------------
# Error reporting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "pos"),
    [
        ("", 0),  # empty: end-of-input where a primary was expected
        ("a +", 3),  # dangling binary operator: end where an operand was expected
        ("* a", 0),  # leading binary operator
    ],
)
def test_error_positions(source: str, pos: int) -> None:
    with pytest.raises(ExpressionSyntaxError) as exc:
        parse(source)
    assert exc.value.pos == pos


def test_double_plus_is_unary_not_an_error() -> None:
    # `a + + b` is legal — the second `+` is unary plus.
    assert parse("a + + b") == BinaryOp(op="+", left=Name(name="a"), right=UnaryOp(op="+", operand=Name(name="b")))


def test_trailing_input_rejected() -> None:
    with pytest.raises(ExpressionSyntaxError) as exc:
        parse("a b")
    assert exc.value.pos == 2


def test_unbalanced_paren_rejected() -> None:
    with pytest.raises(ExpressionSyntaxError):
        parse("(a + b")


def test_unbalanced_bracket_rejected() -> None:
    with pytest.raises(ExpressionSyntaxError):
        parse("[a, b")


def test_unexpected_character_rejected() -> None:
    with pytest.raises(ExpressionSyntaxError) as exc:
        parse("a | b")
    assert exc.value.pos == 2


def test_missing_comma_in_call_rejected() -> None:
    with pytest.raises(ExpressionSyntaxError):
        parse("f(a b)")


def test_error_message_has_caret_and_line() -> None:
    with pytest.raises(ExpressionSyntaxError) as exc:
        parse("a +\n  $")
    rendered = str(exc.value)
    assert "line 2" in rendered
    assert "^" in rendered

"""Tests for expression type-checking (§1.11.5 / §1.8.8).

Bottom-up type inference over the vocabulary: term-slot, initial-condition,
motion, and expression-parameter expressions must produce the type their slot
declares, with no implicit broadcast except the polymorphic zero literal.
Weak-form residuals are intentionally not type-checked (escape hatch).
"""

from __future__ import annotations

from vcell_fenics.formalism import MathDescription, validate
from vcell_fenics.formalism.schema import (
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _errors(md: MathDescription) -> list[str]:
    return [d.message for d in validate(md) if d.severity == "error"]


def _surface_model(
    *,
    diffusion: str = "0.1",
    source: str = "0.0",
    relative_advection: str | None = None,
    ic: str = "1.0",
    parameters: list[ParameterConstant | ParameterExpression] | None = None,
) -> MathDescription:
    terms = {"diffusion": diffusion, "source": source}
    if relative_advection is not None:
        terms["relative_advection"] = relative_advection
    return MathDescription(
        geometry="g",
        subdomains=[Subdomain(name="membrane", kind="surface")],
        variables=[Variable(name="rho", subdomain="membrane")],
        equations=[
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms=terms,
                initial_condition=ic,
            )
        ],
        parameters=list(parameters or []),
    )


# ---------------------------------------------------------------------------
# Slot type conformance.
# ---------------------------------------------------------------------------


def test_scalar_source_accepts_scalar() -> None:
    assert validate(_surface_model(source="rho - 1")) == []


def test_vector_in_scalar_source_is_rejected() -> None:
    # x is a vector; a scalar source slot cannot hold it.
    assert any("expected scalar" in m for m in _errors(_surface_model(source="geom.x")))


def test_relative_advection_requires_vector() -> None:
    # A scalar in the vector relative_advection slot is a no-broadcast error.
    assert any("expected vector" in m for m in _errors(_surface_model(relative_advection="0.5")))


def test_relative_advection_accepts_vector() -> None:
    assert validate(_surface_model(relative_advection="geom.x")) == []


def test_conditional_with_relational_validates() -> None:
    # if(condition, then, else) with a relational condition — the common imported form.
    assert validate(_surface_model(source="if(rho > 0.5, -1.0, 1.0)")) == []


def test_relational_operator_requires_scalar_operands() -> None:
    # geom.x is a vector; comparing it is a type error.
    assert any("requires scalar operands" in m for m in _errors(_surface_model(source="if(geom.x > 0.0, 1.0, 0.0)")))


def test_diffusion_accepts_scalar() -> None:
    assert validate(_surface_model(diffusion="0.1")) == []


# ---------------------------------------------------------------------------
# Arithmetic and the no-broadcast rule (§1.8.8).
# ---------------------------------------------------------------------------


def test_scalar_times_vector_is_a_vector() -> None:
    # `0.5 * x` is the explicit broadcast and is allowed in a vector slot.
    assert validate(_surface_model(relative_advection="0.5 * geom.x")) == []


def test_adding_scalar_and_vector_is_rejected() -> None:
    assert any("no implicit broadcast" in m for m in _errors(_surface_model(relative_advection="geom.x + 1")))


def test_dividing_by_a_vector_is_rejected() -> None:
    assert any("divisor must be scalar" in m for m in _errors(_surface_model(source="1 / geom.x")))


# ---------------------------------------------------------------------------
# The polymorphic zero literal.
# ---------------------------------------------------------------------------


def test_zero_literal_is_accepted_as_a_vector() -> None:
    assert validate(_surface_model(relative_advection="0")) == []


def test_nonzero_scalar_is_not_accepted_as_a_vector() -> None:
    assert any("expected vector" in m for m in _errors(_surface_model(relative_advection="5")))


# ---------------------------------------------------------------------------
# Calculus operator argument/result types (§1.8.5).
# ---------------------------------------------------------------------------


def test_div_of_scalar_is_rejected() -> None:
    # div needs a vector; rho is scalar. (div on a non-governed scalar param.)
    md = _surface_model(source="div(k)", parameters=[ParameterConstant(name="k", value=1.0)])
    assert any("requires a vector argument" in m for m in _errors(md))


def test_grad_surf_of_scalar_yields_vector_in_vector_slot() -> None:
    # grad_surf(p) is a vector and fits relative_advection; p is a scalar param.
    md = _surface_model(
        relative_advection="grad_surf(p)",
        parameters=[ParameterConstant(name="p", value=1.0)],
    )
    assert validate(md) == []


# ---------------------------------------------------------------------------
# Initial-condition and parameter-expression types.
# ---------------------------------------------------------------------------


def test_vector_ic_for_scalar_variable_is_rejected() -> None:
    assert any("expected scalar" in m for m in _errors(_surface_model(ic="geom.x")))


def test_expression_parameter_type_mismatch_is_rejected() -> None:
    # p is declared vector but its body "1.0" is scalar; the parameter body is
    # type-checked whether or not p is used elsewhere.
    md = _surface_model(parameters=[ParameterExpression(name="p", expression="1.0", type="vector")])
    assert any("expected vector" in m for m in _errors(md))


def test_inner_of_mismatched_ranks_is_rejected() -> None:
    md = _surface_model(source="inner(geom.x, 1.0)")  # vector vs scalar
    assert any("matching ranks" in m for m in _errors(md))

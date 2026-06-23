"""Tests for the expression-level validation checks (§1.11.3/§1.11.8/§1.11.9).

These exercise the parse + AST-walk pass: name resolution, the trace direction
rule, parameter-expression rules (no variables, acyclic, helper scoping),
initial-condition content rules, weak-form structure, and the operator-usage
narrow / smoothness rules.

Models are built directly from the schema dataclasses (small and focused)
rather than mutated fixtures, so each test pins exactly one expression rule.
The fixture-level acceptance test (fixtures validate clean) lives in
test_formalism_validator.py and already covers the happy path end-to-end.
"""

from __future__ import annotations

from vcell_fenics.formalism import MathDescription, validate
from vcell_fenics.formalism.schema import (
    MotionUnknown,
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
)


def _errors(md: MathDescription) -> list[str]:
    return [d.message for d in validate(md) if d.severity == "error"]


def _surface_model(
    *,
    source: str = "0.0",
    diffusion: str = "0.1",
    ic: str = "1.0",
    parameters: list[ParameterConstant | ParameterExpression] | None = None,
    extra_variables: list[Variable] | None = None,
) -> MathDescription:
    """A minimal valid single-species surface model, parameterised at the
    expression sites individual tests want to poke."""

    return MathDescription(
        geometry="g",
        subdomains=[Subdomain(name="membrane", kind="surface")],
        variables=[Variable(name="rho", subdomain="membrane"), *(extra_variables or [])],
        equations=[
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": diffusion, "source": source},
                initial_condition=ic,
            )
        ],
        parameters=list(parameters or []),
    )


# ---------------------------------------------------------------------------
# Name resolution (§1.11.3).
# ---------------------------------------------------------------------------


def test_unknown_name_in_slot_is_rejected() -> None:
    assert any("unknown name 'mystery'" in m for m in _errors(_surface_model(source="mystery * rho")))


def test_local_variable_and_parameter_resolve() -> None:
    md = _surface_model(source="k * rho", parameters=[ParameterConstant(name="k", value=1.0)])
    assert validate(md) == []


def test_function_used_without_call_is_rejected() -> None:
    # `sin` is a function; a bare reference is an error.
    assert any("must be called with arguments" in m for m in _errors(_surface_model(source="sin")))


def test_expression_syntax_error_surfaces_as_diagnostic() -> None:
    assert any("expression syntax error" in m for m in _errors(_surface_model(source="rho +")))


# ---------------------------------------------------------------------------
# Cross-subdomain references and the trace rule (§1.8.2, §1.8.6).
# ---------------------------------------------------------------------------


def _bulk_surface_model(*, surface_source: str) -> MathDescription:
    return MathDescription(
        geometry="g",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="membrane", kind="surface"),
        ],
        variables=[
            Variable(name="c", subdomain="cyto"),
            Variable(name="rho", subdomain="membrane"),
        ],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="c",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.5"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.1", "source": surface_source},
                initial_condition="1.0",
            ),
        ],
    )


def test_higher_dim_variable_needs_trace() -> None:
    # `c` lives on the bulk (higher dim); a bare reference from the surface must
    # be flagged with a trace suggestion.
    errs = _errors(_bulk_surface_model(surface_source="c * rho"))
    assert any("trace(c)" in m for m in errs)


def test_trace_of_higher_dim_variable_is_accepted() -> None:
    assert validate(_bulk_surface_model(surface_source="trace(c) * rho")) == []


def test_trace_of_non_higher_dim_variable_is_rejected() -> None:
    # rho lives on the same subdomain as the expression; trace(rho) is invalid.
    errs = _errors(_bulk_surface_model(surface_source="trace(rho)"))
    assert any("does not live on a subdomain of higher dimension" in m for m in errs)


# ---------------------------------------------------------------------------
# Parameter expressions (§1.8.3, §1.11.3, §2.2.3).
# ---------------------------------------------------------------------------


def test_parameter_expression_may_not_reference_variable() -> None:
    md = _surface_model(parameters=[ParameterExpression(name="p", expression="rho + 1")])
    assert any("may not reference the variable 'rho'" in m for m in _errors(md))


def test_parameter_expression_cycle_is_rejected() -> None:
    md = _surface_model(
        parameters=[
            ParameterExpression(name="a", expression="b + 1"),
            ParameterExpression(name="b", expression="a + 1"),
        ]
    )
    assert any("parameter expression cycle" in m for m in _errors(md))


def test_parameter_expression_with_helper_needs_scope() -> None:
    md = _surface_model(parameters=[ParameterExpression(name="p", expression="geom.azimuth")])
    assert any("declares no 'subdomain:' scope" in m for m in _errors(md))


def test_scoped_parameter_helper_is_accepted_on_its_subdomain() -> None:
    md = _surface_model(
        source="p * rho",
        parameters=[ParameterExpression(name="p", expression="geom.azimuth", subdomain="membrane")],
    )
    assert validate(md) == []


def test_scoped_parameter_used_from_wrong_subdomain_is_rejected() -> None:
    md = MathDescription(
        geometry="g",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="membrane", kind="surface"),
        ],
        variables=[
            Variable(name="c", subdomain="cyto"),
            Variable(name="rho", subdomain="membrane"),
        ],
        parameters=[ParameterExpression(name="curv", expression="geom.mean_curvature", subdomain="membrane")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="c",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.5", "source": "curv * c"},  # curv is membrane-scoped
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="1.0",
            ),
        ],
    )
    assert any("is scoped to subdomain 'membrane'" in m for m in _errors(md))


# ---------------------------------------------------------------------------
# Initial-condition content (§1.11.8).
# ---------------------------------------------------------------------------


def test_ic_referencing_time_is_rejected() -> None:
    # Time is `sim.t` (ADR 006); bare `t` is now a free user name, so the IC restriction is on it.
    assert any("may not reference time sim.t" in m for m in _errors(_surface_model(ic="sim.t")))


def test_ic_referencing_state_variable_is_rejected() -> None:
    assert any("may not reference the state variable 'rho'" in m for m in _errors(_surface_model(ic="rho")))


# ---------------------------------------------------------------------------
# Operator-usage rules (§1.11.9).
# ---------------------------------------------------------------------------


def test_narrow_rule_calculus_on_governed_variable_is_rejected() -> None:
    errs = _errors(_surface_model(source="div(grad_surf(rho))"))
    assert any("narrow rule" in m for m in errs)


def test_narrow_rule_allows_calculus_on_other_variable() -> None:
    # grad_surf on a non-governed variable (other) is fine in rho's slot.
    md = _surface_model(
        source="div_surf(grad_surf(other))",
        extra_variables=[Variable(name="other", subdomain="membrane")],
    )
    # `other` also needs its own equation; add it so coverage passes.
    md = MathDescription(
        geometry=md.geometry,
        subdomains=md.subdomains,
        variables=md.variables,
        parameters=md.parameters,
        equations=[
            *md.equations,
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="other",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="0.0",
            ),
        ],
    )
    assert validate(md) == []


def test_smoothness_rule_lapl_on_p1_is_rejected() -> None:
    md = _surface_model(
        source="lapl_beltrami(other)",
        extra_variables=[Variable(name="other", subdomain="membrane", space="lagrange_p1")],
    )
    md = MathDescription(
        geometry=md.geometry,
        subdomains=md.subdomains,
        variables=md.variables,
        parameters=md.parameters,
        equations=[
            *md.equations,
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="other",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="0.0",
            ),
        ],
    )
    assert any("lagrange_p2" in m for m in _errors(md))


def test_smoothness_rule_lapl_on_p2_is_accepted() -> None:
    md = _surface_model(
        source="lapl_beltrami(other)",
        extra_variables=[Variable(name="other", subdomain="membrane", space="lagrange_p2")],
    )
    md = MathDescription(
        geometry=md.geometry,
        subdomains=md.subdomains,
        variables=md.variables,
        parameters=md.parameters,
        equations=[
            *md.equations,
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="other",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="0.0",
            ),
        ],
    )
    assert validate(md) == []


# ---------------------------------------------------------------------------
# Weak-form structure (§1.9.5, §2.3.5).
# ---------------------------------------------------------------------------


def _weak_form_model(*, form: str, temporality: str = "steady_state") -> MathDescription:
    return MathDescription(
        geometry="g",
        subdomains=[Subdomain(name="membrane", kind="surface", motion=MotionUnknown(variable="v"))],
        variables=[Variable(name="v", subdomain="membrane", type="vector")],
        equations=[
            WeakFormEquation(
                variable="v",
                subdomain="membrane",
                temporality=temporality,  # type: ignore[arg-type]
                form=form,
                initial_condition="0",
            )
        ],
    )


def test_weak_form_missing_test_function_is_rejected() -> None:
    errs = _errors(_weak_form_model(form="inner(v, v) * dx_Gamma"))
    assert any("test function v_test" in m for m in errs)


def test_steady_state_weak_form_with_partial_t_is_rejected() -> None:
    errs = _errors(_weak_form_model(form="inner(partial_t(v), v_test) * dx_Gamma"))
    assert any("must not contain partial_t(v)" in m for m in errs)


def test_time_dependent_weak_form_without_partial_t_is_rejected() -> None:
    errs = _errors(_weak_form_model(form="inner(v, v_test) * dx_Gamma", temporality="time_dependent"))
    assert any("must contain partial_t(v)" in m for m in errs)


def test_partial_t_outside_weak_form_is_rejected() -> None:
    # partial_t in a template slot (not a weak form) is invalid.
    assert any("only valid in a weak-form" in m for m in _errors(_surface_model(source="partial_t(rho)")))


# ---------------------------------------------------------------------------
# Random IC primitives (§1.8.9) — normal/uniform, initial-condition only.
# ---------------------------------------------------------------------------


def test_random_primitive_in_initial_condition_is_valid() -> None:
    # A random draw is fine in an IC (composable with spatial structure); the realization stores it
    # once as a fixed field, so it stays a pure function of space.
    assert validate(_surface_model(ic="1.0 + 0.1*geom.x[0] + normal(0, 0.05)")) == []
    assert validate(_surface_model(ic="uniform(0.4, 0.6)")) == []


def test_random_primitive_outside_initial_condition_is_rejected() -> None:
    # A draw in a re-assembled term would not be a fixed function of space (the value at a point
    # would change between assemblies), so it is confined to the once-realized IC.
    assert any("only in an initial_condition" in m for m in _errors(_surface_model(source="normal(0, 1) * rho")))


def test_random_primitive_arity_is_checked() -> None:
    assert any("takes two arguments" in m for m in _errors(_surface_model(ic="normal(0)")))

"""Tests for the structural validation pass (vcell_fenics.formalism.validator).

Two layers:

1. **Acceptance** — the three worked-example fixtures (which are the v1
   reference models) validate with zero errors. The §2.7 fixture exercises the
   unknown-motion-variable IC-on-steady-state exception specifically.
2. **Rejection** — each structural rule is tripped by a minimal mutation of a
   valid model, and the expected error is asserted.

Mutations are built by editing the loaded dataclass tree via dataclasses.replace
(the schema is frozen), keeping each test focused on one rule.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vcell_fenics.formalism import (
    FormalismValidationError,
    MathDescription,
    load_yaml,
    validate,
    validate_or_raise,
)
from vcell_fenics.formalism.schema import (
    BCNeumann,
    ParameterConstant,
    TemplateEquation,
    Variable,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> MathDescription:
    return load_yaml(FIXTURES / name)


def _errors(md: MathDescription) -> list[str]:
    return [d.message for d in validate(md) if d.severity == "error"]


# ---------------------------------------------------------------------------
# Acceptance — the reference models validate cleanly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["section_1_4_5.yaml", "section_1_6_6.yaml", "section_2_7.yaml"])
def test_fixtures_validate_without_errors(name: str) -> None:
    assert validate(_load(name)) == []


def test_steady_state_motion_variable_may_have_ic() -> None:
    # §2.7: v_membrane is an unknown-motion variable governed by a steady_state
    # weak form, and it carries initial_condition "0". That is the documented
    # exception to "steady_state ⇒ no IC" (§1.7.7), so it must validate.
    md = _load("section_2_7.yaml")
    motion_eq = next(e for e in md.equations if e.variable == "v_membrane")
    assert motion_eq.temporality == "steady_state"
    assert motion_eq.initial_condition is not None
    assert validate(md) == []


# ---------------------------------------------------------------------------
# Reference-resolution and declaration rules (§1.11.3).
# ---------------------------------------------------------------------------


def test_reserved_name_as_parameter_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    # `r` is the radial geometric helper (§1.8.4) — reserved.
    bad = dataclasses.replace(md, parameters=[*md.parameters, ParameterConstant(name="r", value=1.0)])
    assert any("reserved name" in m for m in _errors(bad))


def test_parameter_shadowing_a_variable_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    bad = dataclasses.replace(md, parameters=[*md.parameters, ParameterConstant(name="rho_active", value=1.0)])
    assert any("shadows a variable" in m for m in _errors(bad))


def test_duplicate_variable_on_same_subdomain_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    dup = Variable(name="rho_active", subdomain="membrane")
    bad = dataclasses.replace(md, variables=[*md.variables, dup])
    assert any("duplicate variable" in m for m in _errors(bad))


def test_equation_on_undeclared_subdomain_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    broken = dataclasses.replace(eq, subdomain="nonexistent")
    bad = dataclasses.replace(md, equations=[broken, *md.equations[1:]])
    assert any("undeclared subdomain" in m for m in _errors(bad))


def test_unknown_motion_variable_must_be_vector() -> None:
    md = _load("section_2_7.yaml")
    # Retype v_membrane to scalar — a motion variable must be vector (§1.10.3).
    retyped = [dataclasses.replace(v, type="scalar") if v.name == "v_membrane" else v for v in md.variables]
    bad = dataclasses.replace(md, variables=retyped)
    assert any("must be vector-typed" in m for m in _errors(bad))


# ---------------------------------------------------------------------------
# Coverage and temporality (§1.11.4, §1.9.5).
# ---------------------------------------------------------------------------


def test_ungoverned_variable_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    orphan = Variable(name="rho_orphan", subdomain="membrane")
    bad = dataclasses.replace(md, variables=[*md.variables, orphan])
    assert any("undetermined" in m for m in _errors(bad))


def test_time_dependent_equation_requires_ic() -> None:
    md = _load("section_1_4_5.yaml")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    no_ic = dataclasses.replace(eq, initial_condition=None)
    bad = dataclasses.replace(md, equations=[no_ic, *md.equations[1:]])
    assert any("requires an initial_condition" in m for m in _errors(bad))


def test_steady_state_non_motion_equation_rejects_ic() -> None:
    md = _load("section_1_4_5.yaml")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    # Make it steady_state but keep its IC: not a motion variable, so illegal.
    bad_eq = dataclasses.replace(eq, temporality="steady_state")
    bad = dataclasses.replace(md, equations=[bad_eq, *md.equations[1:]])
    assert any("must not have an initial_condition" in m for m in _errors(bad))


# ---------------------------------------------------------------------------
# Template conformance (§1.4.2).
# ---------------------------------------------------------------------------


def test_unknown_template_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    bad_eq = dataclasses.replace(eq, template="does_not_exist")
    bad = dataclasses.replace(md, equations=[bad_eq, *md.equations[1:]])
    assert any("unknown template" in m for m in _errors(bad))


def test_unknown_slot_is_rejected() -> None:
    md = _load("section_1_4_5.yaml")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    bad_eq = dataclasses.replace(eq, terms={**eq.terms, "bogus_slot": "0"})
    bad = dataclasses.replace(md, equations=[bad_eq, *md.equations[1:]])
    assert any("unknown slot" in m for m in _errors(bad))


def test_wrong_subdomain_kind_for_template_is_rejected() -> None:
    md = _load("section_1_6_6.yaml")
    # Move the bulk_radv_diff equation for L onto the surface membrane.
    eq = next(e for e in md.equations if e.variable == "L")
    assert isinstance(eq, TemplateEquation)
    moved = dataclasses.replace(eq, subdomain="membrane")
    others = [e for e in md.equations if e.variable != "L"]
    bad = dataclasses.replace(md, equations=[moved, *others])
    assert any("requires subdomain kind" in m for m in _errors(bad))


def test_bulk_radv_diff_requires_diffusion_or_source() -> None:
    md = _load("section_1_6_6.yaml")
    eq = next(e for e in md.equations if e.variable == "L")
    assert isinstance(eq, TemplateEquation)
    empty = dataclasses.replace(eq, terms={})
    others = [e for e in md.equations if e.variable != "L"]
    bad = dataclasses.replace(md, equations=[empty, *others])
    assert any("at least one of" in m for m in _errors(bad))


# ---------------------------------------------------------------------------
# Boundary-condition consistency (§1.11.7).
# ---------------------------------------------------------------------------


def test_conflicting_bc_kinds_rejected() -> None:
    md = _load("section_1_6_6.yaml")
    # A Dirichlet on (L, outer) already exists; add a Neumann on the same pair.
    clash = BCNeumann(variable="L", boundary="outer", expression="0")
    bad = dataclasses.replace(md, boundary_conditions=[*md.boundary_conditions, clash])
    assert any("conflicting BC kinds" in m for m in _errors(bad))


def test_bc_on_undeclared_variable_rejected() -> None:
    md = _load("section_1_6_6.yaml")
    ghost = BCNeumann(variable="nope", boundary="outer", expression="0")
    bad = dataclasses.replace(md, boundary_conditions=[*md.boundary_conditions, ghost])
    assert any("undeclared variable" in m for m in _errors(bad))


# ---------------------------------------------------------------------------
# validate_or_raise.
# ---------------------------------------------------------------------------


def test_validate_or_raise_passes_clean_model() -> None:
    assert validate_or_raise(_load("section_1_4_5.yaml")) == []


def test_validate_or_raise_raises_on_error() -> None:
    md = _load("section_1_4_5.yaml")
    orphan = Variable(name="rho_orphan", subdomain="membrane")
    bad = dataclasses.replace(md, variables=[*md.variables, orphan])
    with pytest.raises(FormalismValidationError) as exc:
        validate_or_raise(bad)
    assert exc.value.errors


# ---------------------------------------------------------------------------
# Require-dilution warning (validation-and-diagnostics.md registry #1): a
# time-dependent scalar weak form on a moving subdomain that references no
# divergence operator likely forgot the dilution term ρ ∇_Γ·v_Γ.
# ---------------------------------------------------------------------------


def _moving_weak_form_model(form: str, *, moving: bool = True, vtype: str = "scalar") -> str:
    motion = '{ kind: prescribed, velocity: "rate * x / r(x)" }' if moving else "{ kind: none }"
    return f"""
math_description:
  geometry: disk_membrane
  subdomains:
    - {{ name: membrane, kind: surface, motion: {motion} }}
  variables:
    - {{ name: rho, subdomain: membrane, type: {vtype} }}
  parameters:
    - {{ name: rate, value: 1.0 }}
    - {{ name: D, value: 0.1 }}
  equations:
    - template: weak_form
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      form: "{form}"
      initial_condition: "0"
"""


def _warnings(model_yaml: str) -> list[str]:
    return [d.message for d in validate(load_yaml(model_yaml)) if d.severity == "warning"]


_DT_DIFF = "partial_t(rho) * rho_test + D * inner(grad_surf(rho), grad_surf(rho_test))"
_DILUTION = " + rho * div_surf(rate * x / r(x)) * rho_test"


def test_moving_density_weak_form_without_dilution_warns() -> None:
    messages = _warnings(_moving_weak_form_model(_DT_DIFF))
    assert len(messages) == 1
    assert "dilution" in messages[0] and "div_surf" in messages[0]


def test_moving_density_weak_form_with_dilution_is_quiet() -> None:
    assert _warnings(_moving_weak_form_model(_DT_DIFF + _DILUTION)) == []


def test_static_density_weak_form_is_quiet() -> None:
    # Nothing moves, so nothing dilutes — no warning even without a divergence operator.
    assert _warnings(_moving_weak_form_model(_DT_DIFF, moving=False)) == []


def test_moving_vector_weak_form_is_quiet() -> None:
    # A vector weak form is a momentum balance, not a co-moving density — no dilution expected.
    momentum = "partial_t(rho) * inner(rho, rho_test) + D * inner(grad_surf(rho), grad_surf(rho_test))"
    assert _warnings(_moving_weak_form_model(momentum, vtype="vector")) == []


def test_dilution_warning_does_not_block_validation() -> None:
    # The check is a warning, not an error — validate_or_raise still passes (returns the warning).
    warnings = validate_or_raise(load_yaml(_moving_weak_form_model(_DT_DIFF)))
    assert len(warnings) == 1 and warnings[0].severity == "warning"

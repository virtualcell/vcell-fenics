"""T9 `cell_force_balance` in the formalism (ADR 018, staging step 1): its schema, round trip, validator rules
and the runner's refusal.

The base model is ADR 018 decision 2's example, with a nucleus in the first cell: two cells (`cyt__c1` with
`nuc__c1` inside it, and `cyt__c2`) in a medium `ec`, their free membranes, the contact between them, and the
nuclear envelope. Each rule is tripped by one edit of that model's dict.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from vcell_fenics.formalism import dump_yaml, load_dict, load_yaml, validate
from vcell_fenics.formalism.loader import FormalismLoadError
from vcell_fenics.formalism.schema import MathDescription, TemplateEquation
from vcell_fenics.formalism.templates import CELL_FORCE_BALANCE, REGISTRY
from vcell_fenics.runner import RunError, _refuse_cell_force_balance

_BASE = """
math_description:
  geometry: two_cells
  subdomains:
    - { name: cyt__c1, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: nuc__c1, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: cyt__c2, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: ec, kind: volume }
    - { name: pm__c1, kind: surface, motion: { kind: unknown, variable: v } }
    - { name: pm__c2, kind: surface, motion: { kind: unknown, variable: v } }
    - { name: jn__c1_c2, kind: surface, motion: { kind: unknown, variable: v } }
    - { name: ne__c1, kind: surface, motion: { kind: unknown, variable: v } }
  variables:
    - { name: v, subdomain: [cyt__c1, nuc__c1, cyt__c2], type: vector }
  equations:
    - template: cell_force_balance
      variable: v
      subdomain: [cyt__c1, nuc__c1, cyt__c2]
      temporality: steady_state
      volumes:
        cyt__c1: { viscosity: eta, bulk_viscosity: eta_b, drag: zeta, pressure: P_c1 }
        nuc__c1: { viscosity: eta_n, pressure: P_c1 }
        cyt__c2: { viscosity: eta, drag: zeta, pressure: "K * (V0 - region_size(cyt__c2))" }
      surfaces:
        pm__c1: { tension: T_c1 }
        pm__c2: { tension: T_c2 }
        jn__c1_c2: { tension: "Lambda + T_c1 + T_c2" }
  parameters:
    - { name: eta, value: 1.0 }
    - { name: eta_n, value: 10.0 }
    - { name: eta_b, value: 0.5 }
    - { name: zeta, value: 0.1 }
    - { name: K, value: 2.0 }
    - { name: V0, value: 3.0 }
    - { name: K_S, value: 0.2 }
    - { name: S0, value: 6.0 }
    - { name: sigma, value: 1.0 }
    - { name: Lambda, value: -0.5 }
    - { name: P_c1, expression: "K * (V0 - region_size(cyt__c1) - region_size(nuc__c1))" }
    - { name: T_c1, expression: "sigma + K_S * (region_size(pm__c1) + region_size(jn__c1_c2) - S0)" }
    - { name: T_c2, expression: "sigma + K_S * (region_size(pm__c2) + region_size(jn__c1_c2) - S0)" }
"""


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_BASE)
    return raw


def _md(raw: dict[str, Any]) -> MathDescription:
    return load_dict(raw)


def _errors(raw: dict[str, Any]) -> list[str]:
    return [d.message for d in validate(_md(raw)) if d.severity == "error"]


def _equation(raw: dict[str, Any]) -> dict[str, Any]:
    eq: dict[str, Any] = raw["math_description"]["equations"][0]
    return eq


def _subdomain(raw: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in raw["math_description"]["subdomains"] if s["name"] == name)


def _assert_one_error(raw: dict[str, Any], fragment: str) -> None:
    errors = _errors(raw)
    assert any(fragment in e for e in errors), errors


# ---------------------------------------------------------------------------
# Acceptance, schema and round trip
# ---------------------------------------------------------------------------


def test_the_base_model_validates_without_diagnostics() -> None:
    assert validate(_md(_raw())) == []


def test_a_subdomain_list_loads_as_members_with_the_first_as_home() -> None:
    md = _md(_raw())
    (v,) = md.variables
    assert v.subdomain == "cyt__c1"
    assert v.members == ("cyt__c1", "nuc__c1", "cyt__c2")
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    assert eq.subdomain == "cyt__c1"
    assert eq.members == v.members
    assert eq.terms == {}
    assert eq.volumes["nuc__c1"] == {"viscosity": "eta_n", "pressure": "P_c1"}
    assert eq.surfaces["jn__c1_c2"] == {"tension": "Lambda + T_c1 + T_c2"}


def test_the_model_round_trips_through_yaml_with_its_list_and_slot_maps() -> None:
    md = _md(_raw())
    text = dump_yaml(md)
    assert load_yaml(text) == md
    dumped = yaml.safe_load(text)
    assert dumped["math_description"]["variables"][0]["subdomain"] == ["cyt__c1", "nuc__c1", "cyt__c2"]
    assert "terms" not in dumped["math_description"]["equations"][0]


def test_an_empty_subdomain_list_is_a_load_error() -> None:
    raw = _raw()
    raw["math_description"]["variables"][0]["subdomain"] = []
    with pytest.raises(FormalismLoadError, match="non-empty list"):
        _md(raw)


def test_a_single_member_force_balance_needs_no_list() -> None:
    # One cell: `subdomain: cyt` (not a list) is a T9 velocity on one member.
    raw = _raw()
    md = raw["math_description"]
    md["subdomains"] = [s for s in md["subdomains"] if s["name"] in ("cyt__c1", "ec", "pm__c1")]
    md["parameters"] = [p for p in md["parameters"] if "value" in p]  # the others read the removed subdomains
    md["variables"][0]["subdomain"] = "cyt__c1"
    eq = _equation(raw)
    eq["subdomain"] = "cyt__c1"
    eq["volumes"] = {"cyt__c1": {"viscosity": "eta", "drag": "zeta", "pressure": "K * (V0 - region_size(cyt__c1))"}}
    eq["surfaces"] = {"pm__c1": {"tension": "sigma"}}
    assert _errors(raw) == []


def test_the_registry_spec_is_steady_vector_on_volumes_with_slot_maps() -> None:
    spec = REGISTRY[CELL_FORCE_BALANCE]
    assert spec.slot_maps
    assert spec.temporalities == frozenset({"steady_state"})
    assert spec.governed_types == frozenset({"vector"})
    assert [s.name for s in spec.slots] == ["viscosity", "bulk_viscosity", "drag", "pressure", "active_stress"]
    assert [s.name for s in spec.surface_slots] == ["tension", "traction"]
    assert [s.name for s in spec.slots if s.required] == ["viscosity"]


# ---------------------------------------------------------------------------
# The listed variable (decision 2)
# ---------------------------------------------------------------------------


def test_a_listed_variable_must_be_a_vector() -> None:
    raw = _raw()
    raw["math_description"]["variables"][0]["type"] = "scalar"
    _assert_one_error(raw, "only a vector governed by 'cell_force_balance' may live on a list")


def test_a_listed_variable_members_are_volumes() -> None:
    raw = _raw()
    raw["math_description"]["variables"][0]["subdomain"] = ["cyt__c1", "nuc__c1", "cyt__c2", "pm__c1"]
    _assert_one_error(raw, "member 'pm__c1' is a surface")


def test_a_listed_variable_names_each_member_once() -> None:
    raw = _raw()
    raw["math_description"]["variables"][0]["subdomain"] = ["cyt__c1", "cyt__c1", "nuc__c1", "cyt__c2"]
    _assert_one_error(raw, "names a member more than once")


def test_a_listed_variable_is_governed_only_by_t9() -> None:
    raw = _raw()
    eq = _equation(raw)
    eq["template"] = "bulk_radv_diff"
    _assert_one_error(raw, "only the 'cell_force_balance' template governs such a variable")
    _assert_one_error(raw, "for the 'cell_force_balance' template only")


def test_the_equation_must_cover_exactly_the_variables_members() -> None:
    raw = _raw()
    _equation(raw)["subdomain"] = ["cyt__c1", "cyt__c2", "nuc__c1"]  # order is free
    assert _errors(raw) == []
    _equation(raw)["subdomain"] = ["cyt__c1", "cyt__c2"]
    _assert_one_error(raw, "must be the members of 'v'")


def test_another_variable_of_the_same_name_on_a_member_is_a_duplicate() -> None:
    raw = _raw()
    raw["math_description"]["variables"].append({"name": "v", "subdomain": "nuc__c1", "type": "vector"})
    _assert_one_error(raw, "duplicate variable 'v' on subdomain 'nuc__c1'")


def test_the_velocity_resolves_by_bare_name_on_every_member() -> None:
    # A species on the nucleus may read the velocity (here, a drift along it): `v` is in scope on every member.
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "c", "subdomain": "nuc__c1"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "nuc__c1",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1", "relative_advection": "0.1 * v"},
            "initial_condition": "1",
        }
    )
    assert _errors(raw) == []


# ---------------------------------------------------------------------------
# Motion, both ways (decision 2)
# ---------------------------------------------------------------------------


def test_every_member_must_move_with_the_velocity() -> None:
    raw = _raw()
    _subdomain(raw, "nuc__c1")["motion"] = {"kind": "none"}
    _assert_one_error(raw, "member 'nuc__c1' must have motion {kind: unknown, variable: v}")


def test_every_surface_with_a_load_must_move_with_the_velocity_or_be_a_growing_front() -> None:
    raw = _raw()
    del _subdomain(raw, "jn__c1_c2")["motion"]
    _assert_one_error(raw, "surface 'jn__c1_c2' carries a load, so it must have motion")


def test_a_surface_without_a_tension_may_move_with_the_velocity() -> None:
    # ne__c1 (the nuclear envelope) carries no tension and moves with v: it bounds a member.
    assert "ne__c1" not in _equation(_raw())["surfaces"]
    assert _errors(_raw()) == []


def test_a_volume_outside_the_members_cannot_move_with_the_velocity() -> None:
    raw = _raw()
    _subdomain(raw, "ec")["motion"] = {"kind": "unknown", "variable": "v"}
    _assert_one_error(raw, "'ec' (volume) moves with 'v' but is not one of its members")


# ---------------------------------------------------------------------------
# The slot maps (decisions 2 and 3)
# ---------------------------------------------------------------------------


def test_every_member_needs_an_entry_with_a_viscosity() -> None:
    raw = _raw()
    del _equation(raw)["volumes"]["nuc__c1"]
    _assert_one_error(raw, "member 'nuc__c1' has no entry")
    raw = _raw()
    del _equation(raw)["volumes"]["nuc__c1"]["viscosity"]
    _assert_one_error(raw, "requires slot 'viscosity'")


def test_a_volume_entry_must_name_a_member() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["ec"] = {"viscosity": "eta"}
    _assert_one_error(raw, "'ec' is not one of the equation's members")


def test_unknown_slots_are_refused_in_both_maps() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt__c1"]["tension"] = "sigma"
    _equation(raw)["surfaces"]["pm__c1"]["drag"] = "zeta"
    errors = _errors(raw)
    assert any("unknown volume slot 'tension'" in e for e in errors), errors
    assert any("unknown surface slot 'drag'" in e for e in errors), errors


def test_surfaces_must_be_declared_surface_classes() -> None:
    raw = _raw()
    _equation(raw)["surfaces"]["nowhere"] = {"tension": "sigma"}
    _assert_one_error(raw, "'nowhere' is not a declared subdomain")
    raw = _raw()
    _equation(raw)["surfaces"]["ec"] = {"tension": "sigma"}
    _assert_one_error(raw, "'ec' is a volume; 'surfaces:' takes surface classes")


def test_t9_takes_no_terms_map() -> None:
    raw = _raw()
    _equation(raw)["terms"] = {"viscosity": "eta"}
    _assert_one_error(raw, "takes its slots in 'volumes:'")


def test_t9_is_steady_state_only() -> None:
    raw = _raw()
    _equation(raw)["temporality"] = "time_dependent"
    _assert_one_error(raw, "does not support temporality 'time_dependent'")


def test_t9_refuses_an_initial_condition() -> None:
    raw = _raw()
    _equation(raw)["initial_condition"] = "0"
    _assert_one_error(raw, "takes no initial_condition")


def test_slot_expressions_are_type_checked_where_they_are_evaluated() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt__c1"]["drag"] = "[zeta, zeta]"
    _assert_one_error(raw, "expression has type vector, expected scalar")


def test_no_slot_may_read_the_velocity() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt__c1"]["viscosity"] = "eta * (1 + inner(v, v))"
    _assert_one_error(raw, "may not read the velocity 'v'")
    raw = _raw()
    _equation(raw)["surfaces"]["pm__c1"]["tension"] = "T_c1 + inner(trace(v), trace(v))"
    _assert_one_error(raw, "may not read the velocity 'v'")


def test_a_tension_may_read_a_surface_species() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "myosin", "subdomain": "pm__c1"})
    md["equations"].append(
        {
            "template": "surface_pde_with_dilution",
            "variable": "myosin",
            "subdomain": "pm__c1",
            "temporality": "time_dependent",
            "terms": {"diffusion": "0.1"},
            "initial_condition": "1",
        }
    )
    _equation(raw)["surfaces"]["pm__c1"]["tension"] = "T_c1 * (1 + myosin)"
    assert _errors(raw) == []


# ---------------------------------------------------------------------------
# A uniform pressure (decision 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pressure", "culprit"),
    [
        ("K * geom.x[0]", "geom.x"),
        ("P_field", "P_field (through geom.x)"),
        ("K * c", "c"),
    ],
)
def test_a_pressure_that_can_vary_in_space_is_refused(pressure: str, culprit: str) -> None:
    raw = _raw()
    md = raw["math_description"]
    md["parameters"].append({"name": "P_field", "expression": "K * geom.x[0]"})
    md["variables"].append({"name": "c", "subdomain": "cyt__c2"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "cyt__c2",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    _equation(raw)["volumes"]["cyt__c2"]["pressure"] = pressure
    errors = _errors(raw)
    assert any("NonUniformPressure" in e and culprit in e for e in errors), errors


def test_a_pressure_may_read_time_sizes_parameters_and_a_region_variable() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "atp", "subdomain": "cyt__c2", "space": "region"})
    md["equations"].append(
        {
            "template": "region_ode",
            "variable": "atp",
            "subdomain": "cyt__c2",
            "temporality": "time_dependent",
            "terms": {"uniform_rate": "-0.1 * atp"},
            "initial_condition": "1",
        }
    )
    _equation(raw)["volumes"]["cyt__c2"]["pressure"] = "P_c1 * (1 + sim.t) + K * atp"
    assert _errors(raw) == []


# ---------------------------------------------------------------------------
# Well-posedness and the wall (decision 4)
# ---------------------------------------------------------------------------


def _without_drag(raw: dict[str, Any]) -> dict[str, Any]:
    for slots in _equation(raw)["volumes"].values():
        slots.pop("drag", None)
    return raw


def test_rigid_motion_that_nothing_resists_is_refused() -> None:
    _assert_one_error(_without_drag(_raw()), "UnconstrainedRigidMotion")


def test_a_wall_holding_the_velocity_resists_rigid_motion() -> None:
    raw = _without_drag(_raw())
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "left_wall", "expression": "0"}
    ]
    assert _errors(raw) == []


def test_a_wall_must_hold_the_velocity_at_zero() -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "left_wall", "expression": "[1, 0]"}
    ]
    _assert_one_error(raw, "must be the zero vector")


def test_the_velocity_takes_no_other_bc() -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "neumann", "variable": "v", "boundary": "pm__c1", "expression": "0"}
    ]
    _assert_one_error(raw, "takes only a 'dirichlet' BC")


# ---------------------------------------------------------------------------
# The runner refuses T9 until a solver assembles it
# ---------------------------------------------------------------------------


def test_the_runner_refuses_a_t9_model_by_name() -> None:
    with pytest.raises(RunError, match=r"'cell_force_balance' template \(T9, ADR 018\), governing 'v'"):
        _refuse_cell_force_balance(_md(_raw()))


def test_the_runner_lets_other_models_through() -> None:
    raw = copy.deepcopy(_raw())
    md = raw["math_description"]
    md["subdomains"] = [{"name": "cyt", "kind": "volume"}]
    md["variables"] = [{"name": "c", "subdomain": "cyt"}]
    md["equations"] = [
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "cyt",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    ]
    md["parameters"] = []
    _refuse_cell_force_balance(_md(raw))


# ---------------------------------------------------------------------------
# The code review's findings (each a model the first draft mishandled)
# ---------------------------------------------------------------------------


def test_an_equation_list_in_another_order_is_covered() -> None:
    raw = _raw()
    _equation(raw)["subdomain"] = ["nuc__c1", "cyt__c1", "cyt__c2"]
    assert _errors(raw) == []


def test_two_force_balances_for_one_velocity_are_overdetermined() -> None:
    raw = _raw()
    second = copy.deepcopy(_equation(raw))
    second["subdomain"] = ["nuc__c1", "cyt__c1", "cyt__c2"]
    raw["math_description"]["equations"].append(second)
    _assert_one_error(raw, "governed by 2 equations (overdetermined)")


def test_the_velocitys_name_is_unique_in_the_model() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "v", "subdomain": "ec"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "v",
            "subdomain": "ec",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    _assert_one_error(raw, "its name is unique in the MathDescription")


def test_members_are_written_only_as_a_subdomain_list() -> None:
    raw = _raw()
    eq = _equation(raw)
    eq["members"] = eq.pop("subdomain")
    eq["subdomain"] = "nuc__c1"
    with pytest.raises(FormalismLoadError, match="'members' is not written directly"):
        _md(raw)


def test_an_equation_list_names_each_member_once() -> None:
    raw = _raw()
    _equation(raw)["subdomain"] = ["cyt__c1", "nuc__c1", "cyt__c2", "cyt__c2"]
    _assert_one_error(raw, "the equation names a member more than once")


def test_a_tension_may_not_read_geometry() -> None:
    raw = _raw()
    _equation(raw)["surfaces"]["pm__c1"]["tension"] = "T_c1 * (1 + geom.x[0])"
    _assert_one_error(raw, "not geometry; it reads ['geom.x']")


def test_a_membrane_held_by_a_velocity_resists_rigid_motion_and_carries_no_load() -> None:
    raw = _without_drag(_raw())
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "pm__c1", "expression": "0"}
    ]
    errors = _errors(raw)
    assert any("'pm__c1' holds 'v' by a 'dirichlet' condition" in e and "['tension']" in e for e in errors), errors
    assert not any("UnconstrainedRigidMotion" in e for e in errors), errors
    del _equation(raw)["surfaces"]["pm__c1"]
    assert _errors(raw) == []


def test_a_wall_may_be_written_as_a_zero_vector() -> None:
    raw = _without_drag(_raw())
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "left_wall", "expression": "[0, 0]"}
    ]
    assert _errors(raw) == []


def test_members_given_as_a_python_list_compare_equal_to_a_loaded_model() -> None:
    from vcell_fenics.formalism.schema import Variable

    built = Variable(name="v", subdomain="cyt__c1", type="vector", members=["cyt__c1", "nuc__c1", "cyt__c2"])  # type: ignore[arg-type]
    assert built == _md(_raw()).variables[0]
    hash(built)


# ---------------------------------------------------------------------------
# ADR 019: a single cell (active stress, viscous operator, membrane conditions, a growing front)
# ---------------------------------------------------------------------------

# Nickaeen et al. 2017's motile cell (ADR 019's UC-A), zero-stress variant, mechanics only: one member, the
# vector Laplacian, an isotropic active stress, a drag, no tension, and a growing front.
_ONE_CELL = """
math_description:
  geometry: one_cell
  subdomains:
    - { name: cyt, kind: volume, motion: { kind: unknown, variable: u } }
    - { name: ec, kind: volume }
    - name: pm
      kind: surface
      motion:
        kind: prescribed
        velocity: >-
          trace(u) + (v0 * a0 / region_size(cyt)
          - k * (region_size(cyt) - a0 * a0 * a0 / (region_size(cyt) * region_size(cyt)))) * geom.normal
  variables:
    - { name: u, subdomain: cyt, type: vector }
  equations:
    - template: cell_force_balance
      variable: u
      subdomain: cyt
      temporality: steady_state
      viscous_operator: laplacian
      volumes:
        cyt: { viscosity: alpha, drag: "1", active_stress: "beta * (1 + 0.1 * geom.x[0])" }
  parameters:
    - { name: alpha, value: 0.5 }
    - { name: beta, value: 5.0 }
    - { name: v0, value: 2.5 }
    - { name: a0, value: 3.14159 }
    - { name: k, value: 1.5 }
"""


def _one_cell() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_ONE_CELL)
    return raw


def test_a_single_motile_cell_validates_without_diagnostics() -> None:
    assert validate(_md(_one_cell())) == []


def test_the_viscous_operator_round_trips_and_defaults_to_omitted() -> None:
    md = _md(_one_cell())
    eq = md.equations[0]
    assert isinstance(eq, TemplateEquation)
    assert eq.viscous_operator == "laplacian"
    assert load_yaml(dump_yaml(md)) == md
    base = _md(_raw()).equations[0]
    assert isinstance(base, TemplateEquation)
    assert base.viscous_operator is None
    assert "viscous_operator" not in yaml.safe_load(dump_yaml(_md(_raw())))["math_description"]["equations"][0]


def test_an_unknown_viscous_operator_is_a_load_error() -> None:
    raw = _one_cell()
    _equation(raw)["viscous_operator"] = "newtonian"
    with pytest.raises(FormalismLoadError):
        _md(raw)


def test_the_viscous_operator_is_for_t9_only() -> None:
    raw = _raw()
    raw["math_description"]["variables"].append({"name": "c", "subdomain": "ec", "type": "scalar"})
    raw["math_description"]["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "ec",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "0",
            "viscous_operator": "laplacian",
        }
    )
    _assert_one_error(raw, "'viscous_operator' is for the 'cell_force_balance' template only")


def test_an_active_stress_may_vary_in_space_but_not_read_the_velocity() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt__c1"]["active_stress"] = "sigma * geom.x[0]"
    assert _errors(raw) == []
    _equation(raw)["volumes"]["cyt__c1"]["active_stress"] = "div(v)"
    _assert_one_error(raw, "may not read the velocity 'v'")


def test_a_nonuniform_pressure_points_to_the_active_stress() -> None:
    raw = _raw()
    _equation(raw)["volumes"]["cyt__c2"]["pressure"] = "geom.x[0]"
    _assert_one_error(raw, "A stress that varies in space is an 'active_stress'")


def test_a_free_membrane_may_omit_its_tension() -> None:
    raw = _raw()
    del _equation(raw)["surfaces"]["pm__c2"]
    assert _errors(raw) == []


def test_a_traction_is_a_vector_load_that_may_read_geometry() -> None:
    raw = _raw()
    _equation(raw)["surfaces"]["pm__c1"]["traction"] = "f0 * exp(-geom.x[0] * geom.x[0]) * geom.normal"
    raw["math_description"]["parameters"].append({"name": "f0", "value": 0.3})
    assert _errors(raw) == []
    _equation(raw)["surfaces"]["pm__c1"]["traction"] = "f0"
    assert _errors(raw) == ["expression has type scalar, expected vector (§1.11.5)"]
    _equation(raw)["surfaces"]["pm__c1"]["traction"] = "trace(v)"
    _assert_one_error(raw, "may not read the velocity 'v'")


def test_a_growing_front_may_carry_a_traction_but_not_a_tension() -> None:
    raw = _one_cell()
    _equation(raw)["surfaces"] = {"pm": {"traction": "[0.1, 0]"}}
    assert _errors(raw) == []
    _equation(raw)["surfaces"] = {"pm": {"tension": "1"}}
    _assert_one_error(raw, "'pm' is a growing front (a prescribed velocity) and may not carry a tension")


def test_a_surface_with_a_prescribed_displacement_is_not_a_growing_front() -> None:
    raw = _one_cell()
    _subdomain(raw, "pm")["motion"] = {"kind": "prescribed", "displacement": "[0, 0]"}
    _equation(raw)["surfaces"] = {"pm": {"traction": "[0.1, 0]"}}
    _assert_one_error(raw, "surface 'pm' carries a load, so it must have motion")


def test_a_zero_velocity_edge_on_a_growing_front_holds_the_cell() -> None:
    raw = _one_cell()
    _equation(raw)["volumes"]["cyt"].pop("drag")
    _assert_one_error(raw, "UnconstrainedRigidMotion")
    _subdomain(raw, "pm")["motion"]["velocity"] = (
        "(v0 * a0 / region_size(cyt) - k * (region_size(cyt) - a0)) * geom.normal"
    )
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "0"}
    ]
    assert _errors(raw) == []


def test_a_membranes_given_velocity_may_be_a_field_of_position() -> None:
    raw = _one_cell()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "[0.1 * geom.x[1], 0]"}
    ]
    assert _errors(raw) == []


def test_a_membranes_given_velocity_must_be_a_vector() -> None:
    raw = _one_cell()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "0.1"}
    ]
    assert _errors(raw) == ["expression has type scalar, expected vector (§1.11.5)"]


def test_a_membranes_given_velocity_may_not_read_the_velocity_or_a_field_species() -> None:
    raw = _one_cell()
    md = raw["math_description"]
    md["variables"].append({"name": "m", "subdomain": "cyt", "type": "scalar"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "m",
            "subdomain": "cyt",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    md["boundary_conditions"] = [{"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "trace(u)"}]
    assert _errors(raw) == [
        "a membrane's given velocity may not read the velocity 'u' itself",
    ]
    md["boundary_conditions"][0]["expression"] = "trace(m) * geom.normal"
    assert _errors(raw) == [
        "a membrane's given velocity may read constants, sim.t, geom.*, region variables and region_size, not a "
        "field variable; it reads ['m'] (ADR 019 decision 4)",
    ]


def test_a_velocity_is_given_on_a_membrane_or_a_box_face_not_a_volume() -> None:
    raw = _one_cell()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "ec", "expression": "0"}
    ]
    _assert_one_error(raw, "'ec' is a volume")


def test_a_membranes_given_velocity_may_read_region_quantities_and_time() -> None:
    raw = _one_cell()
    md = raw["math_description"]
    md["variables"].append({"name": "R", "subdomain": "cyt", "type": "scalar", "space": "region"})
    md["equations"].append(
        {
            "template": "region_ode",
            "variable": "R",
            "subdomain": "cyt",
            "temporality": "time_dependent",
            "terms": {"uniform_rate": "0"},
            "initial_condition": "1",
        }
    )
    md["boundary_conditions"] = [
        {
            "kind": "dirichlet",
            "variable": "u",
            "boundary": "pm",
            "expression": "trace(R) * sim.t / region_size(cyt) * geom.normal",
        }
    ]
    assert _errors(raw) == []


def test_a_membrane_held_by_a_velocity_carries_no_traction() -> None:
    raw = _one_cell()
    _equation(raw)["surfaces"] = {"pm": {"traction": "[0.1, 0]"}}
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "0"}
    ]
    assert _errors(raw) == [
        "'pm' holds 'u' by a 'dirichlet' condition, which gives its velocity, so it takes no ['traction'] "
        "(ADR 019 decision 4)"
    ]


def test_two_velocities_on_one_membrane_are_refused() -> None:
    raw = _one_cell()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "0"},
        {"kind": "dirichlet", "variable": "u", "boundary": "pm", "expression": "[1, 0]"},
    ]
    _assert_one_error(raw, "a second BC for 'u' on 'pm'")


def test_a_listed_surface_at_rest_is_neither_moving_with_the_velocity_nor_a_growing_front() -> None:
    raw = _one_cell()
    _subdomain(raw, "pm")["motion"] = {"kind": "none"}
    _equation(raw)["surfaces"] = {"pm": {"traction": "[0.1, 0]"}}
    assert _errors(raw) == [
        "surface 'pm' carries a load, so it must have motion {kind: unknown, variable: u} or be a growing front "
        "with a prescribed velocity (ADR 018 decision 2, ADR 019 decision 5)"
    ]


def test_a_volume_in_surfaces_is_reported_once() -> None:
    raw = _raw()
    _equation(raw)["surfaces"]["ec"] = {"tension": "1"}
    errors = _errors(raw)
    assert [e for e in errors if "'ec'" in e] == ["'ec' is a volume; 'surfaces:' takes surface classes"], errors


def test_an_explicit_symmetric_operator_round_trips() -> None:
    raw = _raw()
    _equation(raw)["viscous_operator"] = "symmetric"
    md = _md(raw)
    assert load_yaml(dump_yaml(md)) == md
    assert yaml.safe_load(dump_yaml(md))["math_description"]["equations"][0]["viscous_operator"] == "symmetric"
    assert validate(md) == []


def test_a_misspelt_membrane_is_read_as_a_box_face() -> None:
    raw = _one_cell()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "u", "boundary": "pmm", "expression": "[1, 0]"}
    ]
    _assert_one_error(raw, "'pmm' is not a declared subdomain, so it is read as a box face")

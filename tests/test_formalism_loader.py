"""Tests for the YAML / JSON loader and dumper.

Three layers of coverage:

1. **Fidelity** — each worked-example YAML in tests/fixtures/ loads to the
   same dataclass tree as the programmatic construction in
   test_formalism_schema.py.
2. **Round-trip** — every MathDescription survives YAML → dict → dataclass
   → dict → YAML → dict → dataclass identity. JSON path is verified for
   one of the three.
3. **Error reporting** — malformed inputs raise FormalismLoadError with a
   path string pointing at the offending field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vcell_fenics.formalism import (
    BCDirichlet,
    BCInterfaceFluxBalance,
    BCNeumann,
    FormalismLoadError,
    MathDescription,
    MotionPrescribedVelocity,
    MotionUnknown,
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
    dump_json,
    dump_yaml,
    load_dict,
    load_json,
    load_yaml,
    to_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fidelity — each fixture loads to the structure the schema tests build.
# ---------------------------------------------------------------------------


def test_loads_section_1_4_5_two_species_membrane() -> None:
    md = load_yaml(FIXTURES / "section_1_4_5.yaml")
    assert md.geometry == "disk_radius_1"
    assert len(md.subdomains) == 1
    assert md.subdomains[0].name == "membrane"
    assert isinstance(md.subdomains[0].motion, MotionPrescribedVelocity)
    assert md.subdomains[0].motion.velocity == "r_dot * geom.x / geom.radius"
    assert [v.name for v in md.variables] == ["rho_active", "rho_inactive"]
    # All variables default to scalar / lagrange_p1.
    assert all(v.type == "scalar" and v.space == "lagrange_p1" for v in md.variables)
    assert len(md.equations) == 2
    eq0 = md.equations[0]
    assert isinstance(eq0, TemplateEquation)
    assert eq0.template == "surface_pde_with_dilution"
    assert eq0.terms["source"] == "k_on * rho_inactive - k_off * rho_active"
    assert md.boundary_conditions == []
    assert {p.name for p in md.parameters} == {"k_on", "k_off", "r_dot"}


def test_loads_section_1_6_6_ligand_receptor() -> None:
    md = load_yaml(FIXTURES / "section_1_6_6.yaml")
    assert md.geometry == "cell_with_extracellular"
    assert {s.name for s in md.subdomains} == {"extracellular", "membrane"}
    assert len(md.equations) == 3
    assert len(md.boundary_conditions) == 2
    # First BC is Dirichlet on the outer boundary.
    bc0 = md.boundary_conditions[0]
    assert isinstance(bc0, BCDirichlet)
    assert bc0.variable == "L"
    assert bc0.boundary == "outer"
    assert bc0.expression == "L_reservoir"
    # Second BC is the composable-pattern Neumann on the membrane.
    bc1 = md.boundary_conditions[1]
    assert isinstance(bc1, BCNeumann)
    assert bc1.boundary == "membrane"
    assert "trace(L)" in bc1.expression


def test_loads_section_2_7_end_to_end() -> None:
    md = load_yaml(FIXTURES / "section_2_7.yaml")
    # Unknown motion routed to a vector variable.
    assert isinstance(md.subdomains[0].motion, MotionUnknown)
    assert md.subdomains[0].motion.variable == "v_membrane"
    # Vector-typed variable preserves its declared type.
    v_membrane = next(v for v in md.variables if v.name == "v_membrane")
    assert v_membrane.type == "vector"
    # Mixed-temporality system.
    motion_eq = md.equations[0]
    assert isinstance(motion_eq, WeakFormEquation)
    assert motion_eq.temporality == "steady_state"
    assert "inner(f_active, v_membrane_test)" in motion_eq.form
    receptor_eq = md.equations[1]
    assert isinstance(receptor_eq, TemplateEquation)
    assert receptor_eq.temporality == "time_dependent"
    # Expression-valued parameter with subdomain scope is preserved.
    f_active = next(p for p in md.parameters if p.name == "f_active")
    assert isinstance(f_active, ParameterExpression)
    assert f_active.type == "vector"
    assert f_active.subdomain == "membrane"
    assert f_active.expression == "[f0 * cos(geom.azimuth), 0]"


# ---------------------------------------------------------------------------
# Round-trip identity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    ["section_1_4_5.yaml", "section_1_6_6.yaml", "section_2_7.yaml"],
)
def test_yaml_round_trip_is_identity(fixture: str) -> None:
    original = load_yaml(FIXTURES / fixture)
    text = dump_yaml(original)
    reloaded = load_yaml(text)
    assert reloaded == original


def test_json_round_trip_is_identity() -> None:
    original = load_yaml(FIXTURES / "section_2_7.yaml")
    text = dump_json(original)
    reloaded = load_json(text)
    assert reloaded == original


def test_to_dict_round_trip_is_identity() -> None:
    original = load_yaml(FIXTURES / "section_1_6_6.yaml")
    reloaded = load_dict(to_dict(original))
    assert reloaded == original


def test_dump_yaml_to_file(tmp_path: Path) -> None:
    original = load_yaml(FIXTURES / "section_1_4_5.yaml")
    out = tmp_path / "round_trip.yaml"
    dump_yaml(original, out)
    assert out.is_file()
    assert load_yaml(out) == original


# ---------------------------------------------------------------------------
# Defaults omitted on dump.
# ---------------------------------------------------------------------------


def test_dump_omits_default_motion_none() -> None:
    md = MathDescription(
        geometry="g",
        subdomains=[
            # No motion field → defaults to MotionNone, should round-trip
            # without an explicit `motion:` block.
            Subdomain(name="s", kind="volume")
        ],
        variables=[Variable(name="u", subdomain="s")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="s",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0",
            )
        ],
    )
    out = to_dict(md)
    body = out["math_description"]
    # Motion default omitted.
    assert "motion" not in body["subdomains"][0]
    # Variable type / space defaults omitted.
    assert body["variables"][0] == {"name": "u", "subdomain": "s"}
    # Empty parameters / BC lists omitted.
    assert "parameters" not in body
    assert "boundary_conditions" not in body


def test_dump_constant_parameter_uses_shorthand() -> None:
    p = ParameterConstant(name="k", value=0.5)
    md = MathDescription(
        geometry="g",
        subdomains=[Subdomain(name="s", kind="volume")],
        variables=[Variable(name="u", subdomain="s")],
        parameters=[p],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="s",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0",
            )
        ],
    )
    out = to_dict(md)
    # ParameterConstant emits {name, value}; no explicit kind field.
    assert out["math_description"]["parameters"][0] == {"name": "k", "value": 0.5}


# ---------------------------------------------------------------------------
# Error reporting — path-bearing FormalismLoadError on malformed input.
# ---------------------------------------------------------------------------


def test_error_missing_envelope() -> None:
    with pytest.raises(FormalismLoadError) as exc:
        load_dict({"not_the_envelope": {}})
    assert "math_description" in str(exc.value)


def test_error_missing_required_top_level_field() -> None:
    with pytest.raises(FormalismLoadError) as exc:
        load_dict({"math_description": {"geometry": "g"}})
    assert exc.value.path == "math_description"
    assert "missing required" in exc.value.message


def test_error_unknown_subdomain_kind() -> None:
    raw = {
        "math_description": {
            "geometry": "g",
            "subdomains": [{"name": "s", "kind": "bulk"}],  # 'bulk' is not a kind
            "variables": [{"name": "u", "subdomain": "s"}],
            "equations": [
                {
                    "template": "bulk_radv_diff",
                    "variable": "u",
                    "subdomain": "s",
                    "temporality": "time_dependent",
                    "terms": {"diffusion": "1.0"},
                    "initial_condition": "0",
                }
            ],
        }
    }
    with pytest.raises(FormalismLoadError) as exc:
        load_dict(raw)
    assert exc.value.path == "math_description.subdomains[0].kind"
    assert "bulk" in exc.value.message


def test_error_motion_prescribed_with_both_velocity_and_displacement() -> None:
    raw = {
        "math_description": {
            "geometry": "g",
            "subdomains": [
                {
                    "name": "m",
                    "kind": "surface",
                    "motion": {
                        "kind": "prescribed",
                        "velocity": "0",
                        "displacement": "0",
                    },
                }
            ],
            "variables": [{"name": "rho", "subdomain": "m"}],
            "equations": [
                {
                    "template": "surface_pde_with_dilution",
                    "variable": "rho",
                    "subdomain": "m",
                    "temporality": "time_dependent",
                    "terms": {"diffusion": "0.1"},
                    "initial_condition": "1.0",
                }
            ],
        }
    }
    with pytest.raises(FormalismLoadError) as exc:
        load_dict(raw)
    assert exc.value.path == "math_description.subdomains[0].motion"
    assert "exactly one" in exc.value.message


def test_error_parameter_with_both_value_and_expression() -> None:
    raw = {
        "math_description": {
            "geometry": "g",
            "subdomains": [{"name": "s", "kind": "volume"}],
            "variables": [{"name": "u", "subdomain": "s"}],
            "parameters": [{"name": "p", "value": 1.0, "expression": "t"}],
            "equations": [
                {
                    "template": "bulk_radv_diff",
                    "variable": "u",
                    "subdomain": "s",
                    "temporality": "time_dependent",
                    "terms": {"diffusion": "p"},
                    "initial_condition": "0",
                }
            ],
        }
    }
    with pytest.raises(FormalismLoadError) as exc:
        load_dict(raw)
    assert exc.value.path == "math_description.parameters[0]"
    assert "mutually exclusive" in exc.value.message


def test_error_unknown_field_in_variable() -> None:
    raw = {
        "math_description": {
            "geometry": "g",
            "subdomains": [{"name": "s", "kind": "volume"}],
            "variables": [{"name": "u", "subdomain": "s", "color": "red"}],
            "equations": [
                {
                    "template": "bulk_radv_diff",
                    "variable": "u",
                    "subdomain": "s",
                    "temporality": "time_dependent",
                    "terms": {"diffusion": "1"},
                    "initial_condition": "0",
                }
            ],
        }
    }
    with pytest.raises(FormalismLoadError) as exc:
        load_dict(raw)
    assert exc.value.path == "math_description.variables[0]"
    assert "color" in exc.value.message


def test_error_neumann_bc_missing_expression() -> None:
    raw = {
        "math_description": {
            "geometry": "g",
            "subdomains": [{"name": "s", "kind": "volume"}],
            "variables": [{"name": "u", "subdomain": "s"}],
            "boundary_conditions": [{"kind": "neumann", "variable": "u", "boundary": "outer"}],
            "equations": [
                {
                    "template": "bulk_radv_diff",
                    "variable": "u",
                    "subdomain": "s",
                    "temporality": "time_dependent",
                    "terms": {"diffusion": "1"},
                    "initial_condition": "0",
                }
            ],
        }
    }
    with pytest.raises(FormalismLoadError) as exc:
        load_dict(raw)
    assert exc.value.path == "math_description.boundary_conditions[0]"
    assert "expression" in exc.value.message


# ---------------------------------------------------------------------------
# load_yaml accepts both file paths and inline YAML text.
# ---------------------------------------------------------------------------


def test_load_yaml_accepts_inline_text() -> None:
    text = (FIXTURES / "section_1_4_5.yaml").read_text(encoding="utf-8")
    md = load_yaml(text)
    assert md.geometry == "disk_radius_1"


def test_load_yaml_treats_singleline_existing_path_as_file() -> None:
    # The str-vs-path heuristic: a single-line string that names an existing
    # file resolves as a file; otherwise as inline text.
    p = str(FIXTURES / "section_1_4_5.yaml")
    md = load_yaml(p)
    assert md.geometry == "disk_radius_1"


# ---------------------------------------------------------------------------
# §1.6.6 expression-parameter usage (the time-varying-reservoir example
# from the review pass) round-trips correctly even though it's not in the
# fixture file directly — built programmatically.
# ---------------------------------------------------------------------------


def test_expression_parameter_no_subdomain_round_trips() -> None:
    # `L_reservoir = "1.0 + 0.5 * sin(omega * sim.t)"` from the §1.6.6 example
    # text. No geometric helpers → no subdomain scope required.
    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="extracellular", kind="volume")],
        variables=[Variable(name="L", subdomain="extracellular")],
        parameters=[
            ParameterExpression(name="omega", expression="2 * 3.14159"),
            ParameterExpression(name="L_reservoir", expression="1.0 + 0.5 * sin(omega * sim.t)"),
        ],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L",
                subdomain="extracellular",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="0",
            )
        ],
        boundary_conditions=[BCDirichlet(variable="L", boundary="outer", expression="L_reservoir")],
    )
    reloaded = load_yaml(dump_yaml(md))
    assert reloaded == md
    # And the dumper omits the subdomain field when it's None.
    dumped = to_dict(md)
    assert "subdomain" not in dumped["math_description"]["parameters"][0]
    assert "subdomain" not in dumped["math_description"]["parameters"][1]


# ---------------------------------------------------------------------------
# Heterogeneous BC list (silence the unused-import warning on
# BCInterfaceFluxBalance and exercise the dispatcher).
# ---------------------------------------------------------------------------


def test_round_trip_interface_flux_balance_bc() -> None:
    md = MathDescription(
        geometry="two_bulks",
        subdomains=[
            Subdomain(name="left", kind="volume"),
            Subdomain(name="right", kind="volume"),
        ],
        variables=[
            Variable(name="u_left", subdomain="left"),
            Variable(name="u_right", subdomain="right"),
        ],
        parameters=[ParameterConstant(name="P", value=0.1)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_left",
                subdomain="left",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_right",
                subdomain="right",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFluxBalance(
                variable="u_left",
                partner_variable="u_right",
                boundary="membrane",
                expression="P * (u_left - u_right)",
            )
        ],
    )
    reloaded = load_yaml(dump_yaml(md))
    assert reloaded == md

"""Schema-dataclass tests for the declarative formalism.

Constructs the three worked examples from docs/modeling/declarative-formalism.md
(§1.4.5, §1.6.6, §2.7) programmatically and asserts structural properties:
required fields populated, defaults behaving as documented, frozen
instances rejecting reassignment, and isinstance narrowing working on the
tagged unions.

No parser, loader, or validator is exercised here — those ship separately.
The point is that the in-memory representation can carry the doc's
examples losslessly via plain Python kwargs.
"""

from __future__ import annotations

import dataclasses

import pytest

from vcell_fenics.formalism import (
    BCDirichlet,
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    BCNeumann,
    BCRobin,
    BoundaryCondition,
    MathDescription,
    MotionNone,
    MotionPrescribedDisplacement,
    MotionPrescribedVelocity,
    MotionUnknown,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    ParameterRegionMap,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
)

# ---------------------------------------------------------------------------
# Defaults and immutability.
# ---------------------------------------------------------------------------


def test_subdomain_motion_defaults_to_none() -> None:
    sub = Subdomain(name="cytoplasm", kind="volume")
    assert isinstance(sub.motion, MotionNone)
    assert sub.motion.kind == "none"


def test_variable_defaults() -> None:
    v = Variable(name="c", subdomain="cytoplasm")
    assert v.type == "scalar"
    assert v.space == "lagrange_p1"


def test_mathdescription_optional_lists_default_empty() -> None:
    md = MathDescription(
        geometry="cell_2d",
        subdomains=[Subdomain(name="membrane", kind="surface")],
        variables=[Variable(name="rho", subdomain="membrane")],
        equations=[
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.05"},
                initial_condition="1.0",
            )
        ],
    )
    assert md.parameters == []
    assert md.boundary_conditions == []


def test_frozen_instances_reject_reassignment() -> None:
    v = Variable(name="c", subdomain="cytoplasm")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Motion union — discriminated-union narrowing via isinstance.
# ---------------------------------------------------------------------------


def test_motion_union_isinstance_narrowing() -> None:
    sub_static = Subdomain(name="extracellular", kind="volume")
    sub_prescribed = Subdomain(
        name="membrane",
        kind="surface",
        motion=MotionPrescribedVelocity(velocity="r_dot * geom.x / geom.radius"),
    )
    sub_unknown = Subdomain(
        name="membrane2",
        kind="surface",
        motion=MotionUnknown(variable="v_membrane"),
    )

    # isinstance is the recommended narrowing mechanism — `kind` alone is not
    # sufficient because two MotionPrescribed* variants share kind="prescribed".
    assert isinstance(sub_static.motion, MotionNone)
    assert isinstance(sub_prescribed.motion, MotionPrescribedVelocity)
    assert isinstance(sub_unknown.motion, MotionUnknown)

    # And demonstrate that mypy-style narrowing reads the right field:
    if isinstance(sub_unknown.motion, MotionUnknown):
        assert sub_unknown.motion.variable == "v_membrane"


def test_motion_prescribed_displacement_is_distinct_from_velocity() -> None:
    d = MotionPrescribedDisplacement(displacement="[v_x * t, 0]")
    v = MotionPrescribedVelocity(velocity="[v_x, 0]")
    assert d.kind == "prescribed"
    assert v.kind == "prescribed"
    assert isinstance(d, MotionPrescribedDisplacement)
    assert not isinstance(d, MotionPrescribedVelocity)


# ---------------------------------------------------------------------------
# Parameter union — three forms.
# ---------------------------------------------------------------------------


def test_parameter_constant() -> None:
    p: Parameter = ParameterConstant(name="k_on", value=0.10)
    assert p.kind == "scalar"
    assert isinstance(p, ParameterConstant)
    assert p.value == pytest.approx(0.10)


def test_parameter_expression_with_subdomain_scope() -> None:
    # f_active from §1.5.8 / §1.10.8 / §2.7 — vector-typed, scoped to
    # membrane because the body uses theta(x).
    p: Parameter = ParameterExpression(
        name="f_active",
        type="vector",
        subdomain="membrane",
        expression="[f0 * cos(geom.azimuth), 0]",
    )
    assert p.kind == "expression"
    assert isinstance(p, ParameterExpression)
    assert p.type == "vector"
    assert p.subdomain == "membrane"


def test_parameter_expression_default_scope_is_none() -> None:
    p = ParameterExpression(
        name="L_reservoir",
        expression="1.0 + 0.5 * sin(omega * sim.t)",
    )
    assert p.type == "scalar"
    assert p.subdomain is None


def test_parameter_region_map() -> None:
    p: Parameter = ParameterRegionMap(
        name="D",
        subdomain="cytoplasm",
        values={"cytoplasm_left_cell": 0.10, "cytoplasm_right_cell": 0.25},
    )
    assert p.kind == "region_map"
    assert isinstance(p, ParameterRegionMap)
    assert p.values["cytoplasm_left_cell"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Worked example §1.4.5 — 2-species mass-action on a prescribed-motion membrane.
# ---------------------------------------------------------------------------


def test_constructs_section_1_4_5_two_species_membrane() -> None:
    md = MathDescription(
        geometry="disk_radius_1",
        subdomains=[
            Subdomain(
                name="membrane",
                kind="surface",
                motion=MotionPrescribedVelocity(velocity="r_dot * geom.x / geom.radius"),
            ),
        ],
        variables=[
            Variable(name="rho_active", subdomain="membrane"),
            Variable(name="rho_inactive", subdomain="membrane"),
        ],
        equations=[
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho_active",
                subdomain="membrane",
                temporality="time_dependent",
                terms={
                    "diffusion": "0.1",
                    "source": "k_on * rho_inactive - k_off * rho_active",
                },
                initial_condition="1.0 + 0.5 * cos(2 * geom.azimuth)",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho_inactive",
                subdomain="membrane",
                temporality="time_dependent",
                terms={
                    "diffusion": "0.1",
                    "source": "-k_on * rho_inactive + k_off * rho_active",
                },
                initial_condition="1.0",
            ),
        ],
        parameters=[
            ParameterConstant(name="k_on", value=0.10),
            ParameterConstant(name="k_off", value=0.05),
            ParameterConstant(name="r_dot", value=1.00),
        ],
    )
    assert len(md.variables) == 2
    assert len(md.equations) == 2
    assert md.boundary_conditions == []
    assert isinstance(md.subdomains[0].motion, MotionPrescribedVelocity)


# ---------------------------------------------------------------------------
# Worked example §1.6.6 — ligand-receptor bulk-surface coupling.
# Exercises Dirichlet + Neumann BCs and the composable bulk-surface pattern.
# ---------------------------------------------------------------------------


def test_constructs_section_1_6_6_ligand_receptor() -> None:
    md = MathDescription(
        geometry="cell_with_extracellular",
        subdomains=[
            Subdomain(name="extracellular", kind="volume", motion=MotionNone()),
            Subdomain(
                name="membrane",
                kind="surface",
                motion=MotionPrescribedVelocity(velocity="0"),
            ),
        ],
        variables=[
            Variable(name="L", subdomain="extracellular"),
            Variable(name="rho_f", subdomain="membrane"),
            Variable(name="rho_b", subdomain="membrane"),
        ],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L",
                subdomain="extracellular",
                temporality="time_dependent",
                terms={"diffusion": "0.5"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho_f",
                subdomain="membrane",
                temporality="time_dependent",
                terms={
                    "diffusion": "0.05",
                    "source": "-(k_on * trace(L) * rho_f - k_off * rho_b)",
                },
                initial_condition="0.5",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho_b",
                subdomain="membrane",
                temporality="time_dependent",
                terms={
                    "diffusion": "0.05",
                    "source": "k_on * trace(L) * rho_f - k_off * rho_b",
                },
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCDirichlet(variable="L", boundary="outer", expression="L_reservoir"),
            BCNeumann(
                variable="L",
                boundary="membrane",
                expression="k_on * trace(L) * rho_f - k_off * rho_b",
            ),
        ],
        parameters=[
            ParameterConstant(name="k_on", value=0.10),
            ParameterConstant(name="k_off", value=0.02),
            ParameterConstant(name="L_reservoir", value=1.00),
        ],
    )
    assert len(md.subdomains) == 2
    assert len(md.boundary_conditions) == 2
    assert isinstance(md.boundary_conditions[0], BCDirichlet)
    assert isinstance(md.boundary_conditions[1], BCNeumann)


# ---------------------------------------------------------------------------
# Worked example §2.7 — end-to-end weak-form + unknown motion + expression
# parameter (the most feature-rich of the three; same model as §1.10.8).
# ---------------------------------------------------------------------------


def test_constructs_section_2_7_end_to_end() -> None:
    weak_form = (
        "( eta * inner(v_membrane, v_membrane_test)"
        " + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)"
        " - inner(f_active, v_membrane_test)"
        ") * dx_Gamma"
    )
    md = MathDescription(
        geometry="cell_2d",
        subdomains=[
            Subdomain(
                name="membrane",
                kind="surface",
                motion=MotionUnknown(variable="v_membrane"),
            ),
        ],
        variables=[
            Variable(name="v_membrane", subdomain="membrane", type="vector"),
            Variable(name="rho", subdomain="membrane"),
        ],
        parameters=[
            ParameterConstant(name="eta", value=1.0),
            ParameterConstant(name="sigma_T", value=0.10),
            ParameterConstant(name="k_off", value=0.02),
            ParameterConstant(name="f0", value=0.3),
            ParameterExpression(
                name="f_active",
                type="vector",
                subdomain="membrane",
                expression="[f0 * cos(geom.azimuth), 0]",
            ),
        ],
        equations=[
            WeakFormEquation(
                variable="v_membrane",
                subdomain="membrane",
                temporality="steady_state",
                form=weak_form,
                initial_condition="0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="membrane",
                temporality="time_dependent",
                terms={"diffusion": "0.05", "source": "-k_off * rho"},
                initial_condition="1.0 + 0.3 * cos(2 * geom.azimuth)",
            ),
        ],
    )
    # Mixed-temporality system (§1.9.4): one steady_state weak form, one
    # time_dependent template equation.
    assert md.equations[0].temporality == "steady_state"
    assert md.equations[1].temporality == "time_dependent"

    # Expression-valued parameter carries its scope.
    f_active = md.parameters[-1]
    assert isinstance(f_active, ParameterExpression)
    assert f_active.subdomain == "membrane"

    # Weak-form equation narrows by isinstance.
    motion_eq = md.equations[0]
    assert isinstance(motion_eq, WeakFormEquation)
    assert "partial_t" not in motion_eq.form  # steady_state — no time derivative


# ---------------------------------------------------------------------------
# Boundary-condition coverage — all five kinds construct cleanly.
# ---------------------------------------------------------------------------


def test_all_bc_kinds_construct() -> None:
    # Explicit list[BoundaryCondition] — mypy can't infer the union type
    # from a heterogeneous literal list, so a bare `bcs = [...]` would be
    # inferred as list[object] and the .kind accesses below would fail.
    bcs: list[BoundaryCondition] = [
        BCDirichlet(variable="u", boundary="outer", expression="0"),
        BCNeumann(variable="u", boundary="outer", expression="0"),
        BCRobin(variable="u", boundary="outer", alpha="1.0", beta="0.5", expression="g(geom.x, sim.t)"),
        # Default partition coefficient k is "1".
        BCInterfaceValueEquality(
            variable="u_left",
            partner_variable="u_right",
            boundary="membrane",
        ),
        BCInterfaceFluxBalance(
            variable="u_left",
            partner_variable="u_right",
            boundary="membrane",
            expression="P * (u_left - u_right)",
        ),
    ]
    assert bcs[0].kind == "dirichlet"
    assert bcs[1].kind == "neumann"
    assert bcs[2].kind == "robin"
    assert bcs[3].kind == "interface_value_equality"
    assert isinstance(bcs[3], BCInterfaceValueEquality)
    assert bcs[3].expression == "1"  # default partition coefficient
    assert bcs[4].kind == "interface_flux_balance"

"""Tests for the VCell→formalism bridge (`vcell_fenics.pyvcell_bridge`).

Three layers:

1. **Expression translation** — VCell coordinate/time/power syntax → the formalism's
   namespaced built-ins. Pure, needs no pyvcell.
2. **Structural import** — a VCell `MathDescription` (pyvcell's lowered math model) →
   a formalism `MathDescription` that validates, plus loud rejection of the
   out-of-scope / not-yet-implemented constructs (§2.6.3 / §2.6.2).
3. **End-to-end** — import a VCell reaction-diffusion model and actually run it through
   the FEniCSx backend, proving the import lands on a runnable, well-posed model.

pyvcell (>= 0.3.0) is a normal dependency — its pure-Pydantic `vcml.models_math` data model
installs with a minimal default dependency set, so it is always present in the env.
"""

from __future__ import annotations

import pytest
import pyvcell.vcml.models_math as vm

from vcell_fenics.formalism.schema import TemplateEquation
from vcell_fenics.pyvcell_bridge import (
    VcellImportError,
    import_math_description,
    import_model,
    translate_expression,
)

# --- 1. expression translation -------------------------------------------------


def test_coordinates_and_time_become_namespaced_builtins() -> None:
    assert translate_expression("x") == "geom.x[0]"
    assert translate_expression("y") == "geom.x[1]"
    assert translate_expression("z") == "geom.x[2]"
    assert translate_expression("t") == "sim.t"
    assert translate_expression("1.0 + 0.3*x") == "1.0 + 0.3*geom.x[0]"
    assert translate_expression("sin(t) * exp(-z)") == "sin(sim.t) * exp(-geom.x[2])"


def test_power_operator_is_translated() -> None:
    assert translate_expression("x^2 + y^2") == "geom.x[0]**2 + geom.x[1]**2"


def test_unary_minus_power_precedence_matches_vcell() -> None:
    # VCell's grammar binds the sign inside the power base (`-x^2` means `(-x)^2 = x²`); the
    # formalism binds `**` tighter (`-x**2` means `-(x²)`). The translator inserts parens so the
    # VCell meaning is preserved — verified against VCell's own region masks over the corpus.
    assert translate_expression("-x^2") == "(-geom.x[0])**2"
    assert translate_expression("-x^2 + -y^2") == "(-geom.x[0])**2 + (-geom.x[1])**2"
    # VCell power is left-associative; ours is right-associative.
    assert translate_expression("2^3^2") == "(2**3)**2"
    # A positive base needs no parens (unchanged from the naive translation).
    assert translate_expression("x^2") == "geom.x[0]**2"


def test_coordinate_names_are_matched_only_as_whole_identifiers() -> None:
    # The `x` in `max`/`exp` and a parameter named `Ca_x` must not be rewritten; VCell
    # reserves x/y/z/t as coordinates, so no user symbol legitimately collides.
    assert translate_expression("max(exp(a), b)") == "max(exp(a), b)"
    assert translate_expression("Ca_x * k_yz") == "Ca_x * k_yz"
    assert translate_expression("kt * tx") == "kt * tx"  # not bare t / x


def test_inserted_builtin_is_not_re_translated() -> None:
    # `x` → `geom.x[0]` contains an `x`; a naive sequential replace would loop. One pass.
    assert translate_expression("x") == "geom.x[0]"


# --- 2. structural import ------------------------------------------------------


def test_compartment_pde_maps_to_bulk_radv_diff() -> None:
    vcml = vm.MathDescription(
        name="cell",
        constants=[vm.Constant(name="D", exp="0.2")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cytosol",
                pde_equations=[
                    vm.PdeEquation(name="c", diffusion="D", rate="-k*c", initial="1.0 + 0.3*x", steady=False)
                ],
            )
        ],
    )
    md = import_math_description(vcml, geometry="disk_2d")

    assert md.geometry == "disk_2d"
    assert [(s.name, s.kind) for s in md.subdomains] == [("cytosol", "volume")]
    assert [(v.name, v.subdomain) for v in md.variables] == [("c", "cytosol")]
    (eq,) = md.equations
    assert isinstance(eq, TemplateEquation)
    assert eq.template == "bulk_radv_diff"
    assert eq.temporality == "time_dependent"
    assert eq.terms == {"diffusion": "D", "source": "-k*c"}
    assert eq.initial_condition == "1.0 + 0.3*geom.x[0]"  # coordinate translated


def test_membrane_pde_maps_to_surface_template_and_geometry_defaults_to_name() -> None:
    vcml = vm.MathDescription(
        name="membrane_model",
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="membrane",
                pde_equations=[vm.PdeEquation(name="rho", diffusion="0.1", initial="1.0", steady=False)],
            )
        ],
    )
    md = import_math_description(vcml)
    assert md.geometry == "membrane_model"  # defaulted from the VCell math name
    assert [(s.name, s.kind) for s in md.subdomains] == [("membrane", "surface")]
    (eq,) = md.equations
    assert eq.template == "surface_pde_with_dilution"


def test_steady_pde_is_steady_state_and_ode_maps_to_lumped_ode() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="u", diffusion="1.0", steady=True)],
                ode_equations=[vm.OdeEquation(name="w", rate="k1 - k2*w", initial="0.0")],
            )
        ],
    )
    md = import_math_description(vcml)
    by_var = {e.variable: e for e in md.equations}
    assert by_var["u"].temporality == "steady_state"
    w = by_var["w"]
    assert isinstance(w, TemplateEquation)
    assert w.template == "lumped_ode"
    assert w.terms == {"rate": "k1 - k2*w"}


def test_velocity_becomes_a_relative_advection_vector() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[
                    vm.PdeEquation(name="c", diffusion="1.0", steady=False, velocity=vm.Velocity(x="vx", y="2*y"))
                ],
            )
        ],
    )
    (eq,) = import_math_description(vcml).equations
    assert isinstance(eq, TemplateEquation)
    assert eq.terms["relative_advection"] == "[vx, 2*geom.x[1]]"


def test_numeric_and_symbolic_constants_and_functions() -> None:
    from vcell_fenics.formalism.schema import ParameterConstant, ParameterExpression

    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="D", exp="0.2"), vm.Constant(name="K", exp="a*b")],
        functions=[vm.MathFunction(name="f", exp="x*2")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="D")])
        ],
    )
    params = {p.name: p for p in import_math_description(vcml).parameters}
    assert isinstance(params["D"], ParameterConstant) and params["D"].value == pytest.approx(0.2)
    assert isinstance(params["K"], ParameterExpression) and params["K"].expression == "a*b"
    assert isinstance(params["f"], ParameterExpression) and params["f"].expression == "geom.x[0]*2"


def test_imported_model_validates() -> None:
    from vcell_fenics.formalism import validate

    vcml = vm.MathDescription(
        name="cell",
        constants=[vm.Constant(name="D", exp="0.2"), vm.Constant(name="k", exp="0.5")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cytosol",
                pde_equations=[
                    vm.PdeEquation(name="c", diffusion="D", rate="-k*c", initial="1.0 + 0.3*x", steady=False)
                ],
            )
        ],
    )
    md = import_math_description(vcml, geometry="disk_2d")
    assert [d for d in validate(md) if d.severity == "error"] == []


# --- function inlining + observables -------------------------------------------


def test_variable_referencing_function_is_inlined_and_observed() -> None:
    from vcell_fenics.formalism import validate

    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="k", exp="2.0")],
        functions=[vm.MathFunction(name="J", exp="k * c")],  # references the variable c
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", rate="-J", initial="1.0", steady=False)],
            )
        ],
    )
    result = import_model(vcml, geometry="g")

    (eq,) = result.math.equations
    assert isinstance(eq, TemplateEquation)
    assert "J" not in eq.terms["source"]  # the function was inlined away
    assert "k * c" in eq.terms["source"]
    assert all(p.name != "J" for p in result.math.parameters)  # not a parameter
    assert [(o.name, o.expression) for o in result.observables] == [("J", "k * c")]  # surfaced as an observable
    assert [d for d in validate(result.math) if d.severity == "error"] == []  # now validates clean


def test_pure_function_stays_a_parameter_not_an_observable() -> None:
    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="a", exp="2.0"), vm.Constant(name="b", exp="3.0")],
        functions=[vm.MathFunction(name="g", exp="a + b")],  # references no variable
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="g", initial="1.0", steady=False)]
            )
        ],
    )
    result = import_model(vcml, geometry="g")
    assert result.observables == ()
    assert any(p.name == "g" for p in result.math.parameters)


def test_nested_variable_function_is_fully_inlined() -> None:
    vcml = vm.MathDescription(
        name="m",
        functions=[vm.MathFunction(name="inner", exp="2 * c"), vm.MathFunction(name="outer", exp="inner + 1")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", rate="-outer", initial="1.0", steady=False)],
            )
        ],
    )
    result = import_model(vcml, geometry="g")
    observables = {o.name: o.expression for o in result.observables}
    assert observables["outer"] == "(2 * c) + 1"  # `inner` inlined into `outer`
    (eq,) = result.math.equations
    assert isinstance(eq, TemplateEquation)
    assert "outer" not in eq.terms["source"] and "inner" not in eq.terms["source"]


def test_import_math_description_is_the_math_of_import_model() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", initial="1.0", steady=False)]
            )
        ],
    )
    assert import_math_description(vcml, geometry="g") == import_model(vcml, geometry="g").math


# --- loud rejection of out-of-scope / not-yet constructs ----------------------


def test_stochastic_constructs_are_rejected() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                jump_processes=[vm.JumpProcess(name="birth", probability_rate="k")],
            )
        ],
    )
    with pytest.raises(VcellImportError, match="stochastic jump processes"):
        import_math_description(vcml)


def test_jump_condition_imports_as_neumann_with_traced_bulk_species() -> None:
    # A membrane JumpCondition is a per-side Neumann flux on the bulk species at the membrane
    # (§1.6.5). The flux's bulk-species references are wrapped in trace(·); parameters are direct.
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", initial="0")])
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                jump_conditions=[vm.JumpCondition(name="c", in_flux="k * c", out_flux="0.0")],
            )
        ],
    )
    bcs = import_math_description(vcml).boundary_conditions
    assert len(bcs) == 1
    bc = bcs[0]
    assert type(bc).__name__ == "BCNeumann" and bc.variable == "c" and bc.boundary == "pm"
    assert bc.expression == "k * trace(c)"  # bulk species c traced; parameter k direct


def test_jump_condition_species_on_both_sides_rejected() -> None:
    # A species present in both compartments would need two side-distinguished BCs, which the
    # per-variable BCNeumann(variable, boundary) cannot yet express → reject rather than guess.
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="1.0")]),
            vm.CompartmentSubDomain(name="ec", pde_equations=[vm.PdeEquation(name="c", diffusion="1.0")]),
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                jump_conditions=[vm.JumpCondition(name="c", in_flux="k1", out_flux="k2")],
            )
        ],
    )
    with pytest.raises(NotImplementedError, match=r"lives in\s+both compartments"):
        import_math_description(vcml)


def test_boundary_conditions_need_the_geometry_dimension() -> None:
    # Without `dim`, the importer can't tell which of VCell's six box faces are real, so a PDE
    # carrying any non-default boundary is rejected rather than mis-imported.
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", boundaries=vm.Boundaries(xm="0.5"))],
            )
        ],
    )
    with pytest.raises(NotImplementedError, match="pass dim="):
        import_math_description(vcml)


def test_value_face_imports_as_per_variable_dirichlet() -> None:
    # A `Value` (Dirichlet) face with no explicit value uses the species' initial condition
    # (VCell's default-Dirichlet rule); a `Flux` face with a value is a Neumann BC. The result is
    # per-variable, and z-faces are filtered out at dim 2.
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                boundary_types=[
                    vm.MathBoundaryType(boundary="Xm", type="Value"),
                    vm.MathBoundaryType(boundary="Yp", type="Flux"),
                    vm.MathBoundaryType(boundary="Zm", type="Value"),  # spurious in 2D — must be dropped
                ],
                pde_equations=[
                    vm.PdeEquation(name="c", diffusion="1.0", initial="5.0", boundaries=vm.Boundaries(yp="2.0"))
                ],
            )
        ],
    )
    bcs = import_math_description(vcml, dim=2).boundary_conditions
    by_face = {bc.boundary: bc for bc in bcs}
    assert set(by_face) == {"x_minus", "y_plus"}  # no z_minus
    assert type(by_face["x_minus"]).__name__ == "BCDirichlet" and by_face["x_minus"].expression == "5.0"
    assert type(by_face["y_plus"]).__name__ == "BCNeumann" and by_face["y_plus"].expression == "2.0"
    assert all(bc.variable == "c" for bc in bcs)


def test_box_face_names_match_the_realization() -> None:
    # The importer's 2D box-face names must be exactly those the realization tags, so an imported
    # per-face BC (`boundary=x_minus`) binds to a realized box face at solve time.
    from vcell_fenics.backend.realize import _FACE_NAMES_2D
    from vcell_fenics.pyvcell_bridge.importer import _FACE_NAME_MAP

    assert (_FACE_NAME_MAP["Xm"], _FACE_NAME_MAP["Xp"], _FACE_NAME_MAP["Ym"], _FACE_NAME_MAP["Yp"]) == _FACE_NAMES_2D


def test_default_flux_boundary_types_import_cleanly() -> None:
    # VCell's default no-flux faces (all `Flux`, no expressions) == the formalism's
    # natural zero-Neumann boundary, so they import without producing any BC.
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                boundary_types=[vm.MathBoundaryType(boundary=f, type="Flux") for f in ("Xm", "Xp", "Ym", "Yp")],
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", initial="1.0", steady=False)],
            )
        ],
    )
    md = import_math_description(vcml)
    assert md.boundary_conditions == []


# --- 3. end-to-end: import a VCell model and run it through FEniCSx ------------


def test_imported_reaction_diffusion_model_runs_and_decays() -> None:
    from vcell_fenics.backend import SolverConfiguration, make_disk_geometry, run

    # Uniform initial state with pure first-order decay (-k·c) and no-flux faces: the
    # field stays spatially uniform and relaxes as c0·exp(-k·t). A real FEniCSx solve.
    vcml = vm.MathDescription(
        name="decay",
        constants=[vm.Constant(name="D", exp="0.1"), vm.Constant(name="k", exp="0.5")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cytosol",
                pde_equations=[vm.PdeEquation(name="c", diffusion="D", rate="-k*c", initial="2.0", steady=False)],
            )
        ],
    )
    md = import_math_description(vcml, geometry="disk_2d")
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytosol", radius=1.0, h=0.2)

    problem = run(md, geometry, SolverConfiguration(dt=0.05, t_final=1.0))
    c = problem.unknown.x.array

    assert float(c.min()) > 0.0  # stays positive
    # uniform decay from 2.0: mean ≈ 2·exp(-0.5·1.0) ≈ 1.21, well below the IC.
    assert 1.0 < float(c.mean()) < 1.4

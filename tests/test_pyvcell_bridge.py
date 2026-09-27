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

from vcell_fenics.formalism.schema import ParameterExpression, TemplateEquation
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


def test_membrane_pde_box_face_boundaries_are_dropped() -> None:
    # VCell emits per-face boundary_types for every species, including a membrane species (R_boundaryXm
    # → 0.0 for a cell membrane). A membrane is realized as the *closed* interface curve, which never
    # touches the box, so those faces have nothing to apply to → dropped (not rejected). This is what
    # lets a real receptor model import.
    vcml = vm.MathDescription(
        name="m",
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                pde_equations=[
                    vm.PdeEquation(
                        name="R",
                        diffusion="0.05",
                        initial="0.0",
                        boundaries=vm.Boundaries(xm="0.0", xp="0.0", ym="0.0", yp="0.0", zm="0.0", zp="0.0"),
                    )
                ],
            )
        ],
    )
    md = import_math_description(vcml, dim=2)
    assert [v.name for v in md.variables] == ["R"]
    assert md.boundary_conditions == []  # the spurious membrane box-face BCs are dropped


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


def test_a_velocity_reached_through_functions_keeps_its_parameters() -> None:
    # VCell routes a species velocity through functions to a dotted constant (the furrow model:
    # vobj_Cyt1_velX → vproc_1.velocityX = 0). Regression: the velocity was not a root of the
    # reachable set, so its constant function was dropped as dead and the equation named nothing.
    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="vproc_1.velocityX", exp="0.0"), vm.Constant(name="vproc_1.velocityY", exp="0.0")],
        functions=[
            vm.MathFunction(name="vobj_Cyt1_velX", exp="vproc_1.velocityX"),
            vm.MathFunction(name="vobj_Cyt1_velY", exp="vproc_1.velocityY"),
        ],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="Cyt",
                pde_equations=[
                    vm.PdeEquation(
                        name="Dex",
                        diffusion="1.0",
                        steady=False,
                        velocity=vm.Velocity(x="vobj_Cyt1_velX", y="vobj_Cyt1_velY"),
                    )
                ],
            )
        ],
    )
    md = import_math_description(vcml)
    (eq,) = md.equations
    assert isinstance(eq, TemplateEquation)
    assert eq.terms["relative_advection"] == "[vobj_Cyt1_velX, vobj_Cyt1_velY]"
    params = {p.name: p for p in md.parameters}
    assert isinstance(params["vobj_Cyt1_velX"], ParameterExpression)
    assert params["vobj_Cyt1_velX"].expression == "vproc_1.velocityX"
    assert "vproc_1.velocityX" in params


def _moving_boundary_vcml(membranes: int = 1) -> vm.MathDescription:
    """A cyto disk (with a diffusing species) inside ec, joined by one (or more) membranes whose
    ``inside_compartment`` is the moving interior — the moving-boundary import target."""
    membrane_subdomains = [
        vm.MembraneSubDomain(name=f"cyto_ec_{i}", inside_compartment="cyto", outside_compartment="ec")
        for i in range(membranes)
    ]
    return vm.MathDescription(
        name="cell",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="C", diffusion="10", rate="0", initial="x", steady=False)],
            ),
            vm.CompartmentSubDomain(name="ec"),
        ],
        membrane_subdomains=membrane_subdomains,
    )


def test_front_velocity_imports_as_prescribed_motion_on_the_interior() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    from vcell_fenics.formalism.schema import MotionNone, MotionPrescribedVelocity

    md = import_math_description(
        _moving_boundary_vcml(),
        geometry="disk",
        dim=2,
        front_velocity=FrontVelocity(velocity_x=0.5, velocity_y=0.0),
    )
    motions = {s.name: s.motion for s in md.subdomains}
    # The moving front carries the cell interior (v = v_b); ec stays the static lab frame.
    assert isinstance(motions["cyto"], MotionPrescribedVelocity)
    assert motions["cyto"].velocity == "[0.5, 0.0]"
    assert isinstance(motions["ec"], MotionNone)


def test_front_velocity_expression_components_are_translated() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    from vcell_fenics.formalism.schema import MotionPrescribedVelocity

    md = import_math_description(
        _moving_boundary_vcml(),
        dim=2,
        front_velocity=FrontVelocity(velocity_x="sin(t)", velocity_y="2*y"),
    )
    motion = next(s.motion for s in md.subdomains if s.name == "cyto")
    assert isinstance(motion, MotionPrescribedVelocity)
    assert motion.velocity == "[sin(sim.t), 2*geom.x[1]]"


def test_front_velocity_includes_z_only_for_a_3d_geometry() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    from vcell_fenics.formalism.schema import MotionPrescribedVelocity

    front = FrontVelocity(velocity_x=0.5, velocity_y=0.0, velocity_z=0.25)
    md = import_math_description(_moving_boundary_vcml(), dim=3, front_velocity=front)
    motion = next(s.motion for s in md.subdomains if s.name == "cyto")
    assert isinstance(motion, MotionPrescribedVelocity)
    assert motion.velocity == "[0.5, 0.0, 0.25]"


def test_no_front_velocity_leaves_all_subdomains_static() -> None:
    from vcell_fenics.formalism.schema import MotionNone

    md = import_math_description(_moving_boundary_vcml())
    assert all(isinstance(s.motion, MotionNone) for s in md.subdomains)


def test_front_velocity_surface_name_selects_the_moving_membrane() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    from vcell_fenics.formalism.schema import MotionPrescribedVelocity

    front = FrontVelocity(velocity_x=0.5, velocity_y=0.0, surface_name="cyto_ec_1")
    md = import_math_description(_moving_boundary_vcml(membranes=2), dim=2, front_velocity=front)
    motion = next(s.motion for s in md.subdomains if s.name == "cyto")
    assert isinstance(motion, MotionPrescribedVelocity)


def test_front_velocity_is_ambiguous_without_a_surface_name() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    with pytest.raises(VcellImportError, match="multiple membranes"):
        import_math_description(_moving_boundary_vcml(membranes=2), dim=2, front_velocity=FrontVelocity(velocity_x=0.5))


def test_front_velocity_surface_name_must_match_a_membrane() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    with pytest.raises(VcellImportError, match="matches no membrane"):
        import_math_description(
            _moving_boundary_vcml(), dim=2, front_velocity=FrontVelocity(velocity_x=0.5, surface_name="nope")
        )


def test_front_velocity_requires_a_membrane() -> None:
    from pyvcell.vcml.models_app import FrontVelocity

    vcml = vm.MathDescription(
        name="cell",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="C", diffusion="10", rate="0", initial="x", steady=False)],
            )
        ],
    )
    with pytest.raises(VcellImportError, match="no membrane"):
        import_math_description(vcml, dim=2, front_velocity=FrontVelocity(velocity_x=0.5))


def test_numeric_and_symbolic_constants_and_functions() -> None:
    from vcell_fenics.formalism.schema import ParameterConstant, ParameterExpression

    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="D", exp="0.2"), vm.Constant(name="K", exp="a*b")],
        functions=[
            vm.MathFunction(name="g", exp="D*3"),  # constant-reducible + referenced → stays a parameter
            vm.MathFunction(name="f", exp="x*2"),  # coordinate-referencing → inlined into the equation
        ],
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="c", diffusion="g", rate="f")])
        ],
    )
    md = import_math_description(vcml)
    params = {p.name: p for p in md.parameters}
    assert isinstance(params["D"], ParameterConstant) and params["D"].value == pytest.approx(0.2)
    assert isinstance(params["K"], ParameterExpression) and params["K"].expression == "a*b"
    assert isinstance(params["g"], ParameterExpression) and params["g"].expression == "D*3"
    # A coordinate-dependent function is not a constant parameter — it is inlined where it is used.
    assert "f" not in params
    (eq,) = md.equations
    assert isinstance(eq, TemplateEquation)
    assert eq.terms["source"] == "(geom.x[0]*2)"


def test_unreferenced_pure_function_is_dropped() -> None:
    # VCell emits region-size bookkeeping (`Size_<compartment>`, `vobj_<region>_size`) that calls
    # geometric built-ins like vcRegionVolume('domain') — not part of the PDE problem and not a
    # formalism expression. Nothing references them, so they are dropped (not imported as a
    # parameter that would fail to parse). A *referenced* such function would still be emitted and
    # rejected loudly at validation, so this drops only provably-dead definitions.
    vcml = vm.MathDescription(
        name="m",
        functions=[
            vm.MathFunction(name="Size_cell", exp="vcRegionVolume('domain')"),
            vm.MathFunction(name="vobj_domain0_size", exp="vcRegionVolume('domain')"),
        ],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="domain", pde_equations=[vm.PdeEquation(name="u", diffusion="1.0", initial="1.0")]
            )
        ],
    )
    params = {p.name for p in import_math_description(vcml).parameters}
    assert "Size_cell" not in params
    assert "vobj_domain0_size" not in params


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


def _membrane_flux_model(in_flux: str) -> vm.MathDescription:
    return vm.MathDescription(
        name="flux",
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="u", diffusion="0.1", initial="1")])
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                jump_conditions=[vm.JumpCondition(name="u", in_flux=in_flux, out_flux="0.0")],
            )
        ],
    )


def test_jump_condition_preserves_vcell_flux_sign() -> None:
    # The membrane jump-condition SIGN CONVENTION, confirmed against VCell's FV solver two-sided
    # (cross_validation/membrane_flux_sign.py). Our bridge maps in_flux *directly* (no sign flip) to
    # the BCNeumann expression, and the backend's Neumann sign is verified independently
    # (d(mass)/dt = ∫ h ds, test_neumann_influx_adds_mass_at_predicted_rate). So preserving the sign
    # is what makes our import reproduce the FV mass direction:
    #   efflux: VCell in_flux NEGATIVE (∝ Kf·u) -> FV cytosol mass DECREASES;
    #   influx: VCell in_flux POSITIVE (constant general-kinetics rate) -> FV cytosol mass INCREASES.
    (efflux,) = import_math_description(_membrane_flux_model("-1.0 * Kflux * (Kf * u)")).boundary_conditions
    assert type(efflux).__name__ == "BCNeumann" and efflux.variable == "u" and efflux.boundary == "pm"
    assert efflux.expression == "-1.0 * Kflux * (Kf * trace(u))"  # negative; bulk u traced onto membrane
    assert efflux.expression.lstrip().startswith("-")

    (influx,) = import_math_description(_membrane_flux_model("Kflux * J_influx")).boundary_conditions
    assert influx.expression == "Kflux * J_influx"  # positive constant flux preserved, no trace wrapping
    assert not influx.expression.lstrip().startswith("-")


def test_coupled_jump_conditions_become_single_sided_interface_fluxes() -> None:
    # A species crossing the membrane — two domain-restricted species, one per compartment, whose
    # fluxes reference each other (a permeability flux P·(s_ext − s_cyto)) — is a cross-compartment
    # coupling at an internal interface (§1.6.2, both compartments modelled). VCell's in_flux /
    # out_flux are independent single-sided fluxes, so the importer emits TWO BCInterfaceFlux, one per
    # species, each carrying the flux into its OWN side (taken from that species' own well-posed side:
    # in_flux for the inside species, out_flux for the outside species).
    vcml = vm.MathDescription(
        name="perm",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto", pde_equations=[vm.PdeEquation(name="s_cyto", diffusion="1.0", initial="1")]
            ),
            vm.CompartmentSubDomain(
                name="ec", pde_equations=[vm.PdeEquation(name="s_ext", diffusion="1.0", initial="0")]
            ),
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                jump_conditions=[
                    vm.JumpCondition(name="s_cyto", in_flux="P * (s_ext - s_cyto)", out_flux="0.0"),
                    vm.JumpCondition(name="s_ext", in_flux="0.0", out_flux="-1.0 * P * (s_ext - s_cyto)"),
                ],
            )
        ],
    )
    from vcell_fenics.formalism.schema import BCInterfaceFlux

    bcs = import_math_description(vcml).boundary_conditions  # two BCs, one per side
    assert all(isinstance(bc, BCInterfaceFlux) and bc.boundary == "pm" for bc in bcs)
    by_var = {bc.variable: bc for bc in bcs}
    assert set(by_var) == {"s_cyto", "s_ext"}
    # Each side's own well-posed flux, with both bulk traces wrapped (the adjacent compartment's
    # species is reachable on the membrane through its trace): the inside species takes its in_flux,
    # the outside its out_flux.
    assert by_var["s_cyto"].expression == "P * (trace(s_ext) - trace(s_cyto))"
    assert by_var["s_ext"].expression == "-1.0 * P * (trace(s_ext) - trace(s_cyto))"


def test_membrane_reaction_traces_adjacent_bulk_species() -> None:
    # A membrane receptor R binding ligand from BOTH compartments. The bulk species in the membrane
    # reaction `source` must be wrapped in trace(·) — a volume variable is only defined on the membrane
    # through its trace (§1.6.5/§1.8.2), the same wrapping the jump-condition fluxes get — else the
    # model fails validation. The whole thing then routes to the three-region membrane coupling.
    from vcell_fenics.formalism import validate
    from vcell_fenics.formalism.schema import BCInterfaceFlux

    vcml = vm.MathDescription(
        name="receptor",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto", pde_equations=[vm.PdeEquation(name="L_in", diffusion="1.0", initial="1.0")]
            ),
            vm.CompartmentSubDomain(
                name="ec", pde_equations=[vm.PdeEquation(name="L_out", diffusion="1.0", initial="1.0")]
            ),
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                pde_equations=[
                    vm.PdeEquation(name="R", diffusion="0.05", initial="0.0", rate="kon * (L_in + L_out) * (Rmax - R)")
                ],
                jump_conditions=[
                    vm.JumpCondition(name="L_in", in_flux="-kon * L_in * (Rmax - R)", out_flux="0.0"),
                    vm.JumpCondition(name="L_out", in_flux="0.0", out_flux="-kon * L_out * (Rmax - R)"),
                ],
            )
        ],
        constants=[vm.Constant(name="kon", exp="0.5"), vm.Constant(name="Rmax", exp="2.0")],
    )
    md = import_math_description(vcml, dim=2)
    assert [d for d in validate(md) if d.severity == "error"] == []  # trace-wrapping makes it well-posed

    (surface_eq,) = [eq for eq in md.equations if eq.subdomain == "pm"]
    assert isinstance(surface_eq, TemplateEquation)
    # Bulk species traced in the membrane reaction; the membrane species R is left direct.
    assert surface_eq.terms["source"] == "kon * (trace(L_in) + trace(L_out)) * (Rmax - R)"
    # Each crossing ligand becomes a single-sided interface flux that itself references R directly.
    fluxes = {bc.variable: bc for bc in md.boundary_conditions if isinstance(bc, BCInterfaceFlux)}
    assert set(fluxes) == {"L_in", "L_out"}
    assert fluxes["L_in"].expression == "-kon * trace(L_in) * (Rmax - R)"
    assert fluxes["L_out"].expression == "-kon * trace(L_out) * (Rmax - R)"


def test_jump_condition_species_on_both_sides_rejected() -> None:
    # The legacy case: a single domain-less volume variable defined in BOTH compartments, where VCell's
    # JumpCondition genuinely carried two meaningful per-side fluxes. This needs two side-distinguished
    # BCs, which the per-variable BC(variable, boundary) cannot yet express → reject rather than guess.
    # (Modern VCell math gives each variable one domain, so only one flux side is well-posed.)
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
    with pytest.raises(NotImplementedError, match="legacy domain-less volume variable defined on both sides"):
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


def test_zero_flux_box_faces_are_dropped_nonzero_kept() -> None:
    # A zero D∇u·n is the natural no-flux default, so a Flux face whose value resolves to 0 (VCell's
    # per-face default, e.g. u_boundaryXm = 0) is dropped — both redundant and, for an interior
    # compartment that doesn't touch the box, a constraint it has no boundary for. A non-zero flux
    # stays a Neumann.
    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="zero_bc", exp="0.0"), vm.Constant(name="flux_bc", exp="0.5")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                boundary_types=[
                    vm.MathBoundaryType(boundary="Xm", type="Flux"),
                    vm.MathBoundaryType(boundary="Xp", type="Flux"),
                ],
                pde_equations=[
                    vm.PdeEquation(
                        name="c", diffusion="1.0", initial="1.0", boundaries=vm.Boundaries(xm="zero_bc", xp="flux_bc")
                    )
                ],
            )
        ],
    )
    bcs = import_math_description(vcml, dim=2).boundary_conditions
    neumann = [(bc.boundary, bc.expression) for bc in bcs if type(bc).__name__ == "BCNeumann"]
    assert neumann == [("x_plus", "flux_bc")]  # zero Xm dropped, non-zero Xp kept


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


def test_membrane_receptor_model_imports_geometry_and_math_end_to_end() -> None:
    import pyvcell.vcml.models_geometry as gmod

    from vcell_fenics.backend import assemble_membrane_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled
    from vcell_fenics.pyvcell_bridge import import_geometry, normalize_to_geometry_frame

    # The full receptor–ligand cell imported end-to-end from VCell — BOTH geometry AND math: a disk
    # `cytosol` inside an `extracellular` box meeting at the `pm` membrane (geometry), and two ligands
    # captured by a membrane receptor R (math). The geometry is realized into the coupled substrate
    # (`realize_interface_coupled` — same body-fitted marched mesh the permeability study uses) and the
    # model solves through the three-region membrane-coupled assembler, conserving total ligand. This is
    # the whole VCell→FEniCSx pipeline for a bulk↔membrane↔bulk model, no hand-built geometry or physics.
    vcml_geom = gmod.Geometry(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=[
            gmod.SubVolume(
                name="cytosol", handle=1, subvolume_type=gmod.SubVolumeType.analytic, analytic_expr="x^2 + y^2 < 0.25"
            ),
            gmod.SubVolume(
                name="extracellular", handle=0, subvolume_type=gmod.SubVolumeType.analytic, analytic_expr="1.0"
            ),
        ],
        surface_classes=[gmod.SurfaceClass(name="pm", subvolume_ref_1="cytosol", subvolume_ref_2="extracellular")],
    )
    vcml_math = vm.MathDescription(
        name="cell",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cytosol", pde_equations=[vm.PdeEquation(name="L_in", diffusion="1.0", initial="1.0")]
            ),
            vm.CompartmentSubDomain(
                name="extracellular", pde_equations=[vm.PdeEquation(name="L_out", diffusion="1.0", initial="1.0")]
            ),
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cytosol",
                outside_compartment="extracellular",
                pde_equations=[
                    vm.PdeEquation(name="R", diffusion="0.05", initial="0.0", rate="kon * (L_in + L_out) * (Rmax - R)")
                ],
                jump_conditions=[
                    vm.JumpCondition(name="L_in", in_flux="-kon * L_in * (Rmax - R)", out_flux="0.0"),
                    vm.JumpCondition(name="L_out", in_flux="0.0", out_flux="-kon * L_out * (Rmax - R)"),
                ],
            )
        ],
        constants=[vm.Constant(name="kon", exp="0.5"), vm.Constant(name="Rmax", exp="2.0")],
    )
    gd = import_geometry(vcml_geom)
    md = import_math_description(vcml_math, geometry=gd.name, dim=2)
    gd, md = normalize_to_geometry_frame(gd, md)  # the genuine pipeline (a no-op for this in-plane 2D model)
    geometry = realize_interface_coupled(
        gd,
        inner_subdomain="cytosol",
        outer_subdomain="extracellular",
        membrane_subdomain="pm",
        interface="pm",
        h=0.12,
    )

    problem = assemble_membrane_coupled(md, geometry, dt=0.02)
    assert problem.membrane_species == ["R"]
    total0 = problem.total_mass()
    inner0, outer0 = problem.mass("L_in"), problem.mass("L_out")
    for _ in range(120):
        problem.step()
    # Receptor captured ligand from both compartments (both deplete, R grows); total (free L_in + free
    # L_out + bound R) conserved to round-off.
    assert problem.mass("R") > 1e-2
    assert problem.mass("L_in") < inner0 - 1e-3 and problem.mass("L_out") < outer0 - 1e-3
    assert problem.total_mass() == pytest.approx(total0, abs=1e-9)


# --- 4. geometry-frame normalization (the VCell z-in-2D quirk) ------------------


def test_normalize_to_geometry_frame_binds_out_of_plane_coordinates() -> None:
    # VCell tolerates a 3D coordinate in a 2D model (e.g. add_sphere -> geom.x[2]); it is a unit slice
    # at z = origin.z. normalize_to_geometry_frame binds every out-of-dimension geom.x[axis] (axis >=
    # dim) to origin[axis] in BOTH the geometry and the math, so z never reaches realize / assemble.
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume
    from vcell_fenics.formalism.schema import BCNeumann, MathDescription, ParameterExpression, Subdomain, Variable
    from vcell_fenics.pyvcell_bridge import normalize_to_geometry_frame

    gd = GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(0.0, 0.0, 5.0),  # the slab sits at z = 5
        subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0] + geom.x[1] + geom.x[2]"),),
    )
    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume")],
        variables=[Variable(name="u", subdomain="cyto")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.1", "source": "geom.x[2] * u"},
                initial_condition="geom.x[0] + geom.x[2]",
            )
        ],
        parameters=[ParameterExpression(name="p", expression="geom.x[2] + 1")],
        boundary_conditions=[BCNeumann(variable="u", boundary="x_minus", expression="geom.x[2]")],
    )
    gd2, md2 = normalize_to_geometry_frame(gd, md)

    # z (geom.x[2], axis >= dim) -> origin.z = 5.0, everywhere; the in-plane x / y are untouched.
    assert gd2.subvolumes[0].expression == "geom.x[0] + geom.x[1] + (5.0)"
    (eq,) = md2.equations
    assert isinstance(eq, TemplateEquation)
    assert eq.terms["source"] == "(5.0) * u"
    assert eq.terms["diffusion"] == "0.1"
    assert eq.initial_condition == "geom.x[0] + (5.0)"
    assert md2.parameters[0].expression == "(5.0) + 1"  # type: ignore[union-attr]
    assert md2.boundary_conditions[0].expression == "(5.0)"


def test_normalize_to_geometry_frame_leaves_a_3d_model_unchanged() -> None:
    # In a 3D geometry every coordinate is in-plane, so nothing is bound.
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume
    from vcell_fenics.formalism.schema import MathDescription, Subdomain, Variable
    from vcell_fenics.pyvcell_bridge import normalize_to_geometry_frame

    gd = GeometryDescription(
        name="cell",
        dim=3,
        extent=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[2]"),),
    )
    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume")],
        variables=[Variable(name="u", subdomain="cyto")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition="geom.x[2]",
            )
        ],
    )
    gd2, md2 = normalize_to_geometry_frame(gd, md)
    assert gd2.subvolumes[0].expression == "geom.x[2]"
    (eq,) = md2.equations
    assert isinstance(eq, TemplateEquation)
    assert eq.initial_condition == "geom.x[2]"


def test_volume_region_equation_is_a_region_ode_fed_by_its_jump_condition() -> None:
    # A well-mixed cytosolic species (VCell's VolumeRegionVariable + VolumeRegionEquation, §1.4.2 T5) is a
    # `region`-space variable under `region_ode`; its membrane jump condition is its own-side flux BC —
    # the membrane term of its region balance — and a membrane expression sees it through trace(·).
    vcml = vm.MathDescription(
        name="m",
        variables=[vm.MathVariable(name="ca", var_type=vm.MathVariableType.volume_region, domain="cyto")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                volume_region_equations=[
                    vm.VolumeRegionEquation(name="ca", uniform_rate="0.0", volume_rate="-kd * ca", initial="0.1")
                ],
            )
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                jump_conditions=[vm.JumpCondition(name="ca", in_flux="j * (1.0 - ca)", out_flux="0.0")],
            )
        ],
    )
    md = import_math_description(vcml)
    (ca,) = md.variables
    assert (ca.name, ca.subdomain, ca.space) == ("ca", "cyto", "region")
    (eq,) = md.equations
    assert isinstance(eq, TemplateEquation) and eq.template == "region_ode"
    assert eq.terms == {"region_rate": "-kd * ca"}  # the zero uniform rate is the default, left out
    assert eq.initial_condition == "0.1" and eq.temporality == "time_dependent"
    (bc,) = md.boundary_conditions
    assert type(bc).__name__ == "BCNeumann" and (bc.variable, bc.boundary) == ("ca", "pm")
    assert bc.expression == "j * (1.0 - trace(ca))"


def test_membrane_region_equation_is_a_region_ode_on_the_membrane() -> None:
    # The membrane potential (MembraneRegionVariable + MembraneRegionEquation): C dV/dt = -I on the
    # membrane, a `region_ode` whose rate references the adjacent bulk species through trace(·).
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="k", diffusion="1.0", initial="1")])
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                membrane_region_equations=[
                    vm.MembraneRegionEquation(
                        name="V", uniform_rate="0.0", membrane_rate="-g * (V - log(k)) / C", initial="-70.0"
                    )
                ],
            )
        ],
    )
    md = import_math_description(vcml)
    v = next(var for var in md.variables if var.name == "V")
    assert (v.subdomain, v.space) == ("pm", "region")
    eq = next(e for e in md.equations if e.variable == "V")
    assert isinstance(eq, TemplateEquation) and eq.template == "region_ode"
    assert eq.terms == {"region_rate": "-g * (V - log(trace(k))) / C"}


def test_a_function_used_only_by_a_region_equation_is_kept() -> None:
    # A pure function referenced by nothing but a region equation's rate (a membrane potential's leak
    # conductance, VCell's `Size_membr`) is live: the dead-function pruning must count region equations.
    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="g0", exp="1.5")],
        functions=[vm.MathFunction(name="g_leak", exp="(2.0 * g0)", domain="pm")],
        compartment_subdomains=[
            vm.CompartmentSubDomain(name="cyto", pde_equations=[vm.PdeEquation(name="k", diffusion="1.0", initial="1")])
        ],
        membrane_subdomains=[
            vm.MembraneSubDomain(
                name="pm",
                inside_compartment="cyto",
                outside_compartment="ec",
                membrane_region_equations=[
                    vm.MembraneRegionEquation(
                        name="V", uniform_rate="0.0", membrane_rate="-g_leak * V", initial="-70.0"
                    )
                ],
            )
        ],
    )
    md = import_math_description(vcml)
    assert "g_leak" in {p.name for p in md.parameters}

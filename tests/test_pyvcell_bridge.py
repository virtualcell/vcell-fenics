"""Tests for the VCell→formalism bridge (`vcell_fenics.pyvcell_bridge`).

Three layers:

1. **Expression translation** — VCell coordinate/time/power syntax → the formalism's
   namespaced built-ins. Pure, needs no pyvcell.
2. **Structural import** — a VCell `MathDescription` (pyvcell's lowered math model) →
   a formalism `MathDescription` that validates, plus loud rejection of the
   out-of-scope / not-yet-implemented constructs (§2.6.3 / §2.6.2).
3. **End-to-end** — import a VCell reaction-diffusion model and actually run it through
   the FEniCSx backend, proving the import lands on a runnable, well-posed model.

The pyvcell-dependent tests skip cleanly when pyvcell is not installed (it is a local
sibling, not on PyPI — `pixi run -e dev link-pyvcell` installs it editable/--no-deps).
"""

from __future__ import annotations

import pytest

from vcell_fenics.formalism.schema import TemplateEquation
from vcell_fenics.pyvcell_bridge import VcellImportError, import_math_description, translate_expression

try:
    import pyvcell.vcml.models_math as vm  # lazy import — pulls none of pyvcell's heavy stack

    _HAVE_PYVCELL = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAVE_PYVCELL = False

needs_pyvcell = pytest.mark.skipif(
    not _HAVE_PYVCELL, reason="pyvcell not installed (run `pixi run -e dev link-pyvcell`)"
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


@needs_pyvcell
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


@needs_pyvcell
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


@needs_pyvcell
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


@needs_pyvcell
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


@needs_pyvcell
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


@needs_pyvcell
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


# --- loud rejection of out-of-scope / not-yet constructs ----------------------


@needs_pyvcell
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


@needs_pyvcell
def test_membrane_jump_conditions_are_not_yet_implemented() -> None:
    vcml = vm.MathDescription(
        name="m",
        membrane_subdomains=[
            vm.MembraneSubDomain(name="mem", jump_conditions=[vm.JumpCondition(name="J", in_flux="f")])
        ],
    )
    with pytest.raises(NotImplementedError, match="jump-condition import is a follow-up"):
        import_math_description(vcml)


@needs_pyvcell
def test_explicit_boundary_expressions_are_not_yet_implemented() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0", boundaries=vm.Boundaries(xm="0.0"))],
            )
        ],
    )
    with pytest.raises(NotImplementedError, match="per-face boundary-condition import"):
        import_math_description(vcml)


@needs_pyvcell
def test_non_flux_boundary_type_is_not_yet_implemented() -> None:
    vcml = vm.MathDescription(
        name="m",
        compartment_subdomains=[
            vm.CompartmentSubDomain(
                name="cyto",
                boundary_types=[vm.MathBoundaryType(boundary="Xm", type="Value")],
                pde_equations=[vm.PdeEquation(name="c", diffusion="1.0")],
            )
        ],
    )
    with pytest.raises(NotImplementedError, match="only the default no-flux"):
        import_math_description(vcml)


@needs_pyvcell
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


@needs_pyvcell
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

"""Non-diffusing species (#186) and what VCell's electrophysiology models need beside them.

A VCell `OdeEquation` on a spatial compartment or membrane (T4 `lumped_ode`) is a field without transport:
a value at every point, ``du/dt = rate(u, x, t)`` there, no flux between points. The solvers assemble it
as a transport-free field block (`backend/equations.as_field_equation`). The checks:

- **pointwise exactness:** with no transport, ``M u̇ = −k M u`` is ``u̇ = −k u`` at every node, so a decaying
  ODE matches ``u0(x) e^{−kt}`` node by node, whatever the initial shape;
- **conservation:** a diffusing species binding a non-diffusing buffer conserves the total;
- **membrane ODEs are honored:** a gating variable's ``rate`` reaches the membrane-coupled solver. That
  slot was silently ignored there;
- **``sim.t`` is live** in the coupled solvers (a time-dependent stimulus), not frozen at t = 0;
- **VCell's per-face Dirichlet BCs:** the same value on every box face is a reservoir on the outer wall,
  and an interior compartment's per-face BCs are dropped;
- **refusal:** a non-diffusing species on a moving subdomain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    assemble,
    integrate_interface_coupled,
    integrate_membrane_coupled,
    make_two_bulk_membrane_geometry,
)
from vcell_fenics.backend.geometry import make_disk_geometry
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    MathDescription,
    MotionPrescribedVelocity,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _ode(variable: str, subdomain: str, rate: str, ic: str) -> TemplateEquation:
    return TemplateEquation(
        template="lumped_ode",
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"rate": rate},
        initial_condition=ic,
    )


def _pde(variable: str, subdomain: str, ic: str, **terms: str) -> TemplateEquation:
    template = "surface_pde_with_dilution" if subdomain == "mem" else "bulk_radv_diff"
    return TemplateEquation(
        template=template,
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"diffusion": "1.0", **terms},
        initial_condition=ic,
    )


def _two_compartments() -> InterfaceCoupledGeometry:
    return make_two_bulk_membrane_geometry(
        "cell", inner="cyto", outer_subdomain="ext", membrane="mem", interface="mem", outer="wall", h=0.1
    )


def _integral(field: fem.Function) -> float:
    mesh = field.function_space.mesh
    return float(fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh))).real)


# --- the single-mesh path -------------------------------------------------------------------------------


def test_a_non_diffusing_species_decays_pointwise_on_a_single_mesh() -> None:
    md = MathDescription(
        geometry="disk",
        subdomains=[Subdomain(name="cell", kind="volume")],
        variables=[Variable(name="u", subdomain="cell")],
        parameters=[ParameterConstant(name="k", value=2.0)],
        equations=[_ode("u", "cell", "-k * u", "1.0 + geom.x[0] * geom.x[0]")],
    )
    problem = assemble(md, make_disk_geometry("disk", volume_subdomain="cell", h=0.15), dt=0.05)
    result = integrate_discrete_problem(problem, t_final=0.5, rtol=1e-9, atol=1e-12)
    x = problem.V.tabulate_dof_coordinates()[:, 0]
    ratio = result.solution.x.array / (1.0 + x * x)
    # no transport: every node decays by the same factor, whatever the initial shape ...
    assert np.ptp(ratio) < 1e-9
    # ... and that factor is e^{-kt}, to the single-mesh integrator's BDF cold-start error (≈ k·dt₀)
    assert float(ratio.mean()) == pytest.approx(math.exp(-2.0 * 0.5), rel=5e-4)


def test_a_non_diffusing_species_on_a_moving_subdomain_is_refused() -> None:
    md = MathDescription(
        geometry="disk",
        subdomains=[Subdomain(name="cell", kind="volume", motion=MotionPrescribedVelocity(velocity="[0.1, 0.0]"))],
        variables=[Variable(name="u", subdomain="cell")],
        equations=[_ode("u", "cell", "-u", "1.0")],
    )
    with pytest.raises(NotImplementedError, match="non-diffusing species"):
        assemble(md, make_disk_geometry("disk", volume_subdomain="cell", h=0.2), dt=0.05)


# --- the two-compartment solver ---------------------------------------------------------------------------


def _buffer_model() -> MathDescription:
    """A diffusing c binding an immobile buffer B in the cytosol (c + B ⇌ CB, CB immobile too), exchanging
    with an extracellular u; the ext side also carries a decoupled pointwise decay w."""
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="mem", kind="surface"),
        ],
        variables=[
            Variable(name="c", subdomain="cyto"),
            Variable(name="B", subdomain="cyto"),
            Variable(name="CB", subdomain="cyto"),
            Variable(name="u", subdomain="ext"),
            Variable(name="w", subdomain="ext"),
        ],
        parameters=[ParameterConstant(name="kon", value=3.0), ParameterConstant(name="koff", value=0.5)],
        equations=[
            _pde("c", "cyto", "1.0", source="-kon * c * B + koff * CB"),
            _ode("B", "cyto", "-kon * c * B + koff * CB", "2.0"),
            _ode("CB", "cyto", "kon * c * B - koff * CB", "0.0"),
            _pde("u", "ext", "0.0"),
            _ode("w", "ext", "-w", "1.0 + geom.x[1]"),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="c", boundary="mem", expression="0.5 * (u - c)"),
            BCInterfaceFlux(variable="u", boundary="mem", expression="0.5 * (c - u)"),
        ],
    )


def test_a_buffered_exchange_conserves_calcium_and_decays_pointwise() -> None:
    result = integrate_interface_coupled(_buffer_model(), _two_compartments(), t_final=1.0, rtol=1e-8, atol=1e-11)
    assert result.fields is not None
    fields = result.fields
    geometry_start = integrate_interface_coupled(_buffer_model(), _two_compartments(), t_final=1e-9)
    assert geometry_start.fields is not None

    def total(f: dict[str, fem.Function]) -> float:
        return _integral(f["c"]) + _integral(f["CB"]) + _integral(f["u"])

    assert total(fields) == pytest.approx(total(geometry_start.fields), rel=1e-7)
    assert _integral(fields["CB"]) > 0.1  # the buffer did bind
    # the buffer (free + bound) is immobile and conserved point by point: B + CB = 2 at every node
    assert np.allclose(fields["B"].x.array + fields["CB"].x.array, 2.0, atol=1e-6)
    # the decoupled pointwise decay on the other side: w(x, 1) = (1 + y) e^{-1}
    y = fields["w"].function_space.tabulate_dof_coordinates()[:, 1]
    assert np.allclose(fields["w"].x.array, (1.0 + y) * math.exp(-1.0), rtol=1e-5)


def test_sim_t_is_live_in_the_coupled_solver() -> None:
    # a time-dependent source, dA/dt = sim.t on a well-mixed pool: A(T) = T²/2 (it was frozen at t = 0)
    md = _buffer_model()
    md = MathDescription(
        geometry=md.geometry,
        subdomains=md.subdomains,
        variables=[*md.variables, Variable(name="A", subdomain="cyto", space="region")],
        parameters=md.parameters,
        equations=[
            *md.equations,
            TemplateEquation(
                template="region_ode",
                variable="A",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"uniform_rate": "sim.t"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=md.boundary_conditions,
    )
    result = integrate_interface_coupled(md, _two_compartments(), t_final=1.0, rtol=1e-9, atol=1e-12)
    assert result.fields is not None
    assert float(result.fields["A"].x.array[0]) == pytest.approx(0.5, rel=1e-6)


def test_the_same_value_on_every_box_face_is_the_outer_reservoir() -> None:
    # VCell writes a Value BC per box face, for every compartment. The outer species' (all the same) is the
    # reservoir on the outer wall; the interior cytosol's has no face to act on and is dropped.
    base = _buffer_model()
    faces = ("x_minus", "x_plus", "y_minus", "y_plus")

    def with_bcs(extra: list[BCDirichlet]) -> MathDescription:
        return MathDescription(
            geometry=base.geometry,
            subdomains=base.subdomains,
            variables=base.variables,
            parameters=base.parameters,
            equations=base.equations,
            boundary_conditions=[*base.boundary_conditions, *extra],
        )

    per_face = with_bcs(
        [BCDirichlet(variable="u", boundary=f, expression="0.3") for f in faces]
        + [BCDirichlet(variable="c", boundary=f, expression="9.0") for f in faces]
    )
    wall = with_bcs([BCDirichlet(variable="u", boundary="wall", expression="0.3")])
    a = integrate_interface_coupled(per_face, _two_compartments(), t_final=0.5)
    b = integrate_interface_coupled(wall, _two_compartments(), t_final=0.5)
    assert a.fields is not None and b.fields is not None
    for name in ("c", "u", "CB"):
        assert np.allclose(a.fields[name].x.array, b.fields[name].x.array, rtol=1e-8, atol=1e-10), name

    differing = with_bcs([BCDirichlet(variable="u", boundary=f, expression=str(i)) for i, f in enumerate(faces)])
    with pytest.raises(NotImplementedError, match="same value on every box face"):
        integrate_interface_coupled(differing, _two_compartments(), t_final=0.1)


# --- the membrane-coupled solver ---------------------------------------------------------------------------


def test_a_membrane_gating_ode_is_honored_by_the_membrane_coupled_solver() -> None:
    # dh/dt = (h_inf − h)/tau on the membrane, h_inf varying along it: h(T) = h_inf + (h0 − h_inf) e^{−T/tau}
    # at every membrane node. Its `rate` slot used to be ignored there (the solver read only `source`).
    md = MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="mem", kind="surface"),
        ],
        variables=[
            Variable(name="c", subdomain="cyto"),
            Variable(name="u", subdomain="ext"),
            Variable(name="h", subdomain="mem"),
        ],
        parameters=[ParameterConstant(name="tau", value=0.25)],
        equations=[
            _pde("c", "cyto", "1.0"),
            _pde("u", "ext", "0.0"),
            _ode("h", "mem", "((0.5 + 0.2 * geom.x[0]) - h) / tau", "1.0"),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="c", boundary="mem", expression="0.1 * h * (u - c)"),
            BCInterfaceFlux(variable="u", boundary="mem", expression="0.1 * h * (c - u)"),
        ],
    )
    result = integrate_membrane_coupled(md, _two_compartments(), t_final=0.5, rtol=1e-9, atol=1e-12)
    h = result.field("h")
    x = h.function_space.tabulate_dof_coordinates()[:, 0]
    h_inf = 0.5 + 0.2 * x
    exact = h_inf + (1.0 - h_inf) * math.exp(-0.5 / 0.25)
    assert np.max(np.abs(h.x.array - exact)) < 1e-5

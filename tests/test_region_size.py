"""`region_size(<subdomain>)` (§1.8.4): a subdomain's realized measure, VCell's `vcRegionVolume('X')` /
`vcRegionArea('X')` (#199).

A solver binds each subdomain's measure from its mesh: the compartments' volumes (areas in 2D) and the
membrane's area (length in 2D). The checks:
- the bound numbers are the mesh's own measures;
- an expression that uses them gets them: a region variable whose rate is a size grows by exactly that
  size per unit time, and a bulk source made of a size adds that mass;
- a moving mesh, whose sizes change in time, refuses them (not yet);
- the VCell import translates the built-ins and keeps a size function a region equation references.
"""

from __future__ import annotations

import math

import pytest
import pyvcell.vcml.models_math as vm
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    assemble,
    integrate_interface_coupled,
    make_two_bulk_membrane_geometry,
)
from vcell_fenics.backend.compiler import CompileError
from vcell_fenics.backend.geometry import make_disk_geometry
from vcell_fenics.backend.interface_coupled import _geometry_region_sizes
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism.schema import (
    MathDescription,
    MotionPrescribedVelocity,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
)
from vcell_fenics.pyvcell_bridge import import_math_description


def _two_compartments() -> InterfaceCoupledGeometry:
    return make_two_bulk_membrane_geometry(
        "cell", inner="cyto", outer_subdomain="ext", membrane="mem", interface="membrane", outer="wall", h=0.09
    )


def _integral(mesh: object, integrand: object) -> float:
    return float(fem.assemble_scalar(fem.form(integrand * ufl.dx(domain=mesh))).real)


def test_the_bound_sizes_are_the_meshes_own_measures() -> None:
    geometry = _two_compartments()
    sizes = _geometry_region_sizes(geometry)
    assert sizes["cyto"] == pytest.approx(_integral(geometry.inner_mesh, fem.Constant(geometry.inner_mesh, 1.0)))
    assert sizes["ext"] == pytest.approx(_integral(geometry.outer_mesh, fem.Constant(geometry.outer_mesh, 1.0)))
    membrane = geometry.membrane_mesh
    assert sizes["mem"] == pytest.approx(_integral(membrane, fem.Constant(membrane, 1.0)))
    # and they are the disk's: area π r² and circumference 2π r, to the polygon's resolution
    assert sizes["cyto"] == pytest.approx(math.pi * 0.25, rel=1e-2)
    assert sizes["mem"] == pytest.approx(2.0 * math.pi * 0.5, rel=1e-2)


def _pde(variable: str, subdomain: str, **terms: str) -> TemplateEquation:
    return TemplateEquation(
        template="bulk_radv_diff",
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"diffusion": "1.0", **terms},
        initial_condition="0.0",
    )


def test_a_region_variable_whose_rate_is_a_size_grows_by_that_size() -> None:
    # dA/dt = region_size(cyto) + region_size(mem) (a uniform rate): A(T) = (|cyto| + |mem|)·T, exactly
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
            Variable(name="A", subdomain="cyto", space="region"),
        ],
        parameters=[ParameterExpression(name="Size_mem", expression="region_size(mem)")],
        equations=[
            _pde("c", "cyto"),
            _pde("u", "ext"),
            TemplateEquation(
                template="region_ode",
                variable="A",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"uniform_rate": "region_size(cyto) + Size_mem"},
                initial_condition="0.0",
            ),
        ],
    )
    geometry = _two_compartments()
    sizes = _geometry_region_sizes(geometry)
    result = integrate_interface_coupled(md, geometry, t_final=0.5)
    assert result.fields is not None
    value = float(result.fields["A"].x.array[0])
    assert value == pytest.approx((sizes["cyto"] + sizes["mem"]) * 0.5, rel=1e-8)


def _disk_source_model(*, moving: bool) -> MathDescription:
    """u_t = Δu + region_size(cell) on a disk: the mass grows at |cell|² per unit time."""
    motion = MotionPrescribedVelocity(velocity="[0.1, 0.0]") if moving else None
    subdomain = (
        Subdomain(name="cell", kind="volume", motion=motion) if motion else Subdomain(name="cell", kind="volume")
    )
    return MathDescription(
        geometry="disk",
        subdomains=[subdomain],
        variables=[Variable(name="u", subdomain="cell")],
        equations=[_pde("u", "cell", source="region_size(cell)")],
    )


def test_a_single_mesh_source_made_of_a_size_adds_that_mass() -> None:
    geometry = make_disk_geometry("disk", volume_subdomain="cell", h=0.1)
    problem = assemble(_disk_source_model(moving=False), geometry, dt=0.05)
    area = _integral(problem.V.mesh, fem.Constant(problem.V.mesh, 1.0))
    result = integrate_discrete_problem(problem, t_final=0.4)
    mass = _integral(problem.V.mesh, result.solution)
    assert mass == pytest.approx(area * area * 0.4, rel=1e-8)


def test_a_moving_mesh_refuses_region_sizes() -> None:
    # on a moving mesh a size changes in time; that is a later increment, refused with a clear message
    geometry = make_disk_geometry("disk", volume_subdomain="cell", h=0.2)
    with pytest.raises(CompileError, match="region_size"):
        assemble(_disk_source_model(moving=True), geometry, dt=0.05)


def test_the_vcell_built_ins_translate_and_a_size_a_region_equation_uses_is_kept() -> None:
    vcml = vm.MathDescription(
        name="m",
        constants=[vm.Constant(name="AreaPerUnitArea_pm", exp="1.0")],
        functions=[
            vm.MathFunction(name="Size_pm", exp="(AreaPerUnitArea_pm * vcRegionArea('pm'))", domain="pm"),
            vm.MathFunction(name="Size_cyto", exp="vcRegionVolume('cyto')", domain="cyto"),  # unreferenced: dropped
        ],
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
                        name="V", uniform_rate="0.0", membrane_rate="-V / Size_pm", initial="-70.0"
                    )
                ],
            )
        ],
    )
    md = import_math_description(vcml)
    expressions = {p.name: getattr(p, "expression", None) for p in md.parameters}
    assert expressions["Size_pm"] == "(AreaPerUnitArea_pm * region_size(pm))"
    assert "Size_cyto" not in expressions

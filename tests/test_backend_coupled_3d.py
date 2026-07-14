"""3D interface-coupled reaction–diffusion — the fvsolver-comparison prerequisite (dim 3).

`realize_interface_coupled` now realizes a 3D geometry (Netgen multi-region, ADR 008 §8), and the
coupled solver is dimension-generic (it derives gdim/tdim from the mesh). This exercises the whole
path end-to-end in 3D: two bulk species (inside + outside a sphere) coupled by a membrane permeability
flux, integrated to equilibrium. The checks are the two fvsolver-comparison metrics that hold on a
closed system: **mass conservation** (exact) and correct **mass-weighted equilibrium**.
"""

from __future__ import annotations

import math

import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh

from vcell_fenics.backend import assemble_membrane_coupled, integrate_interface_coupled
from vcell_fenics.backend.geometry import InterfaceCoupledGeometry
from vcell_fenics.backend.realize import realize_interface_coupled
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _sphere_in_box(radius: float = 1.0) -> GeometryDescription:
    r2 = radius * radius
    return GeometryDescription(
        name="cell3d",
        dim=3,
        extent=(4.0, 4.0, 4.0),
        origin=(-2.0, -2.0, -2.0),
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < {r2}"),
            SubVolume(
                name="extracellular", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 > {r2}"
            ),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
    )


def _mass(field: fem.Function) -> float:
    return float(fem.assemble_scalar(fem.form(field * ufl.dx)).real)


def _volume(mesh: Mesh) -> float:
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh))).real)


@pytest.mark.integration
def test_3d_interface_coupled_conserves_mass_and_equilibrates() -> None:
    radius = 1.0
    geometry = realize_interface_coupled(
        _sphere_in_box(radius),
        inner_subdomain="cytosol",
        outer_subdomain="extracellular",
        membrane_subdomain="pm",
        interface="pm",
        h=0.4,
    )
    md = MathDescription(
        geometry="cell3d",
        subdomains=[Subdomain(name="cytosol", kind="volume"), Subdomain(name="extracellular", kind="volume")],
        variables=[Variable(name="u", subdomain="cytosol"), Variable(name="v", subdomain="extracellular")],
        parameters=[ParameterConstant(name="P", value=0.5)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="cytosol",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="v",
                subdomain="extracellular",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u", boundary="pm", expression="P * (trace(v) - trace(u))"),
            BCInterfaceFlux(variable="v", boundary="pm", expression="P * (trace(u) - trace(v))"),
        ],
    )
    result = integrate_interface_coupled(md, geometry, t_final=8.0)

    v_in = _volume(result.inner.function_space.mesh)
    v_out = _volume(result.outer.function_space.mesh)
    m_in, m_out = _mass(result.inner), _mass(result.outer)

    # The two compartments tile the box exactly, and the initial mass (u ≡ 1 in cytosol) is conserved.
    assert v_in + v_out == pytest.approx(64.0, rel=1e-6)
    assert m_in + m_out == pytest.approx(v_in, rel=1e-4)  # closed system: total mass = initial cytosol mass

    # The permeability flux drives both compartments to the mass-weighted equilibrium concentration.
    eq = (m_in + m_out) / (v_in + v_out)
    assert m_in / v_in == pytest.approx(eq, rel=5e-2)
    assert m_out / v_out == pytest.approx(eq, rel=5e-2)
    assert eq == pytest.approx(4.0 / 3.0 * math.pi * radius**3 / 64.0, rel=0.15)  # ≈ V_sphere / V_box, coarse mesh


def _receptor_model() -> MathDescription:
    """Two bulk ligands (inside/outside) and a membrane receptor `R` that irreversibly captures ligand
    from both compartments (a `surface_pde_with_dilution` species: surface diffusion + binding). The bulk
    fluxes are the exact negation of the surface production, so total ligand (free in + free out + bound)
    is conserved."""
    return MathDescription(
        geometry="cell3d",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[
            Variable(name="L_in", subdomain="cyto"),
            Variable(name="L_out", subdomain="ext"),
            Variable(name="R", subdomain="pm"),
        ],
        parameters=[ParameterConstant(name="kon", value=0.5), ParameterConstant(name="Rmax", value=2.0)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05", "source": "kon * (trace(L_in) + trace(L_out)) * (Rmax - R)"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="L_in", boundary="pm", expression="-kon * trace(L_in) * (Rmax - R)"),
            BCInterfaceFlux(variable="L_out", boundary="pm", expression="-kon * trace(L_out) * (Rmax - R)"),
        ],
    )


def _receptor_geometry() -> InterfaceCoupledGeometry:
    r2 = 1.0
    gd = GeometryDescription(
        name="cell3d",
        dim=3,
        extent=(4.0, 4.0, 4.0),
        origin=(-2.0, -2.0, -2.0),
        subvolumes=(
            SubVolume(name="cyto", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 < {r2}"),
            SubVolume(name="ext", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 + geom.x[2]**2 > {r2}"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cyto", outside="ext"),),
    )
    return realize_interface_coupled(
        gd, inner_subdomain="cyto", outer_subdomain="ext", membrane_subdomain="pm", interface="pm", h=0.45
    )


@pytest.mark.integration
def test_3d_membrane_surface_species_captures_and_conserves() -> None:
    # A surface PDE on the 2D-in-3D membrane: the receptor binds ligand (bound mass grows) while total
    # ligand across both bulks + the membrane is conserved to round-off.
    problem = assemble_membrane_coupled(_receptor_model(), _receptor_geometry(), dt=0.02)
    total0 = problem.total_mass()
    bound0 = problem.mass("R")
    for _ in range(40):
        problem.step()

    assert problem.mass("R") > bound0 + 1.0  # the membrane receptor captured ligand
    assert problem.total_mass() == pytest.approx(total0, abs=1e-8)  # closed system, conserved to round-off

"""The multi-compartment solver: any number of compartments and membranes on one partition.

The checks:
- **exact conservation** across two membranes (nucleus | cytosol | extracellular space), and with a
  membrane species binding ligand from one side;
- **the well-mixed limit:** with fast diffusion the mean concentrations follow the three-pool exchange ODE
  (from the realized volumes and areas), and the gap shrinks as D grows;
- **parity:** the two-compartment receptor model gives what `integrate_membrane_coupled` gives on the same mesh;
- **box faces:** a Dirichlet value holds on the compartment's own share of a face, per face;
- **a region variable** (a well-mixed nucleus) exchanges with a cytosolic field and conserves the total;
- **touching cells:** two cells sharing a membrane, each also facing the outside, meet at junction points.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sla
import ufl
from dolfinx import fem

from vcell_fenics.backend.interface_coupled import integrate_membrane_coupled
from vcell_fenics.backend.multi_compartment import (
    MultiCompartmentGeometry,
    integrate_multi_compartment,
    realize_multi_compartment,
    species_mass,
)
from vcell_fenics.backend.realize import realize_interface_coupled
from vcell_fenics.cli import load_pair
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)

_CV = Path(__file__).parent.parent / "cross_validation"


def _nested() -> GeometryDescription:
    """A nucleus (r = 0.3) in a cytosol (r = 0.7) in a 2 × 2 box of extracellular space."""

    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="nuc", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.09"),
            SubVolume(name="cyt", type="analytic", expression="geom.x[0] ** 2 + geom.x[1] ** 2 < 0.49"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="ne", inside="nuc", outside="cyt"),
            SurfaceClass(name="pm", inside="cyt", outside="ec"),
        ),
    )


def _pde(variable: str, subdomain: str, ic: str, diffusion: str = "D", **terms: str) -> TemplateEquation:
    template = "surface_pde_with_dilution" if subdomain in ("ne", "pm") else "bulk_radv_diff"
    return TemplateEquation(
        template=template,
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"diffusion": diffusion, **terms},
        initial_condition=ic,
    )


def _subdomains(volumes: tuple[str, ...], surfaces: tuple[str, ...]) -> list[Subdomain]:
    return [Subdomain(name=n, kind="volume") for n in volumes] + [Subdomain(name=n, kind="surface") for n in surfaces]


def _exchange(diffusion: float) -> MathDescription:
    """n (nucleus) ⇌ c (cytosol) ⇌ e (outside) by permeabilities P1 across ne and P2 across pm."""

    return MathDescription(
        geometry="cell",
        subdomains=_subdomains(("nuc", "cyt", "ec"), ("ne", "pm")),
        variables=[
            Variable(name="n", subdomain="nuc"),
            Variable(name="c", subdomain="cyt"),
            Variable(name="e", subdomain="ec"),
        ],
        parameters=[
            ParameterConstant(name="D", value=diffusion),
            ParameterConstant(name="P1", value=0.5),
            ParameterConstant(name="P2", value=0.3),
        ],
        equations=[_pde("n", "nuc", "3.0"), _pde("c", "cyt", "1.0 + geom.x[0]"), _pde("e", "ec", "0.0")],
        boundary_conditions=[
            BCInterfaceFlux(variable="n", boundary="ne", expression="P1 * (c - n)"),
            BCInterfaceFlux(variable="c", boundary="ne", expression="P1 * (n - c)"),
            BCInterfaceFlux(variable="c", boundary="pm", expression="P2 * (e - c)"),
            BCInterfaceFlux(variable="e", boundary="pm", expression="P2 * (c - e)"),
        ],
    )


@pytest.fixture(scope="module")
def nested() -> MultiCompartmentGeometry:
    return realize_multi_compartment(_nested(), h=0.06)


def _total(fields: dict[str, float]) -> float:
    return sum(fields.values())


def test_the_partition_carries_every_compartment_and_membrane(nested: MultiCompartmentGeometry) -> None:
    assert sorted(nested.compartments) == ["cyt", "ec", "nuc"]
    assert sorted(nested.membranes) == ["ne", "pm"]
    assert nested.sizes["nuc"] == pytest.approx(np.pi * 0.09, rel=1e-2)
    assert nested.sizes["cyt"] == pytest.approx(np.pi * (0.49 - 0.09), rel=5e-3)
    assert nested.sizes["pm"] == pytest.approx(2 * np.pi * 0.7, rel=5e-3)
    # only the outside reaches the box
    assert set(nested.compartments["ec"].faces) == {"x_minus", "x_plus", "y_minus", "y_plus"}
    assert nested.compartments["nuc"].faces == {} and nested.compartments["cyt"].faces == {}


def test_an_exchange_across_two_membranes_conserves(nested: MultiCompartmentGeometry) -> None:
    md = _exchange(1.0)
    start = integrate_multi_compartment(md, nested, t_final=1e-9)
    end = integrate_multi_compartment(md, nested, t_final=1.0, rtol=1e-8, atol=1e-11)
    before = {name: species_mass(start, name) for name in "nce"}
    after = {name: species_mass(end, name) for name in "nce"}
    assert _total(after) == pytest.approx(_total(before), rel=1e-12)
    assert after["e"] > 0.05 * _total(before)  # it crossed both membranes


def test_fast_diffusion_approaches_the_three_pool_exchange(nested: MultiCompartmentGeometry) -> None:
    s = nested.sizes
    vn, vc, ve, a1, a2 = s["nuc"], s["cyt"], s["ec"], s["ne"], s["pm"]
    rates = np.array(
        [
            [-0.5 * a1 / vn, 0.5 * a1 / vn, 0.0],
            [0.5 * a1 / vc, -(0.5 * a1 + 0.3 * a2) / vc, 0.3 * a2 / vc],
            [0.0, 0.3 * a2 / ve, -0.3 * a2 / ve],
        ]
    )
    start = integrate_multi_compartment(_exchange(1.0), nested, t_final=1e-9)
    means0 = np.array([species_mass(start, x) for x in "nce"]) / np.array([vn, vc, ve])
    exact = sla.expm(rates * 0.5) @ means0
    gaps = []
    for diffusion in (20.0, 200.0):
        end = integrate_multi_compartment(_exchange(diffusion), nested, t_final=0.5, rtol=1e-9, atol=1e-12)
        means = np.array([species_mass(end, x) for x in "nce"]) / np.array([vn, vc, ve])
        gaps.append(float(np.max(np.abs(means / exact - 1.0))))
    assert gaps[1] < 2e-3
    assert gaps[1] < 0.2 * gaps[0]  # the well-mixed limit: the gap closes like 1/D


def _receptor(nested_md: bool = False) -> tuple[GeometryDescription, MathDescription]:
    model = load_pair(_CV / "receptor_math.yaml", _CV / "receptor_geom.yaml")
    return model.geometry, model.math


def test_the_receptor_model_matches_the_membrane_coupled_solver() -> None:
    description, md = _receptor()
    two = realize_interface_coupled(
        description,
        inner_subdomain="cyto_dom",
        outer_subdomain="ext_dom",
        membrane_subdomain="mem_dom",
        interface="mem_dom",
        h=0.1,
    )
    reference = integrate_membrane_coupled(md, two, t_final=0.5, rtol=1e-9, atol=1e-12)
    many = realize_multi_compartment(description, h=0.1)
    result = integrate_multi_compartment(md, many, t_final=0.5, rtol=1e-9, atol=1e-12)
    for name in ("s_cyto", "s_ext", "R"):
        expected = reference.mass(name)
        assert species_mass(result, name) == pytest.approx(expected, rel=1e-6), name


def test_a_membrane_species_binding_from_outside_conserves(nested: MultiCompartmentGeometry) -> None:
    # ligand e binds receptors R on pm; the nucleus and the cytosol carry an inert exchange
    md = MathDescription(
        geometry="cell",
        subdomains=_subdomains(("nuc", "cyt", "ec"), ("ne", "pm")),
        variables=[
            Variable(name="n", subdomain="nuc"),
            Variable(name="c", subdomain="cyt"),
            Variable(name="e", subdomain="ec"),
            Variable(name="R", subdomain="pm"),
        ],
        parameters=[ParameterConstant(name="D", value=1.0), ParameterConstant(name="kon", value=2.0)],
        equations=[
            _pde("n", "nuc", "1.0"),
            _pde("c", "cyt", "0.0"),
            _pde("e", "ec", "1.0"),
            _pde("R", "pm", "0.0", diffusion="0.01", source="kon * trace(e) * (1.0 - R)"),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="n", boundary="ne", expression="0.5 * (c - n)"),
            BCInterfaceFlux(variable="c", boundary="ne", expression="0.5 * (n - c)"),
            BCInterfaceFlux(variable="e", boundary="pm", expression="-kon * trace(e) * (1.0 - R)"),
        ],
    )
    start = integrate_multi_compartment(md, nested, t_final=1e-9)
    end = integrate_multi_compartment(md, nested, t_final=0.5, rtol=1e-8, atol=1e-11)
    bound, free = (species_mass(r, "e") + species_mass(r, "R") for r in (start, end))
    assert free == pytest.approx(bound, rel=1e-9)
    assert species_mass(end, "R") > 0.1
    inert = [species_mass(r, "n") + species_mass(r, "c") for r in (start, end)]
    assert inert[1] == pytest.approx(inert[0], rel=1e-12)


def test_a_box_face_value_holds_on_the_compartments_share_of_that_face() -> None:
    # a cytosol cut by the box face x = -1 (a half cell); e held at 2 on x_plus only, c at 1 on x_minus only
    description = GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cyt", type="analytic", expression="(geom.x[0] + 1) ** 2 + geom.x[1] ** 2 < 0.49"),
            SubVolume(name="nuc", type="analytic", expression="(geom.x[0] - 0.4) ** 2 + geom.x[1] ** 2 < 0.04"),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="pm", inside="cyt", outside="ec"),
            SurfaceClass(name="ne", inside="nuc", outside="ec"),
        ),
    )
    geometry = realize_multi_compartment(description, h=0.08)
    md = MathDescription(
        geometry="cell",
        subdomains=_subdomains(("cyt", "nuc", "ec"), ("pm", "ne")),
        variables=[Variable(name="c", subdomain="cyt"), Variable(name="e", subdomain="ec")],
        parameters=[ParameterConstant(name="D", value=1.0)],
        equations=[_pde("c", "cyt", "0.0"), _pde("e", "ec", "0.0")],
        boundary_conditions=[
            BCDirichlet(variable="e", boundary="x_plus", expression="2.0"),
            BCDirichlet(variable="c", boundary="x_minus", expression="1.0"),
            BCDirichlet(variable="c", boundary="x_plus", expression="5.0"),  # c never reaches x_plus: dropped
        ],
    )
    result = integrate_multi_compartment(md, geometry, t_final=20.0, rtol=1e-8, atol=1e-11)
    # no membrane flux: each compartment relaxes to its own face's value (the nucleus is a hole in ec)
    assert np.allclose(result.fields["c"].x.array, 1.0, atol=1e-4)
    assert np.allclose(result.fields["e"].x.array, 2.0, atol=1e-4)


def test_a_well_mixed_nucleus_exchanges_with_the_cytosol(nested: MultiCompartmentGeometry) -> None:
    md = MathDescription(
        geometry="cell",
        subdomains=_subdomains(("nuc", "cyt", "ec"), ("ne", "pm")),
        variables=[
            Variable(name="N", subdomain="nuc", space="region"),
            Variable(name="c", subdomain="cyt"),
            Variable(name="e", subdomain="ec"),
        ],
        parameters=[ParameterConstant(name="D", value=1.0)],
        equations=[
            TemplateEquation(
                template="region_ode",
                variable="N",
                subdomain="nuc",
                temporality="time_dependent",
                terms={"uniform_rate": "0.0"},
                initial_condition="4.0",
            ),
            _pde("c", "cyt", "0.0"),
            _pde("e", "ec", "0.0"),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="N", boundary="ne", expression="0.5 * (c - N)"),
            BCInterfaceFlux(variable="c", boundary="ne", expression="0.5 * (N - c)"),
            BCInterfaceFlux(variable="c", boundary="pm", expression="0.2 * (e - c)"),
            BCInterfaceFlux(variable="e", boundary="pm", expression="0.2 * (c - e)"),
        ],
    )
    start = integrate_multi_compartment(md, nested, t_final=1e-9)
    end = integrate_multi_compartment(md, nested, t_final=1.0, rtol=1e-8, atol=1e-11)
    volume = nested.sizes["nuc"]

    def total(r: object) -> float:
        assert hasattr(r, "fields")
        n = float(r.fields["N"].x.array[0]) * volume
        return n + species_mass(r, "c") + species_mass(r, "e")  # type: ignore[arg-type]

    assert total(end) == pytest.approx(total(start), rel=1e-10)
    assert float(end.fields["N"].x.array[0]) < 4.0 * 0.9


def test_two_touching_cells_meet_at_junctions() -> None:
    # two cells side by side, sharing a membrane (gap junction) and each facing the outside
    description = GeometryDescription(
        name="pair",
        dim=2,
        extent=(3.0, 2.0, 1.0),
        origin=(-1.5, -1.0, 0.0),
        subvolumes=(
            SubVolume(
                name="a", type="analytic", expression="(geom.x[0] + 0.45) ** 2 + geom.x[1] ** 2 < 0.36 && geom.x[0] < 0"
            ),
            SubVolume(
                name="b",
                type="analytic",
                expression="(geom.x[0] - 0.45) ** 2 + geom.x[1] ** 2 < 0.36 && geom.x[0] >= 0",
            ),
            SubVolume(name="ec", type="analytic", expression="1"),
        ),
        surfaces=(
            SurfaceClass(name="gj", inside="a", outside="b"),
            SurfaceClass(name="pa", inside="a", outside="ec"),
            SurfaceClass(name="pb", inside="b", outside="ec"),
        ),
    )
    geometry = realize_multi_compartment(description, h=0.05)
    assert sorted(geometry.membranes) == ["gj", "pa", "pb"]
    md = MathDescription(
        geometry="pair",
        subdomains=_subdomains(("a", "b", "ec"), ("gj", "pa", "pb")),
        variables=[
            Variable(name="u", subdomain="a"),
            Variable(name="v", subdomain="b"),
            Variable(name="e", subdomain="ec"),
        ],
        parameters=[ParameterConstant(name="D", value=1.0)],
        equations=[_pde("u", "a", "1.0"), _pde("v", "b", "0.0"), _pde("e", "ec", "0.0")],
        boundary_conditions=[
            BCInterfaceFlux(variable="u", boundary="gj", expression="1.0 * (v - u)"),
            BCInterfaceFlux(variable="v", boundary="gj", expression="1.0 * (u - v)"),
            BCInterfaceFlux(variable="u", boundary="pa", expression="0.1 * (e - u)"),
            BCInterfaceFlux(variable="e", boundary="pa", expression="0.1 * (u - e)"),
            BCInterfaceFlux(variable="v", boundary="pb", expression="0.1 * (e - v)"),
            BCInterfaceFlux(variable="e", boundary="pb", expression="0.1 * (v - e)"),
        ],
    )
    start = integrate_multi_compartment(md, geometry, t_final=1e-9)
    end = integrate_multi_compartment(md, geometry, t_final=1.0, rtol=1e-8, atol=1e-11)
    total = [sum(species_mass(r, x) for x in "uve") for r in (start, end)]
    assert total[1] == pytest.approx(total[0], rel=1e-12)
    # both routes carried it: through the gap junction into b, and out through both cells' membranes
    assert species_mass(end, "v") > 0.1 and species_mass(end, "e") > 0.1


def _field_integral(f: fem.Function) -> float:
    return float(fem.assemble_scalar(fem.form(f * ufl.dx(domain=f.function_space.mesh))).real)


def test_a_membrane_species_nothing_else_reads(nested: MultiCompartmentGeometry) -> None:
    # a surface species that only diffuses, beside a bulk exchange: its column has no coupling block, which
    # DOLFINx can't place without a (structural) diagonal — found by the coverage survey (101449802)
    md = replace(
        _exchange(1.0),
        variables=[*_exchange(1.0).variables, Variable(name="L", subdomain="ne")],
        equations=[*_exchange(1.0).equations, _pde("L", "ne", "1.0 + geom.x[0]", diffusion="0.1")],
    )
    start = integrate_multi_compartment(md, nested, t_final=1e-9)
    end = integrate_multi_compartment(md, nested, t_final=0.5, rtol=1e-8, atol=1e-11)
    assert species_mass(end, "L") == pytest.approx(species_mass(start, "L"), rel=1e-10)
    spread = end.fields["L"].x.array
    assert float(spread.max() - spread.min()) < 0.6  # it diffused along the envelope (started 0.6 apart)

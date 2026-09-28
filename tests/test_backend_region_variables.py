"""Region variables (§1.4.2 T5) in the two-compartment method-of-lines solver (#196).

A region variable is one value per region, a Real unknown in the blocked system. It's either a well-mixed
species (a compartment's) or a membrane potential (the membrane's). The checks:

- **The well-mixed limit.** A well-mixed cytosolic species is what a cytosolic species becomes as its
  diffusion grows without bound, so its trajectory must match a fast-diffusing PDE twin; and the exchange
  conserves ∫u_out + |cyto|·c exactly.
- **An exact membrane potential.** C dV/dt = −g (V − E) has V(t) = E + (V0 − E) e^{−g t / C}; the error must
  be the time integrator's alone (it shrinks with the tolerance).
- **A potential-gated flux.** A permeability that depends on V couples the Real block both ways and still
  conserves mass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    integrate_interface_coupled,
    integrate_membrane_coupled,
    make_two_bulk_membrane_geometry,
)
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)

INNER_RADIUS, OUTER_RADIUS = 0.5, 1.0
TIMES = (0.0, 0.25, 0.5, 1.0, 2.0)


def _geometry() -> InterfaceCoupledGeometry:
    return make_two_bulk_membrane_geometry(
        "cell",
        inner="cyto",
        outer_subdomain="ext",
        membrane="mem",
        interface="membrane",
        outer="wall",
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        h=0.09,
    )


def _areas() -> tuple[float, float]:
    """The compartments' areas on the mesh (a polygon: slightly under π r²), the weights of the mass."""
    geometry = _geometry()

    def area(mesh: object) -> float:
        form = fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))
        return float(geometry.parent_mesh.comm.allreduce(fem.assemble_scalar(form).real))

    return area(geometry.inner_mesh), area(geometry.outer_mesh)


def _pde(variable: str, subdomain: str, diffusion: str, ic: str) -> TemplateEquation:
    return TemplateEquation(
        template="bulk_radv_diff",
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"diffusion": diffusion},
        initial_condition=ic,
    )


def _region(variable: str, subdomain: str, rate: str | None, ic: str) -> TemplateEquation:
    return TemplateEquation(
        template="region_ode",
        variable=variable,
        subdomain=subdomain,
        temporality="time_dependent",
        terms={"region_rate": rate} if rate is not None else {},
        initial_condition=ic,
    )


def _exchange_model(*, well_mixed: bool, inner_diffusion: str = "1.0", permeability: str = "P") -> MathDescription:
    """A cytosolic species `c` (well-mixed, or a PDE with `inner_diffusion`) exchanging with a diffusing
    extracellular `u` through a permeability flux: P·(u − c) into the cytosol, its negation outward."""
    c_space = "region" if well_mixed else "lagrange_p1"
    c_equation = _region("c", "cyto", None, "1.0") if well_mixed else _pde("c", "cyto", inner_diffusion, "1.0")
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="mem", kind="surface"),
        ],
        variables=[Variable(name="c", subdomain="cyto", space=c_space), Variable(name="u", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=2.0)],
        equations=[c_equation, _pde("u", "ext", "1.0", "0.0")],
        boundary_conditions=[
            BCInterfaceFlux(variable="c", boundary="membrane", expression=f"{permeability} * (u - c)"),
            BCInterfaceFlux(variable="u", boundary="membrane", expression=f"{permeability} * (c - u)"),
        ],
    )


def _with_potential(md: MathDescription, *, gate_permeability: bool) -> MathDescription:
    """Add a membrane potential V (C dV/dt = −g (V − E), V0 = −20, E = −60); optionally gate the permeability
    on it, P·(1 + 0.01 (V + 60)), which couples V into the species fluxes."""
    from dataclasses import replace

    potential = _region("V", "mem", "-g * (V - E) / C", "-20.0")
    permeability = "P * (1.0 + 0.01 * (V + 60.0))" if gate_permeability else "P"
    return replace(
        md,
        variables=[*md.variables, Variable(name="V", subdomain="mem", space="region")],
        parameters=[
            *md.parameters,
            ParameterConstant(name="g", value=2.0),
            ParameterConstant(name="C", value=0.5),
            ParameterConstant(name="E", value=-60.0),
        ],
        equations=[*md.equations, potential],
        boundary_conditions=[
            BCInterfaceFlux(variable="c", boundary="membrane", expression=f"{permeability} * (u - c)"),
            BCInterfaceFlux(variable="u", boundary="membrane", expression=f"{permeability} * (c - u)"),
        ],
    )


def _run(md: MathDescription, *, rtol: float = 1.0e-6, atol: float = 1.0e-8) -> dict[float, dict[str, float]]:
    """Each species' mean over its own domain at each output time (a region variable's value is its mean)."""

    means: dict[float, dict[str, float]] = {}

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        row: dict[str, float] = {}
        for name, field in fields.items():
            mesh = field.function_space.mesh
            if field.function_space.dofmap.index_map.size_global == 1:  # a region variable (a Real)
                n_owned = field.function_space.dofmap.index_map.size_local
                row[name] = mesh.comm.allreduce(float(np.sum(field.x.array[:n_owned].real)))
            else:
                size = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))
                total = fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh)))
                row[name] = mesh.comm.allreduce(float(total.real)) / mesh.comm.allreduce(float(size.real))
        means[t] = row

    integrate_interface_coupled(
        md, _geometry(), t_final=TIMES[-1], output_times=TIMES, on_output_fields=record, rtol=rtol, atol=atol
    )
    return means


def test_well_mixed_species_is_the_fast_diffusion_limit_and_conserves_mass() -> None:
    area_in, area_out = _areas()
    well_mixed = _run(_exchange_model(well_mixed=True))
    fast = _run(_exchange_model(well_mixed=False, inner_diffusion="1000.0"))
    for t in TIMES:
        # the same trajectory as an infinitely fast-diffusing cytosolic species
        assert well_mixed[t]["c"] == pytest.approx(fast[t]["c"], rel=2e-4)
        # exchange only: the cytosol's |cyto|·c plus the exterior's ∫u is the initial |cyto|·1
        total = area_in * well_mixed[t]["c"] + area_out * well_mixed[t]["u"]
        assert total == pytest.approx(area_in, rel=1e-6)
    assert well_mixed[TIMES[-1]]["c"] < 0.5  # it did exchange


def test_membrane_potential_matches_its_exact_relaxation() -> None:
    # C dV/dt = −g (V − E): V(t) = E + (V0 − E) e^{−g t / C}. The formulation adds no error of its own (the rate
    # is uniform, so its membrane average is exact), so what is left is the adaptive integrator's: small, and
    # shrinking sharply as the tolerance tightens.
    def error(rtol: float) -> float:
        means = _run(_with_potential(_exchange_model(well_mixed=False), gate_permeability=False), rtol=rtol, atol=1e-12)
        return max(abs(means[t]["V"] - (-60.0 + 40.0 * math.exp(-4.0 * t))) for t in TIMES)

    loose, tight = error(1.0e-6), error(1.0e-9)
    assert loose < 1.0e-2  # on a 40 mV relaxation
    assert tight < loose / 50.0


def test_potential_gated_permeability_conserves_mass() -> None:
    area_in, area_out = _areas()
    gated = _run(_with_potential(_exchange_model(well_mixed=True), gate_permeability=True))
    ungated = _run(_with_potential(_exchange_model(well_mixed=True), gate_permeability=False))
    for t in TIMES:
        assert area_in * gated[t]["c"] + area_out * gated[t]["u"] == pytest.approx(area_in, rel=1e-6)
    # the gate opens the membrane while V is depolarised (V > −60), so the exchange runs ahead
    assert gated[0.5]["c"] < ungated[0.5]["c"]


# --- the membrane-coupled solver: a membrane potential next to membrane species ----------------------------


def _receptor_model(*, gated: bool) -> MathDescription:
    """A receptor R on the membrane captures ligand from both compartments (L_in, L_out), with a membrane
    potential V (C dV/dt = −g (V − E)); ``gated`` makes the binding rate depend on V, kon·(1 + 0.01 (V + 60)).
    Total ligand ∫L_in + ∫L_out + ∫R is conserved either way."""
    kon = "kon * (1.0 + 0.01 * (V + 60.0))" if gated else "kon"
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[
            Variable(name="L_in", subdomain="cyto"),
            Variable(name="L_out", subdomain="ext"),
            Variable(name="R", subdomain="pm"),
            Variable(name="V", subdomain="pm", space="region"),
        ],
        parameters=[
            ParameterConstant(name="kon", value=0.5),
            ParameterConstant(name="Rmax", value=2.0),
            ParameterConstant(name="g", value=2.0),
            ParameterConstant(name="C", value=0.5),
            ParameterConstant(name="E", value=-60.0),
        ],
        equations=[
            _pde("L_in", "cyto", "1.0", "1.0"),
            _pde("L_out", "ext", "1.0", "1.0"),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05", "source": f"{kon} * (trace(L_in) + trace(L_out)) * (Rmax - R)"},
                initial_condition="0.0",
            ),
            _region("V", "pm", "-g * (V - E) / C", "-20.0"),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="L_in", boundary="pm", expression=f"-{kon} * trace(L_in) * (Rmax - R)"),
            BCInterfaceFlux(variable="L_out", boundary="pm", expression=f"-{kon} * trace(L_out) * (Rmax - R)"),
        ],
    )


def _receptor_geometry() -> InterfaceCoupledGeometry:
    return make_two_bulk_membrane_geometry(
        "cell", inner="cyto", outer_subdomain="ext", membrane="pm", interface="pm", outer="wall", h=0.13
    )


def _integral(field: fem.Function) -> float:
    mesh = field.function_space.mesh
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(field * ufl.dx(domain=mesh))).real))


def _receptor_run(md: MathDescription, *, rtol: float, atol: float) -> tuple[dict[float, float], list[float]]:
    """V at each output time (read through ``on_output_regions``)."""
    potential: dict[float, float] = {}
    integrate_membrane_coupled(
        md,
        _receptor_geometry(),
        t_final=TIMES[-1],
        output_times=TIMES,
        on_output_regions=lambda t, values: potential.__setitem__(t, values["V"]),
        rtol=rtol,
        atol=atol,
    )
    return potential, []


def test_membrane_potential_beside_membrane_species_matches_its_exact_relaxation() -> None:
    potential, _ = _receptor_run(_receptor_model(gated=False), rtol=1.0e-8, atol=1.0e-12)
    assert set(potential) == set(TIMES)
    error = max(abs(potential[t] - (-60.0 + 40.0 * math.exp(-4.0 * t))) for t in TIMES)
    assert error < 1.0e-3


def test_potential_gated_binding_conserves_ligand() -> None:
    def ligand(md: MathDescription) -> tuple[float, float, float]:
        geometry = _receptor_geometry()
        start = integrate_membrane_coupled(md, geometry, t_final=1.0e-9)
        result = integrate_membrane_coupled(md, _receptor_geometry(), t_final=TIMES[-1])
        total0 = sum(_integral(start.field(name)) for name in ("L_in", "L_out", "R"))
        total = sum(_integral(result.field(name)) for name in ("L_in", "L_out", "R"))
        return total0, total, _integral(result.field("R"))

    total0, total, bound_gated = ligand(_receptor_model(gated=True))
    assert total == pytest.approx(total0, rel=1e-7)
    _, _, bound_plain = ligand(_receptor_model(gated=False))
    assert bound_gated > bound_plain  # the gate speeds binding while V > −60


def _two_cell_geometry() -> InterfaceCoupledGeometry:
    """Two separate cells in one exterior: the cytosol class is two disks, the membrane two circles."""
    from vcell_fenics.backend.realize import realize_interface_coupled
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    two_cells = GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(
                name="cyto",
                type="analytic",
                expression=(
                    "((geom.x[0] - 0.45)**2 + geom.x[1]**2 < 0.09) || ((geom.x[0] + 0.45)**2 + geom.x[1]**2 < 0.09)"
                ),
            ),
            SubVolume(name="ext", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="mem", inside="cyto", outside="ext"),),
    )
    return realize_interface_coupled(
        two_cells, inner_subdomain="cyto", outer_subdomain="ext", membrane_subdomain="mem", interface="membrane", h=0.08
    )


def test_region_counts_see_disconnected_cells() -> None:
    from vcell_fenics.backend.interface_coupled import connected_region_count

    geometry = _two_cell_geometry()
    assert connected_region_count(geometry.inner_mesh) == 2  # two cytosols
    assert connected_region_count(geometry.membrane_mesh) == 2  # two membranes
    assert connected_region_count(geometry.outer_mesh) == 1  # one exterior around both


def test_a_region_variable_on_disconnected_regions_is_refused() -> None:
    # one Real would couple the two cells' well-mixed pools into one — refused until per-region instancing
    with pytest.raises(NotImplementedError, match="2 disconnected regions"):
        integrate_interface_coupled(_exchange_model(well_mixed=True), _two_cell_geometry(), t_final=0.1)

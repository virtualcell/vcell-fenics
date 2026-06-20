"""Foundational cross-mesh plumbing for two-bulk + membrane coupling (§1.6.2 / §1.6.6).

The first increment of cross-compartment coupling is the *geometry*, not the physics: prove that a
form integrated on the membrane can reach the traces of **both** bulk variables at once — a membrane
equation in `trace(u_inner)` and `trace(u_outer)`, or one bulk side's interface flux referencing the
partner trace. The existing `CoupledGeometry` (one bulk + one surface, a single entity map) reaches
only one bulk natively, and integrating a term *on the membrane* that references a bulk function
(codim −1 from the membrane) is rejected by ffcx (`codim >= 0`).

`InterfaceCoupledGeometry` carries both bulk submeshes, the membrane, and the three entity maps to the
shared parent. The coupling integral is taken on the parent's **interior** interface facets (`dS`),
and the orientation-robust trace of a bulk variable is `membrane_trace(f) = f('+') + f('-')` — each
bulk function is non-zero only on its own side, so the sum is its value regardless of the arbitrary
`dS` side labelling. These tests pin that: each trace integrates to value × membrane length, and a
two-sided combination assembles correctly. No new physics — just the substrate the assembler builds on.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    assemble_interface_coupled,
    make_two_bulk_membrane_geometry,
    membrane_trace,
)
from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.formalism.schema import (
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _geometry(*, inner_radius: float = 0.5, outer_radius: float = 1.0, h: float = 0.06) -> InterfaceCoupledGeometry:
    return make_two_bulk_membrane_geometry(
        "cell",
        inner="cyto",
        outer_subdomain="ext",
        membrane="mem",
        interface="membrane",
        outer="wall",
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        h=h,
    )


def _membrane_ds(geometry: InterfaceCoupledGeometry) -> UflExpr:
    parent = geometry.parent_mesh
    parent.topology.create_connectivity(parent.topology.dim - 1, parent.topology.dim)
    return ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)(geometry.interface_tag)


def _assemble_membrane_scalar(geometry: InterfaceCoupledGeometry, integrand: UflExpr) -> float:
    """∫_membrane integrand dS, with all three submesh functions reachable via their entity maps."""
    emaps = [geometry.inner_entity_map, geometry.outer_entity_map, geometry.membrane_entity_map]
    form = fem.form(integrand * _membrane_ds(geometry), entity_maps=emaps)
    return float(fem.assemble_scalar(form).real)


def test_geometry_exposes_two_bulks_a_membrane_and_three_entity_maps() -> None:
    geometry = _geometry()
    assert geometry.kind_of("cyto") == "volume"
    assert geometry.kind_of("ext") == "volume"
    assert geometry.kind_of("mem") == "surface"
    # Each subdomain resolves to its own mesh of the right topological dimension.
    assert geometry.mesh_of("cyto").topology.dim == 2
    assert geometry.mesh_of("ext").topology.dim == 2
    assert geometry.mesh_of("mem").topology.dim == 1
    # The three maps relate the submeshes to one shared parent (the cross-mesh substrate).
    for subdomain in ("cyto", "ext", "mem"):
        assert geometry.entity_map_of(subdomain) is not None


def test_both_bulk_traces_are_reachable_on_the_membrane() -> None:
    # The core de-risking result: a membrane integral can reference BOTH bulk traces, each with the
    # right value. u_inner = 1 on the cytosol, u_outer = 2 on the extracellular; the membrane is the
    # inner circle of length 2π·r_inner. membrane_trace picks each bulk's own-side value.
    inner_radius = 0.5
    geometry = _geometry(inner_radius=inner_radius)
    length = 2.0 * math.pi * inner_radius

    u_inner = fem.Function(fem.functionspace(geometry.inner_mesh, ("Lagrange", 1)))
    u_outer = fem.Function(fem.functionspace(geometry.outer_mesh, ("Lagrange", 1)))
    u_inner.x.array[:] = 1.0
    u_outer.x.array[:] = 2.0

    inner_integral = _assemble_membrane_scalar(geometry, membrane_trace(u_inner))
    outer_integral = _assemble_membrane_scalar(geometry, membrane_trace(u_outer))
    both = _assemble_membrane_scalar(geometry, membrane_trace(u_inner) + 3.0 * membrane_trace(u_outer))

    assert inner_integral == pytest.approx(1.0 * length, rel=2e-2)  # ∫ trace(u_inner) ds = 1·|Γ|
    assert outer_integral == pytest.approx(2.0 * length, rel=2e-2)  # ∫ trace(u_outer) ds = 2·|Γ|
    # A single integrand mixing both traces (a membrane reaction f(trace_in, trace_out)) assembles.
    assert both == pytest.approx((1.0 + 3.0 * 2.0) * length, rel=2e-2)


def test_membrane_trace_picks_the_own_side_value_not_an_average() -> None:
    # A spatially varying bulk field: u_inner = x on the cytosol. Its membrane trace integrates to
    # ∫_Γ x ds = 0 over the centred circle (antisymmetric), and ∫_Γ x² ds = π·r³ — confirming the trace
    # is the genuine boundary value of the inner field, not a cross-side blend.
    inner_radius = 0.5
    geometry = _geometry(inner_radius=inner_radius)
    u_inner = fem.Function(fem.functionspace(geometry.inner_mesh, ("Lagrange", 1)))
    u_inner.interpolate(lambda p: p[0])  # u = x

    trace = membrane_trace(u_inner)
    assert _assemble_membrane_scalar(geometry, trace) == pytest.approx(0.0, abs=2e-2)  # ∫ x ds = 0
    # ∫_Γ x² ds = ∫_0^2π (r cosθ)² r dθ = π r³
    assert _assemble_membrane_scalar(geometry, trace * trace) == pytest.approx(math.pi * inner_radius**3, rel=5e-2)


# ---------------------------------------------------------------------------
# Flux-balance interface coupling (§1.6.2) — backward Euler.
# ---------------------------------------------------------------------------


def _flux_balance_model(*, permeability: str = "P", inner_ic: str = "1.0", outer_ic: str = "0.0") -> MathDescription:
    """Two bulk diffusion species coupled by a permeability flux P·(u_out − u_in) at the membrane."""
    return MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="u_in", subdomain="cyto"), Variable(name="u_out", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=2.0)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition=inner_ic,
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "0.1"},
                initial_condition=outer_ic,
            ),
        ],
        boundary_conditions=[
            BCInterfaceFluxBalance(
                variable="u_in",
                partner_variable="u_out",
                boundary="membrane",
                expression=f"{permeability} * (u_out - u_in)",
            )
        ],
    )


def test_flux_balance_equilibrates_and_conserves_mass() -> None:
    # A permeability flux P·(u_out − u_in) across the membrane drives the two compartments to a uniform
    # equilibrium with no external flux, conserving total mass. Starting u_in=1, u_out=0, the steady
    # value is the mass-weighted mean u_eq = (A_in·1 + A_out·0)/(A_in + A_out) = A_in/(A_in + A_out).
    inner_radius, outer_radius = 0.5, 1.0
    geometry = _geometry(inner_radius=inner_radius, outer_radius=outer_radius, h=0.09)
    area_in = math.pi * inner_radius**2
    area_out = math.pi * (outer_radius**2 - inner_radius**2)
    u_eq = area_in / (area_in + area_out)

    problem = assemble_interface_coupled(_flux_balance_model(), geometry, dt=0.05)
    mass0 = problem.total_mass()
    for _ in range(400):
        problem.step()

    assert problem.inner.x.array.mean() == pytest.approx(u_eq, rel=2e-2)
    assert problem.outer.x.array.mean() == pytest.approx(u_eq, rel=2e-2)
    # No external flux ⇒ total ligand is conserved to round-off across the whole run.
    assert problem.total_mass() == pytest.approx(mass0, rel=1e-9)


def test_no_permeability_leaves_the_compartments_uncoupled() -> None:
    # Negative control: P = 0 ⇒ no interface flux ⇒ the compartments do NOT equilibrate; each just
    # diffuses internally to its own (conserved) mean. This pins that the equilibration above is driven
    # by the coupling term, not by some spurious cross-talk in the assembly.
    geometry = _geometry(h=0.1)
    model = _flux_balance_model()
    no_flux = replace(model, parameters=[ParameterConstant(name="P", value=0.0)])

    problem = assemble_interface_coupled(no_flux, geometry, dt=0.05)
    for _ in range(200):
        problem.step()
    # u_in started at 1 and u_out at 0; with no coupling they stay near their own means, far apart.
    assert problem.inner.x.array.mean() == pytest.approx(1.0, rel=1e-6)
    assert problem.outer.x.array.mean() == pytest.approx(0.0, abs=1e-6)


def test_value_equality_constraint_is_rejected() -> None:
    # The other interface kind — the u = k·partner constraint — needs a different mechanism (a
    # constrained solve, not a flux term) and is a follow-up; it must fail loudly, not silently.
    model = replace(
        _flux_balance_model(),
        boundary_conditions=[
            BCInterfaceValueEquality(variable="u_in", partner_variable="u_out", boundary="membrane", expression="1")
        ],
    )
    with pytest.raises(NotImplementedError, match="value-equality"):
        assemble_interface_coupled(model, _geometry(h=0.2), dt=0.05)

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

import pytest
import ufl
from dolfinx import fem

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    make_two_bulk_membrane_geometry,
    membrane_trace,
)
from vcell_fenics.backend._typing import UflExpr


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

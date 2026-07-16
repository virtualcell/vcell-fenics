"""Foundational cross-mesh plumbing for two-bulk + membrane coupling (§1.6.2 / §1.6.6).

The first increment of cross-compartment coupling is the *geometry*, not the physics: prove that a
form integrated on the membrane can reach the traces of **both** bulk variables at once — a membrane
equation in `trace(u_inner)` and `trace(u_outer)`, or one bulk side's interface flux referencing the
adjacent compartment's trace. The existing `CoupledGeometry` (one bulk + one surface, a single entity map) reaches
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

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import compute_midpoints, create_submesh, create_unit_square, exterior_facet_indices, meshtags
from mpi4py import MPI

from vcell_fenics.backend import (
    InterfaceCoupledGeometry,
    InterfaceCoupledResult,
    assemble_interface_coupled,
    integrate_interface_coupled,
    make_two_bulk_membrane_geometry,
    membrane_trace,
)
from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
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


def _split_box_coupled_geometry(n: int) -> InterfaceCoupledGeometry:
    """The unit square split at x = 0.5 into `cyto` (left) and `ext` (right) compartments meeting at a
    straight `membrane`, as an `InterfaceCoupledGeometry`. Built inline as an *exact, nestable* test
    fixture (a structured n×n mesh with an even n so x=0.5 lands on facets): the straight interface has
    no geometric error and doubling n bisects every cell, so the meshes nest. This is the coupled
    analogue of `_square_geometry` — for measuring the two-mesh solver's spatial order without the
    mesh-topology noise a re-meshed disk-in-annulus at each h would inject (project_gmsh_isolation_followups)."""

    parent = create_unit_square(MPI.COMM_WORLD, n, n)
    tdim = parent.topology.dim
    ncells = parent.topology.index_map(tdim).size_local
    midpoints = compute_midpoints(parent, tdim, np.arange(ncells, dtype=np.int32))
    cyto_tag, ext_tag = 1, 2
    cell_values = np.where(midpoints[:, 0] < 0.5, cyto_tag, ext_tag).astype(np.int32)
    cell_tags = meshtags(parent, tdim, np.arange(ncells, dtype=np.int32), cell_values)

    # Membrane = interior facets straddling a cyto and an ext cell (the x=0.5 line); wall = box exterior.
    parent.topology.create_connectivity(tdim - 1, tdim)
    f2c = parent.topology.connectivity(tdim - 1, tdim)
    membrane_facets = np.array(
        [
            f
            for f in range(parent.topology.index_map(tdim - 1).size_local)
            if len(cells := f2c.links(f)) == 2 and cell_values[cells[0]] != cell_values[cells[1]]
        ],
        dtype=np.int32,
    )
    # The reservoir wall is where the OUTER (ext) compartment meets the box exterior — an exterior facet
    # whose one cell is an ext cell. (Restricting to ext-adjacent, not all exterior, matters for the BE
    # solver: it places the weak-Dirichlet penalty as a block form on the outer submesh, which cannot carry
    # a row on a cyto-adjacent facet where the outer space has no dof.)
    wall_facets = np.array(
        [f for f in exterior_facet_indices(parent.topology) if cell_values[f2c.links(f)[0]] == ext_tag],
        dtype=np.int32,
    )
    interface_tag, wall_tag = 100, 300
    idx = np.concatenate([membrane_facets, wall_facets]).astype(np.int32)
    val = np.concatenate(
        [np.full(membrane_facets.size, interface_tag, np.int32), np.full(wall_facets.size, wall_tag, np.int32)]
    )
    order = np.argsort(idx)
    facet_tags = meshtags(parent, tdim - 1, idx[order], val[order])

    inner_mesh, inner_emap, *_ = create_submesh(parent, tdim, cell_tags.find(cyto_tag))
    outer_mesh, outer_emap, *_ = create_submesh(parent, tdim, cell_tags.find(ext_tag))
    membrane_mesh, membrane_emap, *_ = create_submesh(parent, tdim - 1, membrane_facets)
    return InterfaceCoupledGeometry(
        name="cell",
        inner_subdomain="cyto",
        outer_subdomain="ext",
        membrane_subdomain="mem",
        inner_mesh=inner_mesh,
        outer_mesh=outer_mesh,
        membrane_mesh=membrane_mesh,
        inner_entity_map=inner_emap,
        outer_entity_map=outer_emap,
        membrane_entity_map=membrane_emap,
        parent_mesh=parent,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        inner_region_tag=cyto_tag,
        outer_region_tag=ext_tag,
        interface="membrane",
        interface_tag=interface_tag,
        outer="wall",
        outer_tag=wall_tag,
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
# Single-sided interface flux coupling (§1.6.2) — backward Euler.
# ---------------------------------------------------------------------------


def _permeability_model(
    *, permeability: str = "P", diffusion: str = "0.1", inner_ic: str = "1.0", outer_ic: str = "0.0"
) -> MathDescription:
    """Two bulk diffusion species coupled by a permeability flux at the membrane, expressed as a pair of
    independent single-sided fluxes: P·(u_out − u_in) into the inner side and its negation into the
    outer side (together conserving mass, the equal-and-opposite VCell permeability pair)."""
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
                terms={"diffusion": diffusion},
                initial_condition=inner_ic,
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": diffusion},
                initial_condition=outer_ic,
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u_in", boundary="membrane", expression=f"{permeability} * (u_out - u_in)"),
            BCInterfaceFlux(variable="u_out", boundary="membrane", expression=f"{permeability} * (u_in - u_out)"),
        ],
    )


def test_permeability_equilibrates_and_conserves_mass() -> None:
    # A permeability flux P·(u_out − u_in) across the membrane drives the two compartments to a uniform
    # equilibrium with no external flux, conserving total mass. Starting u_in=1, u_out=0, the steady
    # value is the mass-weighted mean u_eq = (A_in·1 + A_out·0)/(A_in + A_out) = A_in/(A_in + A_out).
    inner_radius, outer_radius = 0.5, 1.0
    geometry = _geometry(inner_radius=inner_radius, outer_radius=outer_radius, h=0.09)
    area_in = math.pi * inner_radius**2
    area_out = math.pi * (outer_radius**2 - inner_radius**2)
    u_eq = area_in / (area_in + area_out)

    problem = assemble_interface_coupled(_permeability_model(), geometry, dt=0.05)
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
    model = _permeability_model()
    no_flux = replace(model, parameters=[ParameterConstant(name="P", value=0.0)])

    problem = assemble_interface_coupled(no_flux, geometry, dt=0.05)
    for _ in range(200):
        problem.step()
    # u_in started at 1 and u_out at 0; with no coupling they stay near their own means, far apart.
    assert problem.inner.x.array.mean() == pytest.approx(1.0, rel=1e-6)
    assert problem.outer.x.array.mean() == pytest.approx(0.0, abs=1e-6)


def test_value_equality_constraint_is_rejected() -> None:
    # The other interface kind — the u = k·u_adjacent constraint — needs a different mechanism (a
    # constrained solve, not a flux term) and is a follow-up; it must fail loudly, not silently.
    model = replace(
        _permeability_model(),
        boundary_conditions=[
            BCInterfaceValueEquality(variable="u_in", adjacent_variable="u_out", boundary="membrane", expression="1")
        ],
    )
    with pytest.raises(NotImplementedError, match="value-equality"):
        assemble_interface_coupled(model, _geometry(h=0.2), dt=0.05)


# ---------------------------------------------------------------------------
# Method-of-lines coupled integrator (PETSc TS, blocked two-mesh).
# ---------------------------------------------------------------------------


def _inner_mean(result: InterfaceCoupledResult) -> float:
    inner = result.inner
    mesh = inner.function_space.mesh
    area = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))
    value = fem.assemble_scalar(fem.form(inner * ufl.dx(domain=mesh)))
    return float((value / area).real)


def test_mol_equilibrates_and_conserves_mass() -> None:
    # The method-of-lines coupled integrator (PETSc TS adaptive BDF over the blocked two-mesh system)
    # reaches the same analytic mass-weighted equilibrium as backward Euler, with mass conserved — the
    # blocked TS residual + Jacobian (and the assemble_matrix-holder / SAME_NONZERO_PATTERN fix) is
    # consistent. MOL is the accurate path (≈0 time error) and handles a nonlinear coupling flux.
    inner_radius, outer_radius = 0.5, 1.0
    geometry = _geometry(inner_radius=inner_radius, outer_radius=outer_radius, h=0.09)
    area_in = math.pi * inner_radius**2
    area_out = math.pi * (outer_radius**2 - inner_radius**2)
    u_eq = area_in / (area_in + area_out)

    # Fast intra-compartment diffusion (D=1, diffusion time r²/D ≈ 0.25) so the system equilibrates by
    # t = 4 (~16 diffusion times) — keeps the test cheap while reaching the steady state.
    result = integrate_interface_coupled(_permeability_model(diffusion="1.0"), geometry, t_final=4.0)
    assert result.time == pytest.approx(4.0)
    assert _inner_mean(result) == pytest.approx(u_eq, rel=2e-2)
    assert result.total_mass() == pytest.approx(area_in, rel=2e-2)  # init mass = 1·A_in, conserved


def test_mol_transient_is_second_order_in_space() -> None:
    # A convergence study at the FEniCSx layer: the MID-TRANSIENT functional (mean u_in at t=0.3, before
    # equilibrium — sensitive to the coupling rate AND the spatial profile, not just the steady state)
    # self-converges at the P1 rate O(h²). Refine over a NESTED sequence of structured split-box meshes
    # (`_split_box_coupled_geometry`, n = 8/16/32) — the straight membrane meshes exactly and doubling n
    # bisects every cell, so both compartments and the interface refine together with no mesh-topology
    # noise. (Independently re-meshing the disk-in-annulus at each h injects that noise once the geometry
    # is near-exact, which flattens the signal — hence the structured fixture; see
    # project_gmsh_isolation_followups.) The definitive cross-solver check is a joint refinement vs
    # VCell's FV solver (a follow-up); this pins the spatial order without an external reference.
    model = replace(_permeability_model(), parameters=[ParameterConstant(name="P", value=1.0)])

    def functional(n: int) -> float:
        return _inner_mean(integrate_interface_coupled(model, _split_box_coupled_geometry(n), t_final=0.3))

    values = [functional(n) for n in (8, 16, 32)]  # doubling n halves h; the meshes nest
    diff_coarse = abs(values[0] - values[1])
    diff_fine = abs(values[1] - values[2])
    order = math.log2(diff_coarse / diff_fine)
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order spatial self-convergence, got {order:.2f} ({values})"


def test_be_coupled_is_second_order_against_manufactured_solution() -> None:
    # Method of manufactured solutions on the BACKWARD-EULER two-bulk solver with the interface flux ACTIVE
    # (P > 0), an in-bulk source, and an outer-wall Dirichlet all engaged at once — the absolute-coupled-value
    # check that the equilibration and MOL-order tests never made. u_in* = 2 + x², u_out* = A + x² with
    # A = 2 + D/P: on both sides −D∇²u* = −2 = source, and at the straight membrane x = 0.5 the diffusive flux
    # D·∂ₓu_in* = D exactly balances P(u_out* − u_in*) = P·(A − 2) = D, so u* is the exact steady solution.
    # It self-converges at the P1 rate O(h²) over the NESTED split-box (n = 8/16/32).
    #
    # This is the test that pins the interface coupling to an analytical value. Assembling that coupling as an
    # implicit bilinear matrix silently over-counts it in DOLFINx 0.10/0.11 (both restrictions of a submesh
    # *argument* alias to the same cell); the solver instead assembles the flux from the state coefficients
    # (exact) and applies the implicit Jacobian matrix-free. Without that, this test collapses to ~O(1) error.
    diffusion, permeability = 1.0, 2.0
    wall_level = 2.0 + diffusion / permeability  # A: the outer offset that makes u* consistent

    model = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="u_in", subdomain="cyto"), Variable(name="u_out", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=permeability)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion), "source": str(-2.0 * diffusion)},
                initial_condition="2 + geom.x[0]**2",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion), "source": str(-2.0 * diffusion)},
                initial_condition=f"{wall_level} + geom.x[0]**2",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u_in", boundary="membrane", expression="P*(u_out - u_in)"),
            BCInterfaceFlux(variable="u_out", boundary="membrane", expression="P*(u_in - u_out)"),
            BCDirichlet(variable="u_out", boundary="wall", expression=f"{wall_level} + geom.x[0]**2"),
        ],
    )

    def l2_error(n: int) -> float:
        geometry = _split_box_coupled_geometry(n)
        problem = assemble_interface_coupled(model, geometry, dt=0.02)
        for _ in range(200):  # backward-Euler to steady state (u* is the manufactured steady solution)
            problem.step()
        squared = 0.0
        for field, offset in ((problem.inner, 2.0), (problem.outer, wall_level)):
            mesh = field.function_space.mesh
            exact = fem.Function(field.function_space)
            exact.interpolate(
                fem.Expression(
                    offset + ufl.SpatialCoordinate(mesh)[0] ** 2, field.function_space.element.interpolation_points
                )
            )
            squared += float(fem.assemble_scalar(fem.form((field - exact) ** 2 * ufl.dx(domain=mesh))).real)
        return math.sqrt(squared)

    errors = [l2_error(n) for n in (8, 16, 32)]  # doubling n halves h; the split-box meshes nest
    order = math.log2(errors[0] / errors[-1]) / 2.0  # two 2× refinements
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order MMS convergence, got {order:.2f} ({errors})"


def test_mol_coupled_is_second_order_against_manufactured_solution() -> None:
    # The METHOD-OF-LINES counterpart of the BE MMS above, and the parity check that keeps the two coupled
    # solvers from silently diverging (the failure mode that hid the interface-flux over-count). Same
    # manufactured steady solution u_in* = 2 + x², u_out* = A + x² (A = 2 + D/P) with the flux, an in-bulk
    # source, and a wall Dirichlet all active. The MOL integrator (adaptive BDF) drives the time error to ≈0,
    # so integrating to a steady time isolates the P1 spatial order O(h²) over the nested split-box. This also
    # exercises the `source` and outer-wall Dirichlet that `integrate_interface_coupled` gained here (it
    # assembles the flux as a residual, so it was always over-count-free — only these terms were missing).
    diffusion, permeability = 1.0, 2.0
    wall_level = 2.0 + diffusion / permeability

    model = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="u_in", subdomain="cyto"), Variable(name="u_out", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=permeability)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion), "source": str(-2.0 * diffusion)},
                initial_condition="2 + geom.x[0]**2",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": str(diffusion), "source": str(-2.0 * diffusion)},
                initial_condition=f"{wall_level} + geom.x[0]**2",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u_in", boundary="membrane", expression="P*(u_out - u_in)"),
            BCInterfaceFlux(variable="u_out", boundary="membrane", expression="P*(u_in - u_out)"),
            BCDirichlet(variable="u_out", boundary="wall", expression=f"{wall_level} + geom.x[0]**2"),
        ],
    )

    def l2_error(n: int) -> float:
        result = integrate_interface_coupled(model, _split_box_coupled_geometry(n), t_final=2.0)  # to steady
        squared = 0.0
        for field, offset in ((result.inner, 2.0), (result.outer, wall_level)):
            mesh = field.function_space.mesh
            exact = fem.Function(field.function_space)
            exact.interpolate(
                fem.Expression(
                    offset + ufl.SpatialCoordinate(mesh)[0] ** 2, field.function_space.element.interpolation_points
                )
            )
            squared += float(fem.assemble_scalar(fem.form((field - exact) ** 2 * ufl.dx(domain=mesh))).real)
        return math.sqrt(squared)

    errors = [l2_error(n) for n in (8, 16, 32)]
    order = math.log2(errors[0] / errors[-1]) / 2.0
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order MMS convergence, got {order:.2f} ({errors})"


def test_in_bulk_source_and_outer_wall_dirichlet() -> None:
    # The two new features together (a manufactured-solution-style check). On the DECOUPLED (P=0) outer
    # annulus, the steady field s* = 2 − 0.5(r−Rm)² is held by the in-bulk source f = −D∇²s* = 2 − 0.5/r
    # (a spatial forcing mass-action cannot express) plus a weak outer-wall Dirichlet s*(1) = 1.875 pinning
    # the level. The solver reproduces s* to the P1 discretisation error — verifying that a spatial `source`
    # and an outer `BCDirichlet` are both assembled (before this, `assemble_interface_coupled` silently
    # dropped both).
    model = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="u_in", subdomain="cyto"), Variable(name="u_out", subdomain="ext")],
        parameters=[ParameterConstant(name="P", value=0.0)],  # decoupled: isolates the outer compartment
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0", "source": "2 - 0.5/geom.radius"},
                initial_condition="2 - 0.5*(geom.radius - 0.5)**2",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="u_in", boundary="membrane", expression="P*(u_out - u_in)"),
            BCInterfaceFlux(variable="u_out", boundary="membrane", expression="P*(u_in - u_out)"),
            BCDirichlet(variable="u_out", boundary="wall", expression="1.875"),
        ],
    )
    geometry = _geometry(h=0.06)
    problem = assemble_interface_coupled(model, geometry, dt=0.01)
    for _ in range(500):
        problem.step()
    coords = fem.functionspace(geometry.outer_mesh, ("Lagrange", 1)).tabulate_dof_coordinates()
    r = np.hypot(coords[:, 0], coords[:, 1])
    exact = 2.0 - 0.5 * (r - 0.5) ** 2
    assert np.abs(problem.outer.x.array - exact).max() < 5e-3  # held to the P1 discretisation error

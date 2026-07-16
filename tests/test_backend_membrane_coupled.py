"""A membrane species coupled to BOTH bulk compartments — the receptor–ligand cell (§1.6.5/§1.6.6).

`assemble_membrane_coupled` solves a three-field system: two bulk diffusion species (one per
compartment) and one surface species on their shared membrane, coupled by (1) each bulk's
`BCInterfaceFlux` referencing the surface density and (2) the surface reaction referencing both bulk
traces. The defining checks are mass conservation across the bulk-free + membrane-bound pools (the
fully-lagged binding is termwise conservative), capture from BOTH compartments (the increment's point),
and the negative control that pins the transfer to the coupling.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import compute_midpoints, create_submesh, create_unit_square, exterior_facet_indices, meshtags
from mpi4py import MPI

from vcell_fenics.backend import (
    MembraneCoupledProblem,
    MembraneCoupledResult,
    assemble_membrane_coupled,
    integrate_membrane_coupled,
)
from vcell_fenics.backend.diagnostics import NonlinearTermError
from vcell_fenics.backend.geometry import InterfaceCoupledGeometry, make_two_bulk_membrane_geometry
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    MathDescription,
    ParameterConstant,
    ParameterExpression,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _model(
    *, kon: float = 0.5, rmax: float = 2.0, diffusion: str = "1.0", surf_diffusion: str = "0.05", r_ic: str = "0.0"
) -> MathDescription:
    """A receptor on the membrane irreversibly captures ligand from both compartments. The bulk fluxes
    are the exact negation of the per-side production, so total ligand (free L_in + free L_out + bound R)
    is conserved. `kon`·`rmax` is kept gentle so the explicit (IMEX) binding is stable."""
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
        ],
        parameters=[ParameterConstant(name="kon", value=kon), ParameterConstant(name="Rmax", value=rmax)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": diffusion},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": diffusion},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": surf_diffusion, "source": "kon * (trace(L_in) + trace(L_out)) * (Rmax - R)"},
                initial_condition=r_ic,
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="L_in", boundary="pm", expression="-kon * trace(L_in) * (Rmax - R)"),
            BCInterfaceFlux(variable="L_out", boundary="pm", expression="-kon * trace(L_out) * (Rmax - R)"),
        ],
    )


def _geom(h: float = 0.13):  # type: ignore[no-untyped-def]
    return make_two_bulk_membrane_geometry(
        "cell", inner="cyto", outer_subdomain="ext", membrane="pm", interface="pm", outer="wall", h=h
    )


def _split_box_membrane_geometry(n: int) -> InterfaceCoupledGeometry:
    """The unit square split at x = 0.5 into `cyto` (left) / `ext` (right) meeting at a straight `pm`
    membrane, as an `InterfaceCoupledGeometry` carrying a membrane submesh — the *nestable* structured
    analogue of `_geom`'s disk-in-annulus (doubling n bisects every cell, so both bulk regions AND the
    membrane refine together with no mesh-topology noise). Used for the surface-diffusion SPATIAL-order
    check; the disk-in-annulus re-meshes independently at each h, which flattens the rate (the same reason
    the two-bulk convergence tests use a structured fixture; see project_gmsh_isolation_followups)."""
    parent = create_unit_square(MPI.COMM_WORLD, n, n)
    tdim = parent.topology.dim
    ncells = parent.topology.index_map(tdim).size_local
    midpoints = compute_midpoints(parent, tdim, np.arange(ncells, dtype=np.int32))
    cyto_tag, ext_tag = 1, 2
    cell_values = np.where(midpoints[:, 0] < 0.5, cyto_tag, ext_tag).astype(np.int32)
    cell_tags = meshtags(parent, tdim, np.arange(ncells, dtype=np.int32), cell_values)

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
    # The reservoir wall is the ext-adjacent exterior (unused here — no wall BC — but the geometry needs it).
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
        membrane_subdomain="pm",
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
        interface="pm",
        interface_tag=interface_tag,
        outer="wall",
        outer_tag=wall_tag,
    )


def _solve(model: MathDescription, *, steps: int, dt: float = 0.02, h: float = 0.13) -> MembraneCoupledProblem:
    problem = assemble_membrane_coupled(model, _geom(h), dt=dt)
    for _ in range(steps):
        problem.step()
    return problem


# ---------------------------------------------------------------------------
# 1. construction / dispatch
# ---------------------------------------------------------------------------


def test_assemble_returns_a_three_field_problem() -> None:
    problem = assemble_membrane_coupled(_model(), _geom(), dt=0.02)
    assert isinstance(problem, MembraneCoupledProblem)
    assert (problem.inner_species, problem.outer_species, problem.membrane_species) == (["L_in"], ["L_out"], ["R"])
    # `field` resolves each by name; the membrane field lives on the (1-D) membrane mesh.
    assert problem.field("R").function_space.mesh.topology.dim == 1
    assert problem.field("L_in").function_space.mesh.topology.dim == 2


# ---------------------------------------------------------------------------
# 2. total-ligand conservation (the headline — termwise-conservative binding)
# ---------------------------------------------------------------------------


def test_total_ligand_is_conserved_to_round_off() -> None:
    problem = assemble_membrane_coupled(_model(), _geom(), dt=0.02)
    total0 = problem.total_mass()
    for _ in range(120):
        problem.step()
    # Free ligand in both compartments + membrane-bound receptor: conserved to round-off because the
    # bulk fluxes are the exact negation of the surface production (same lagged expression).
    assert problem.total_mass() == pytest.approx(total0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. capture from BOTH compartments (the increment's defining behaviour)
# ---------------------------------------------------------------------------


def test_binding_captures_ligand_from_both_compartments() -> None:
    problem = assemble_membrane_coupled(_model(), _geom(), dt=0.02)
    inner0, outer0, bound0 = problem.mass("L_in"), problem.mass("L_out"), problem.mass("R")
    for _ in range(120):
        problem.step()
    # Both bulks lose ligand (capture reaches across the membrane from each side) and the membrane gains.
    assert problem.mass("L_in") < inner0 - 1e-3
    assert problem.mass("L_out") < outer0 - 1e-3
    assert problem.mass("R") > bound0 + 1e-3


def test_no_binding_leaves_the_fields_uncoupled() -> None:
    # Negative control: kon = 0 ⇒ no capture ⇒ the receptor stays at its initial 0 and neither bulk
    # transfers mass to the membrane (each is conserved on its own). Pins the capture above to the
    # coupling, not spurious cross-talk in the three-mesh assembly.
    problem = assemble_membrane_coupled(_model(kon=0.0), _geom(), dt=0.02)
    inner0, outer0 = problem.mass("L_in"), problem.mass("L_out")
    for _ in range(120):
        problem.step()
    assert problem.mass("R") == pytest.approx(0.0, abs=1e-12)
    assert problem.mass("L_in") == pytest.approx(inner0, rel=1e-9)
    assert problem.mass("L_out") == pytest.approx(outer0, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. analytic steady state — irreversible capture exhausts the ligand
# ---------------------------------------------------------------------------


def test_irreversible_capture_reaches_the_analytic_steady_state() -> None:
    # With irreversible binding (no k_off) and more receptor sites than ligand, the steady state is
    # ALL ligand captured: the bound total → the initial free total and the bulks → ~0. (Rmax·|Γ| ≈
    # 2·π ≈ 6.3 exceeds the initial ligand ≈ 3.14, so the receptor never saturates.)
    problem = assemble_membrane_coupled(_model(), _geom(h=0.15), dt=0.02)
    total0 = problem.total_mass()
    for _ in range(300):
        problem.step()
    assert problem.mass("L_in") + problem.mass("L_out") == pytest.approx(0.0, abs=0.1)
    assert problem.mass("R") == pytest.approx(total0, rel=0.05)


# ---------------------------------------------------------------------------
# 5. temporal order — the IMEX scheme is first order in dt
# ---------------------------------------------------------------------------


def test_first_order_in_time() -> None:
    # The mid-transient bound total (before saturation) self-converges under dt-refinement at the
    # backward-Euler / IMEX rate O(dt): successive differences shrink ~2x per halving. Pins temporal
    # consistency (a wrong lagging would not converge to a single value).
    def bound_at_t(dt: float) -> float:
        steps = round(0.4 / dt)
        return _solve(_model(), steps=steps, dt=dt, h=0.15).mass("R")

    values = [bound_at_t(dt) for dt in (0.04, 0.02, 0.01)]
    diff_coarse = abs(values[0] - values[1])
    diff_fine = abs(values[1] - values[2])
    assert diff_fine < 0.7 * diff_coarse  # ~first-order (halving would give 0.5; allow slack)


# ---------------------------------------------------------------------------
# 5b. analytical MMS parity — both solvers vs a closed-form binding ODE
# ---------------------------------------------------------------------------


def _analytical_binding_model() -> MathDescription:
    """A membrane binding reaction with a closed-form solution, for a convergence check. With NO interface
    flux the bulk ligand never depletes — it stays at its uniform IC (L_in = L_out = 1), so trace(L) = 1 and
    the (spatially uniform) surface receptor obeys the ODE dR/dt = kon·(L_in + L_out)·(Rmax − R) =
    2·kon·(Rmax − R), whose exact solution is R(t) = Rmax·(1 − e^{−2·kon·t}) (R₀ = 0). Surface diffusion is
    inert on the uniform R, so this isolates the binding coupling + time integration against a closed form."""
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
        boundary_conditions=[],  # no interface flux ⇒ bulk stays uniform, and R follows the closed-form ODE
    )


def _R_exact(t: float, *, kon: float = 0.5, rmax: float = 2.0) -> float:
    return float(rmax * (1.0 - np.exp(-2.0 * kon * t)))


def test_be_membrane_coupled_is_first_order_against_the_analytical_binding_ode() -> None:
    # The IMEX (lagged-binding) backward-Euler step converges to the exact receptor ODE at O(dt¹): refining
    # dt at fixed final time, the error vs R(T) = Rmax(1 − e^{−2·kon·t}) halves with dt. This complements the
    # self-convergence check above by pinning the scheme to an ANALYTICAL value; R stays uniform to round-off
    # (asserted), so it is purely the binding coupling + time discretisation, with no spatial error.
    t_final = 1.5
    exact = _R_exact(t_final)

    def error(dt: float) -> float:
        problem = assemble_membrane_coupled(_analytical_binding_model(), _geom(), dt=dt)
        for _ in range(round(t_final / dt)):
            problem.step()
        r = problem.field("R").x.array
        assert float(r.std()) < 1e-10  # R stays spatially uniform (no gradient to diffuse away)
        return abs(float(r.mean()) - exact)

    errors = [error(dt) for dt in (0.02, 0.01, 0.005)]
    order = float(np.log2(errors[0] / errors[-1]) / 2.0)
    assert 0.8 <= order <= 1.3, f"expected ~1st-order temporal convergence, got {order:.2f} ({errors})"


def test_mol_membrane_coupled_matches_the_analytical_binding_ode() -> None:
    # The MOL parity check: the adaptive-BDF integrator (matrix-free implicit binding) reproduces the SAME
    # exact receptor ODE to its time tolerance — the analytical counterpart of the BE test above, so the two
    # membrane-coupled solvers are pinned to one closed-form solution and cannot silently diverge on it.
    t_final = 1.5
    result = integrate_membrane_coupled(_analytical_binding_model(), _geom(), t_final=t_final)
    r = result.field("R").x.array
    assert float(r.std()) < 1e-10  # spatially uniform under the adaptive solve too
    assert float(r.mean()) == pytest.approx(_R_exact(t_final), abs=2e-3)


def test_mol_membrane_coupled_surface_diffusion_is_second_order_in_space() -> None:
    # The SPATIAL complement to the temporal binding-ODE parity above: there R was uniform, so the surface
    # Laplacian ∇_Γ·(D_s ∇_Γ R) was inert; here R varies along the membrane, exercising it. A surface-diffusion
    # eigenmode on the straight membrane, R(y, t) = cos(π y)·e^{−D_s π² t} (zero-flux ends at y = 0, 1), is the
    # exact solution, with the bulk uniform and decoupled. Refined over the NESTED split-box with the adaptive
    # MOL integrator (time error ≈ 0, so the spatial order is not polluted), the surface L2 error converges at
    # the P1 rate O(h²). The disk-in-annulus re-meshes independently at each h and would flatten the rate.
    surf_diffusion = 0.05
    t_final = 0.5
    decay = float(np.exp(-surf_diffusion * np.pi**2 * t_final))

    model = MathDescription(
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
        ],
        parameters=[],
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
                terms={"diffusion": str(surf_diffusion)},
                initial_condition=f"cos({np.pi} * geom.x[1])",  # eigenmode; zero-flux at the membrane ends
            ),
        ],
        boundary_conditions=[],  # decoupled ⇒ bulk stays uniform, R is a pure surface-diffusion eigenmode
    )

    def surface_error(n: int) -> float:
        # Tight time tolerance so the adaptive time error stays well below the spatial error being measured.
        result = integrate_membrane_coupled(
            model, _split_box_membrane_geometry(n), t_final=t_final, rtol=1e-9, atol=1e-11
        )
        field = result.field("R")
        mesh = field.function_space.mesh
        y = ufl.SpatialCoordinate(mesh)[1]
        exact = fem.Function(field.function_space)
        exact.interpolate(fem.Expression(ufl.cos(np.pi * y) * decay, field.function_space.element.interpolation_points))
        return float(np.sqrt(float(fem.assemble_scalar(fem.form((field - exact) ** 2 * ufl.dx(domain=mesh))).real)))

    errors = [surface_error(n) for n in (8, 16, 32)]  # nested: doubling n bisects the membrane too
    order = float(np.log2(errors[0] / errors[-1]) / 2.0)
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order surface-diffusion convergence, got {order:.2f} ({errors})"


# ---------------------------------------------------------------------------
# 6. multiple species per compartment AND on the membrane (the general case)
# ---------------------------------------------------------------------------


def _multi_model() -> MathDescription:
    """Two cytosolic species A, B; one extracellular Lo; two membrane receptors Ra, Rb. Site `a`
    captures A (from cyto) AND Lo (from ext) → Ra; site `b` captures B → Rb. So Ra is fed from BOTH
    compartments, and each bulk species' flux is the exact negation of its contribution to a surface
    source — the conserved pools are (A + Lo + Ra) and (B + Rb)."""
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[
            Variable(name="A", subdomain="cyto"),
            Variable(name="B", subdomain="cyto"),
            Variable(name="Lo", subdomain="ext"),
            Variable(name="Ra", subdomain="pm"),
            Variable(name="Rb", subdomain="pm"),
        ],
        parameters=[
            ParameterConstant(name="ka", value=0.4),
            ParameterConstant(name="kb", value=0.3),
            ParameterConstant(name="Rmax", value=3.0),
        ],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="A",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="B",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.8",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="Lo",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="Ra",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05", "source": "ka * (trace(A) + trace(Lo)) * (Rmax - Ra)"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="Rb",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05", "source": "kb * trace(B) * (Rmax - Rb)"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="A", boundary="pm", expression="-ka * trace(A) * (Rmax - Ra)"),
            BCInterfaceFlux(variable="Lo", boundary="pm", expression="-ka * trace(Lo) * (Rmax - Ra)"),
            BCInterfaceFlux(variable="B", boundary="pm", expression="-kb * trace(B) * (Rmax - Rb)"),
        ],
    )


def test_multiple_species_per_region_construct_in_order() -> None:
    problem = assemble_membrane_coupled(_multi_model(), _geom(), dt=0.02)
    assert problem.inner_species == ["A", "B"]
    assert problem.outer_species == ["Lo"]
    assert problem.membrane_species == ["Ra", "Rb"]


def test_each_conserved_pool_is_conserved_with_multiple_species() -> None:
    # Two independent binding sites, one fed from both compartments. Each conserved pool — (A + Lo + Ra)
    # and (B + Rb) — is conserved to round-off (every flux is the exact negation of its production), and
    # the per-species symbol table keeps them from cross-contaminating.
    problem = assemble_membrane_coupled(_multi_model(), _geom(), dt=0.02)
    pool_a0 = problem.mass("A") + problem.mass("Lo") + problem.mass("Ra")
    pool_b0 = problem.mass("B") + problem.mass("Rb")
    for _ in range(120):
        problem.step()
    assert problem.mass("A") + problem.mass("Lo") + problem.mass("Ra") == pytest.approx(pool_a0, abs=1e-10)
    assert problem.mass("B") + problem.mass("Rb") == pytest.approx(pool_b0, abs=1e-10)


def test_a_membrane_site_captures_from_both_compartments() -> None:
    # Site `a` is fed by A (cytosolic) and Lo (extracellular): the receptor Ra grows while BOTH A and Lo
    # deplete — the general "list of same-compartment + adjacent-compartment + membrane species" coupling.
    problem = assemble_membrane_coupled(_multi_model(), _geom(), dt=0.02)
    a0, lo0, b0 = problem.mass("A"), problem.mass("Lo"), problem.mass("B")
    for _ in range(120):
        problem.step()
    assert problem.mass("Ra") > 1e-2 and problem.mass("Rb") > 1e-2  # both sites captured
    assert problem.mass("A") < a0 - 1e-3 and problem.mass("Lo") < lo0 - 1e-3  # Ra drew from both sides
    assert problem.mass("B") < b0 - 1e-3


# ---------------------------------------------------------------------------
# 7. method-of-lines integrator (PETSc TS adaptive BDF, matrix-free Newton)
# ---------------------------------------------------------------------------


def test_mol_integrates_conserves_and_captures() -> None:
    # The fully-implicit MOL integrator reaches t_final, captures ligand from both compartments, and
    # conserves total ligand to round-off — the exact cross-mesh Jacobian is un-assemblable (the
    # ∂(binding)/∂ρ block is zero in DOLFINx 0.10/0.11), so the implicit Jacobian is applied matrix-free.
    geom = _geom(h=0.15)
    model = _model()
    total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()  # the t=0 total (no stepping)
    result = integrate_membrane_coupled(model, geom, t_final=4.0)
    assert isinstance(result, MembraneCoupledResult)
    assert result.time == pytest.approx(4.0)
    assert result.membrane_species == ["R"]
    assert result.mass("R") > 1e-2  # captured onto the membrane
    assert result.total_mass() == pytest.approx(total0, abs=1e-9)


def test_mol_handles_stiff_binding() -> None:
    # kon·Rmax = 10 — fast binding that over-consumes the boundary layer in one explicit (IMEX) step and
    # blows the backward-Euler/`assemble_membrane_coupled` path up. The fully-implicit MOL integrator
    # (matrix-free Newton, adaptive BDF) handles it: all ligand captured, total conserved to round-off.
    geom = _geom(h=0.15)
    model = _model(kon=1.0, rmax=10.0)
    total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
    result = integrate_membrane_coupled(model, geom, t_final=4.0)
    assert result.total_mass() == pytest.approx(total0, abs=1e-8)
    assert result.mass("R") > 0.9 * total0  # near-complete capture (Rmax·|Γ| ≫ ligand)


def test_mol_conserves_each_pool_with_multiple_species() -> None:
    # The general multi-species case under MOL: two independent conserved pools (A+Lo+Ra) and (B+Rb),
    # each conserved to round-off through the matrix-free implicit solve.
    geom = _geom(h=0.15)
    model = _multi_model()
    be = assemble_membrane_coupled(model, geom, dt=0.02)
    pool_a0 = be.mass("A") + be.mass("Lo") + be.mass("Ra")
    pool_b0 = be.mass("B") + be.mass("Rb")
    result = integrate_membrane_coupled(model, geom, t_final=3.0)
    assert result.inner_species == ["A", "B"] and result.membrane_species == ["Ra", "Rb"]
    assert result.mass("A") + result.mass("Lo") + result.mass("Ra") == pytest.approx(pool_a0, abs=1e-9)
    assert result.mass("B") + result.mass("Rb") == pytest.approx(pool_b0, abs=1e-9)


# ---------------------------------------------------------------------------
# 7b. moving membrane under the method-of-lines integrator (the migrating cell)
# ---------------------------------------------------------------------------


def test_mol_velocity_none_is_the_static_integration() -> None:
    # Opt-in: omitting `velocity` is the original single-TS static solve — round-off conservation, no motion.
    geom = _geom(h=0.13)
    area0 = _inner_area(geom)
    model = _model()
    total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
    result = integrate_membrane_coupled(model, geom, t_final=2.0, velocity=None)
    assert _inner_area(geom) == pytest.approx(area0, abs=1e-12)  # mesh did not move
    assert result.total_mass() == pytest.approx(total0, abs=1e-9)  # static ⇒ round-off conservation


def test_mol_conserves_total_ligand_under_motion() -> None:
    # The migrating-cell headline: the stiff binding integrates adaptively WHILE the cell deforms, and the
    # GCL-consistent effective dilution rate keeps total ligand conserved. The dilution itself is now
    # conserved to ~round-off (see test_mol_moving_coupled_dilution_conserves_without_binding); the residual
    # O(motion-step) drift here is the *lagged interface coupling flux* across the moving membrane (a
    # separate operator split, not the dilution). Both expansion and shrinkage.
    for velocity in ("[0.3 * geom.x[0], 0.3 * geom.x[1]]", "[-0.3 * geom.x[0], -0.3 * geom.x[1]]"):
        geom = _geom(h=0.11)
        area0 = _inner_area(geom)
        model = _model()
        total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
        result = integrate_membrane_coupled(model, geom, t_final=0.3, velocity=velocity, motion_steps=10)
        assert abs(_inner_area(geom) / area0 - 1.0) > 0.15  # the cell deformed substantially
        assert result.total_mass() == pytest.approx(total0, rel=2e-3)  # ≪ the ~0.2 a no-dilution run drifts


def test_mol_motion_splitting_error_is_first_order() -> None:
    # With the effective dilution rate the move/dilute split is conserved to ~round-off, so the residual
    # conservation error under motion is the *lagged interface coupling flux* on the moving membrane —
    # still first order in the outer motion step. Halving the interval (doubling motion_steps) halves the
    # drift; this is the clean, predictable convergence the BE/IMEX path can't give.
    velocity = "[0.3 * geom.x[0], 0.3 * geom.x[1]]"
    model = _model()

    def drift(motion_steps: int) -> float:
        geom = _geom(h=0.11)
        total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
        result = integrate_membrane_coupled(model, geom, t_final=0.3, velocity=velocity, motion_steps=motion_steps)
        return abs(result.total_mass() / total0 - 1.0)

    coarse, fine = drift(10), drift(20)
    assert coarse / fine == pytest.approx(2.0, abs=0.3)  # first-order: ~2× reduction per halved interval


def test_mol_handles_stiff_binding_under_motion() -> None:
    # The reason MOL-under-motion exists: stiff binding (kon·Rmax large) that the explicit IMEX-BE step
    # cannot take while the membrane also moves. The adaptive implicit solve captures strongly and stays
    # conserved on the deforming cell.
    geom = _geom(h=0.11)
    model = _model(kon=1.0, rmax=10.0)
    total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
    result = integrate_membrane_coupled(
        model, geom, t_final=0.3, velocity="[0.3 * geom.x[0], 0.3 * geom.x[1]]", motion_steps=10
    )
    assert result.mass("R") > 2.0  # strong stiff capture (≫ the gentle-kon ~1.1)
    assert result.total_mass() == pytest.approx(total0, rel=5e-3)


def test_be_moving_coupled_conserves_each_field_to_roundoff() -> None:
    # The conservative ALE correction on the backward-Euler coupled path, isolated from the binding
    # coupling (kon=0, but R seeded so the membrane dilutes too). Each co-moving field just diffuses +
    # dilutes on its own moving mesh, and the (u − r·uⁿ)·w time term — r = the per-cell (bulk) / per-facet
    # (membrane) swept ratio |Kⁿ|/|Kⁿ⁺¹| — conserves each field's substance to *solver precision*, exactly
    # and independently of dt (the term telescopes). The coupled analogue of the single-mesh bulk/membrane
    # conservative form.
    v = "[0.3 * geom.x[0], 0.3 * geom.x[1]]"
    model = _model(kon=0.0, r_ic="1.0")
    totals = []
    for dt, nsteps in ((0.03, 10), (0.0075, 40)):
        geom = _geom(h=0.11)
        p = assemble_membrane_coupled(model, geom, dt=dt, velocity=v)
        m0 = {s: p.mass(s) for s in ("L_in", "L_out", "R")}
        tot0 = p.total_mass()
        for _ in range(nsteps):
            p.step()
        for s in ("L_in", "L_out", "R"):
            assert abs(p.mass(s) / m0[s] - 1.0) < 1e-10  # each field's substance conserved to round-off
        totals.append(p.total_mass() / tot0 - 1.0)
    assert abs(totals[0]) < 1e-10 and abs(totals[1] - totals[0]) < 1e-10  # exact + dt-independent


def test_mol_moving_coupled_dilution_conserves_without_binding() -> None:
    # The MOL coupled path's dilution in isolation (kon=0, R seeded). The strided TS can't telescope, so it
    # uses the GCL-consistent effective rate ln(|Kⁿ⁺¹|/|Kⁿ|)/interval per mesh: the continuous over-a-stride
    # decay exactly cancels the discrete mesh jump, conserving total substance to ~round-off (≫ tighter than
    # the ~0.1 % lagged-coupling drift of the binding case). Confirms the dilution — not the fix's target
    # coupling flux — is what the effective rate makes conservative.
    v = "[0.3 * geom.x[0], 0.3 * geom.x[1]]"
    model = _model(kon=0.0, r_ic="1.0")
    geom = _geom(h=0.11)
    total0 = assemble_membrane_coupled(model, geom, dt=0.02).total_mass()
    result = integrate_membrane_coupled(model, geom, t_final=0.3, velocity=v, motion_steps=10)
    assert result.total_mass() == pytest.approx(total0, rel=1e-3)  # dilution conserved to ~round-off, no coupling


def test_motion_grows_the_cell_to_the_analytic_homothety() -> None:
    # Guards the mesh-motion node→dof permutation: it is topological/fixed and must be computed ONCE on the
    # initial geometry, NOT re-queried from the moved node positions each step (which mis-matches and
    # progressively distorts/under-grows the mesh). Under v = a·x the cell is a pure homothety, so its area
    # must reach a₀·e^{2aT}; the permutation bug under-grew it by ~9% (independent of dt — a geometric, not
    # time-stepping, error). Only the legitimate O(dt) forward-Euler motion error should remain.
    import numpy as np

    from vcell_fenics.backend.interface_coupled import MembraneCoupledMeshMotion

    a, dt, steps = 0.3, 0.02, 50
    geom = _geom(h=0.13)
    area0 = _inner_area(geom)
    motion = MembraneCoupledMeshMotion(_model(), geom, velocity=f"[{a} * geom.x[0], {a} * geom.x[1]]", dt=dt)
    for _ in range(steps):
        motion.advance()
    assert _inner_area(geom) == pytest.approx(area0 * np.exp(2 * a * dt * steps), rel=5e-3)


# ---------------------------------------------------------------------------
# 7c. in-bulk volume reactions (a `source` on a bulk species)
# ---------------------------------------------------------------------------


def _reacting_model(*, src_a: str, src_b: str, src_c: str | None = None) -> MathDescription:
    """A cyto compartment with reacting species A, B (+ optional C), an inert ext ligand, and a membrane
    receptor with no binding (kon implicit 0). The reaction lives entirely in the bulk via each species'
    `source`; conserved pools depend on the stoichiometry the caller encodes in the source strings."""
    cyto_vars = [Variable(name="A", subdomain="cyto"), Variable(name="B", subdomain="cyto")]
    cyto_eqs = [
        TemplateEquation(
            template="bulk_radv_diff",
            variable="A",
            subdomain="cyto",
            temporality="time_dependent",
            terms={"diffusion": "1.0", "source": src_a},
            initial_condition="1.0",
        ),
        TemplateEquation(
            template="bulk_radv_diff",
            variable="B",
            subdomain="cyto",
            temporality="time_dependent",
            terms={"diffusion": "1.0", "source": src_b},
            initial_condition="0.8",
        ),
    ]
    if src_c is not None:
        cyto_vars.append(Variable(name="C", subdomain="cyto"))
        cyto_eqs.append(
            TemplateEquation(
                template="bulk_radv_diff",
                variable="C",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0", "source": src_c},
                initial_condition="0.0",
            )
        )
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[*cyto_vars, Variable(name="L_out", subdomain="ext"), Variable(name="R", subdomain="pm")],
        parameters=[
            ParameterConstant(name="k1", value=0.8),
            ParameterConstant(name="k2", value=0.3),
            ParameterConstant(name="kf", value=1.0),
        ],
        equations=[
            *cyto_eqs,
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.5",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[],
    )


def test_affine_in_bulk_reaction_conserves_the_reacting_pool() -> None:
    # Affine within-compartment A ⇌ B (k1·A reverse, k2·B forward) under backward Euler: assembled
    # implicitly into the block matrix, so it is unconditionally stable. Mass shifts A→B but the closed
    # pool A+B is conserved to round-off, and the ratio approaches detailed balance A/B → k2/k1.
    model = _reacting_model(src_a="-k1*A + k2*B", src_b="k1*A - k2*B")
    problem = assemble_membrane_coupled(model, _geom(h=0.13), dt=0.02)
    pool0, a0 = problem.mass("A") + problem.mass("B"), problem.mass("A")
    for _ in range(300):
        problem.step()
    assert problem.mass("A") < a0 - 1e-3  # A is consumed into B
    assert problem.mass("A") + problem.mass("B") == pytest.approx(pool0, abs=1e-10)  # closed reaction
    assert problem.mass("A") / problem.mass("B") == pytest.approx(0.3 / 0.8, abs=0.02)  # detailed balance k2/k1


def test_nonlinear_in_bulk_reaction_is_rejected_by_backward_euler() -> None:
    # The BE assembler can only lower an affine source. A mass-action product A*B is nonlinear in the
    # unknowns ⇒ a loud `NonlinearTermError` (named, before form compilation) pointing at the MOL — not a
    # deep UFL arity traceback or a silently-wrong solve.
    model = _reacting_model(src_a="-kf*A*B", src_b="-kf*A*B", src_c="kf*A*B")
    with pytest.raises(NonlinearTermError, match="nonlinear"):
        assemble_membrane_coupled(model, _geom(h=0.13), dt=0.02)


def test_mol_handles_nonlinear_in_bulk_mass_action() -> None:
    # The MOL counterpart: a nonlinear mass-action A + B → C in the bulk (rate kf·A·B). The matrix-free
    # Newton differences the residual, so the nonlinearity is handled implicitly; the conserved pools (A+C)
    # and (B+C) — each consumed/produced one-for-one — hold to round-off.
    model = _reacting_model(src_a="-kf*A*B", src_b="-kf*A*B", src_c="kf*A*B")
    geom = _geom(h=0.13)
    area0 = _inner_area(geom)
    pool_ac0, pool_bc0 = 1.0 * area0, 0.8 * area0  # uniform ICs: (A+C)=1.0·area, (B+C)=0.8·area at t=0
    result = integrate_membrane_coupled(model, geom, t_final=1.0)
    assert result.mass("C") > 0.1  # the reaction actually ran
    assert result.mass("A") + result.mass("C") == pytest.approx(pool_ac0, rel=1e-9)
    assert result.mass("B") + result.mass("C") == pytest.approx(pool_bc0, rel=1e-9)


def test_mol_nonlinear_in_bulk_reaction_conserves_under_motion() -> None:
    # The migrating cell with a nonlinear in-bulk reaction: A + B → C while the membrane expands. The
    # conserved pool (A+C) holds to the O(motion-step) split error (no external flux through the membrane),
    # combining the implicit dilution, the moving mesh, and the matrix-free reaction in one solve.
    model = _reacting_model(src_a="-kf*A*B", src_b="-kf*A*B", src_c="kf*A*B")
    geom = _geom(h=0.13)
    area0 = _inner_area(geom)
    pool_ac0 = 1.0 * area0
    result = integrate_membrane_coupled(
        model, geom, t_final=0.6, velocity="[0.25 * geom.x[0], 0.25 * geom.x[1]]", motion_steps=24
    )
    assert abs(_inner_area(geom) / area0 - 1.0) > 0.15  # the cell deformed substantially
    assert result.mass("A") + result.mass("C") == pytest.approx(pool_ac0, rel=1e-2)


# ---------------------------------------------------------------------------
# 7d. force-balance velocity — the membrane moves under its own surface tension
# ---------------------------------------------------------------------------


def _aspect(geom) -> float:  # type: ignore[no-untyped-def]
    x = geom.inner_mesh.geometry.x
    return float((x[:, 0].max() - x[:, 0].min()) / (x[:, 1].max() - x[:, 1].min()))


def test_force_balance_circle_is_a_fixed_point_with_conserving_biochemistry() -> None:
    # The membrane velocity is SOLVED from a surface-tension Stokes force balance on the cyto, not
    # prescribed. A circle under uniform tension is the Laplace fixed point: it stays circular (aspect ≈ 1,
    # no spurious deformation). The biochemistry rides along and total ligand is conserved to round-off —
    # the dilution is self-consistent with whatever the force balance does to the mesh.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    geom = _geom(h=0.07)
    motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02)
    problem = assemble_membrane_coupled(_model(), geom, dt=0.02, motion=motion)
    total0 = problem.total_mass()
    for _ in range(10):
        problem.step()
    assert _aspect(geom) == pytest.approx(1.0, abs=0.02)  # stayed circular — a fixed point, no spurious flow
    assert problem.total_mass() == pytest.approx(total0, abs=1e-5)  # biochemistry conserved through the motion


def test_force_balance_relaxes_a_deformed_cell_with_biochemistry_riding_along() -> None:
    # The migrating-cell mechanics: a pre-deformed (elliptical) cell relaxes back toward the
    # minimal-perimeter circle under its own surface tension — the aspect ratio decreases — while the
    # receptor binding runs and total ligand stays conserved. Pre-deform with a prescribed area-preserving
    # strain, then hand the substrate to the force-balance driver.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion, MembraneCoupledMeshMotion

    geom = _geom(h=0.07)
    pre = MembraneCoupledMeshMotion(_model(), geom, velocity="[0.3 * geom.x[0], -0.3 * geom.x[1]]", dt=0.05)
    for _ in range(5):
        pre.advance()
    aspect0 = _aspect(geom)
    assert aspect0 > 1.1  # genuinely deformed into an ellipse

    motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02)
    problem = assemble_membrane_coupled(_model(), geom, dt=0.02, motion=motion)
    total0 = problem.total_mass()
    for _ in range(25):
        problem.step()
    assert _aspect(geom) < aspect0 - 0.02  # relaxed toward the circle (tension did mechanical work)
    assert problem.total_mass() == pytest.approx(total0, abs=1e-4)  # ligand conserved while the shape changed


def test_mol_force_balance_circle_is_a_fixed_point() -> None:
    # Force balance under the adaptive MOL: the circle stays a fixed point and total ligand is conserved to
    # the O(motion-step) split error while the stiff binding integrates adaptively. The driver's dt must
    # equal the outer interval t_final / motion_steps.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    geom = _geom(h=0.07)
    total0 = assemble_membrane_coupled(_model(), geom, dt=0.02).total_mass()
    motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.3 / 10)
    result = integrate_membrane_coupled(_model(), geom, t_final=0.3, motion=motion, motion_steps=10)
    assert _aspect(geom) == pytest.approx(1.0, abs=0.02)  # fixed point
    assert result.total_mass() == pytest.approx(total0, rel=1e-3)  # conserved through the adaptive solve


def _cyto_area(geom) -> float:  # type: ignore[no-untyped-def]
    import ufl
    from dolfinx import fem
    from petsc4py import PETSc

    one = fem.Constant(geom.inner_mesh, PETSc.ScalarType(1.0))  # type: ignore[operator]
    return float(fem.assemble_scalar(fem.form(one * ufl.dx)).real)


def _membrane_roughness(geom) -> float:  # type: ignore[no-untyped-def]
    # Node-scale radial roughness about the centroid: max |r_i − ½(r_{i−1}+r_{i+1})| over the angle-sorted
    # membrane loop — a detector for the leading-edge spike (a single node lurching off the smooth front).
    x = geom.membrane_mesh.geometry.x[:, :2]
    c = x.mean(axis=0)
    order = np.argsort(np.arctan2(x[:, 1] - c[1], x[:, 0] - c[0]))
    r = np.linalg.norm(x[order] - c, axis=1)
    smooth = 0.5 * (np.roll(r, 1) + np.roll(r, -1))
    return float(np.max(np.abs(r - smooth)))


def test_force_balance_area_correction_conserves_cell_area() -> None:
    # The P2 Stokes velocity is divergence-free (∮u·n=0), but its P1 interpolation is not — moving the
    # polygon vertices by it leaks a spurious O(h) inward flux that shrinks the cell. `area_correction=True`
    # cancels exactly that flux each step (a radial α(x−c) with α=−∮v1·n/(2A)), holding the area to ~machine
    # precision over a long run; the negative control (no correction) shrinks by several percent.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    geom = _geom(h=0.09)
    motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02, area_correction=True)
    a0 = _cyto_area(geom)
    for _ in range(40):
        motion.advance()
    assert _cyto_area(geom) == pytest.approx(a0, rel=1e-5)  # flux cancellation conserves area to ~round-off

    geom_raw = _geom(h=0.09)
    raw = ForceBalanceMeshMotion(geom_raw, tension=0.5, dt=0.02)  # no correction
    b0 = _cyto_area(geom_raw)
    for _ in range(40):
        raw.advance()
    assert _cyto_area(geom_raw) < 0.97 * b0  # the uncorrected drift is real and several percent


def test_force_balance_tension_smoothing_damps_the_leading_edge_spike() -> None:
    # A node-scale spike in the tension (γ = base + α·signal where the signal localizes) drives a node-scale
    # Laplace–Beltrami force that lurches one membrane node out and kinks the front — the leading-edge
    # instability. The surface-Helmholtz tension filter removes the sub-mesh-scale forcing and keeps the
    # membrane smooth; area correction is on in both runs so the comparison isolates the roughness.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    def run(smoothing: float) -> float:
        geom = _geom(h=0.07)
        motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02, tension_smoothing=smoothing, area_correction=True)
        coords = motion.tension.function_space.tabulate_dof_coordinates()
        lead = int(np.argmin(np.linalg.norm(coords[:, :2] - np.array([0.5, 0.0]), axis=1)))  # leading-edge dof
        spike = np.full(coords.shape[0], 0.5)
        spike[lead] = 6.0  # a single-dof (node-scale) tension spike
        for _ in range(50):
            motion.tension.x.array[:] = spike  # re-force it each step
            motion.advance()
        return _membrane_roughness(geom)

    rough_raw = run(0.0)
    rough_smoothed = run(0.21)  # smoothing length 3·h
    assert rough_raw > 0.1  # without the filter the spike genuinely blows up — a node lurches off the front
    assert rough_smoothed < 0.01  # with it the membrane stays smooth (the spike is damped before it loads)


def _centroid(geom) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    x = geom.membrane_mesh.geometry.x
    return float(x[:, 0].mean()), float(x[:, 1].mean())


def _mechano_model(r_ic: str) -> MathDescription:
    """A receptor R on the membrane with a *prescribed asymmetric* initial density (no binding) — the
    knob that drives mechano-chemical migration once the tension is coupled to R."""
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
        ],
        parameters=[ParameterConstant(name="kon", value=0.0)],
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
                initial_condition="0.5",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.02"},
                initial_condition=r_ic,
            ),
        ],
        boundary_conditions=[],
    )


def test_mechano_chemical_coupling_migrates_the_cell() -> None:
    # The migration North Star: the membrane tension is coupled to the receptor density (γ = base +
    # sensitivity·R). An asymmetric R (higher on +x) makes the tension asymmetric, the membrane contracts
    # harder there, and the cell MIGRATES toward the high-R side. The discrimination is the coupling: with
    # sensitivity = 0 (same asymmetric R, no tension coupling) the cell stays put.
    import math

    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    def migrate(sensitivity: float) -> float:
        geom = _geom(h=0.06)
        problem = assemble_membrane_coupled(
            _mechano_model("1.0 + 0.8 * geom.x[0]"),  # receptor density higher on the +x side
            geom,
            dt=0.02,
            motion=(motion := ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02)),
        )
        motion.update_tension_from_receptor(problem.field("R"), base=0.5, sensitivity=sensitivity)
        (x0, y0) = _centroid(geom)
        for _ in range(25):
            problem.step()
            motion.update_tension_from_receptor(problem.field("R"), base=0.5, sensitivity=sensitivity)
        (x1, y1) = _centroid(geom)
        return math.hypot(x1 - x0, y1 - y0)

    coupled, uncoupled = migrate(0.4), migrate(0.0)
    assert coupled > 0.01  # the cell migrated under the tension asymmetry
    assert coupled > 20 * uncoupled  # and only because of the coupling (uncoupled stays put)


def _chemotaxis_model(lout_ic: str, *, kon: float = 0.8, rmax: float = 3.0) -> MathDescription:
    """A receptor that binds ligand from both compartments, with a parameterizable extracellular ligand
    IC. An asymmetric `lout_ic` is the ONLY asymmetry input — the receptor polarization that follows is
    self-generated by binding, not prescribed."""
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
        ],
        parameters=[ParameterConstant(name="kon", value=kon), ParameterConstant(name="Rmax", value=rmax)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "0.3"},
                initial_condition=lout_ic,
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.02", "source": "kon * (trace(L_in) + trace(L_out)) * (Rmax - R)"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="L_in", boundary="pm", expression="-kon * trace(L_in) * (Rmax - R)"),
            BCInterfaceFlux(variable="L_out", boundary="pm", expression="-kon * trace(L_out) * (Rmax - R)"),
        ],
    )


def _x_polarization(problem: MembraneCoupledProblem) -> float:
    """Mean receptor density on the +x half of the membrane minus the −x half (the emergent polarity)."""
    field = problem.field("R")
    xc = field.function_space.tabulate_dof_coordinates()[:, 0]
    values = field.x.array
    return float(values[xc > 0].mean() - values[xc < 0].mean())


def test_self_organizing_chemotaxis_migrates_up_a_ligand_gradient() -> None:
    # The migration North Star, fully self-organizing: an external LIGAND gradient is the only asymmetry
    # input. Binding samples more ligand on the +x side, so the receptor R POLARIZES there (emergent, not
    # prescribed — and it grows over time); the R-coupled tension polarizes; and the cell MIGRATES up the
    # gradient (+x). A uniform ligand is the negative control — no polarization, no migration. The whole
    # chain (ligand → binding → polarity → tension → motion) runs while total ligand stays conserved.
    from vcell_fenics.backend.interface_coupled import ForceBalanceMeshMotion

    def run(lout_ic: str) -> tuple[list[float], float, float]:
        geom = _geom(h=0.06)
        motion = ForceBalanceMeshMotion(geom, tension=0.5, dt=0.02)
        problem = assemble_membrane_coupled(_chemotaxis_model(lout_ic), geom, dt=0.02, motion=motion)
        total0, (x0, _) = problem.total_mass(), _centroid(geom)
        polarity = []
        for i in range(40):
            problem.step()
            motion.update_tension_from_receptor(problem.field("R"), base=0.5, sensitivity=0.5)
            if i in (5, 20, 39):
                polarity.append(_x_polarization(problem))
        (x1, _) = _centroid(geom)
        return polarity, x1 - x0, abs(problem.total_mass() / total0 - 1.0)

    (early, mid, late), shift, drift = run("0.6 + 0.6 * geom.x[0]")  # ligand higher on +x
    assert 0.0 < early < mid < late  # the receptor polarity EMERGES toward +x and grows (self-generated)
    assert shift > 0.005  # the cell migrated up the gradient
    assert drift < 1e-5  # total ligand conserved through the whole self-organizing loop

    flat_polarity, flat_shift, _ = run("1.0")  # uniform ligand — the control
    assert abs(flat_polarity[-1]) < 1e-3  # no polarity emerges without a gradient
    assert abs(flat_shift) < 0.2 * shift  # and the cell does not migrate


# ---------------------------------------------------------------------------
# 7e. sustained ligand reservoir — a Dirichlet on the outer box wall
# ---------------------------------------------------------------------------

_RESERVOIR_GRADIENT = "0.6 + 0.6 * geom.x[0]"


def _reservoir_model(*, reservoir: bool, kon: float = 0.8, rmax: float = 3.0) -> MathDescription:
    """The chemotaxis model with an optional Dirichlet reservoir holding L_out at a gradient on the wall."""
    bcs: list[object] = [
        BCInterfaceFlux(variable="L_in", boundary="pm", expression="-kon * trace(L_in) * (Rmax - R)"),
        BCInterfaceFlux(variable="L_out", boundary="pm", expression="-kon * trace(L_out) * (Rmax - R)"),
    ]
    if reservoir:
        bcs.append(BCDirichlet(variable="L_out", boundary="wall", expression=_RESERVOIR_GRADIENT))
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
        ],
        parameters=[ParameterConstant(name="kon", value=kon), ParameterConstant(name="Rmax", value=rmax)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_in",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L_out",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "0.3"},
                initial_condition=_RESERVOIR_GRADIENT,
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.02", "source": "kon * (trace(L_in) + trace(L_out)) * (Rmax - R)"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=bcs,  # type: ignore[arg-type]
    )


def _wall_ligand_error(problem: MembraneCoupledProblem | MembraneCoupledResult) -> tuple[float, float]:
    """(max |L_out − reservoir target| on the wall, mean L_out on the wall) — how well the wall is held."""
    import numpy as np

    field = problem.field("L_out")
    coords = field.function_space.tabulate_dof_coordinates()
    on_wall = np.sqrt((coords[:, :2] ** 2).sum(axis=1)) > 0.98  # the outer boundary at r = 1
    target = 0.6 + 0.6 * coords[on_wall, 0]
    values = field.x.array[on_wall]
    return float(np.abs(values - target).max()), float(values.mean())


def test_reservoir_dirichlet_holds_the_wall_against_depletion() -> None:
    # The sustained ligand reservoir: a Dirichlet on the outer box wall holds L_out at a fixed gradient,
    # enforced weakly by a penalty. Binding continuously consumes ligand, so WITHOUT the reservoir the wall
    # concentration depletes far below its initial value; WITH it the wall stays pinned at the gradient —
    # the supply that lets a chemotactic gradient persist instead of washing out.
    with_reservoir = assemble_membrane_coupled(_reservoir_model(reservoir=True), _geom(h=0.07), dt=0.02)
    without = assemble_membrane_coupled(_reservoir_model(reservoir=False), _geom(h=0.07), dt=0.02)
    for _ in range(60):
        with_reservoir.step()
        without.step()
    held_err, held_mean = _wall_ligand_error(with_reservoir)
    _, depleted_mean = _wall_ligand_error(without)
    assert held_err < 1e-2  # the wall is pinned to the reservoir gradient
    assert depleted_mean < 0.5 * held_mean  # without it, binding drains the wall well below the held level


def test_mol_reservoir_holds_the_wall() -> None:
    # The reservoir under the adaptive MOL — implicit in the residual and the assembled Jacobian.
    geom = _geom(h=0.07)
    result = integrate_membrane_coupled(_reservoir_model(reservoir=True), geom, t_final=1.0)
    held_err, held_mean = _wall_ligand_error(result)
    assert held_err < 1e-2
    assert held_mean == pytest.approx(0.6, abs=0.05)  # mean of 0.6 + 0.6x over the symmetric wall


def test_reservoir_dirichlet_on_an_unreachable_target_is_rejected() -> None:
    # Only the outer compartment touches the box wall, so a Dirichlet on an inner species (or any non-outer
    # boundary) is a modelling error — rejected loudly rather than silently ignored.
    inner_target = replace(
        _reservoir_model(reservoir=False),
        boundary_conditions=[BCDirichlet(variable="L_in", boundary="wall", expression="1.0")],
    )
    with pytest.raises(NotImplementedError, match="outer"):
        assemble_membrane_coupled(inner_target, _geom(), dt=0.02)


# ---------------------------------------------------------------------------
# 7f. bulk advection (a `relative_advection` velocity on a bulk species)
# ---------------------------------------------------------------------------


def _advection_model(advection: str | None) -> MathDescription:
    """A cyto blob (off-centre at x ≈ −0.2) optionally advected by a prescribed velocity — the bulk
    `relative_advection` slot that VCell's `<Velocity>` lowers to (e.g. the Rac/Rho moving-boundary models)."""
    cyto_terms: dict[str, str] = {"diffusion": "0.05"}
    if advection is not None:
        cyto_terms["relative_advection"] = advection
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[
            Variable(name="a", subdomain="cyto"),
            Variable(name="L", subdomain="ext"),
            Variable(name="R", subdomain="pm"),
        ],
        parameters=[ParameterConstant(name="k", value=0.0)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="a",
                subdomain="cyto",
                temporality="time_dependent",
                terms=cyto_terms,
                initial_condition="exp(-((geom.x[0] + 0.2) * (geom.x[0] + 0.2) + geom.x[1] * geom.x[1]) / 0.04)",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="L",
                subdomain="ext",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.5",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="R",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": "0.05"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[],
    )


def _blob_centroid_x(problem: MembraneCoupledProblem | MembraneCoupledResult) -> float:
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    a = problem.field("a")
    mesh = a.function_space.mesh
    num = fem.assemble_scalar(fem.form(a * ufl.SpatialCoordinate(mesh)[0] * ufl.dx(domain=mesh)))
    den = fem.assemble_scalar(fem.form(a * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(num, op=MPI.SUM) / mesh.comm.allreduce(den, op=MPI.SUM))


def test_bulk_advection_transports_the_species_along_the_velocity() -> None:
    # A prescribed bulk velocity transports the species: a blob starting off-centre at x ≈ −0.2 is carried
    # in +x by v = [0.5, 0]. The advected centroid moves much further than the diffusion-only control (which
    # only relaxes toward the centre). This is the `relative_advection` term the Rac/Rho models lower to.
    advected = assemble_membrane_coupled(_advection_model("[0.5, 0.0]"), _geom(h=0.05), dt=0.01)
    diffusive = assemble_membrane_coupled(_advection_model(None), _geom(h=0.05), dt=0.01)
    start = _blob_centroid_x(advected)
    for _ in range(30):
        advected.step()
        diffusive.step()
    advected_shift = _blob_centroid_x(advected) - start
    diffusive_shift = _blob_centroid_x(diffusive) - start
    assert advected_shift > 0.06  # carried downstream in +x
    assert advected_shift > 3 * diffusive_shift  # ≫ the diffusion-only relaxation


def test_mol_bulk_advection_transports_the_species() -> None:
    # The same advective transport under the adaptive MOL (the term enters the residual and the Jacobian).
    geom = _geom(h=0.05)
    start = _blob_centroid_x(assemble_membrane_coupled(_advection_model("[0.5, 0.0]"), geom, dt=0.01))
    result = integrate_membrane_coupled(_advection_model("[0.5, 0.0]"), geom, t_final=0.3)
    assert _blob_centroid_x(result) - start > 0.06  # advected downstream


# ---------------------------------------------------------------------------
# 8. expression-valued parameters (VCell unit factors)
# ---------------------------------------------------------------------------


def test_expression_valued_unit_factor_binds_in_the_coupling() -> None:
    # An imported VCell membrane model carries volume↔membrane unit factors as *expression* parameters
    # (e.g. KFlux = Area/Volume, UnitFactor = pow(KMOLE, 1)). The coupled solver must bind those — not
    # only bare constants — in the coupling context. Here `uf` (= 2·0.5 = 1) multiplies the binding; the
    # solve reproduces the no-factor result, proving `_param_symbols` compiles the expression.
    model = replace(
        _model(),
        parameters=[
            ParameterConstant(name="kon", value=0.5),
            ParameterConstant(name="Rmax", value=2.0),
            ParameterExpression(name="uf", expression="2.0 * 0.5"),  # a unit factor that reduces to 1
        ],
        boundary_conditions=[
            BCInterfaceFlux(variable="L_in", boundary="pm", expression="-uf * kon * trace(L_in) * (Rmax - R)"),
            BCInterfaceFlux(variable="L_out", boundary="pm", expression="-uf * kon * trace(L_out) * (Rmax - R)"),
        ],
    )
    problem = assemble_membrane_coupled(model, _geom(), dt=0.02)
    total0 = problem.total_mass()
    for _ in range(120):
        problem.step()
    assert problem.mass("R") > 1e-2  # binding happened — uf bound as a parameter
    assert problem.total_mass() == pytest.approx(total0, abs=1e-10)  # uf=1 on both sides → conserved


# ---------------------------------------------------------------------------
# 9. moving membrane — the three-region ALE mesh-motion substrate
# ---------------------------------------------------------------------------


def _inner_area(geom) -> float:  # type: ignore[no-untyped-def]
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    mesh = geom.inner_mesh
    local = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local.real, op=MPI.SUM))


def test_membrane_mesh_motion_moves_the_substrate_coherently() -> None:
    # The moving-membrane substrate: a prescribed outward radial velocity expands the cell. The membrane
    # moves by dt·v, the parent's interior follows by harmonic extension, and the two bulk submeshes
    # inherit it — all four meshes by the same field at coincident nodes, so the topological entity maps
    # stay valid. The inner disk grows, and the coupled assembler still builds (and steps) on the
    # deformed geometry — the foundation the moving-membrane solve is built on. (Nothing else in the
    # codebase moves a parent + multiple submeshes coherently.)
    from vcell_fenics.backend.interface_coupled import MembraneCoupledMeshMotion

    geom = _geom(h=0.13)
    area0 = _inner_area(geom)
    motion = MembraneCoupledMeshMotion(_model(), geom, velocity="[0.3 * geom.x[0], 0.3 * geom.x[1]]", dt=0.05)
    for _ in range(3):
        motion.advance()
    assert _inner_area(geom) > area0 * 1.02  # the membrane expanded outward → the disk grew

    # The coupling form re-assembles + steps on the DEFORMED geometry (entity maps survived the move).
    problem = assemble_membrane_coupled(_model(kon=0.0), geom, dt=0.02)
    problem.step()
    assert problem.mass("L_in") > 0.0  # a sane solve on the deformed mesh


def test_moving_membrane_dilutes_the_bulk_to_the_analytic_value() -> None:
    # The bulk dilution term, verified against an exact solution. Decouple binding (kon=0) and expand the
    # cell radially with v = a·x. In the cyto disk the harmonic mesh velocity equals a·x exactly, so a
    # uniform ligand field (diffusion of a constant is zero) stays uniform and must dilute as the area
    # grows: c(T) = c₀·e^{-2aT} (∇·v = 2a in 2D). Without the `c ∇·v_mesh` term the value would not change
    # at all (the node-following BE update leaves a uniform field fixed) and mass would be created.
    import numpy as np
    from mpi4py import MPI

    a, dt, steps = 0.3, 0.02, 15
    geom = _geom(h=0.1)
    problem = assemble_membrane_coupled(_model(kon=0.0), geom, dt=dt, velocity=f"[{a} * geom.x[0], {a} * geom.x[1]]")
    for _ in range(steps):
        problem.step()
    values = problem.field("L_in").x.array
    comm = geom.inner_mesh.comm
    cmax = float(comm.allreduce(values.max(), op=MPI.MAX))
    cmin = float(comm.allreduce(values.min(), op=MPI.MIN))
    assert cmax - cmin < 1e-3  # stayed uniform — pure dilution, no spurious gradients
    assert cmax == pytest.approx(np.exp(-2 * a * dt * steps), rel=1e-3)  # the analytic dilution factor


def test_moving_membrane_conserves_total_ligand_under_expansion_and_shrinkage() -> None:
    # The headline conservation check under motion, both directions. As the cell deforms (area changes by
    # ~20%), the bulk `c ∇·v_mesh` and the MANDATORY surface `ρ ∇_Γ·v_Γ` dilution keep every co-moving
    # pool's integral consistent with its geometry — so total ligand (free L_in + free L_out + bound R) is
    # conserved to ~1e-4 (the lagged-dilution O(dt) error). The discrimination is the magnitude: WITHOUT
    # the dilution terms the total would drift by the full volume change (~0.2), 1000× the tolerance here.
    for velocity in ("[0.3 * geom.x[0], 0.3 * geom.x[1]]", "[-0.3 * geom.x[0], -0.3 * geom.x[1]]"):
        geom = _geom(h=0.1)
        area0 = _inner_area(geom)
        problem = assemble_membrane_coupled(_model(), geom, dt=0.02, velocity=velocity)
        total0 = problem.total_mass()
        for _ in range(15):
            problem.step()
        assert abs(_inner_area(geom) / area0 - 1.0) > 0.15  # the cell actually deformed substantially
        assert problem.total_mass() == pytest.approx(total0, rel=1e-3)  # ≪ the ~0.2 a no-dilution run drifts


def test_velocity_none_is_the_static_solve() -> None:
    # The moving-membrane path is opt-in: omitting `velocity` (or passing None) leaves the membrane fixed
    # and recovers the static, round-off-conservative solve — the matrix is never re-assembled.
    geom = _geom(h=0.12)
    area0 = _inner_area(geom)
    problem = assemble_membrane_coupled(_model(), geom, dt=0.02, velocity=None)
    total0 = problem.total_mass()
    for _ in range(5):
        problem.step()
    assert _inner_area(geom) == pytest.approx(area0, abs=1e-12)  # mesh did not move
    assert problem.total_mass() == pytest.approx(total0, abs=1e-10)  # static ⇒ round-off conservation


# ---------------------------------------------------------------------------
# 10. loud rejections
# ---------------------------------------------------------------------------


def test_missing_surface_equation_is_rejected() -> None:
    # A valid TWO-bulk model (no membrane species) handed to the three-field assembler must fail
    # loudly rather than silently solving without the surface coupling — that is the two-bulk
    # `assemble_interface_coupled`'s job.
    two_bulk = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cyto", kind="volume"), Subdomain(name="ext", kind="volume")],
        variables=[Variable(name="L_in", subdomain="cyto"), Variable(name="L_out", subdomain="ext")],
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
                initial_condition="0.0",
            ),
        ],
    )
    with pytest.raises(NotImplementedError, match="surface equation"):
        assemble_membrane_coupled(two_bulk, _geom(), dt=0.02)


def test_value_equality_constraint_is_rejected() -> None:
    model = replace(
        _model(),
        boundary_conditions=[
            BCInterfaceValueEquality(variable="L_in", adjacent_variable="L_out", boundary="pm", expression="1")
        ],
    )
    with pytest.raises(NotImplementedError, match="value-equality"):
        assemble_membrane_coupled(model, _geom(), dt=0.02)

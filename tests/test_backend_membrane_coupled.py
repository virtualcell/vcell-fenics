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

import pytest

from vcell_fenics.backend import MembraneCoupledProblem, assemble_membrane_coupled
from vcell_fenics.backend.geometry import make_two_bulk_membrane_geometry
from vcell_fenics.backend.interface_coupled import _mass_of
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    MathDescription,
    ParameterConstant,
    Subdomain,
    TemplateEquation,
    Variable,
)


def _model(
    *, kon: float = 0.5, rmax: float = 2.0, diffusion: str = "1.0", surf_diffusion: str = "0.05"
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
                initial_condition="0.0",
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
    assert (problem.inner_var, problem.outer_var, problem.membrane_var) == ("L_in", "L_out", "R")
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
    inner0, outer0, bound0 = _mass_of(problem.inner), _mass_of(problem.outer), _mass_of(problem.membrane)
    for _ in range(120):
        problem.step()
    # Both bulks lose ligand (capture reaches across the membrane from each side) and the membrane gains.
    assert _mass_of(problem.inner) < inner0 - 1e-3
    assert _mass_of(problem.outer) < outer0 - 1e-3
    assert _mass_of(problem.membrane) > bound0 + 1e-3


def test_no_binding_leaves_the_fields_uncoupled() -> None:
    # Negative control: kon = 0 ⇒ no capture ⇒ the receptor stays at its initial 0 and neither bulk
    # transfers mass to the membrane (each is conserved on its own). Pins the capture above to the
    # coupling, not spurious cross-talk in the three-mesh assembly.
    problem = assemble_membrane_coupled(_model(kon=0.0), _geom(), dt=0.02)
    inner0, outer0 = _mass_of(problem.inner), _mass_of(problem.outer)
    for _ in range(120):
        problem.step()
    assert _mass_of(problem.membrane) == pytest.approx(0.0, abs=1e-12)
    assert _mass_of(problem.inner) == pytest.approx(inner0, rel=1e-9)
    assert _mass_of(problem.outer) == pytest.approx(outer0, rel=1e-9)


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
    assert _mass_of(problem.inner) + _mass_of(problem.outer) == pytest.approx(0.0, abs=0.1)
    assert _mass_of(problem.membrane) == pytest.approx(total0, rel=0.05)


# ---------------------------------------------------------------------------
# 5. temporal order — the IMEX scheme is first order in dt
# ---------------------------------------------------------------------------


def test_first_order_in_time() -> None:
    # The mid-transient bound total (before saturation) self-converges under dt-refinement at the
    # backward-Euler / IMEX rate O(dt): successive differences shrink ~2x per halving. Pins temporal
    # consistency (a wrong lagging would not converge to a single value).
    def bound_at_t(dt: float) -> float:
        steps = round(0.4 / dt)
        return _mass_of(_solve(_model(), steps=steps, dt=dt, h=0.15).membrane)

    values = [bound_at_t(dt) for dt in (0.04, 0.02, 0.01)]
    diff_coarse = abs(values[0] - values[1])
    diff_fine = abs(values[1] - values[2])
    assert diff_fine < 0.7 * diff_coarse  # ~first-order (halving would give 0.5; allow slack)


# ---------------------------------------------------------------------------
# 6. loud rejections
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

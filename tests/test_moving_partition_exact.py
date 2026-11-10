"""A constructed exact solution on a moving partition (ADR 015): the moments carried through each half-stride move.

The membrane, a circle, expands along its normal: `v = a n`, so its radius is `R(t) = 0.5 + a t`. Inside it a bulk
species at rest, `u = e^{-t} (2 + cos(π r²))`, driven by `F = ∂_t u − D Δu`; on it a species riding the
membrane and diluting as it stretches, `ρ = 1 + t²` (on a circle `∇_Γ·v = a / R`). They exchange mass through a
Robin flux on the bulk side, `D ∂_n u + a u = c(t) trace(u)` — the total flux relative to the moving membrane,
as the solver states it — with `c(t) = D u_r / u + a` at `r = R(t)`, and the membrane gives up what the bulk
receives (`−c(t) trace(u)` in its source, plus the known remainder that keeps `ρ` exact).

The geometry is traced at eight times the mesh's density, so the error measured is the solver's, not the
traced circle's (the realizer's default contour grid would cap the convergence below h ≈ extent / 50).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem

from tests.test_moving_partition import _cell
from vcell_fenics.backend.multi_compartment import (
    MultiCompartmentResult,
    integrate_multi_compartment,
    realize_multi_compartment,
)
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    MathDescription,
    MotionPrescribedVelocity,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
)

_A, _D, _DG, _PI = 0.2, 0.5, 0.1, math.pi
_R = f"(0.5 + {_A} * sim.t)"
_S = "(geom.x[0]**2 + geom.x[1]**2)"
_SOURCE = (
    f"exp(-sim.t) * (-(2 + cos({_PI} * {_S})) "
    f"+ {_D} * (4 * {_PI} * sin({_PI} * {_S}) + 4 * {_PI}**2 * {_S} * cos({_PI} * {_S})))"
)
_C = f"({_D} * (-2 * {_PI} * {_R} * sin({_PI} * {_R}**2)) / (2 + cos({_PI} * {_R}**2)) + {_A})"
_U_ON_MEMBRANE = f"(exp(-sim.t) * (2 + cos({_PI} * {_R}**2)))"
_RHO_SOURCE = f"2 * sim.t + (1 + sim.t**2) * {_A} / {_R} - {_C} * trace(u) + {_C} * {_U_ON_MEMBRANE}"


def _exact_u(x: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(math.exp(-t) * (2.0 + np.cos(_PI * (x[0] ** 2 + x[1] ** 2))))


def _model() -> MathDescription:
    return MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cyto", kind="volume"),
            Subdomain(name="ext", kind="volume"),
            Subdomain(name="pm", kind="surface", motion=MotionPrescribedVelocity(velocity=f"{_A} * geom.normal")),
        ],
        variables=[Variable(name="u", subdomain="cyto"), Variable(name="rho", subdomain="pm")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": str(_D), "source": _SOURCE},
                initial_condition=f"2 + cos({_PI} * {_S})",
            ),
            TemplateEquation(
                template="surface_pde_with_dilution",
                variable="rho",
                subdomain="pm",
                temporality="time_dependent",
                terms={"diffusion": str(_DG), "source": _RHO_SOURCE},
                initial_condition="1.0",
            ),
        ],
        boundary_conditions=[BCInterfaceFlux(variable="u", boundary="pm", expression=f"{_C} * trace(u)")],
    )


def _solve(h: float, strides: int, t_final: float = 1.0) -> MultiCompartmentResult:
    geometry = realize_multi_compartment(_cell(), h=h, compartments={"cyto"}, resolution=round(2.0 / h * 8) + 1)
    return integrate_multi_compartment(
        _model(),
        geometry,
        t_final=t_final,
        motion_membrane="pm",
        stride_ends=[t_final * k / strides for k in range(1, strides + 1)],
    )


def _errors(result: MultiCompartmentResult, t: float = 1.0) -> tuple[float, float]:
    """The relative L2 error of u (against the exact solution interpolated at degree 3) and the largest error of ρ."""
    u = result.fields["u"]
    exact = fem.Function(fem.functionspace(u.function_space.mesh, ("Lagrange", 3)))
    exact.interpolate(lambda x: _exact_u(x, t))
    error = math.sqrt(float(np.real(fem.assemble_scalar(fem.form((u - exact) ** 2 * ufl.dx)))))
    norm = math.sqrt(float(np.real(fem.assemble_scalar(fem.form(exact**2 * ufl.dx)))))
    return error / norm, float(np.abs(result.fields["rho"].x.array - (1.0 + t**2)).max())


def test_the_solution_converges_to_the_exact_one_under_refinement() -> None:
    # h halved with the strides quadrupled (the study: u 1.14e-2 → 4.8e-3, ρ 7.8e-3 → 2.4e-3, toward second order)
    coarse = _errors(_solve(0.2, 5))
    fine = _errors(_solve(0.1, 20))
    assert fine[0] < 6e-3 and fine[1] < 3e-3
    assert coarse[0] / fine[0] > 2.0  # u
    assert coarse[1] / fine[1] > 2.8  # ρ


def test_the_membrane_species_is_second_order_in_the_stride() -> None:
    # on one mesh, against a fine-stride reference: the symmetric split's time error (the study: rate 2.0)
    reference = _solve(0.2, 40).fields["rho"].x.array
    coarse = np.abs(_solve(0.2, 5).fields["rho"].x.array - reference).max()
    fine = np.abs(_solve(0.2, 10).fields["rho"].x.array - reference).max()
    assert math.log2(coarse / fine) > 1.7


@pytest.mark.parametrize("strides", [3, 7])
def test_mass_is_exchanged_exactly(strides: int) -> None:
    # without the manufactured sources only the exchange moves mass: the total is kept to round-off at any stride
    from dataclasses import replace

    md = _model()
    equations: list[WeakFormEquation | TemplateEquation] = []
    for eq in md.equations:
        assert isinstance(eq, TemplateEquation)
        terms = {k: v for k, v in eq.terms.items() if k != "source"}
        if eq.variable == "rho":
            terms["source"] = f"-{_C} * trace(u)"
        equations.append(replace(eq, terms=terms))
    totals: dict[float, float] = {}

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        totals[t] = sum(float(np.real(fem.assemble_scalar(fem.form(fields[name] * ufl.dx)))) for name in ("u", "rho"))

    geometry = realize_multi_compartment(_cell(), h=0.2, compartments={"cyto"}, resolution=81)
    integrate_multi_compartment(
        replace(md, equations=equations),
        geometry,
        t_final=1.0,
        output_times=[0.0, 1.0],
        on_output=record,
        motion_membrane="pm",
        stride_ends=[k / strides for k in range(1, strides + 1)],
    )
    assert totals[1.0] == pytest.approx(totals[0.0], rel=1e-12)

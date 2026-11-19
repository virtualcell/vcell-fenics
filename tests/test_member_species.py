"""A cell's species in the multi-compartment solver, carried by its force balance (ADR 021 decision 8).

The force balance is assembled on the member compartment's own mesh, and the member's species ride its velocity
`v` in the conservative form `−∫ u (v + w)·∇q` (ADR 020 decision 2). The run strides on the fixed partition: `v`
is solved from the species before each stride and held while a fresh integrator covers it (ADR 021 decision 1).

The model is Nickaeen et al. 2017's zero-velocity cell in a fixed circle. These tests port step 3b's tests of the T9
driver's own species problem, which retired once the two paths agreed (ADR 021 decision 7: their difference fell
as backward Euler's error, ratios 0.58, 0.54, 0.52 per halving of Δt). Without the crowding cut-offs it is
singular past the instability, so these tests look at the onset only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend.cell_force_balance import (
    ForceBalanceError,
    assemble_cell_force_balance,
    build_members_mesh,
)
from vcell_fenics.backend.diffusion_check import DiffusionSignError
from vcell_fenics.backend.multi_compartment import (
    MultiCompartmentGeometry,
    MultiCompartmentResult,
    integrate_multi_compartment,
    realize_multi_compartment,
)
from vcell_fenics.formalism import load_dict
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.validator import FormalismValidationError


def _disk() -> GeometryDescription:
    return GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(5.0, 4.0, 1.0),
        origin=(-2.0, -2.0, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )


_MYOSIN = """
math_description:
  geometry: disk_in_medium
  subdomains:
    - { name: ec, kind: volume }
    - { name: cell, kind: volume, motion: { kind: unknown, variable: v } }
    - { name: pm, kind: surface, motion: { kind: unknown, variable: v } }
  variables:
    - { name: m, subdomain: cell }
    - { name: v, subdomain: cell, type: vector }
  equations:
    - template: bulk_radv_diff
      variable: m
      subdomain: cell
      temporality: time_dependent
      terms: { diffusion: "1" }
      initial_condition: "1.5 + 0.01 * geom.x[0]"
    - template: cell_force_balance
      variable: v
      subdomain: cell
      temporality: steady_state
      viscous_operator: laplacian
      volumes:
        cell: { viscosity: alpha, drag: "1", active_stress: "beta * m" }
  boundary_conditions:
    - { kind: dirichlet, variable: v, boundary: pm, expression: "0" }
  parameters:
    - { name: alpha, value: 0.5 }
    - { name: beta, value: 5.0 }
"""


def _raw(beta: float = 5.0) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_MYOSIN)
    raw["math_description"]["parameters"][1]["value"] = beta
    return raw


def _myosin(raw: dict[str, Any]) -> dict[str, Any]:
    eq: dict[str, Any] = raw["math_description"]["equations"][0]
    return eq


_GEOMETRY: dict[tuple[float, bool], MultiCompartmentGeometry] = {}


def _geometry(h: float, medium: bool = False) -> MultiCompartmentGeometry:
    """One realization per h (a fixed partition is never changed by a run); the medium only when it has species."""

    key = (h, medium)
    if key not in _GEOMETRY:
        _GEOMETRY[key] = realize_multi_compartment(_disk(), h=h, compartments={"cell", "ec"} if medium else {"cell"})
    return _GEOMETRY[key]


def _strides(t_final: float, dt: float) -> list[float]:
    n = round(t_final / dt)
    return [t_final * k / n for k in range(1, n + 1)]


def _integrate(
    raw: dict[str, Any],
    t_final: float,
    dt: float,
    *,
    h: float = 0.15,
    medium: bool = False,
    outputs: tuple[float, ...] = (),
    on_output: Callable[[float, dict[str, fem.Function]], None] | None = None,
    override: Callable[[float, fem.Function], None] | None = None,
    rtol: float = 1e-8,
    persistent: bool = False,
) -> MultiCompartmentResult:
    return integrate_multi_compartment(
        load_dict(raw),
        _geometry(h, medium),
        t_final=t_final,
        stride_ends=_strides(t_final, dt),
        output_times=outputs,
        on_output=on_output,
        velocity_override=override,
        rtol=rtol,
        atol=rtol * 1e-2,
        persistent_integrator=persistent,
    )


def _integral(form: Any, mesh: Any) -> float:
    return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(form)).real))


def _spread(m: fem.Function) -> float:
    """‖m − mean m‖ over the cell: the size of the perturbation."""

    mesh = m.function_space.mesh
    dx = ufl.dx(domain=mesh)
    mean = _integral(m * dx, mesh) / _integral(fem.Constant(mesh, 1.0) * dx, mesh)
    return math.sqrt(_integral((m - mean) ** 2 * dx, mesh))


def _totals(
    raw: dict[str, Any], t_final: float, dt: float, names: tuple[str, ...] = ("m",), **options: Any
) -> tuple[MultiCompartmentResult, list[dict[str, float]]]:
    """A run, and each species' total at its start and its end."""

    rows: list[dict[str, float]] = []

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        rows.append(
            {
                n: _integral(fields[n] * ufl.dx(domain=fields[n].function_space.mesh), fields[n].function_space.mesh)
                for n in names
            }
        )

    result = _integrate(raw, t_final, dt, outputs=(0.0, t_final), on_output=record, **options)
    return result, rows


# -- conservation, the uniform state, the onset ---------------------------------------------------------------


def test_myosin_is_conserved_while_the_flow_compresses_it() -> None:
    """ADR 020 decision 2: the conservative form, with the edge held, conserves ∫m, to the solvers' tolerance
    under the method of lines (ADR 021 decision 1), though the compressible flow `∇·v ≠ 0` concentrates it."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    totals: list[float] = []

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        m = fields["m"]
        totals.append(_integral(m * ufl.dx(domain=m.function_space.mesh), m.function_space.mesh))

    result = _integrate(raw, 0.3, 0.02, outputs=(0.0, 0.1, 0.2, 0.3), on_output=record)
    assert result.strides == 15
    assert np.allclose(totals, totals[0], rtol=1e-11, atol=0.0), np.array(totals) / totals[0] - 1
    assert result.velocity is not None
    mesh = result.velocity.function_space.mesh
    assert _integral(ufl.div(result.velocity) ** 2 * ufl.dx(domain=mesh), mesh) > 1e-8  # a dropped m∇·v would show


def test_a_uniform_myosin_exerts_no_force_and_stays_uniform() -> None:
    """A uniform active stress on a velocity held at the edge loads nothing, so `v = 0` and `m` stays as it was."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1.5"
    result = _integrate(raw, 0.1, 0.05)
    assert result.velocity is not None
    assert np.abs(result.velocity.x.array).max() < 1e-12
    assert np.allclose(result.fields["m"].x.array, 1.5, atol=1e-10)


def test_the_instability_sets_in_above_threshold_and_decays_below() -> None:
    """ADR 020 decision 6: Nickaeen's zero-velocity instability, before any collapse. At E2's β = 5 (β m₀ = 7.5)
    the perturbation grows; at β = 0.5 it decays. A carrier of the wrong sign would disperse the myosin and
    suppress the growth. Myosin stays positive."""

    spreads = {}
    outputs = tuple(0.05 * k for k in range(11))
    for beta in (0.5, 5.0):
        rows: list[tuple[float, float]] = []

        def record(t: float, fields: dict[str, fem.Function], rows: list[tuple[float, float]] = rows) -> None:
            rows.append((_spread(fields["m"]), float(fields["m"].x.array.min())))

        _integrate(_raw(beta), 0.5, 0.025, outputs=outputs, on_output=record)
        spreads[beta] = rows[-1][0] / rows[0][0]
        assert min(low for _, low in rows) > 0.0
    assert spreads[5.0] > 2.0, spreads
    assert spreads[0.5] < 0.9, spreads


def test_each_output_velocity_is_the_one_solved_from_that_outputs_species() -> None:
    """ADR 020 decision 5 and ADR 021 decision 6: at `tⁿ` the bundle's `v` is the one solved from `mⁿ`, not the
    lagged one that carried the stride ending there. Checked against a force balance solved afresh."""

    raw = _raw()
    kept: list[tuple[float, np.ndarray, np.ndarray]] = []

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        kept.append((t, fields["m"].x.array.copy(), fields["v"].x.array.copy()))

    _integrate(raw, 0.2, 0.05, outputs=(0.0, 0.1, 0.2), on_output=record)
    md = load_dict(raw)
    geometry = _geometry(0.15)
    m = fem.Function(fem.functionspace(geometry.compartments["cell"].mesh, ("Lagrange", 1)))
    balance = assemble_cell_force_balance(md, geometry, prebuilt=build_members_mesh(md, geometry), fields={"m": m})
    for t, species, velocity in kept:
        m.x.array[:] = species
        expected = balance.solve(t).x.array
        assert np.allclose(velocity, expected, rtol=0.0, atol=1e-10 * np.abs(expected).max()), t
    assert np.abs(kept[-1][2] - kept[0][2]).max() > 1e-4  # the velocity changes between the rows


# -- the transport against a known answer (a prescribed velocity) ---------------------------------------------


def _steady_error(h: float, sign: float = 1.0) -> float:
    """Carried by a prescribed `v = ∇Φ`, `Φ = r²/2`, with `D = 1` and zero total flux at the edge, `m` relaxes
    to `C exp(Φ)` (the flux `−∇m + m∇Φ` vanishes everywhere). The relative L² error of the steady state."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1"

    def prescribed(t: float, v: fem.Function) -> None:
        v.interpolate(lambda x: np.vstack([sign * x[0], sign * x[1]]))

    result = integrate_multi_compartment(
        load_dict(raw),
        realize_multi_compartment(_disk(), h=h, compartments={"cell"}),
        t_final=15.0,
        stride_ends=[15.0],
        velocity_override=prescribed,
        rtol=1e-9,
        atol=1e-11,
    )
    m = result.fields["m"]
    mesh = m.function_space.mesh
    dx = ufl.dx(domain=mesh)
    exact = fem.Function(fem.functionspace(mesh, ("Lagrange", 3)))
    exact.interpolate(lambda x: np.exp((x[0] ** 2 + x[1] ** 2) / 2))
    exact.x.array[:] *= _integral(m * dx, mesh) / _integral(exact * dx, mesh)
    return math.sqrt(_integral((m - exact) ** 2 * dx, mesh) / _integral(exact**2 * dx, mesh))


def test_the_carrier_reaches_the_known_steady_state_at_second_order() -> None:
    """ADR 020 decision 6: catches a wrong sign or direction of the carrier, which conservation cannot."""

    coarse, fine = _steady_error(0.05), _steady_error(0.025)
    assert fine < 1e-4
    assert math.log2(coarse / fine) > 1.85, (coarse, fine)
    assert _steady_error(0.1, sign=-1.0) > 0.2  # the opposite carrier piles myosin at the centre instead


# -- a stationary front ------------------------------------------------------------------------------------------


def test_a_stationary_front_lets_the_material_through_and_keeps_the_myosin() -> None:
    """ADR 020 decisions 2 and 4: a front with a prescribed velocity of zero does not move, while the flow crosses
    it (`v·n ≠ 0`), and the run keeps ∫m."""

    raw = _raw()
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0"}
    raw["math_description"]["boundary_conditions"] = []
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    result, (start, end) = _totals(raw, 0.2, 0.02)
    assert result.velocity is not None
    mesh = result.velocity.function_space.mesh
    outflow = _integral(abs(ufl.dot(result.velocity, ufl.FacetNormal(mesh))) * ufl.ds(domain=mesh), mesh)
    assert outflow > 1e-6
    assert abs(end["m"] - start["m"]) < 1e-11 * start["m"]


# -- species beside the cell (ADR 021 decision 3) ----------------------------------------------------------------


def _with_medium(raw: dict[str, Any], k: float = 2.0) -> dict[str, Any]:
    """A species `c` in the medium, exchanged with the cell's myosin across the held membrane at rate `k`."""

    md = raw["math_description"]
    md["variables"].append({"name": "c", "subdomain": "ec"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "ec",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "0",
        }
    )
    md["parameters"].append({"name": "k", "value": k})
    md["boundary_conditions"] += [
        {"kind": "interface_flux", "variable": "m", "boundary": "pm", "expression": "-k * m"},
        {"kind": "interface_flux", "variable": "c", "boundary": "pm", "expression": "k * m"},
    ]
    return raw


def test_a_species_in_the_medium_exchanges_with_the_cells_across_the_held_membrane() -> None:
    """ADR 021 decision 3: with the species in one solver, a species beside the cell couples to the cell's across
    the held membrane by its flux conditions (on a held membrane `v = 0`, so the flux is relative to the material).
    The total of the two is conserved, and myosin has moved into the medium."""

    raw = _with_medium(_raw())
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    _, (start, end) = _totals(raw, 0.2, 0.02, ("m", "c"), medium=True)
    total = start["m"] + start["c"]
    assert start["c"] == 0.0
    assert end["c"] > 0.05 * total
    assert abs(end["m"] + end["c"] - total) < 1e-10 * total, (end["m"] + end["c"]) / total - 1


# -- a transport slot that reads the species and the velocity (ADR 021 decision 2) -------------------------------


def test_the_crowding_cut_offs_run_and_keep_the_myosin() -> None:
    """E4's cut-offs, nonlinear in `m`, with the drift reading the lagged `v`: they compile in the member's
    context and run, and ∫m is kept. Above the threshold the drift's cut-off slows the aggregation that the
    uncut model shows."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    plain = _integrate(raw, 0.3, 0.025)
    _myosin(raw)["terms"] = {"diffusion": "1 - m / 125", "relative_advection": "(max(0, 1 - m / 2) - 1) * v"}
    cut, (start, end) = _totals(raw, 0.3, 0.025)
    assert abs(end["m"] - start["m"]) < 1e-11 * start["m"]
    assert cut.fields["m"].x.array.max() < plain.fields["m"].x.array.max()


def test_the_drift_cut_off_saturates_the_myosin_near_its_kink() -> None:
    """ADR 021 decision 7: E4a's peak (13.3) never reaches `m_max,u` (15), so the cut-off's kink is driven here.
    A prescribed `v = −x` converges on the centre; with the cut-off `max(0, 1 − m/m_max,u)` on the carrier, the
    material stops riding `v` where `m` reaches `m_max,u = 1.5` (mean 1), and the strides still converge.

    Saturation is not a bound of the continuous model (at `m = m_max,u` the transport contributes `∇m·v` to
    `∂m/∂t`), so the band is set from a measurement (2026-11-19, D = 0.02): at h = 0.1 the peak is 1.549 by t = 2
    and settles at 1.586 by t = 5 (1.500 at h = 0.05), where without the cut-off it reaches 14.7 by t = 2 and 24
    by t = 5. ∫m drifts by 2e-13."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1"
    peaks: dict[str, float] = {}
    for label, terms in (
        ("cut", {"diffusion": "0.02", "relative_advection": "(max(0, 1 - m / 1.5) - 1) * v"}),
        ("uncut", {"diffusion": "0.02"}),
    ):
        _myosin(raw)["terms"] = terms
        result, (start, end) = _totals(
            raw, 2.0, 0.05, h=0.1, override=lambda _t, v: v.interpolate(lambda x: np.vstack([-x[0], -x[1]]))
        )
        assert abs(end["m"] - start["m"]) < 1e-10 * start["m"]
        peaks[label] = float(result.fields["m"].x.array.max())
    assert 1.5 < peaks["cut"] < 1.65, peaks  # near m_max,u, a few percent over at this h
    assert peaks["uncut"] > 10.0, peaks


# -- the scope's refusals (ADR 021 decision 3) -------------------------------------------------------------------


def test_a_member_diffusion_that_turns_negative_stops_the_run_naming_the_species() -> None:
    """ADR 021 decision 4 on a member: with no flow (`beta = 0`) and a uniform source, `m = 1.5 + t`, and its
    diffusion `1 − m/2` turns negative at t = 0.5. The check runs in every stride's integrator."""
    raw = _raw(beta=0.0)
    _myosin(raw)["terms"] = {"diffusion": "1 - m/2", "source": "1"}
    _myosin(raw)["initial_condition"] = "1.5"
    with pytest.raises(DiffusionSignError, match=r"the diffusion of 'm' became negative at t ≈ 0\.5"):
        _integrate(raw, 1.0, 0.1)


def test_a_membrane_that_would_move_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"] = []
    with pytest.raises(ForceBalanceError, match="MovingSkeleton: the membrane 'pm'"):
        _integrate(raw, 0.1, 0.05)


def test_a_held_membrane_that_grows_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0.3 * geom.x"}
    with pytest.raises(ForceBalanceError, match="MovingSkeleton: the membrane 'pm' is a growing front"):
        _integrate(raw, 0.1, 0.05)


def test_a_flux_on_a_member_species_at_a_stationary_front_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0"}
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "neumann", "variable": "m", "boundary": "pm", "expression": "0.1"}
    ]
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: a neumann BC on 'm' at the stationary front"):
        _integrate(raw, 0.1, 0.05)


def test_a_membrane_species_on_the_cells_membrane_is_refused() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "r", "subdomain": "pm"})
    md["equations"].append(
        {
            "template": "surface_pde_with_dilution",
            "variable": "r",
            "subdomain": "pm",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: 'r' lives on the membrane 'pm'"):
        _integrate(raw, 0.1, 0.05)


def test_a_flux_on_a_medium_species_at_a_stationary_front_is_refused() -> None:
    """Decision 3: a flux across a stationary front waits for its own decision, whichever side writes it."""

    raw = _with_medium(_raw())
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0"}
    md = raw["math_description"]
    md["boundary_conditions"] = [
        bc for bc in md["boundary_conditions"] if bc["variable"] != "m" and bc["variable"] != "v"
    ]
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: a interface_flux BC on 'c' at the stationary"):
        _integrate(raw, 0.1, 0.05, medium=True)


def test_a_non_diffusing_species_on_the_member_is_refused() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "a", "subdomain": "cell"})
    md["equations"].append(
        {
            "template": "lumped_ode",
            "variable": "a",
            "subdomain": "cell",
            "temporality": "time_dependent",
            "terms": {"rate": "0"},
            "initial_condition": "1",
        }
    )
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: 'a' on the member 'cell' is governed by"):
        _integrate(raw, 0.1, 0.05)


def test_the_pressure_may_not_read_a_species() -> None:
    raw = _raw()
    raw["math_description"]["equations"][1]["volumes"]["cell"]["pressure"] = "m"
    with pytest.raises(
        FormalismValidationError, match="NonUniformPressure"
    ):  # the validator, before the backend sees it
        _integrate(raw, 0.1, 0.05)


def test_a_box_face_the_member_touches_must_be_at_rest() -> None:
    description = GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(4.0, 2.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression="geom.x[0] < 1.5"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )
    geometry = realize_multi_compartment(description, h=0.25, compartments={"cell"})
    raw = _raw()
    faces = sorted(geometry.compartments["cell"].faces)
    raw["math_description"]["boundary_conditions"] += [
        {"kind": "dirichlet", "variable": "v", "boundary": face, "expression": "0"} for face in faces
    ]
    run = {"t_final": 0.05, "stride_ends": [0.05]}
    integrate_multi_compartment(load_dict(raw), geometry, **run)  # type: ignore[arg-type]  # every wall at rest
    raw["math_description"]["boundary_conditions"][1]["expression"] = "[0.5, 0]"
    with pytest.raises(FormalismValidationError, match="the box does not move"):  # the validator, first
        integrate_multi_compartment(load_dict(raw), geometry, **run)  # type: ignore[arg-type]


def test_an_output_between_stride_ends_is_refused() -> None:
    with pytest.raises(ValueError, match=r"output times \[0.07\] are not stride ends"):
        _integrate(_raw(), 0.1, 0.05, outputs=(0.0, 0.07, 0.1), on_output=lambda t, f: None)


def test_a_force_balance_needs_strides() -> None:
    with pytest.raises(ValueError, match="stride_ends must rise strictly"):
        integrate_multi_compartment(load_dict(_raw()), _geometry(0.15), t_final=0.1)


# -- the persistent integrator (a solver option) ------------------------------------------------------------------


def test_one_integrator_across_the_strides_keeps_the_myosin_and_the_same_stagger() -> None:
    """`persistent_integrator`: one integrator across the strides, its BDF history carried over each new `v`. It
    integrates the same staggered model, so it agrees with a fresh integrator per stride to the integrators'
    error, and the conservative form still keeps ∫m. Measured 2026-11-18 (h = 0.15, Δt = 0.025, rtol 1e-6): the
    persistent run's relative L² error against a tight run of the same stagger was 3.4e-5, a fresh integrator's
    9.1e-6, both far inside the stagger's own 7.5e-4."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    totals: list[float] = []

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        m = fields["m"]
        totals.append(_integral(m * ufl.dx(domain=m.function_space.mesh), m.function_space.mesh))

    kept = _integrate(raw, 0.3, 0.02, outputs=(0.0, 0.3), on_output=record, rtol=1e-6, persistent=True)
    fresh = _integrate(raw, 0.3, 0.02, rtol=1e-6)
    assert kept.strides == fresh.strides == 15
    assert kept.steps > 0
    assert np.allclose(totals, totals[0], rtol=1e-11, atol=0.0), np.array(totals) / totals[0] - 1
    a, b = kept.fields["m"], fresh.fields["m"]
    difference = fem.Function(a.function_space)
    difference.x.array[:] = a.x.array - b.x.array
    mesh = a.function_space.mesh
    relative = math.sqrt(
        _integral(difference**2 * ufl.dx(domain=mesh), mesh) / _integral(b**2 * ufl.dx(domain=mesh), mesh)
    )
    assert relative < 1e-3, relative


def test_one_integrator_across_a_moving_membranes_strides_is_refused() -> None:
    with pytest.raises(ValueError, match="persistent_integrator with a moving membrane 'pm'"):
        integrate_multi_compartment(
            load_dict(_raw()),
            _geometry(0.15),
            t_final=0.1,
            stride_ends=[0.1],
            motion_membrane="pm",
            persistent_integrator=True,
        )


# -- vcell-core's E2, when the sibling checkout is present ----------------------------------------------------

_E2 = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "math" / "resolved-cell.jsonl"


@pytest.mark.skipif(not _E2.exists(), reason="vcell-core's E2 document needs the sibling checkout")
def test_vcell_cores_e2_runs_and_conserves_its_myosin() -> None:
    """vcell-core's own lowering of E2 (#251), read from the sibling checkout, not copied: it runs as written,
    with no carrier slot and no BC for `m`."""

    records = [json.loads(line) for line in _E2.read_text().splitlines() if line.strip()]
    document = next(r for r in records if r.get("case") == "e2")["fenics"]
    description = GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(5.0, 4.0, 1.0),
        origin=(-2.0, -2.0, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="cell_ec_membrane", inside="cell", outside="ec"),),
    )
    geometry = realize_multi_compartment(description, h=0.15, compartments={"cell"})
    rows: list[tuple[float, float]] = []

    def record(t: float, fields: dict[str, fem.Function]) -> None:
        m = fields["m"]
        rows.append((_integral(m * ufl.dx(domain=m.function_space.mesh), m.function_space.mesh), _spread(m)))

    integrate_multi_compartment(
        load_dict(document),
        geometry,
        t_final=0.3,
        stride_ends=_strides(0.3, 0.025),
        output_times=(0.0, 0.3),
        on_output=record,
        rtol=1e-8,
        atol=1e-10,
    )
    (total, spread), (total_end, spread_end) = rows
    assert abs(total_end - total) < 1e-11 * total
    assert spread_end > spread  # above threshold: the perturbation grows

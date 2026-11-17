"""A cell force balance with species on its member, run in time on a fixed skeleton (ADR 018 step 3b; ADR 020).

The model is Nickaeen et al. 2017's zero-velocity cell in a fixed circle, as vcell-core's E2 lowers it (written
here independently): myosin `m` on the cell, carried by the solved velocity with no carrier slot, driving an
active stress `β m`, with the vector Laplacian, a drag and the membrane held at `v = 0`. Without the crowding
cut-offs it is singular past the instability (ADR 020, Context), so these tests look at the onset only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend.cell_force_balance import ForceBalanceError
from vcell_fenics.backend.cell_force_balance_run import (
    CellForceBalanceRun,
    assemble_cell_force_balance_run,
    run_cell_force_balance,
)
from vcell_fenics.backend.diagnostics import NonlinearTermError
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
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


def _balance(raw: dict[str, Any]) -> dict[str, Any]:
    eq: dict[str, Any] = raw["math_description"]["equations"][1]
    return eq


_GEOMETRY: dict[float, MultiCompartmentGeometry] = {}


def _geometry(h: float) -> MultiCompartmentGeometry:
    # one realization per h (the run never moves or changes the mesh)
    if h not in _GEOMETRY:
        _GEOMETRY[h] = realize_multi_compartment(_disk(), h=h)
    return _GEOMETRY[h]


def _run(raw: dict[str, Any], h: float = 0.15) -> CellForceBalanceRun:
    return assemble_cell_force_balance_run(load_dict(raw), _geometry(h))


def _spread(run: CellForceBalanceRun) -> float:
    """‖m − mean m‖ over the cell: the size of the perturbation."""

    mesh = run.balance.members.mesh
    m = run.concentration("m")

    def integral(form: Any) -> float:
        return float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(form)).real))

    mean = fem.Constant(mesh, run.total("m") / integral(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh)))
    return math.sqrt(integral((m - mean) ** 2 * ufl.dx(domain=mesh)))


# -- conservation, the uniform state, the onset ---------------------------------------------------------------


def test_myosin_is_conserved_to_round_off_while_the_flow_compresses_it() -> None:
    """ADR 020 decision 2: the conservative form, with the edge held, conserves ∫m exactly (the check #251
    offers), though the compressible flow `∇·v ≠ 0` concentrates it."""

    run = _run(_raw())
    total = run.total("m")
    run.advance(0.3, 0.02)
    assert abs(run.total("m") - total) < 1e-12 * total
    divergence = fem.assemble_scalar(fem.form(ufl.div(run.velocity) ** 2 * ufl.dx)).real
    assert divergence > 1e-8  # the flow is compressible, so a dropped m∇·v would show


def test_a_uniform_myosin_exerts_no_force_and_stays_uniform() -> None:
    """A uniform active stress on a velocity held at the edge loads nothing (`σ_a∮φ·n = 0` for φ = 0 there), so
    `v = 0` and `m` stays as it was."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1.5"
    run = _run(raw)
    v = run.solve_velocity()
    assert np.abs(v.x.array).max() < 1e-12
    run.advance(0.1, 0.05)
    assert np.allclose(run.species.unknown.x.array, 1.5, atol=1e-12)


def test_the_instability_sets_in_above_threshold_and_decays_below() -> None:
    """ADR 020 decision 6: Nickaeen's zero-velocity instability, before any collapse. At E2's β = 5 (β m₀ = 7.5)
    the perturbation grows; at β = 0.5 it decays. A carrier of the wrong sign would disperse the myosin and
    suppress the growth. Myosin stays positive."""

    spreads = {}
    for beta in (0.5, 5.0):
        run = _run(_raw(beta))
        before = _spread(run)
        lowest = [run.minimum("m")]

        def record(r: CellForceBalanceRun, lowest: list[float] = lowest) -> None:
            lowest.append(r.minimum("m"))

        run.advance(0.5, 0.025, record)
        spreads[beta] = _spread(run) / before
        assert min(lowest) > 0.0
    assert spreads[5.0] > 2.0, spreads
    assert spreads[0.5] < 0.9, spreads


# -- the transport against a known answer (a prescribed velocity) ---------------------------------------------


def _steady_error(h: float, sign: float = 1.0) -> float:
    """Carried by a prescribed `v = ∇Φ`, `Φ = r²/2`, with `D = 1` and zero total flux at the edge, `m` relaxes
    to `C exp(Φ)` (the flux `−∇m + m∇Φ` vanishes everywhere). The relative L² error of the steady state."""

    raw = _raw()
    _myosin(raw)["initial_condition"] = "1"
    run = assemble_cell_force_balance_run(load_dict(raw), realize_multi_compartment(_disk(), h=h))
    run.velocity.interpolate(lambda x: np.vstack([sign * x[0], sign * x[1]]))  # in place of the solve
    total = run.total("m")
    run.species.dt.value = 0.25
    for _ in range(60):
        run.species.step()
    mesh = run.balance.members.mesh
    exact = fem.Function(fem.functionspace(mesh, ("Lagrange", 3)))
    exact.interpolate(lambda x: np.exp((x[0] ** 2 + x[1] ** 2) / 2))
    exact.x.array[:] *= total / fem.assemble_scalar(fem.form(exact * ufl.dx)).real
    m = run.concentration("m")
    error = math.sqrt(fem.assemble_scalar(fem.form((m - exact) ** 2 * ufl.dx)).real)
    return error / math.sqrt(fem.assemble_scalar(fem.form(exact**2 * ufl.dx)).real)


def test_the_carrier_reaches_the_known_steady_state_at_second_order() -> None:
    """ADR 020 decision 6: catches a wrong sign or direction of the carrier, which conservation cannot."""

    coarse, fine = _steady_error(0.05), _steady_error(0.025)  # 1.76 at 0.1 → 0.05: pre-asymptotic
    assert fine < 1e-4
    assert math.log2(coarse / fine) > 1.85, (coarse, fine)
    assert _steady_error(0.1, sign=-1.0) > 0.2  # the opposite carrier piles myosin at the centre instead


# -- a stationary front ------------------------------------------------------------------------------------------


def test_a_stationary_front_lets_the_material_through_and_keeps_the_myosin() -> None:
    """ADR 020 decisions 2 and 4: a front with a prescribed velocity of zero does not move, while the flow
    crosses it (`v·n ≠ 0`), and the run keeps ∫m. Conservation follows from the form's structure whatever the
    edge does; the edge's condition `n·(−D∇m + v m) = 0` itself is checked by the known steady state above,
    where `v·n = 1` at the edge."""

    raw = _raw()
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0"}
    raw["math_description"]["boundary_conditions"] = []
    _myosin(raw)["initial_condition"] = "1.5 + 0.5 * geom.x[0]"
    run = _run(raw)
    total = run.total("m")
    run.advance(0.2, 0.02)
    n = ufl.FacetNormal(run.balance.members.mesh)
    outflow = fem.assemble_scalar(fem.form(abs(ufl.dot(run.velocity, n)) * ufl.ds)).real
    assert outflow > 1e-6
    assert abs(run.total("m") - total) < 1e-12 * total


# -- the scope's refusals (ADR 020 decision 4) -----------------------------------------------------------------


def test_a_membrane_that_would_move_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"] = []
    with pytest.raises(ForceBalanceError, match="MovingSkeleton: the membrane 'pm'"):
        _run(raw)


def test_a_bc_on_a_member_species_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"].append(
        {"kind": "neumann", "variable": "m", "boundary": "pm", "expression": "0.1"}
    )
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: a neumann BC on 'm'"):
        _run(raw)


def test_a_species_off_the_member_is_refused() -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "c", "subdomain": "ec"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "c",
            "subdomain": "ec",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    with pytest.raises(ForceBalanceError, match="SpeciesNotYetSupported: 'c' lives on 'ec'"):
        _run(raw)


def test_nonlinear_transport_waits_for_stage_5() -> None:
    """The crowding cut-offs make the diffusion read `m`: backward Euler refuses it."""

    raw = _raw()
    _myosin(raw)["terms"]["diffusion"] = "1 - m / 125"
    with pytest.raises(NonlinearTermError):
        _run(raw)  # refused when the run is assembled, not at its first step


def test_a_slot_may_read_the_species_but_the_pressure_may_not() -> None:
    raw = _raw()
    _balance(raw)["volumes"]["cell"]["drag"] = "1 + 0.1 * m"
    run = _run(raw)
    run.step(0.01)
    raw = _raw()
    _balance(raw)["volumes"]["cell"]["pressure"] = "m"
    with pytest.raises(FormalismValidationError, match="NonUniformPressure"):
        run_cell_force_balance(load_dict(raw), _disk(), h=0.3)
    with pytest.raises(ForceBalanceError, match=r"FieldsNotYetSupported: volumes.cell.pressure reads \['m'\]"):
        _run(raw)  # the backend refuses it too, for math that skipped validation


def test_parameters_may_read_the_member_species_and_the_velocity() -> None:
    """VCell's lowered math reaches species through function parameters: `sa = beta * m` in the active stress,
    `k = 0.1 * m` in a source, `vx = v[0]` in a drift."""

    raw = _raw()
    md = raw["math_description"]
    md["parameters"] += [
        {"name": "sa", "expression": "beta * m"},
        {"name": "k", "expression": "0.1 * m"},
        {"name": "vx", "expression": "0.1 * v[0]"},
    ]
    _balance(raw)["volumes"]["cell"]["active_stress"] = "sa"
    _myosin(raw)["terms"]["source"] = "-k"
    _myosin(raw)["terms"]["relative_advection"] = "[vx, 0]"
    reference = _run(_raw())
    run = _run(raw)
    reference.step(0.02)
    run.step(0.02)
    assert np.allclose(run.velocity.x.array, reference.velocity.x.array, atol=1e-12)  # sa ≡ beta * m
    assert run.total("m") < reference.total("m")  # the source k removes myosin


def test_a_held_membrane_that_grows_is_refused() -> None:
    raw = _raw()
    raw["math_description"]["subdomains"][2]["motion"] = {"kind": "prescribed", "velocity": "0.3 * geom.x"}
    with pytest.raises(ForceBalanceError, match="MovingSkeleton: the membrane 'pm' is a growing front"):
        _run(raw)


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
    geometry = realize_multi_compartment(description, h=0.25)
    raw = _raw()
    faces = sorted(geometry.compartments["cell"].faces)
    raw["math_description"]["boundary_conditions"] += [
        {"kind": "dirichlet", "variable": "v", "boundary": face, "expression": "0"} for face in faces
    ]
    assemble_cell_force_balance_run(load_dict(raw), geometry)  # every wall at rest: accepted
    raw["math_description"]["boundary_conditions"][1]["expression"] = "[0.5, 0]"
    with pytest.raises(ForceBalanceError, match=f"MovingSkeleton: the box face {faces[0]!r}"):
        assemble_cell_force_balance_run(load_dict(raw), geometry)


# -- vcell-core's E2, when the sibling checkout is present ----------------------------------------------------

_E2 = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "math" / "resolved-cell.jsonl"


@pytest.mark.skipif(not _E2.exists(), reason="vcell-core's E2 document needs the sibling checkout")
def test_vcell_cores_e2_runs_and_conserves_its_myosin() -> None:
    """vcell-core's own lowering of E2 (#251), read from the sibling checkout, not copied: it runs as written,
    with no carrier slot and no BC for `m`."""

    records = [json.loads(line) for line in _E2.read_text().splitlines() if line.strip()]
    document = next(r for r in records if r.get("case") == "e2")["fenics"]
    geometry = realize_multi_compartment(
        GeometryDescription(
            name="disk_in_medium",
            dim=2,
            extent=(5.0, 4.0, 1.0),
            origin=(-2.0, -2.0, 0.0),
            subvolumes=(
                SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
                SubVolume(name="ec", type="analytic", expression="1.0"),
            ),
            surfaces=(SurfaceClass(name="cell_ec_membrane", inside="cell", outside="ec"),),
        ),
        h=0.15,
    )
    run = assemble_cell_force_balance_run(load_dict(document), geometry)
    total, spread = run.total("m"), _spread(run)
    run.advance(0.3, 0.025)
    assert abs(run.total("m") - total) < 1e-12 * total
    assert _spread(run) > spread  # above threshold: the perturbation grows

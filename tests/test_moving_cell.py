"""One cell moving in 2D: a cell force balance whose membrane moves, with no species (ADR 023; stage 4a).

Two kinds of membrane:
- **a growing front:** vcell-core's E3 law (ZS: zero traction; ZV: held at `v = 0`), the front velocity
  `trace(v) + v_p(a)·n` with `v_p = v₀a₀/a − k(a − a₀(a₀/a)²)` and `a` the cell's area (`region_size`);
- **a material membrane** with a tension `γ` and a pressure that reads the area, `P = c/a`: the disk of step 3a,
  whose radial velocity `u(R) = A I₁(R/ℓ)` is exact (`tests/test_backend_cell_force_balance.py`), so its area obeys
  `da/dt = 2√(πa) u(√(a/π))`, with a stable radius where `P = γ/R`.

Measured 2026-11-19 (ADR 023, "Implementation and measurement"): E3 at h = 0.1 is second order in `Δt` over its
transient (Richardson ratios 4.87 for ZS, 4.81 for ZV); at `t = 1` ZV's area is exact to 4e-13 at every h (its
front speed reads only the area, which the mesh measures exactly) and ZS's is within 1.2e-4; the material disk is
first order in `Δt` (the semi-implicit tension, ADR 023 decision 4), relative errors 3.0e-3, 1.4e-3, 6.4e-4 at
`Δt` = 0.1, 0.05, 0.025.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import iv

from vcell_fenics.backend.cell_force_balance import (
    ForceBalanceError,
    assemble_cell_force_balance,
    build_members_mesh,
)
from vcell_fenics.backend.moving_cell import (
    integrate_moving_cell,
    moving_cell_scope,
    moving_membrane_of,
)
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.formalism import load_dict
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

_E3 = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "math" / "moving-resolved-cell.jsonl"

# E3's front law (vcell-core `moving-resolved-cell.jsonl`), in area units: v0 = 2.5, a0 = π, k = 1.5
V0, A0, K = 2.5, math.pi, 1.5
VP = f"({V0} * {A0} / region_size(cell) - {K} * (region_size(cell) - {A0}**3 / region_size(cell)**2))"


def _disk(
    radius: float = 1.0, *, box: tuple[float, float, float, float] = (-2.0, 3.0, -2.0, 2.0)
) -> GeometryDescription:
    x0, x1, y0, y1 = box
    return GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(x1 - x0, y1 - y0, 1.0),
        origin=(x0, y0, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {radius * radius}"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )


def _front(held: bool = False) -> dict[str, Any]:
    """E3's ZS (free) or ZV (held) front, written natively."""
    md: dict[str, Any] = {
        "geometry": "disk_in_medium",
        "subdomains": [
            {"name": "ec", "kind": "volume"},
            {"name": "cell", "kind": "volume", "motion": {"kind": "unknown", "variable": "v"}},
            {
                "name": "pm",
                "kind": "surface",
                "motion": {"kind": "prescribed", "velocity": f"trace(v) + {VP} * geom.normal"},
            },
        ],
        "variables": [{"name": "v", "subdomain": "cell", "type": "vector"}],
        "equations": [
            {
                "template": "cell_force_balance",
                "variable": "v",
                "subdomain": "cell",
                "temporality": "steady_state",
                "viscous_operator": "laplacian",
                "volumes": {"cell": {"viscosity": "0.5", "drag": "1", "active_stress": "1"}},
            }
        ],
        "boundary_conditions": [{"kind": "dirichlet", "variable": "v", "boundary": "pm", "expression": "0"}]
        if held
        else [],
    }
    return {"math_description": md}


# the material disk: η = 0.5, ζ = 1, γ = 0.5, P = c/a with c = π/2, so the stable radius is c/(πγ) = 1
ETA, ZETA, GAMMA, C = 0.5, 1.0, 0.5, 0.5 * math.pi


def _material() -> dict[str, Any]:
    raw = _front()
    md = raw["math_description"]
    md["subdomains"][2]["motion"] = {"kind": "unknown", "variable": "v"}
    md["equations"][0]["volumes"] = {
        "cell": {"viscosity": str(ETA), "drag": str(ZETA), "pressure": f"{C} / region_size(cell)"}
    }
    md["equations"][0]["surfaces"] = {"pm": {"tension": str(GAMMA)}}
    return raw


def _disk_rate(a: float) -> float:
    """da/dt of the material disk (the module docstring)."""
    R = math.sqrt(a / math.pi)
    ell = math.sqrt(ETA / ZETA)
    x = R / ell
    i1, di1 = float(iv(1, x)), float(iv(0, x)) - float(iv(1, x)) / x
    u = (C / (math.pi * R * R) - GAMMA / R) / (ETA * di1 / ell) * i1
    return 2.0 * math.sqrt(math.pi * a) * u


def _strides(t_final: float, dt: float) -> list[float]:
    n = round(t_final / dt)
    return [t_final * k / n for k in range(1, n + 1)]


def _run(
    raw: dict[str, Any], geometry: MultiCompartmentGeometry, t_final: float, dt: float
) -> list[tuple[float, float]]:
    """The cell's area at every stride end, from t = 0."""
    md = load_dict(raw)
    membrane = moving_membrane_of(md)
    assert membrane is not None
    areas = [(0.0, geometry.sizes["cell"])]
    ends = _strides(t_final, dt)
    integrate_moving_cell(
        md,
        geometry,
        membrane,
        t_final=t_final,
        stride_ends=ends,
        output_times=ends,
        on_output=lambda t, _f: areas.append((t, geometry.sizes["cell"])),
    )
    return areas


# -- the scope (decision 7) -------------------------------------------------------------------------------------


def _scope_error(raw: dict[str, Any], description: GeometryDescription | None = None) -> str:
    md = load_dict(raw)
    geometry = realize_multi_compartment(description or _disk(), h=0.3)
    with pytest.raises(ForceBalanceError, match="MovingCellScope") as refused:
        moving_cell_scope(md, geometry, "pm")
    return str(refused.value)


def test_a_species_on_the_moving_cell_is_refused_until_5b() -> None:
    raw = _front()
    md = raw["math_description"]
    md["variables"].append({"name": "m", "subdomain": "cell"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "m",
            "subdomain": "cell",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    assert "['m'] on a moving cell wait for stage 5b" in _scope_error(raw)


def test_a_front_with_a_tension_or_a_traction_is_refused() -> None:
    raw = _front()
    raw["math_description"]["equations"][0]["surfaces"] = {"pm": {"tension": "1"}}
    assert "carries a tension" in _scope_error(raw)
    raw["math_description"]["equations"][0]["surfaces"] = {"pm": {"traction": "[1, 0]"}}
    assert "applied traction on the growing front" in _scope_error(raw)


def test_a_material_membrane_given_a_velocity_is_refused() -> None:
    raw = _material()
    raw["math_description"]["boundary_conditions"] = [
        {"kind": "dirichlet", "variable": "v", "boundary": "pm", "expression": "[0.1, 0]"}
    ]
    assert "velocity condition on the material membrane" in _scope_error(raw)


def test_a_cell_touching_the_box_or_beside_another_membrane_is_refused() -> None:
    raw = _front()
    assert "touches the box" in _scope_error(raw, _disk(box=(-0.8, 0.8, -0.8, 0.8)))  # the box cuts the disk
    nucleus = GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(5.0, 4.0, 1.0),
        origin=(-2.0, -2.0, 0.0),
        subvolumes=(
            SubVolume(name="nucleus", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.16"),
            SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(
            SurfaceClass(name="pm", inside="cell", outside="ec"),
            SurfaceClass(name="ne", inside="nucleus", outside="cell"),
        ),
    )
    assert "must be the only one" in _scope_error(raw, nucleus)


# -- the motion (decisions 2, 3, 5, 6) ----------------------------------------------------------------------------


def test_a_held_front_settles_on_its_equilibrium_area_exactly() -> None:
    """ZV: the edge is held, so the front's speed is `v_p(a)` alone, and the mesh measures `a` exactly; the area
    settles on the root of `v_p`, to round-off, at any `h` (decision 5's re-measured `region_size`)."""
    root = brentq(lambda a: V0 * A0 / a - K * (a - A0**3 / a**2), 1.0, 10.0)
    areas = _run(_front(held=True), realize_multi_compartment(_disk(), h=0.3), 1.0, 0.02)
    assert abs(areas[-1][1] - root) < 1e-9 * root


def test_a_solve_after_moves_equals_a_fresh_assembly_on_the_moved_mesh() -> None:
    """Decision 6: nothing is rebuilt after a move; the force balance on the moved submesh is the one a fresh
    assembly there gives."""
    raw = _front()
    md = load_dict(raw)
    geometry = realize_multi_compartment(_disk(), h=0.3)
    result = integrate_moving_cell(md, geometry, "pm", t_final=0.06, stride_ends=_strides(0.06, 0.02))
    moved = geometry.compartments["cell"].mesh.geometry.x.copy()
    fresh = assemble_cell_force_balance(md, geometry, prebuilt=build_members_mesh(md, geometry))
    v = fresh.solve(result.final_time)
    assert np.allclose(geometry.compartments["cell"].mesh.geometry.x, moved)  # the same moved mesh
    assert result.strides == 3 and result.solves == 2 * 3 + 1  # two per stride, and the final velocity
    scale = float(np.abs(v.x.array).max())
    assert scale > 0.1
    assert float(np.abs(result.velocity.x.array - v.x.array).max()) < 1e-10 * scale


@pytest.mark.parametrize("radius", [1.2, 0.8])
def test_a_material_disk_with_a_tension_relaxes_on_its_area_ode(radius: float) -> None:
    """Decisions 2, 4 and 5: the membrane moves with `trace(v)`, its tension semi-implicit over each half-move, and
    the pressure reads the re-measured area. Shrinking from 1.2 and growing from 0.8 towards the stable radius 1."""
    geometry = realize_multi_compartment(_disk(radius), h=0.15)
    areas = _run(_material(), geometry, 0.5, 0.05)
    exact = solve_ivp(lambda _t, a: [_disk_rate(a[0])], (0.0, 0.5), [areas[0][1]], rtol=1e-12, atol=1e-14).y[0, -1]
    moved = areas[-1][1] - areas[0][1]
    assert abs(moved) > 0.1  # it has moved, towards the stable radius
    assert np.sign(moved) == np.sign(math.pi - areas[0][1])
    assert abs(areas[-1][1] - exact) < 5e-3 * exact  # measured 3.2e-3 growing from 0.8, first order in Δt


# -- through the runner (decision 7) --------------------------------------------------------------------------------

_GEOMETRY = """geometry_description:
  name: disk_in_medium
  dim: 2
  extent: [5.0, 4.0, 1.0]
  origin: [-2.0, -2.0, 0.0]
  subvolumes:
    - { name: cell, type: analytic, expression: "geom.x[0]**2 + geom.x[1]**2 < 1.0" }
    - { name: ec, type: analytic, expression: "1.0" }
  surfaces:
    - { name: pm, inside: cell, outside: ec }
"""


def test_the_cli_moves_a_cell_and_writes_its_velocity_and_centroid(tmp_path: Path) -> None:
    from vcell_fenics.cli import main
    from vcell_fenics.results import Bundle

    (tmp_path / "math.yaml").write_text(yaml.safe_dump(_front()))
    (tmp_path / "geom.yaml").write_text(_GEOMETRY)
    argv = ["--math", str(tmp_path / "math.yaml"), "--geometry", str(tmp_path / "geom.yaml"), "--h", "0.3"]
    out = tmp_path / "out"
    assert main([*argv, "--dt", "0.02", "--t-final", "0.04", "--output-dt", "0.02", "--out", str(out)]) == 0
    bundle = Bundle.open(out / "results.fenics")
    assert [v.name for v in bundle.manifest.variables] == ["v_x", "v_y"]  # the member holds the velocity
    assert bundle.manifest.segments[0].motion == "ale"
    centroid = bundle.centroid("cell")
    assert centroid is not None and centroid.shape == (3, 2)
    assert np.abs(centroid).max() < 0.02  # a symmetric cell does not translate (the mesh is not quite symmetric)
    run = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())["run"]
    assert run["moving_membrane"] == "pm" and run["strides"] == 2 and run["solves"] == 5


def test_the_runner_refuses_more_than_a_moving_cell_by_name(tmp_path: Path) -> None:
    from vcell_fenics.cli import main

    raw = _front()
    raw["math_description"]["equations"][0]["surfaces"] = {"pm": {"tension": "1"}}
    (tmp_path / "math.yaml").write_text(yaml.safe_dump(raw))
    (tmp_path / "geom.yaml").write_text(_GEOMETRY)
    argv = ["--math", str(tmp_path / "math.yaml"), "--geometry", str(tmp_path / "geom.yaml"), "--h", "0.3"]
    assert main([*argv, "--t-final", "0.02", "--out", str(tmp_path / "out")]) != 0
    assert not (tmp_path / "out" / "results.fenics").exists()


# -- convergence (decision 8) ---------------------------------------------------------------------------------------


def _e3(case: str) -> dict[str, Any]:
    records = [json.loads(line) for line in _E3.read_text().splitlines() if line.strip()]
    line: dict[str, Any] = next(r for r in records if r.get("case") == case)
    return line


def _e3_run(case: str, h: float, dt: float, t_final: float) -> float:
    """vcell-core's own document (`fenics`), on the oracle's geometry: the area at `t_final`."""
    document = _e3(case)["fenics"]
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
        h=h,
    )
    return _run(document, geometry, t_final, dt)[-1][1]


@pytest.mark.integration
@pytest.mark.skipif(not _E3.exists(), reason="vcell-core's E3 document needs the sibling checkout")
@pytest.mark.parametrize("case", ["zs", "zv"])
def test_vcell_cores_e3_is_second_order_in_dt_over_its_transient(case: str) -> None:
    """Decision 3's explicit midpoint rule: the area at t = 0.05 (the transient; both settle by t ≈ 0.2) at a fixed
    h, `Δt` halving twice; Richardson's ratio of successive differences is 4 for second order."""
    areas = [_e3_run(case, 0.1, dt, 0.05) for dt in (0.0125, 0.00625, 0.003125)]
    ratio = (areas[1] - areas[0]) / (areas[2] - areas[1])
    assert math.log2(ratio) >= 1.8, (areas, ratio)


@pytest.mark.integration
@pytest.mark.skipif(not _E3.exists(), reason="vcell-core's E3 document needs the sibling checkout")
@pytest.mark.parametrize(("case", "tolerance"), [("zs", 2e-4), ("zv", 1e-9)])
def test_vcell_cores_e3_reaches_its_exact_area_at_t1(case: str, tolerance: float) -> None:
    """vcell-core's criterion: the area at t = 1 against `exact`. Both are at equilibrium there, so it measures
    space only; ZV's is exact (its front reads only the area), ZS's carries the force balance's edge error."""
    exact = _e3(case)["exact"]["area"]
    assert abs(_e3_run(case, 0.1, 0.01, 1.0) - exact) < tolerance * exact


@pytest.mark.integration
def test_a_material_disk_is_first_order_in_dt() -> None:
    """Decision 4: the semi-implicit tension perturbs the force balance by `O(Δt)`, so a material membrane with a
    tension is first order in `Δt`, against the exact area ODE at h = 0.1."""
    errors = []
    for dt in (0.1, 0.05, 0.025):
        geometry = realize_multi_compartment(_disk(1.2), h=0.1)
        areas = _run(_material(), geometry, 1.0, dt)
        exact = solve_ivp(lambda _t, a: [_disk_rate(a[0])], (0.0, 1.0), [areas[0][1]], rtol=1e-12, atol=1e-14)
        errors.append(abs(areas[-1][1] - exact.y[0, -1]))
    rates = [math.log2(a / b) for a, b in pairwise(errors)]
    assert all(r > 0.8 for r in rates), (errors, rates)

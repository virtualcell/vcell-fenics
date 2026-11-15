"""A cell force balance with species on its member, through the runner and the CLI (ADR 020 decision 5).

The runner routes a T9 model in ADR 020 decision 4's scope to the staggered run (a fixed skeleton, one member,
species on it, linear transport) and writes each species and the velocity, by component, to the results bundle
(ADR 010 §3, "Vector fields"). Anything outside the scope is refused by name before a bundle is started.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend.cell_force_balance_run import assemble_cell_force_balance_run
from vcell_fenics.backend.multi_compartment import realize_multi_compartment
from vcell_fenics.cli import main
from vcell_fenics.formalism import load_dict
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.results import Bundle
from vcell_fenics.runner import ModelInput, RunError, RunOptions, run_model, uniform_output_times

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

# Nickaeen et al. 2017's zero-velocity cell (as in tests/test_backend_cell_force_balance_run.py)
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
      initial_condition: "1.5 + 0.5 * geom.x[0]"
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


def _raw() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(_MYOSIN)
    return raw


def _options(**overrides: Any) -> RunOptions:
    settings: dict[str, Any] = {
        "h": 0.15,
        "dt": 0.05,
        "t_final": 0.2,
        "output_times": uniform_output_times(0.2, 0.1),
        "fe_degree": 1,
        "time_integration": "backward_euler",
    }
    return RunOptions(**(settings | overrides))


def _run(raw: dict[str, Any], out: Path, **overrides: Any) -> dict[str, Any]:
    model = ModelInput(geometry=_disk(), math=load_dict(raw), source="native")
    return run_model(model, _options(**overrides), out)


def test_the_runner_writes_the_species_and_the_velocity_by_component(tmp_path: Path) -> None:
    summary = _run(_raw(), tmp_path)
    assert summary["run"]["backend"] == "cell_force_balance"
    assert summary["run"]["steps"] == 4

    bundle = Bundle.open(tmp_path / "results.fenics")
    manifest = bundle.manifest
    assert manifest.status == "completed"
    assert bundle.times == (0.0, 0.1, 0.2)
    assert list(manifest.domains) == ["cell"]
    assert [v.name for v in manifest.variables] == ["m", "v_x", "v_y"]
    assert manifest.variable("cell", "m").vector is None
    components = manifest.vectors("cell")
    assert list(components) == ["v"]
    assert [(c.name, c.vector.component if c.vector else None) for c in components["v"]] == [("v_x", 0), ("v_y", 1)]

    # the transport keeps ∫m (ADR 020 decision 2), row after row
    totals = bundle.stats("cell", "m")[:, 1]
    assert np.allclose(totals, totals[0], rtol=1e-10, atol=0.0)

    assert summary["species"] == ["m"]
    assert summary["vectors"] == {"v": ["v_x", "v_y"]}

    # each row's velocity is the one solved from that row's myosin (decision 5), against an independent run
    run = assemble_cell_force_balance_run(load_dict(_raw()), realize_multi_compartment(_disk(), h=0.15))
    dx = ufl.dx(domain=run.balance.members.mesh)
    for row in range(3):
        if row:
            run.advance(0.1 * row, 0.05)
        v = run.solve_velocity()
        for i, name in enumerate(("v_x", "v_y")):
            expected = fem.assemble_scalar(fem.form(v[i] * dx)).real
            assert bundle.stats("cell", name)[row, 1] == pytest.approx(expected, rel=1e-8, abs=1e-12), (row, name)
    assert np.abs(bundle.field("cell", "v_x", 0)).max() > 1e-3  # the gradient in m drives a flow


def test_a_moving_cell_is_refused_before_a_bundle_starts(tmp_path: Path) -> None:
    raw = _raw()
    raw["math_description"]["boundary_conditions"] = []  # a free membrane: the cell would move (stage 4)
    with pytest.raises(RunError, match=r"ADR 020 decision 4's scope.*MovingSkeleton"):
        _run(raw, tmp_path)
    assert not (tmp_path / "results.fenics").exists()


def test_a_prescribed_motion_elsewhere_is_refused(tmp_path: Path) -> None:
    """The review's finding: a motion on the medium would otherwise be dropped without a word."""

    raw = _raw()
    raw["math_description"]["subdomains"][0]["motion"] = {"kind": "prescribed", "velocity": "[1.0, 0.0]"}
    with pytest.raises(RunError, match=r"'ec' has a prescribed motion"):
        _run(raw, tmp_path)
    assert not (tmp_path / "results.fenics").exists()


def test_a_model_error_is_not_blamed_on_the_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a scope refusal (SpeciesNotYetSupported, MovingSkeleton) points at ADR 020 decision 4."""

    from vcell_fenics.backend import cell_force_balance_run
    from vcell_fenics.backend.cell_force_balance import ForceBalanceError

    def failing(*args: Any) -> None:
        raise ForceBalanceError("surface 'pm' has no facets in the realized geometry")

    monkeypatch.setattr(cell_force_balance_run, "assemble_cell_force_balance_run", failing)
    with pytest.raises(RunError, match="cannot be assembled: surface 'pm' has no facets") as caught:
        _run(_raw(), tmp_path)
    assert "decision 4" not in str(caught.value)


def test_a_velocity_with_no_species_is_refused(tmp_path: Path) -> None:
    raw = _raw()
    md = raw["math_description"]
    md["equations"] = md["equations"][1:]
    md["variables"] = md["variables"][1:]
    md["equations"][0]["volumes"]["cell"]["active_stress"] = "beta"
    with pytest.raises(RunError, match="needs a species"):
        _run(raw, tmp_path)


def test_a_higher_degree_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="P1 velocity"):
        _run(_raw(), tmp_path, fe_degree=2)


def test_a_species_named_like_a_velocity_component_is_refused(tmp_path: Path) -> None:
    raw = _raw()
    md = raw["math_description"]
    md["variables"].append({"name": "v_y", "subdomain": "cell"})
    md["equations"].append(
        {
            "template": "bulk_radv_diff",
            "variable": "v_y",
            "subdomain": "cell",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1"},
            "initial_condition": "1",
        }
    )
    with pytest.raises(RunError, match="written to the bundle as v_y"):
        _run(raw, tmp_path)


def test_the_cli_runs_a_force_balance(tmp_path: Path) -> None:
    (tmp_path / "math.yaml").write_text(_MYOSIN)
    (tmp_path / "geom.yaml").write_text(_GEOMETRY)
    argv = ["--math", str(tmp_path / "math.yaml"), "--geometry", str(tmp_path / "geom.yaml")]
    out = tmp_path / "out"
    assert (
        main([*argv, "--h", "0.2", "--dt", "0.05", "--t-final", "0.1", "--output-dt", "0.05", "--out", str(out)]) == 0
    )
    bundle = Bundle.open(out / "results.fenics")
    assert bundle.times == (0.0, 0.05, 0.1)
    assert set(bundle.manifest.vectors("cell")) == {"v"}


_E2 = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "math" / "resolved-cell.jsonl"


@pytest.mark.skipif(not _E2.exists(), reason="vcell-core's E2 document needs the sibling checkout")
def test_vcell_cores_e2_runs_through_the_cli(tmp_path: Path) -> None:
    """vcell-core's E2 (#251), read from the sibling checkout, through the CLI as vcell-core runs E1 (#249)."""

    records = [json.loads(line) for line in _E2.read_text().splitlines() if line.strip()]
    document = next(r for r in records if r.get("case") == "e2")["fenics"]
    (tmp_path / "geom.yaml").write_text(_GEOMETRY.replace("name: pm,", "name: cell_ec_membrane,"))
    (tmp_path / "math.json").write_text(json.dumps(document))
    argv = ["--math", str(tmp_path / "math.json"), "--geometry", str(tmp_path / "geom.yaml")]
    out = tmp_path / "out"
    assert (
        main([*argv, "--h", "0.2", "--dt", "0.01", "--t-final", "0.02", "--output-dt", "0.01", "--out", str(out)]) == 0
    )
    bundle = Bundle.open(out / "results.fenics")
    assert [v.name for v in bundle.manifest.variables] == ["m", "v_cyt_x", "v_cyt_y"]  # E2 names it v_cyt
    totals = bundle.stats("cell", "m")[:, 1]
    assert np.allclose(totals, totals[0], rtol=1e-10, atol=0.0)

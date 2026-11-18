"""A diffusion or a drift that reads the species, on every route (ADR 021 decision 2).

Both assemblers bind the species in the transport slots, as they always did in `source`. Such a slot is
nonlinear, so backward Euler refuses it (`NonlinearTermError`) and the method of lines runs it. The runner
chooses the scheme from the formalism's linearity check (`formalism.linearity`), before anything is assembled.

**The known answer.** On the unit disk, `c* = 1 + x` with the diffusion `1 + c`, the drift `[c, 0]` and the source
`1 + 2x`, held at `c*` on the boundary:
- `−∇·((1 + c*)∇c*) = −|∇c*|² = −1`;
- `∇·(c* [c*, 0]) = 2 c* ∂ₓc* = 2(1 + x)`.

So `c*` is the steady state. It is linear, so P1 holds it exactly and the Galerkin residual vanishes at it: the
discrete steady state is `c*` to the solver's tolerance. A slot that did not read `c`, or a wrong sign, leaves an
O(1) error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import ufl
import yaml
from dolfinx import fem

from vcell_fenics.backend import assemble, make_disk_geometry
from vcell_fenics.backend.diagnostics import NonlinearTermError
from vcell_fenics.backend.multi_compartment import integrate_multi_compartment, realize_multi_compartment
from vcell_fenics.backend.reaction_diffusion import integrate_discrete_problem
from vcell_fenics.formalism import load_dict, load_yaml
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.linearity import affine, nonlinear_expressions, reads
from vcell_fenics.formalism.parser import parse

_MODEL = """
math_description:
  geometry: disk_2d
  subdomains:
    - { name: cyto, kind: volume, motion: { kind: none } }
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "1 + c", relative_advection: "[c, 0]", source: "1 + 2*geom.x[0]" }
      initial_condition: "1.0"
  boundary_conditions:
    - { kind: dirichlet, variable: c, boundary: wall, expression: "1 + geom.x[0]" }
"""


def test_a_diffusion_and_a_drift_that_read_the_species_reach_the_exact_steady_state() -> None:
    md = load_yaml(_MODEL)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.2)
    problem = assemble(md, geometry, dt=0.1, fe_degree=1)
    integrate_discrete_problem(problem, t_final=10.0, rtol=1e-9, atol=1e-11)
    x = problem.V.tabulate_dof_coordinates()[:, 0]
    assert np.abs(problem.unknown.x.array - (1.0 + x)).max() < 1e-6


def test_backward_euler_names_the_nonlinear_slot_rather_than_failing_to_compile() -> None:
    md = load_yaml(_MODEL)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    problem = assemble(md, geometry, dt=0.1, fe_degree=1)
    with pytest.raises(NonlinearTermError):
        problem.step()


def test_an_unrelated_value_error_is_not_taken_for_a_nonlinear_model() -> None:
    md = load_yaml(_MODEL.replace('relative_advection: "[c, 0]", ', ""))
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", boundary="wall", radius=1.0, h=0.3)
    problem = assemble(md, geometry, dt=0.1, fe_degree=1)

    class Broken:
        def compose(self, _problem: object) -> None:
            raise ValueError("a defect elsewhere")

    problem.scheme = Broken()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="a defect elsewhere"):
        problem.step()


def test_the_runner_switches_a_nonlinear_model_to_the_method_of_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vcell_fenics.cli import main

    geometry = """geometry_description:
  name: disk_2d
  # a square, despite the math's name for it: the zero-flux check needs only a closed box
  dim: 2
  extent: [2.0, 2.0, 1.0]
  origin: [-1.0, -1.0, 0.0]
  subvolumes:
    - { name: cyto, type: analytic, expression: "1.0" }
"""
    document = yaml.safe_load(_MODEL)
    document["math_description"]["boundary_conditions"] = []
    document["math_description"]["equations"][0]["terms"] = {"diffusion": "1 + c", "source": "0"}
    document["math_description"]["equations"][0]["initial_condition"] = "1 + geom.x[0]"
    (tmp_path / "math.yaml").write_text(yaml.safe_dump(document))
    (tmp_path / "geom.yaml").write_text(geometry)
    argv = ["--math", str(tmp_path / "math.yaml"), "--geometry", str(tmp_path / "geom.yaml"), "--h", "0.4"]
    out = tmp_path / "out"
    assert main([*argv, "--t-final", "0.1", "--time-integration", "backward_euler", "--out", str(out)]) == 0
    assert "runs by the method of lines" in capsys.readouterr().err

    from vcell_fenics.results import Bundle

    bundle = Bundle.open(out / "results.fenics")
    assert bundle.manifest.solver.options["time_integration"] == "method_of_lines"
    import json

    run = json.loads((out / "results.fenics" / "provenance" / "summary.json").read_text())["run"]
    assert run["dt"] is None  # one adaptive integration: no fixed step (valid JSON, not NaN)
    assert (run["rtol"], run["atol"]) == (1e-6, 1e-8)
    totals = bundle.stats("cyto", "c")[:, 1]
    assert np.allclose(totals, totals[0], rtol=1e-6)  # zero flux: the nonlinear diffusion keeps ∫c


def test_a_linear_model_keeps_backward_euler(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from vcell_fenics.runner import ModelInput, RunOptions, _scheme_for_nonlinear

    md = load_yaml(_MODEL.replace("1 + c", "2.0").replace("[c, 0]", "[1, 0]"))
    options = RunOptions(
        h=0.3, dt=0.1, t_final=0.1, output_times=(0.0, 0.1), fe_degree=1, time_integration="backward_euler"
    )
    model = ModelInput(geometry=_disk_description(), math=md, source="native")
    assert _scheme_for_nonlinear(model, options).time_integration == "backward_euler"
    assert "method of lines" not in capsys.readouterr().err


def _disk_description() -> GeometryDescription:
    return GeometryDescription(
        name="disk_2d",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(SubVolume(name="cyto", type="analytic", expression="1.0"),),
    )


# --- the multi-compartment solver ---------------------------------------------------------------------------

_TWO = {
    "math_description": {
        "geometry": "disk_in_medium",
        "subdomains": [
            {"name": "ec", "kind": "volume"},
            {"name": "cell", "kind": "volume"},
            {"name": "pm", "kind": "surface"},
        ],
        "variables": [{"name": "c", "subdomain": "cell"}, {"name": "e", "subdomain": "ec"}],
        "equations": [
            {
                "template": "bulk_radv_diff",
                "variable": "c",
                "subdomain": "cell",
                "temporality": "time_dependent",
                "terms": {"diffusion": "DIFFUSION"},
                "initial_condition": "1 + geom.x[0]",
            },
            {
                "template": "bulk_radv_diff",
                "variable": "e",
                "subdomain": "ec",
                "temporality": "time_dependent",
                "terms": {"diffusion": "1"},
                "initial_condition": "0",
            },
        ],
    }
}


def _disk_in_medium() -> GeometryDescription:
    return GeometryDescription(
        name="disk_in_medium",
        dim=2,
        extent=(3.0, 3.0, 1.0),
        origin=(-1.5, -1.5, 0.0),
        subvolumes=(
            SubVolume(name="cell", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 1.0"),
            SubVolume(name="ec", type="analytic", expression="1.0"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),),
    )


def _spread(diffusion: str) -> tuple[float, float, float]:
    """∫c at t = 0 and t = 0.05, and the spread max − min of c at t = 0.05."""

    raw = yaml.safe_load(yaml.safe_dump(_TWO).replace("DIFFUSION", diffusion))
    geometry = realize_multi_compartment(_disk_in_medium(), h=0.2)
    totals: list[float] = []
    spread: list[float] = []

    def on_output(t: float, fields: dict[str, fem.Function]) -> None:
        c = fields["c"]
        dx = ufl.dx(domain=c.function_space.mesh)
        totals.append(float(fem.assemble_scalar(fem.form(c * dx)).real))
        spread.append(float(c.x.array.max() - c.x.array.min()))

    integrate_multi_compartment(
        load_dict(raw), geometry, t_final=0.05, output_times=(0.0, 0.05), on_output=on_output, rtol=1e-8, atol=1e-10
    )
    return totals[0], totals[-1], spread[-1]


def test_the_multi_compartment_solver_reads_the_species_in_a_diffusion() -> None:
    # c starts at 1 + x on the disk, with no flux out. D = 1 + c spreads it faster where c is larger than the
    # constant D = 1 does, and both keep ∫c.
    start, end, nonlinear = _spread("1 + c")
    assert end == pytest.approx(start, rel=1e-8)
    _, _, constant = _spread("1")
    assert nonlinear < 0.95 * constant  # D = 1 + c ≥ 1 everywhere, so the profile flattens faster


# --- the linearity check -------------------------------------------------------------------------------------

_U = frozenset({"m", "n"})


@pytest.mark.parametrize(
    ("text", "is_affine"),
    [
        ("2*m + n - 3", True),
        ("k*m/V", True),
        ("grad(m)", True),
        ("trace(m)", True),
        ("m**1", False),  # affine, but UFL cannot split an argument out of a power
        ("[m, n]", True),
        ("[m, 0]", False),  # nor out of a literal with a constant component
        ("m*n", False),
        ("m**2", False),
        ("1/(1 + m)", False),
        ("exp(m)", False),
        ("max(0, 1 - m/15)", False),
        ("(m > 1) * 3", False),
        ("inner(grad(m), [1, 0])", True),
        ("inner([m, n], [n, m])", False),
        ("sin(geom.x[0]) * m", True),
        ("pow(m, 2)", False),
    ],
)
def test_affine_in_the_unknowns(text: str, is_affine: bool) -> None:
    assert affine(parse(text), _U) is is_affine


def test_reads_follows_a_trace() -> None:
    assert reads(parse("k * trace(m)"), _U)
    assert not reads(parse("k * geom.x[0]"), _U)


def _e4a_like(**terms: str) -> dict[str, object]:
    return {
        "math_description": {
            "geometry": "disk_in_medium",
            "subdomains": [
                {"name": "ec", "kind": "volume"},
                {"name": "cell", "kind": "volume", "motion": {"kind": "unknown", "variable": "v"}},
                {"name": "pm", "kind": "surface", "motion": {"kind": "unknown", "variable": "v"}},
            ],
            "variables": [{"name": "m", "subdomain": "cell"}, {"name": "v", "subdomain": "cell", "type": "vector"}],
            "equations": [
                {
                    "template": "bulk_radv_diff",
                    "variable": "m",
                    "subdomain": "cell",
                    "temporality": "time_dependent",
                    "terms": terms,
                    "initial_condition": "1",
                },
                {
                    "template": "cell_force_balance",
                    "variable": "v",
                    "subdomain": "cell",
                    "temporality": "steady_state",
                    "viscous_operator": "laplacian",
                    "volumes": {"cell": {"viscosity": "1", "drag": "1", "active_stress": "m"}},
                },
            ],
            "boundary_conditions": [{"kind": "dirichlet", "variable": "v", "boundary": "pm", "expression": "0"}],
        }
    }


def test_a_slot_that_reads_only_the_lagged_velocity_is_linear() -> None:
    # ADR 020 decision 3: v is solved in its own step and read lagged
    assert nonlinear_expressions(load_dict(_e4a_like(diffusion="1", relative_advection="0.1 * v"))) == []


def test_the_crowding_cut_offs_are_nonlinear() -> None:
    found = nonlinear_expressions(
        load_dict(_e4a_like(diffusion="1 - m/125", relative_advection="(max(0, 1 - m/15) - 1) * v"))
    )
    assert [e.where for e in found] == ["equation 'm' slot 'diffusion'", "equation 'm' slot 'relative_advection'"]


def test_a_nonlinear_boundary_flux_and_a_region_variable_are_found() -> None:
    raw = yaml.safe_load(_MODEL)
    md = raw["math_description"]
    md["equations"][0]["terms"] = {"diffusion": "1", "source": "0"}
    md["boundary_conditions"] = [{"kind": "neumann", "variable": "c", "boundary": "wall", "expression": "-c/(1 + c)"}]
    found = nonlinear_expressions(load_dict(raw))
    assert [e.where for e in found] == ["neumann condition on 'c' at 'wall'"]

    md["boundary_conditions"] = []
    md["variables"].append({"name": "R", "subdomain": "cyto", "space": "region"})
    md["equations"][0]["terms"]["source"] = "c * R"
    assert [e.where for e in nonlinear_expressions(load_dict(raw))] == ["equation 'c' slot 'source'"]


def test_a_membrane_diffusion_reads_its_own_species() -> None:
    # two species in one membrane block (the matrix-free path); D = 1 + r reads its own
    def run(diffusion: str) -> tuple[float, float, float]:
        raw = {
            "math_description": {
                "geometry": "disk_in_medium",
                "subdomains": [
                    {"name": "ec", "kind": "volume"},
                    {"name": "cell", "kind": "volume"},
                    {"name": "pm", "kind": "surface"},
                ],
                "variables": [
                    {"name": "c", "subdomain": "cell"},
                    {"name": "r", "subdomain": "pm"},
                    {"name": "q", "subdomain": "pm"},
                ],
                "equations": [
                    {
                        "template": "bulk_radv_diff",
                        "variable": "c",
                        "subdomain": "cell",
                        "temporality": "time_dependent",
                        "terms": {"diffusion": "1"},
                        "initial_condition": "1",
                    },
                    {
                        "template": "surface_pde_with_dilution",
                        "variable": "r",
                        "subdomain": "pm",
                        "temporality": "time_dependent",
                        "terms": {"diffusion": diffusion},
                        "initial_condition": "1 + geom.x[0]",
                    },
                    {
                        "template": "surface_pde_with_dilution",
                        "variable": "q",
                        "subdomain": "pm",
                        "temporality": "time_dependent",
                        "terms": {"diffusion": "1"},
                        "initial_condition": "1",
                    },
                ],
            }
        }
        out: list[tuple[float, float]] = []

        def on_output(t: float, fields: dict[str, fem.Function]) -> None:
            r = fields["r"]
            total = float(fem.assemble_scalar(fem.form(r * ufl.dx(domain=r.function_space.mesh))).real)
            out.append((total, float(r.x.array.max() - r.x.array.min())))

        geometry = realize_multi_compartment(_disk_in_medium(), h=0.2)
        integrate_multi_compartment(
            load_dict(raw), geometry, t_final=0.05, output_times=(0.0, 0.05), on_output=on_output, rtol=1e-8, atol=1e-10
        )
        return out[0][0], out[-1][0], out[-1][1]

    start, end, nonlinear = run("1 + r*q")
    assert end == pytest.approx(start, rel=1e-10)
    _, _, constant = run("1")
    assert nonlinear < 0.98 * constant  # 0.955 of it measured: D = 1 + r ≥ 1 flattens r faster


def test_a_membrane_diffusion_that_reads_a_bulk_species_is_refused_by_name() -> None:
    raw = yaml.safe_load(yaml.safe_dump(_TWO).replace("DIFFUSION", "1"))
    md = raw["math_description"]
    md["variables"].append({"name": "r", "subdomain": "pm"})
    md["equations"].append(
        {
            "template": "surface_pde_with_dilution",
            "variable": "r",
            "subdomain": "pm",
            "temporality": "time_dependent",
            "terms": {"diffusion": "1 + trace(c)"},
            "initial_condition": "1",
        }
    )
    geometry = realize_multi_compartment(_disk_in_medium(), h=0.3)
    with pytest.raises(NotImplementedError, match="membrane species 'r' reads a bulk species"):
        integrate_multi_compartment(load_dict(raw), geometry, t_final=0.01)


def test_a_bulk_diffusion_reads_a_region_variable() -> None:
    raw = yaml.safe_load(yaml.safe_dump(_TWO).replace("DIFFUSION", "1 + R"))
    md = raw["math_description"]
    md["variables"].append({"name": "R", "subdomain": "cell", "space": "region"})
    md["equations"].append(
        {
            "template": "region_ode",
            "variable": "R",
            "subdomain": "cell",
            "temporality": "time_dependent",
            "terms": {"uniform_rate": "1"},
            "initial_condition": "0",
        }
    )
    geometry = realize_multi_compartment(_disk_in_medium(), h=0.2)
    totals: list[float] = []

    def on_output(t: float, fields: dict[str, fem.Function]) -> None:
        c = fields["c"]
        totals.append(float(fem.assemble_scalar(fem.form(c * ufl.dx(domain=c.function_space.mesh))).real))

    integrate_multi_compartment(
        load_dict(raw), geometry, t_final=0.05, output_times=(0.0, 0.05), on_output=on_output, rtol=1e-8, atol=1e-10
    )
    assert totals[-1] == pytest.approx(totals[0], rel=1e-8)


# --- the moving single mesh (ALE) -----------------------------------------------------------------------------


def test_a_diffusion_that_reads_the_species_runs_on_a_translating_mesh() -> None:
    """A rigid translation keeps the area, so ∫c is conserved; D = 0.1 (1 + c) reads the species. Backward Euler
    refuses it; the strided method of lines runs it."""

    from vcell_fenics.backend import integrate_discrete_problem_moving

    text = _MODEL.replace("motion: { kind: none }", 'motion: { kind: prescribed, velocity: "[0.5, 0.0]" }')
    text = text.replace(
        'terms: { diffusion: "1 + c", relative_advection: "[c, 0]", source: "1 + 2*geom.x[0]" }',
        'terms: { diffusion: "0.1 * (1 + c)" }',
    )
    text = text.replace('initial_condition: "1.0"', 'initial_condition: "1.0 + 0.3*geom.x[0]"')
    text = text[: text.index("  boundary_conditions:")]
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.15)

    refused = assemble(load_yaml(text), geometry, dt=0.01)
    with pytest.raises(NonlinearTermError):
        refused.step()

    problem = assemble(
        load_yaml(text), make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.15), dt=0.01
    )
    before = problem.total_mass()
    integrate_discrete_problem_moving(problem, t_final=0.2, motion_steps=10)
    assert problem.total_mass() == pytest.approx(before, rel=1e-6)
    assert problem.V.mesh.geometry.x[:, 0].mean() == pytest.approx(0.1, abs=0.02)  # moved by v·T on average

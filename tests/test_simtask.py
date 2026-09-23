"""Reading VCell's SimulationTask document (ADR 011 §1) and resolving MathOverrides exactly as VCell does.

The fixtures under `tests/fixtures/simtask/` are real VCell documents: the fvsolver smoke task
(`SimID_1585623750_0__0`, 2D, four species) and the Slurm fixtures of `vcell-server` — a finite-volume
task, a moving-boundary task, a Langevin task and a non-spatial Runge–Kutta task whose simulation scans
`Kf`. Cases the fixtures do not cover (plain overrides, list scans, explicit / KeepEvery output, the
FEniCSx options block) are synthesized by editing the smoke fixture's XML.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from lxml import etree

from vcell_fenics.cli import _resolve, build_parser, load_simtask, main
from vcell_fenics.pyvcell_bridge.overrides import (
    MathOverrides,
    OverrideError,
    ScanSpec,
    apply_overrides,
    parse_math_overrides,
    resolve_overrides,
    scan_coordinates,
)
from vcell_fenics.pyvcell_bridge.simtask import SimulationTaskError, check_supported, read_simtask
from vcell_fenics.results import Bundle

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "simtask"
_SMOKE = _FIXTURES / "SimID_1585623750_0__0.simtask.xml"
_NS = "http://sourceforge.net/projects/vcell/vcml"


def _edit(tmp_path: Path, edit: object, *, name: str = "SimID_1585623750_0__0.simtask.xml") -> Path:
    """Apply ``edit(root)`` to a copy of the smoke fixture and return the new file."""

    tree = etree.parse(str(_SMOKE))
    edit(tree.getroot())  # type: ignore[operator]
    path = tmp_path / name
    tree.write(str(path))
    return path


def _find(root: etree._Element, path: str) -> etree._Element:
    found = root.find("/".join(f"{{{_NS}}}{part}" for part in path.split("/")))
    assert found is not None, path
    return found


# -- the scan odometer and ConstantArraySpec values (ports of the Java) --------------------------------


def test_scan_coordinates_first_axis_slowest() -> None:
    # A 2×3 scan: bounds (1, 2). Java's scanIndexToScanParameterCoordinate: index = i·3 + j.
    assert [scan_coordinates(k, (1, 2)) for k in range(6)] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert scan_coordinates(23, (1, 2, 3)) == (1, 2, 3)
    with pytest.raises(OverrideError, match="out of range"):
        scan_coordinates(6, (1, 2))


def test_scanned_names_are_sorted_and_the_job_index_wraps() -> None:
    # Declared b-then-a, but VCell sorts the names: `a` (2 values) is the slow axis, `b` (3) the fast one.
    overrides = MathOverrides(
        plain={"k": "5.0"}, scans=(ScanSpec("b", ("b0", "b1", "b2")), ScanSpec("a", ("a0", "a1")))
    )
    assert overrides.scan_count == 6
    picks = [(resolve_overrides(overrides, j)["a"], resolve_overrides(overrides, j)["b"]) for j in range(6)]
    assert picks == [("a0", "b0"), ("a0", "b1"), ("a0", "b2"), ("a1", "b0"), ("a1", "b1"), ("a1", "b2")]
    assert resolve_overrides(overrides, 7) == resolve_overrides(overrides, 1)  # scanIndex = jobIndex % scanCount
    assert resolve_overrides(overrides, 3)["k"] == "5.0"  # plain overrides apply to every job


def test_interval_and_list_specs() -> None:
    parsed = parse_math_overrides(
        [
            ("lin", "1001", "0.0 to 1.0, 5 values"),
            ("log", "1001", "0.01 to 10.0, log, 4 values"),
            ("old", "1000", "1.0, 2.0, 3.0"),
            ("new", "1000", '"KMOLE", "KMOLE*2"'),
            ("k", None, " 7.5 "),
        ]
    )
    scans = {scan.name: scan.values for scan in parsed.scans}
    assert [float(v) for v in scans["lin"]] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert [float(v) for v in scans["log"]] == pytest.approx([0.01, 0.1, 1.0, 10.0])
    assert scans["old"] == ("1.0", "2.0", "3.0")
    assert scans["new"] == ("KMOLE", "KMOLE*2")
    assert parsed.plain == {"k": "7.5"}


def test_symbolic_interval_bounds_stay_expressions() -> None:
    (scan,) = parse_math_overrides([("k", "1001", "KMOLE to 2*KMOLE, 3 values")]).scans
    assert scan.values[0] == "((KMOLE) + (((2*KMOLE) - (KMOLE)) * 0.0))"
    assert len(scan.values) == 3


@pytest.mark.parametrize(
    "spec,body,match",
    [
        ("1001", "1 to 10", "invalid"),
        ("1001", "-1 to 10, log, 3 values", "log interval"),
        ("1000", "1.0", "at least two"),
        ("1234", "1", "unknown ConstantArraySpec"),
        (None, "", "no expression"),
    ],
)
def test_malformed_overrides_are_rejected(spec: str | None, body: str, match: str) -> None:
    with pytest.raises(OverrideError, match=match):
        parse_math_overrides([("k", spec, body)])


def test_overriding_an_unknown_constant_is_an_error() -> None:
    task = read_simtask(_SMOKE)
    with pytest.raises(OverrideError, match="does not define"):
        apply_overrides(task.math, {"no_such_constant": "1.0"})


# -- reading real SimulationTask documents --------------------------------------------------------------


def test_reads_the_fvsolver_smoke_task() -> None:
    task = read_simtask(_SMOKE)
    assert (task.task_id, task.job_index, task.sim_key) == (0, 0, "1585623750")
    assert task.job_prefix == "SimID_1585623750_0_"
    assert (task.start_time, task.end_time) == (0.0, 0.01)
    assert task.output_times() == pytest.approx((0.0, 0.005, 0.01))
    assert (task.abs_tol, task.rel_tol, task.dt_default) == (1e-9, 1e-7, 0.05)
    assert task.mesh_size == (51, 51, 1)
    assert task.geometry.dim == 2
    assert {c.name for c in task.math.constants} >= {"Kf", "Kr"}
    warnings = check_supported(task)
    assert len(warnings) == 1 and "solving it with FEniCSx anyway" in warnings[0]


def test_the_job_index_selects_the_scan_point(tmp_path: Path) -> None:
    def add_scan(root: etree._Element) -> None:
        root.set("JobIndex", "1")
        overrides = _find(root, "Simulation/MathOverrides")
        constant = etree.SubElement(overrides, f"{{{_NS}}}Constant", Name="Kf", ConstantArraySpec="1000")
        constant.text = "1.0, 3.0"
        plain = etree.SubElement(overrides, f"{{{_NS}}}Constant", Name="Kr")
        plain.text = "500.0"

    task = read_simtask(_edit(tmp_path, add_scan, name="SimID_1585623750_1__0.simtask.xml"))
    assert task.job_prefix == "SimID_1585623750_1_"
    assert task.resolved_overrides == {"Kf": "3.0", "Kr": "500.0"}
    constants = {c.name: c.exp for c in task.math.constants}
    assert (constants["Kf"], constants["Kr"]) == ("3.0", "500.0")


def test_scan_fixture_resolves_job_zero_to_the_interval_minimum() -> None:
    task = read_simtask(_FIXTURES / "SimID_274631114_0__0.simtask.xml")  # Kf: 0.01 to 10.0, log, 4 values
    assert task.overrides.scan_count == 4
    assert task.resolved_overrides == {"Kf": "0.01"}


@pytest.mark.parametrize(
    "attributes,expected",
    [
        ({"OutputTimes": "0.002,0.004,0.008"}, (0.0, 0.002, 0.004, 0.008)),  # explicit: ends at the last one
        ({"KeepEvery": "1", "KeepAtMost": "4"}, (0.0, 0.0025, 0.005, 0.0075, 0.01)),  # thinned to 4 outputs
    ],
)
def test_output_option_variants(tmp_path: Path, attributes: dict[str, str], expected: tuple[float, ...]) -> None:
    def set_output(root: etree._Element) -> None:
        options = _find(root, "Simulation/SolverTaskDescription/OutputOptions")
        options.attrib.clear()
        options.attrib.update(attributes)
        # KeepEvery counts solver steps: at dt = 0.001, every step over [0, 0.01] is 10 outputs,
        # which KeepAtMost = 4 thins to 4 evenly spaced ones.
        _find(root, "Simulation/SolverTaskDescription/TimeStep").set("DefaultTime", "0.001")

    task = read_simtask(_edit(tmp_path, set_output))
    assert task.output_times() == pytest.approx(expected)


def test_fenicsx_options_block(tmp_path: Path) -> None:
    def add_options(root: etree._Element) -> None:
        description = _find(root, "Simulation/SolverTaskDescription")
        description.set("Solver", "FEniCSx")
        block = etree.SubElement(
            description, f"{{{_NS}}}FEniCSxSolverOptions", ElementDegree="2", TimeIntegration="MethodOfLines"
        )
        etree.SubElement(block, f"{{{_NS}}}MaxElementSize").text = "0.05"
        etree.SubElement(block, f"{{{_NS}}}Preconditioner").text = "hypre"

    task = read_simtask(_edit(tmp_path, add_options))
    assert task.fenicsx is not None
    assert (task.fenicsx.element_degree, task.fenicsx.max_element_size, task.fenicsx.time_integration) == (
        2,
        0.05,
        "method_of_lines",
    )
    warnings = check_supported(task)
    assert warnings == ["ignoring unknown FEniCSxSolverOptions entries: Preconditioner"]


# -- what the solver refuses rather than mis-solves ---------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,match",
    [
        ("SimID_274631114_0__0.simtask.xml", "non-spatial"),
    ],
)
def test_unsupported_real_tasks_are_refused(fixture: str, match: str) -> None:
    with pytest.raises(SimulationTaskError, match=match):
        check_supported(read_simtask(_FIXTURES / fixture))


@pytest.mark.parametrize(
    "edit,match",
    [
        (lambda r: _find(r, "Simulation/SolverTaskDescription/TimeBound").set("StartTime", "1.0"), "StartTime"),
        (lambda r: _find(r, "Simulation/SolverTaskDescription").set("TaskType", "Steady"), "Unsteady"),
        (
            lambda r: r.insert(0, etree.Element(f"{{{_NS}}}FieldFunctionIdentifierSpec")),
            "field data",
        ),
    ],
)
def test_unsupported_task_features_are_refused(tmp_path: Path, edit: object, match: str) -> None:
    def apply(root: etree._Element) -> None:
        edit(root)  # type: ignore[operator]
        spec = root.find(f"{{{_NS}}}FieldFunctionIdentifierSpec")
        if spec is not None:
            spec.text = "field1,fd,1,0,Volume"

    with pytest.raises(SimulationTaskError, match=match):
        check_supported(read_simtask(_edit(tmp_path, apply)))


def test_a_document_that_is_not_a_simulation_task(tmp_path: Path) -> None:
    path = tmp_path / "model.vcml"
    path.write_text('<vcml xmlns="http://sourceforge.net/projects/vcell/vcml"><BioModel Name="m"/></vcml>')
    with pytest.raises(SimulationTaskError, match="not <SimulationTask>"):
        read_simtask(path)


# -- the --simtask input kind, end to end -----------------------------------------------------------------


def _simtask_args(path: Path, *extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["--simtask", str(path), *extra])


def test_simtask_settings_become_the_run_defaults() -> None:
    model = load_simtask(_SMOKE)
    options, overrides = _resolve(model, _simtask_args(_SMOKE))
    assert options.output_times == pytest.approx((0.0, 0.005, 0.01))
    assert options.time_integration == "method_of_lines"  # nonlinear VCell kinetics: MOL by default
    assert (options.rtol, options.atol) == (1e-7, 1e-9)  # the task's ErrorTolerance
    assert options.h == pytest.approx(min(model.geometry.extent[:2]) / 51)  # its 51×51 mesh
    assert overrides == {}
    assert model.suggested_prefix == "SimID_1585623750_0_"
    assert model.suggested_out_dir == _SMOKE.parent


def test_explicit_flags_override_the_task_and_are_recorded(tmp_path: Path) -> None:
    status = main(["--simtask", str(_SMOKE), "--out", str(tmp_path), "--h", "0.4", "--no-fields"])
    assert status == 0
    bundle = Bundle.open(tmp_path / "SimID_1585623750_0_.fenics")
    assert bundle.manifest.solver.overrides["h"]["flag"] == 0.4
    assert bundle.manifest.solver.options["h"] == 0.4


def test_a_vcell_solver_task_runs_end_to_end(tmp_path: Path) -> None:
    """The fvsolver smoke task, solved by FEniCSx: VCell's name for the job, VCell's domain name, every
    output time, and the task's identity in the manifest."""

    assert main(["--simtask", str(_SMOKE), "--out", str(tmp_path)]) == 0
    bundle = Bundle.open(tmp_path / "SimID_1585623750_0_.fenics")
    assert bundle.status == "completed"
    assert bundle.times == pytest.approx((0.0, 0.005, 0.01))
    assert set(bundle.manifest.domains) == {"subdomain1"}
    assert {v.name for v in bundle.manifest.variables} == {"RanC_cyt", "Ran_cyt", "C_cyt", "RanC_nuc"}
    source = bundle.manifest.source
    assert (source.kind, source.sim_key, source.job_index, source.task_id) == ("simtask", "1585623750", 0, 0)
    # RanC → Ran + C: the two products appear in equal amounts at every output time.
    ran, c = bundle.stats("subdomain1", "Ran_cyt")[:, 1], bundle.stats("subdomain1", "C_cyt")[:, 1]
    assert ran == pytest.approx(c, rel=1e-9)
    assert ran[-1] > 0.0


def test_the_bundle_lands_next_to_the_task_by_default(tmp_path: Path) -> None:
    task = tmp_path / _SMOKE.name
    task.write_bytes(_SMOKE.read_bytes())
    assert main(["--simtask", str(task), "--no-fields"]) == 0
    assert (tmp_path / "SimID_1585623750_0_.fenics" / ".zattrs").is_file()


@pytest.mark.parametrize(
    "fixture,message",
    [
        ("SimID_274631114_0__0.simtask.xml", "non-spatial"),
        ("SimID_274672135_0__0.simtask.xml", "particle"),  # Langevin: the bridge refuses particle math
    ],
)
def test_unsupported_tasks_exit_2_with_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fixture: str, message: str
) -> None:
    assert main(["--simtask", str(_FIXTURES / fixture), "--out", str(tmp_path)]) == 2
    assert message in capsys.readouterr().err


# -- moving boundaries: the front velocity and what the moving path accepts ---------------------------------

_MB = _FIXTURES / "SimID_274641196_0__0.simtask.xml"


def _mb_variant(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    """The real moving-boundary fixture with textual edits (each must apply exactly once)."""

    text = _MB.read_text()
    for old, new in replacements:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path = tmp_path / _MB.name
    path.write_text(text)
    return path


def test_the_front_velocity_is_read_from_the_membrane() -> None:
    # pyvcell drops MembraneSubDomain/<Velocity>; the adapter reads it from the raw XML
    from vcell_fenics.pyvcell_bridge.simtask import front_velocity_dependence

    task = read_simtask(_MB)
    assert task.moving_boundary
    (front,) = task.front_velocities
    assert front.surface_name == "cell_ec_membrane"
    assert (front.velocity_x, front.velocity_y) == ("sobj_cell1_ec0_velX", "sobj_cell1_ec0_velY")
    # sobj_cell1_ec0_velX -> sproc_0.velocityX -> sin(t): time only
    assert front_velocity_dependence(task) == "prescribed"


def test_the_front_moves_the_interior_compartment() -> None:
    # threaded to the importer, the front becomes a prescribed motion of the volume it encloses (v = v_b)
    from vcell_fenics.formalism.schema import MotionPrescribedVelocity, TemplateEquation
    from vcell_fenics.pyvcell_bridge import import_math_description

    task = read_simtask(_MB)
    math = import_math_description(task.math, geometry="g", dim=2, front_velocity=task.front_velocities[0])
    motion = {s.name: s.motion for s in math.subdomains}
    assert motion["cell"] == MotionPrescribedVelocity(velocity="[((sin(sim.t))), ((cos(sim.t)))]")
    assert motion["ec"].kind == "none"
    # the front moves the frame only: the species keep VCell's lab-frame velocity (none), so each takes
    # an explicit zero lab-frame advection rather than riding with the cell
    for equation in math.equations:
        assert isinstance(equation, TemplateEquation)
        assert equation.terms.get("advection") == "[0.0, 0.0]" and "relative_advection" not in equation.terms


def test_a_species_dependent_front_is_species_coupled(tmp_path: Path) -> None:
    from vcell_fenics.pyvcell_bridge.simtask import front_velocity_dependence

    task = read_simtask(
        _mb_variant(
            tmp_path,
            (
                '<Function Name="sproc_0.velocityX" Domain="cell_ec_membrane">sin(t)</Function>',
                '<Function Name="sproc_0.velocityX" Domain="cell_ec_membrane">(0.1 * C_cyt)</Function>',
            ),
        )
    )
    assert front_velocity_dependence(task) == "species-coupled"
    assert any("previous step" in w for w in check_supported(task))


def test_a_supported_moving_task_is_accepted() -> None:
    # a prescribed front: nothing to warn about beyond the solver name
    warnings = check_supported(read_simtask(_MB))
    assert not any("front velocity" in w for w in warnings), warnings


def test_a_moving_task_without_a_front_velocity_is_refused(tmp_path: Path) -> None:
    task = read_simtask(
        _mb_variant(
            tmp_path,
            (
                """      <Velocity>
        <X>sobj_cell1_ec0_velX</X>
        <Y>sobj_cell1_ec0_velY</Y>
      </Velocity>
""",
                "",
            ),
        )
    )
    assert task.moving_boundary and not task.front_velocities
    with pytest.raises(SimulationTaskError, match="has no membrane <Velocity>"):
        check_supported(task)


def test_a_moving_task_on_an_image_geometry_is_refused(tmp_path: Path) -> None:
    # the moving-boundary path is analytic-geometry only for now; an image geometry (here the fixture's
    # disk and ec as a 4×4 segmented image) is refused, not realized and then solved on its initial shape
    import zlib

    pixels = bytes([1] * 5 + [2, 2] + [1, 1] + [2, 2] + [1] * 5)
    data = zlib.compress(pixels)
    image = (
        f'<Image Name="seg"><ImageData X="4" Y="4" Z="1" CompressedSize="{len(data)}">{data.hex().upper()}</ImageData>'
        '<PixelClass Name="ec" ImagePixelValue="1" /><PixelClass Name="cell" ImagePixelValue="2" /></Image>\n'
    )
    task = read_simtask(
        _mb_variant(
            tmp_path,
            (
                """    <SubVolume Name="cell" Handle="1" Type="Analytical" KeyValue="109369390">
      <AnalyticExpression>((((x - 5.0) ^ 2.0) + ((y - 5.0) ^ 2.0)) &lt; (3.0 ^ 2.0))</AnalyticExpression>
    </SubVolume>
    <SubVolume Name="ec" Handle="0" Type="Analytical" KeyValue="109369391">
      <AnalyticExpression>1.0</AnalyticExpression>
    </SubVolume>
""",
                image
                + '    <SubVolume Name="cell" Handle="1" Type="Image" ImagePixelValue="2" />\n'
                + '    <SubVolume Name="ec" Handle="0" Type="Image" ImagePixelValue="1" />\n',
            ),
        )
    )
    with pytest.raises(SimulationTaskError, match="on an image-based geometry"):
        check_supported(task)


def test_species_on_the_moving_membrane_are_refused(tmp_path: Path) -> None:
    task = read_simtask(
        _mb_variant(
            tmp_path,
            (
                '    <Function Name="s2" Domain="cell_ec_membrane">s2_init_molecules_um_2</Function>\n',
                '    <MembraneVariable Name="s2" Domain="cell_ec_membrane" />\n',
            ),
            (
                '    <MembraneSubDomain Name="cell_ec_membrane" InsideCompartment="cell" OutsideCompartment="ec">\n',
                '    <MembraneSubDomain Name="cell_ec_membrane" InsideCompartment="cell" OutsideCompartment="ec">\n'
                '      <OdeEquation Name="s2" SolutionType="Unknown">'
                "<Rate>0.0</Rate><Initial>0.0</Initial></OdeEquation>\n",
            ),
        )
    )
    with pytest.raises(SimulationTaskError, match="species on the moving membrane"):
        check_supported(task)


def test_value_identifiers_do_not_match_inside_call_names() -> None:
    # regression: the identifier regex backtracked into call names (`sin(t)` yielded `si`)
    from vcell_fenics.pyvcell_bridge.inlining import referenced_names

    assert referenced_names("sin(t)") == {"t"}
    assert referenced_names("s * sin(s) + exp(x)") == {"s", "x"}
    assert referenced_names("sproc_0.velocityX + a.b") == {"sproc_0.velocityX", "a.b"}

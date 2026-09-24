"""Read VCell's **SimulationTask** document — the one input VCell hands every solver ([ADR 011] §1).

VCell writes ``SimID_<simKey>_<jobIndex>__<taskId>.simtask.xml`` into the user's data directory
(``XmlHelper.simTaskToXML``)::

    <SimulationTask TaskId JobIndex isPowerUser>
      [<ComputeResource/>] <FieldFunctionIdentifierSpec/>*
      <MathDescription/>                     the lowered math — parsed by pyvcell's own visitor
      <Simulation Name>                      <SolverTaskDescription>, <MathOverrides>, <MeshSpecification>
      <Geometry Dimension/>                  parsed by pyvcell's own visitor
    </SimulationTask>

pyvcell reads only ``<BioModel>`` documents, but its ``BiomodelVisitor.visit_MathDescription`` and
``visit_Geometry`` work on the bare elements against a stub ``Application``; this adapter reuses them
and parses the ``<Simulation>`` itself, because pyvcell's ``visit_Simulation`` drops what a solver needs
(``StartTime``, ``TimeStep``, ``ErrorTolerance``, the non-uniform ``OutputOptions``, ``MathOverrides``).
The job's parameter-scan point is applied to the math here (:mod:`.overrides`), so :attr:`SimulationTask.math`
is the math *this job* solves.

Upstreaming this reader into pyvcell is a planned later step.

[ADR 011]: ../../../docs/decisions/011-vcell-solver-contract.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vcell_fenics.pyvcell_bridge.overrides import (
    MathOverrides,
    apply_overrides,
    parse_math_overrides,
    resolve_overrides,
)

# The SolverTaskDescription@Solver value VCell's Java side will use for this solver (ADR 011 §6).
FENICSX_SOLVER_NAME = "FEniCSx"
_FILENAME = re.compile(r"^(SimID_\d+_\d+_)_\d+\.simtask\.xml$")
_NAMESPACE = "{http://sourceforge.net/projects/vcell/vcml}"


class SimulationTaskError(ValueError):
    """The document is not a SimulationTask this solver can run (a user/model error, exit 2)."""


@dataclass(frozen=True)
class OutputSpec:
    """VCell's ``OutputOptions``: uniform (``OutputTimeStep``), explicit (``OutputTimes``), or the
    solver-step based default (``KeepEvery`` / ``KeepAtMost``)."""

    kind: Literal["uniform", "explicit", "default"]
    step: float | None = None
    times: tuple[float, ...] = ()
    keep_every: int | None = None
    keep_at_most: int | None = None


@dataclass(frozen=True)
class FenicsxOptions:
    """``<FEniCSxSolverOptions ElementDegree MaxElementSize TimeIntegration/>`` — the solver-specific block
    the VCell Java side adds (ADR 011 §2). Absent fields fall back to the generic task description."""

    element_degree: int | None = None
    max_element_size: float | None = None
    time_integration: str | None = None  # "backward_euler" | "method_of_lines"
    unknown: tuple[str, ...] = ()  # attributes/children this reader does not know (reported, not fatal)


@dataclass(frozen=True)
class SimulationTask:
    path: Path
    task_id: int
    job_index: int
    sim_name: str
    sim_key: str | None
    solver: str
    task_type: str
    start_time: float
    end_time: float
    dt_default: float | None
    dt_min: float | None
    dt_max: float | None
    abs_tol: float | None
    rel_tol: float | None
    output: OutputSpec
    num_processors: int
    mesh_size: tuple[int, int, int] | None
    fenicsx: FenicsxOptions | None
    overrides: MathOverrides
    resolved_overrides: dict[str, str]
    field_data: tuple[str, ...]
    moving_boundary: bool
    math: Any  # pyvcell MathDescription, this job's overrides applied
    geometry: Any  # pyvcell Geometry
    warnings: tuple[str, ...] = field(default=())
    # A moving-boundary task's front kinematics, one per membrane that carries a <Velocity> (pyvcell
    # FrontVelocity, the raw VCell expressions; the importer inlines them). Empty for a fixed geometry.
    front_velocities: tuple[Any, ...] = ()

    @property
    def job_prefix(self) -> str:
        """``SimID_<simKey>_<jobIndex>_`` — VCell's name for this job's files
        (``SimulationJob.createSimulationJobID``)."""

        if self.sim_key is not None:
            return f"SimID_{self.sim_key}_{self.job_index}_"
        match = _FILENAME.match(self.path.name)
        if match is None:
            raise SimulationTaskError(
                f"{self.path.name}: no Simulation/Version@KeyValue and the file name is not "
                "SimID_<key>_<job>__<task>.simtask.xml — pass --output-prefix"
            )
        return match.group(1)

    def output_times(self) -> tuple[float, ...]:
        """The rows to write, starting with the initial state at 0.

        - uniform: ``0, Δ, …`` to ``EndTime`` (the last interval shortened to land on it);
        - explicit: ``0`` plus the listed times — the run *ends* at the last one, since VCell keeps no
          data after it;
        - default (``KeepEvery N`` steps of ``TimeStep``): uniform at ``N·dt``, thinned to at most
          ``KeepAtMost`` outputs — an approximation for adaptive integrators, which have no fixed step.
        """

        end = self.end_time
        spec = self.output
        if spec.kind == "explicit":
            times = sorted({t for t in spec.times if t > 0.0})
            if not times:
                raise SimulationTaskError("explicit OutputTimes lists no time after 0")
            if times[-1] > end * (1 + 1e-12):
                raise SimulationTaskError(f"explicit output time {times[-1]} is beyond EndTime {end}")
            return (0.0, *times)
        if spec.kind == "uniform":
            assert spec.step is not None
            step = spec.step
        else:
            if self.dt_default is None or spec.keep_every is None:
                raise SimulationTaskError("KeepEvery output needs a TimeStep DefaultTime")
            step = spec.keep_every * self.dt_default
            if spec.keep_at_most is not None and spec.keep_at_most > 0 and end / step > spec.keep_at_most:
                step = end / spec.keep_at_most
        if step <= 0.0:
            raise SimulationTaskError(f"output interval must be positive, got {step}")
        n = max(1, round(end / step))
        return (*(end * k / n for k in range(n)), end)


def read_simtask(path: str | Path) -> SimulationTask:
    """Parse a SimulationTask XML file (see the module docstring). Raises :class:`SimulationTaskError`
    for a document that is not a SimulationTask; solver-capability checks are :func:`check_supported`."""

    from lxml import etree

    source = Path(path)
    try:
        root = etree.parse(str(source)).getroot()
    except (OSError, etree.XMLSyntaxError) as error:
        raise SimulationTaskError(f"{source}: cannot parse as XML: {error}") from error
    if _tag(root) != "SimulationTask":
        raise SimulationTaskError(f"{source}: root element is <{_tag(root)}>, not <SimulationTask>")

    children = {_tag(child): child for child in root}
    for required in ("MathDescription", "Simulation", "Geometry"):
        if required not in children:
            raise SimulationTaskError(f"{source}: <SimulationTask> has no <{required}>")
    math, geometry = _visit_math_and_geometry(children["MathDescription"], children["Geometry"])
    field_data = tuple((child.text or "").strip() for child in root if _tag(child) == "FieldFunctionIdentifierSpec")

    sim = children["Simulation"]
    task = _child(sim, "SolverTaskDescription")
    if task is None:
        raise SimulationTaskError(f"{source}: <Simulation> has no <SolverTaskDescription>")
    time_bound, time_step, tolerance = (
        _child(task, "TimeBound"),
        _child(task, "TimeStep"),
        _child(task, "ErrorTolerance"),
    )
    if time_bound is None:
        raise SimulationTaskError(f"{source}: <SolverTaskDescription> has no <TimeBound>")
    version = _child(sim, "Version")
    mesh = _child(sim, "MeshSpecification")
    size = _child(mesh, "Size") if mesh is not None else None
    processors = _child(task, "NumberProcessors")

    overrides_element = _child(sim, "MathOverrides")
    overrides = parse_math_overrides(
        (constant.get("Name", ""), constant.get("ConstantArraySpec"), constant.text or "")
        for constant in (overrides_element if overrides_element is not None else ())
        if _tag(constant) == "Constant"
    )
    job_index = int(root.get("JobIndex", "0"))
    resolved = resolve_overrides(overrides, job_index)

    return SimulationTask(
        path=source,
        task_id=int(root.get("TaskId", "0")),
        job_index=job_index,
        sim_name=sim.get("Name", "unnamed"),
        sim_key=version.get("KeyValue") if version is not None else None,
        solver=task.get("Solver", ""),
        task_type=task.get("TaskType", "Unsteady"),
        start_time=float(time_bound.get("StartTime", "0")),
        end_time=float(time_bound.get("EndTime", "0")),
        dt_default=_float(time_step, "DefaultTime"),
        dt_min=_float(time_step, "MinTime"),
        dt_max=_float(time_step, "MaxTime"),
        abs_tol=_float(tolerance, "Absolut"),
        rel_tol=_float(tolerance, "Relative"),
        output=_output_spec(_child(task, "OutputOptions"), source),
        num_processors=int((processors.text or "1").strip()) if processors is not None else 1,
        mesh_size=(int(size.get("X", "1")), int(size.get("Y", "1")), int(size.get("Z", "1")))
        if size is not None
        else None,
        fenicsx=_fenicsx_options(_child(task, "FEniCSxSolverOptions")),
        overrides=overrides,
        resolved_overrides=resolved,
        field_data=field_data,
        moving_boundary=task.get("Solver") == "MovingB" or _child(task, "MovingBoundarySolverOptions") is not None,
        math=apply_overrides(math, resolved),
        geometry=geometry,
        front_velocities=_front_velocities(children["MathDescription"]),
    )


def _front_velocities(math_element: Any) -> tuple[Any, ...]:
    """The ``<MembraneSubDomain><Velocity><X>…</X><Y>…</Y>[<Z>…</Z>]</Velocity>`` front kinematics of a
    moving-boundary math description. pyvcell's reader drops this element (it models ``<Velocity>`` only
    as a PDE's advection velocity, as attributes), so it is read from the raw XML here."""

    from pyvcell.vcml.models_app import FrontVelocity

    fronts = []
    for membrane in math_element:
        if _tag(membrane) != "MembraneSubDomain":
            continue
        velocity = _child(membrane, "Velocity")
        if velocity is None:
            continue
        components = {
            axis: (_child(velocity, axis).text or "0").strip() if _child(velocity, axis) is not None else "0.0"
            for axis in ("X", "Y", "Z")
        }
        fronts.append(
            FrontVelocity(
                velocity_x=components["X"],
                velocity_y=components["Y"],
                velocity_z=components["Z"],
                surface_name=membrane.get("Name"),
            )
        )
    return tuple(fronts)


VelocityDependence = Literal["prescribed", "species-coupled"]


def front_velocity_dependence(task: SimulationTask) -> VelocityDependence:
    """Whether the front velocity depends only on space, time and constants (``prescribed``) or also on
    the solved species (``species-coupled``: the backend evaluates it with the species one step behind)."""

    from vcell_fenics.pyvcell_bridge.importer import _collect_variable_names
    from vcell_fenics.pyvcell_bridge.inlining import referenced_names, resolve_functions

    variables = _collect_variable_names(task.math)
    resolution = resolve_functions(list(task.math.functions), variables)
    for front in task.front_velocities:
        for component in (front.velocity_x, front.velocity_y, front.velocity_z):
            if referenced_names(resolution.inline(str(component)) or "") & variables:
                return "species-coupled"
    return "prescribed"


def check_supported(task: SimulationTask) -> list[str]:
    """Refuse what this solver would otherwise mis-solve (ADR 011 §1). Returns non-fatal warnings."""

    name = task.path.name
    warnings: list[str] = []
    if task.moving_boundary or task.front_velocities:
        warnings.extend(_check_moving_boundary(task))
    if task.field_data:
        raise SimulationTaskError(f"{name}: field data (FieldFunctionIdentifierSpec) is not supported yet")
    if task.task_type.lower() != "unsteady":
        raise SimulationTaskError(f"{name}: TaskType={task.task_type!r}; only time-dependent (Unsteady) tasks run")
    if task.start_time != 0.0:
        raise SimulationTaskError(f"{name}: StartTime={task.start_time}; only simulations starting at 0 are supported")
    if task.end_time <= 0.0:
        raise SimulationTaskError(f"{name}: EndTime={task.end_time} must be positive")
    if task.mesh_size is None:
        raise SimulationTaskError(f"{name}: no MeshSpecification — a non-spatial simulation, not a PDE task")
    if task.solver != FENICSX_SOLVER_NAME:
        warnings.append(
            f"the task names solver {task.solver!r}, not {FENICSX_SOLVER_NAME!r}; solving it with FEniCSx anyway"
        )
    if task.fenicsx is not None and task.fenicsx.unknown:
        warnings.append(f"ignoring unknown FEniCSxSolverOptions entries: {', '.join(task.fenicsx.unknown)}")
    return warnings


def _check_moving_boundary(task: SimulationTask) -> list[str]:
    """What the FEniCSx moving-boundary path solves (vcell-fenics tracker "Moving boundaries"): a 2D or
    3D analytic geometry, one moving membrane with a front <Velocity>, and species only in the
    volume it encloses (the interior rides the mesh, ``v = v_b``). Everything else is refused, each with
    its own reason, rather than solved on the initial shape. Returns non-fatal notes."""

    name = task.path.name
    prefix = f"{name}: a moving-boundary simulation (Solver={task.solver!r})"
    if not task.front_velocities:
        raise SimulationTaskError(f"{prefix} has no membrane <Velocity>, so the front's motion is unknown")
    dim = getattr(task.geometry, "dim", None)
    if dim not in (2, 3):
        raise SimulationTaskError(f"{prefix} in {dim}D; the FEniCSx moving-boundary path is 2D or 3D")
    images = [sv.name for sv in getattr(task.geometry, "subvolumes", ()) if sv.subvolume_type.value == "image"]
    if images:
        raise SimulationTaskError(
            f"{prefix} on an image-based geometry (subvolumes {', '.join(images)}); the moving-boundary path "
            "does not yet run on images"
        )
    if len(task.front_velocities) > 1:
        moving = ", ".join(repr(f.surface_name) for f in task.front_velocities)
        raise SimulationTaskError(f"{prefix} moves several membranes ({moving}); one moving front is supported")
    front = task.front_velocities[0]
    membrane = next((m for m in task.math.membrane_subdomains if m.name == front.surface_name), None)
    if membrane is None:
        raise SimulationTaskError(
            f"{prefix}: the <Velocity> names membrane {front.surface_name!r}, which the math lacks"
        )
    if membrane.pde_equations or membrane.ode_equations:
        raise SimulationTaskError(
            f"{prefix} has species on the moving membrane {membrane.name!r}; the FEniCSx moving-boundary path "
            "does not solve membrane species on a moving front yet"
        )
    for compartment in task.math.compartment_subdomains:
        if compartment.name != membrane.inside_compartment and (compartment.pde_equations or compartment.ode_equations):
            raise SimulationTaskError(
                f"{prefix} has species in {compartment.name!r}, outside the moving front; the FEniCSx "
                f"moving-boundary path solves species inside it ({membrane.inside_compartment!r}) only"
            )
    if front_velocity_dependence(task) == "species-coupled":
        return [
            f"the front velocity of {membrane.name!r} depends on the species; it is evaluated with the "
            "species of the previous step (an explicit, first-order coupling)"
        ]
    return []


def _visit_math_and_geometry(math_element: Any, geometry_element: Any) -> tuple[Any, Any]:
    import pyvcell.vcml.models as vc
    import pyvcell.vcml.models_geometry as vcg
    from pyvcell.vcml.vcml_reader import BiomodelVisitor

    app = vc.Application(name="simtask", stochastic=False, geometry=vcg.Geometry(name="_", dim=0))
    visitor = BiomodelVisitor(vc.VCMLDocument())
    visitor.visit_Geometry(geometry_element, app)
    visitor.visit_MathDescription(math_element, app)
    if app.math_description is None:
        raise SimulationTaskError("the <MathDescription> is empty")
    return app.math_description, app.geometry


def _output_spec(element: Any, source: Path) -> OutputSpec:
    if element is None:
        raise SimulationTaskError(f"{source}: <SolverTaskDescription> has no <OutputOptions>")
    if element.get("OutputTimes") is not None:
        times = tuple(float(token) for token in element.get("OutputTimes").split(",") if token.strip())
        return OutputSpec(kind="explicit", times=times)
    if element.get("OutputTimeStep") is not None:
        return OutputSpec(kind="uniform", step=float(element.get("OutputTimeStep")))
    if element.get("KeepEvery") is not None:
        at_most = element.get("KeepAtMost")
        return OutputSpec(
            kind="default", keep_every=int(element.get("KeepEvery")), keep_at_most=int(at_most) if at_most else None
        )
    raise SimulationTaskError(f"{source}: <OutputOptions> has none of OutputTimeStep / OutputTimes / KeepEvery")


def _fenicsx_options(element: Any) -> FenicsxOptions | None:
    if element is None:
        return None
    values: dict[str, str] = {key: value for key, value in element.attrib.items()}
    values |= {_tag(child): (child.text or "").strip() for child in element}
    known = {"ElementDegree", "MaxElementSize", "TimeIntegration"}
    integration = values.get("TimeIntegration")
    return FenicsxOptions(
        element_degree=int(values["ElementDegree"]) if values.get("ElementDegree") else None,
        max_element_size=float(values["MaxElementSize"]) if values.get("MaxElementSize") else None,
        time_integration=_integration_name(integration) if integration else None,
        unknown=tuple(sorted(set(values) - known)),
    )


def _integration_name(value: str) -> str:
    normalized = value.strip().replace("-", "_").lower()
    aliases = {"backwardeuler": "backward_euler", "methodoflines": "method_of_lines", "mol": "method_of_lines"}
    name = aliases.get(normalized.replace("_", ""), normalized)
    if name not in ("backward_euler", "method_of_lines"):
        raise SimulationTaskError(
            f"FEniCSxSolverOptions TimeIntegration={value!r} is not BackwardEuler or MethodOfLines"
        )
    return name


def _child(element: Any, tag: str) -> Any:
    return next((child for child in element if _tag(child) == tag), None) if element is not None else None


def _tag(element: Any) -> str:
    tag = element.tag
    return tag.replace(_NAMESPACE, "") if isinstance(tag, str) else ""


def _float(element: Any, attribute: str) -> float | None:
    if element is None or element.get(attribute) is None:
        return None
    return float(element.get(attribute))


__all__ = [
    "FENICSX_SOLVER_NAME",
    "FenicsxOptions",
    "OutputSpec",
    "SimulationTask",
    "SimulationTaskError",
    "check_supported",
    "read_simtask",
]

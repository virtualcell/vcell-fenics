# ADR 011 — The VCell solver contract: SimulationTask in, results bundle out, VCell status protocol

**Date:** 2026-09-22
**Status:** Accepted for the vcell-fenics side. The VCell Java side (§6) is a follow-up.
**Context doc:** [`docs/integration/vcell-solver-integration.md`](../integration/vcell-solver-integration.md)

## Context

VCell runs every solver the same way:

- It writes a **SimulationTask XML** into the user's data directory:
  `SimID_<simKey>_<jobIndex>__<taskId>.simtask.xml` (`XmlHelper.simTaskToXML`).
  - Root `<SimulationTask TaskId JobIndex>`.
  - `<MathDescription>`.
  - `<Simulation>`, containing `<SolverTaskDescription Solver=…>`, `<MathOverrides>` and
    `<MeshSpecification>`.
  - `<Geometry>`.
  - Optionally `<FieldFunctionIdentifierSpec>`.
- It launches the solver:
  - **HPC:** `HtcSimulationWorker` → `SlurmProxy`, inside a per-solver Apptainer SIF:
    `singularity run --containall … $sif <exe> <args> -tid <taskId>`.
  - **Desktop:** a local process.
- It tracks progress through a **status protocol**: stdout markers locally, REST WorkerEvents on the
  cluster.

vcell-fenics becomes a VCell solver by honouring that contract without VCell-specific shims inside
the numerics. The user's steer: don't overfit VCell's legacy seams. FEniCSx gets its own results
format ([ADR 010](010-results-bundle-vtu-zarr.md)) and its own viewing path.

## Decision

### 1. Input: the SimulationTask document

`vcell-fenics --simtask <file>` is a fourth input kind, alongside `--vcml`, the VCell YAML pair and
the native pair.

- **Parsing** (`pyvcell_bridge/simtask.py`) reuses pyvcell's `BiomodelVisitor`
  (`visit_MathDescription`, `visit_Geometry`, `visit_Simulation`) against a stub `Application`. It
  parses what pyvcell drops itself:
  - `TimeBound@StartTime/EndTime` and `TimeStep@DefaultTime/MinTime/MaxTime`;
  - `ErrorTolerance@Absolut/Relative`;
  - the `OutputOptions` variants (`@OutputTimeStep`, `@KeepEvery/@KeepAtMost`, `@OutputTimes`);
  - `NumberProcessors`;
  - the new FEniCSx options element (§2).

  Upstreaming this reader to pyvcell is a later step.
- **MathOverrides and parameter scans** (`pyvcell_bridge/overrides.py`):
  - Plain overrides are `<Constant Name="k">expr</Constant>`.
  - Scans are `<Constant Name="k" ConstantArraySpec="1000|1001">`, with list, interval or log
    content.
  - Resolution matches Java exactly: scan index = `jobIndex % scanCount`, then
    `MathOverrides.scanIndexToScanParameterCoordinate` over the **sorted** scanned names, with the
    first name slowest.
- **Identity:** the sim key is `Simulation/Version@KeyValue` and the job index is
  `SimulationTask@JobIndex`. The bundle name is `SimID_<key>_<job>_.fenics`, unless
  `--output-prefix` overrides it. The filename is the fallback source of these values.
- **Rejected with exit 2** (never silently mis-solved):
  - `Solver="MovingB"` or `<MovingBoundarySolverOptions>`, which would otherwise import as a fixed
    geometry;
  - field data (`FieldFunctionIdentifierSpec`);
  - `TaskType="Steady"`;
  - `StartTime ≠ 0`;
  - a missing `MeshSpecification`;
  - any math the bridge cannot represent.

  *Amended — see [Amendments](#amendments) below: moving-boundary tasks in a defined scope have been
  accepted since 2026-09-23, and FastSystem is refused since 2026-09-27.*
- **Solver name:** if `Solver` isn't the FEniCSx database name, log a warning and run anyway, so
  finite-volume simtasks can be cross-validated.

### 2. Settings precedence

From highest to lowest:
1. an explicit CLI flag;
2. `<FEniCSxSolverOptions ElementDegree MaxElementSize TimeIntegration …/>` inside
   `SolverTaskDescription` (a new element the Java side adds; absent means defaults);
3. the generic `SolverTaskDescription` (TimeStep, ErrorTolerance, OutputOptions);
4. `h` derived from `MeshSpecification/Size`;
5. built-in defaults.

In simtask mode the default integrator is method of lines (adaptive BDF), driven by
`ErrorTolerance`. Real VCell kinetics are routinely nonlinear, which backward Euler rejects
(`NonlinearTermError`). Every value a flag overrides is logged and recorded in the manifest
(`solver.overrides`).

### 3. Output

The [ADR 010](010-results-bundle-vtu-zarr.md) bundle, written to `--out` (in simtask mode the
default is the simtask's own directory). Rank 0 writes; all ranks participate in the gathers.

### 4. Status protocol

The solver reports through a `StatusReporter` with these events:

| event | stdout marker | REST WorkerEvent |
|---|---|---|
| starting | — | 999, persistent, TTL 600000 |
| phase (entering one) | `[[[progress:<phase>:NN.N%]]]` | 1001 with `WorkerEvent_StatusMsg=WORKEREVENT_PROGRESS\|<phase>` |
| progress | `[[[progress:<phase>:NN.N%]]]` (`[[[progress:NN.N%]]]` before any phase) | 1001, nonpersistent, TTL 60000 (with the phase's status message) |
| data (an output row landed) | `[[[data:<t>]]]` | 1000, nonpersistent, TTL 60000 |
| completed | `[[[progress:100%]]]` | **1003**, persistent, TTL 600000 |
| failed | (stderr message) | 1002, persistent, TTL 600000 |

- **`--vc-print-status`** writes the markers to stdout (rank 0, flushed). Stdout carries **only**
  markers: VCell's `MathExecutable` throws on anything else between `[[[` and `]]]`, and Netgen and
  PETSc print to stdout. So the CLI keeps a duplicate of fd 1 for markers and points fd 1 at stderr.
  - `data:` values are parsed after the last `:`.
  - `progress:` values are parsed between the last `:` and `%`.
  - Newlines are irrelevant.
- **`--vc-send-status-config=FILE`** posts REST WorkerEvents. The config file is in the Langevin
  properties format: `broker_host`, `broker_port`, `broker_username`, `broker_password`,
  `vc_username`, `simKey`, `taskID`, `jobIndex`. The request is
  `POST http://<host>:<port>/api/message/workerEvent?type=queue&JMSPriority=5&JMSTimeToLive=…&JMSDeliveryMode=…&MessageType=WorkerEvent&UserName=…&HostName=…&SimKey=…&TaskID=…&JobIndex=…&WorkerEvent_Status=…[&WorkerEvent_StatusMsg=…]&WorkerEvent_Progress=…&WorkerEvent_TimePoint=…`,
  with HTTP Basic auth. This matches `../vcell/docker/build/batch/entrypoint.sh` and
  `LangevinNoVis01`'s `VCellMessagingRest`.
  - Progress is throttled to one message per 5 s.
  - The status message is truncated to 2048 characters.
  - Messaging failures are logged once and **never abort the solve**.
- **Why the solver sends 1003 itself:** `SimulationStateMachine` completes a job only on 1003, and
  VCell's postprocessor sends only worker-exit.
- **COMPLETED** is sent only after the bundle is finalized (manifest `status: completed`).
- **Phases.** A run reports which phase it is in, in this order: `loading model`, `meshing`,
  `compiling` (JIT-compiled kernels and assembly, up to the first time step), `solving` (progress is the
  fraction of simulated time), `writing results` (at 100%). A phase is entered at once — never throttled
  — and every later progress report carries it, so VCell can show "meshing" or "solving 37%" instead of a
  bare "0%" during the minutes before the first step. The extension is compatible both ways:
  - an older VCell parses `[[[progress:<phase>:NN.N%]]]` as plain progress (the number is still between
    the last `:` and the `%`, so a phase never contains `:`, `%`, `[` or `]`), and an older broker
    already turns `WORKEREVENT_PROGRESS|<phase>` into the job's status message
    (`SimulationMessage.fromSerializedMessage` in `WorkerEventMessage`), which its client ignores while a
    progress bar is shown;
  - a VCell that shows phases falls back to the bare percentage when a solver sends none.
  - Completion stays the plain `[[[progress:100.0%]]]`.
- **`-tid N`** is accepted, since `HtcSimulationWorker` appends it; the CLI warns if it disagrees
  with `TaskId`.
- **Exit codes:**
  - `0`: success.
  - `2`: a model or user error; also sends FAILURE.
  - `1`: a crash; also sends FAILURE, and `comm.Abort(1)` under MPI.
  - `143`: SIGTERM (Slurm cancel/timeout); the manifest is marked failed.

### 5. Container contract

- **Image:** `ghcr.io/virtualcell/vcell-fenics:<version>`, multi-arch (linux/amd64, linux/arm64).
- **SIF:** `oras://ghcr.io/virtualcell/vcell-fenics_singularity:<version>`, which follows
  `../vcell/docs/apptainer-image-build.md` (pre-pulled to the cluster's shared image directory).
- The entrypoint runs `vcell-fenics …` (or any command, e.g. `mpiexec -n N vcell-fenics …`)
  inside the pixi environment.
- The image runs as any uid. Under Apptainer the image is read-only, so the entrypoint copies the
  pre-warmed FFCx cache into a writable per-uid cache (`$VCELL_FENICS_CACHE`, or one under
  `$TMPDIR`).
- **Paths:**
  - HPC: VCell binds the data directories and passes container-side paths.
  - Local Docker: bind the user's data directory at the **same path** inside the container, so the
    simtask's own paths and `--out` need no translation.

## 6. Follow-up — VCell Java side (`../vcell`)

Not implemented by this ADR; listed so the contract above has a known consumer.

- **Solver definition:**
  - `SolverDescription.FEniCSx` (database name `FEniCSx`; features Spatial and Deterministic;
    uniform and explicit output times; analytic geometries only for now; no moving boundary,
    stochastic, fast system, field data or periodic boundaries) — *as implemented, `Feature_Moving`
    and image geometries were added later (vcell #2093, #2094; Amendments below)*;
  - a `SolverFactory` maker;
  - a `FenicsSolver` class: `initialize()` writes `SimID_*_.fenicsMessagingConfig`, and the argv is
    `vcell-fenics --simtask <file> --out <dir> [--vc-send-status-config=…|--vc-print-status]`.
- **Options:** a `FenicsSolverOptions` on `SolverTaskDescription` → `XMLTags`, `Xmlproducer`,
  `XmlReader`, and VCML `getVCML`/`readVCML` (the database stores the task description as VCML).
- **HPC:**
  - `SlurmProxy` gets `vcell.htc.vcellfenics.apptainer.image` and `.solver.list`, plus `ntasks`
    for MPI;
  - `HtcSimulationWorker` property wiring;
  - `vcell-fluxcd` gets `submit.env` and SIF prepull-job entries.
- **Desktop:**
  - Docker detection;
  - `docker run --rm -v <userdir>:<userdir> --user <uid>:<gid> <image> vcell-fenics --simtask … --vc-print-status`
    from the quick-run path;
  - `SimulationListPanel.canQuickRun` allows FEniCSx when Docker is present;
  - UX for the first-time image pull.
- **UI:**
  - a self-hiding FEniCSx options panel in `SolverTaskDescriptionAdvancedPanel`;
  - a FEM mesh-size panel in `MeshTabPanel`;
  - estimates in `SimulationWorkspace.checkSimulationParameters`.
- **Viewing:**
  - a FEniCSx data source in `FieldViewerServer` that reads the bundle (fixed profile → `STATIC`,
    segmented → `TIME_VARYING`);
  - **point-data** support in `/info`, `/field`, `/timeseries` and `/stats`, and in `webapp-viewer`;
  - `VtuGridParser`: `VTK_LINE`, 3D triangle area, and rejecting compressed data rather than
    misreading it;
  - FEniCSx results open the field viewer directly (the Swing `PDEDataViewer` is Cartesian-only),
    which requires viewer packaging (vcell #1851);
  - remote byte serving of bundle files;
  - sim-data cleanup must remove the `.fenics/` directory.
- **pyvcell:** a `FenicsResult` bundle reader; upstream the SimulationTask reader.

## Consequences

- The simtask is the one input VCell needs to produce. No Java-side solver input writer
  (`*FileWriter`) exists or is required for FEniCSx, which makes it the first VCell solver to
  consume the simtask XML directly.
- Standalone use (cross-validation, pyvcell, CI) runs the same code path as VCell. A finite-volume
  simtask can be solved by FEniCSx for comparison.
- The CLI owns stdout hygiene and messaging robustness. Both are tested by porting VCell's parsers
  and a captive HTTP server (`tests/test_status.py`).

## Amendments

The contract above is the 2026-09-22 decision. Later work changed the refusal list in §1 without
changing the principle (refuse what would be mis-solved); the integration tracker's status tables and
progress log are the record.

- **2026-09-23 — moving-boundary tasks accepted in a defined scope** (tracker M1–M5; VCell side
  vcell #2093, `Feature_Moving`). `Solver="MovingB"` / `<MovingBoundarySolverOptions>` tasks run when
  the math has one moving front with a membrane `<Velocity>`, species only inside the moving
  compartment, and no membrane species; the velocity may depend on the species (one-step lag). Each
  of those conditions is a specific exit-2 refusal in `simtask._check_moving_boundary`. Boundary
  conditions on the moving compartment are not refused up front; they fail at the first remesh.
- **2026-09-24 — 3D moving boundaries** (tracker 3M1–3M3; vcell `fenicsx/mb-3d`): the 2D-only
  refusal is lifted for analytic geometries; image subvolumes on a moving front stay refused.
- **2026-09-23 — image geometries accepted** (tracker I1–I6, ADR 012; vcell #2094 lifts the Java-side
  image refusal, keeping it for moving-boundary applications).
- **2026-09-27 — FastSystem refused** (progress entry of that date): pyvcell's reader drops the
  `FastSystem` element, so the simtask and VCML loaders read it from the XML and refuse with exit 2
  rather than solving the unbuffered species wrongly.
- **2026-09-28 — multi-compartment routing**: every fixed model whose equations span two or more
  subdomains runs through the multi-compartment solver (membrane species, region variables, box-face
  values per compartment); region variables on the single-mesh and moving paths are refused.
- **2026-10-01 — run phases** in the status protocol (§4, edited in place that day).

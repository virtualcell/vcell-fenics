# vcell-fenics as a VCell solver — design and progress

**Started:** 2026-09-22 · **Workflow:** one small, fully-gated PR per step, stacked in order
(`container-runner` → `vcell-solver-design` → `realize-mpi` → `mol-output-times` → `results-bundle` → …);
work continues on the local `vcell-solver-integration` branch.
**Status:** in progress — see [Progress](#progress). This is a living document: update the status
table and the progress log as steps land.

Related records: [ADR 010 — results bundle](../decisions/010-results-bundle-vtu-zarr.md),
[ADR 011 — VCell solver contract](../decisions/011-vcell-solver-contract.md),
[`docker/README.md`](../../docker/README.md) (container reference).

## Goal

Run vcell-fenics as a first-class VCell solver:

- **HPC** — under Apptainer/Singularity on the Slurm cluster, launched by VCell's
  `HtcSimulationWorker` → `SlurmProxy` like every other solver image.
- **Desktop** — in Docker when Docker is available, launched by the VCell client's local run.
- **Input** — the **SimulationTask XML** VCell hands every solver
  (`SimID_<simKey>_<jobIndex>__<taskId>.simtask.xml`: MathDescription + Simulation/SolverTaskDescription
  + Geometry).
- **Output** — a results bundle (VTK unstructured-grid meshes + zarr field data) that the new
  **3D field viewer** (`../vcell/webapp-viewer` + `FieldViewerServer`) and pyvcell can read, including
  while the run is still in progress.
- **Status** — progress/data/completion reported the way VCell expects, locally and on the cluster.

The design deliberately does **not** squeeze FEniCSx through VCell's legacy seams (the Cartesian
`.log`/`.mesh`/`.zip` dataset, the Swing `PDEDataViewer`, COMSOL-style `VtuVarInfo` RPCs). FEniCSx gets
its own self-describing output and a dedicated viewing path; VCell reuses only what fits naturally
(the SimulationTask input, the Slurm/Apptainer job plumbing, solver registration, the field viewer).

## Discovery summary (2026-09-22)

What exists today, and what shaped the design:

| Area | Finding | Where |
|---|---|---|
| Solver input | `SimulationTask` root: `TaskId`, `JobIndex`; children `MathDescription`, `Simulation` (`SolverTaskDescription`, `MathOverrides`, `MeshSpecification`), `Geometry`, optional `FieldFunctionIdentifierSpec`. Sim key = `Simulation/Version@KeyValue`. | `../vcell/vcell-core/.../xml/XmlHelper.java` `simTaskToXML`/`XMLToSimTask` |
| Python parsing | **Nothing reads `<SimulationTask>`.** pyvcell's `VcmlReader` handles `<BioModel>` only and drops StartTime, TimeStep, non-uniform OutputOptions, MathOverrides; its `visit_MathDescription/visit_Geometry/visit_Simulation` are reusable against a stub `Application`. | `../pyvcell/pyvcell/vcml/vcml_reader.py` |
| Parameter scans | Scan index = `jobIndex % scanCount`; odometer over **sorted** scanned-constant names, first name slowest. "Immortalized by stored datasets" — must match exactly. | `MathOverrides.java` `scanIndexToScanParameterCoordinate` |
| Local status | `MathExecutable` scans the solver's **stdout** for `[[[data:<t>]]]` / `[[[progress:NN%]]]`; anything else inside `[[[…]]]` throws. | `AbstractCompiledSolver.java`, `LangevinSolver.getApplicationMessage` |
| HPC status | Solvers post REST WorkerEvents (STARTING 999, DATA 1000, PROGRESS 1001, FAILURE 1002, COMPLETED 1003). A job only completes on **1003**; the postprocessor sends worker-exit, not 1003. Langevin's `--vc-send-status-config` is the Python-friendly precedent. | `WorkerEvent.java`, `SimulationStateMachine.java`, `LangevinSolver.java`, `../vcell/docker/build/batch/entrypoint.sh` |
| HPC images | One SIF per solver image, `oras://ghcr.io/virtualcell/<img>_singularity:<tag>`, pre-pulled to the cluster; `SlurmProxy` picks the image per solver from `vcell.htc.*.apptainer.image` + `.solver.list`. | `../vcell/docs/apptainer-image-build.md`, `SlurmProxy.generateScript` |
| Local execution | No Docker/Apptainer use from Java today; quick run requires a native executable in `localsolvers/`. Docker launch is new code. | `ClientSimManager.createQuickRunSolver`, `SimulationListPanel.canQuickRun` |
| Viewer | `FieldViewerServer` (vtk.wasm) already serves solver-produced body-fitted meshes through a "VTU seam" (Chombo static, MovingBoundary per-timestep) — **cell data only** today. `VtuGridParser`: one piece, little-endian, inline binary-uncompressed or ascii; no `VTK_LINE`; triangle area ignores z. | `../vcell/vcell-client/.../viz/FieldViewerServer.java`, `VtuGridParser.java`, `../vcell/webapp-viewer/README.md` |
| Viewer design intent | VCell's renderer design already anticipates FEniCSx: body-fitted meshes, topology piecewise-constant between remeshes, thin/stateless serving. | `../vcell/docs/3d-renderer-design.md`, `salad-3d-renderer-design.md` §8.8–8.13 |
| Precedents | COMSOL (retired FEM solver) wrote `.vtu` + point data and derived its variable list from the simtask. pyvcell / `pythonData` store FV results as zarr `(t,c,z,y,x)` with `axes/channels/times` attrs. | `ComsolVtkFileWriter.java`, `../pyvcell/pyvcell/_internal/simdata/zarr_writer.py` |
| vcell-fenics today | `cli.py` takes `.vcml`/YAML, writes XDMF + `summary.json`; MOL and interface-coupled paths record only t=0 and t_final; no progress markers. | `src/vcell_fenics/cli.py` |

## Decisions

| Topic | Decision |
|---|---|
| Results, fixed mesh (the common case) | **VTU mesh per domain, written once + zarr v2 field arrays** `(T, N)` of P1 point data + per-time stats |
| Results, moving / remeshed mesh (the exception) | **Same format + segments**: a remesh starts a new segment with its own VTU; ALE coordinates are a per-segment zarr array. A fixed run is the one-segment case. The schema reserves this now; it is implemented when the ALE/remesh drivers are wired to the CLI |
| ParaView | Via an **export** command (bundle → PVD or XDMF), not natively |
| SimulationTask parsing | A **vcell-fenics adapter** reusing pyvcell's visitors; upstream to pyvcell later |
| Scope of the first execution | Design docs + the **vcell-fenics side**; the VCell Java side is a follow-up (below) |

### Why VTU + zarr (full comparison goes in ADR 010)

| Format | static | ALE | remesh | multi-domain | readers | verdict |
|---|---|---|---|---|---|---|
| XDMF3 + HDF5 | ✓ | format ✓, dolfinx writer ✗ | ✓ (custom writer) | file per domain | ParaView, h5py, jhdf | close; XDMF upstream dormant, two files, HDF5 not HTTP/mid-run friendly |
| VTKHDF | ✓ | ✓ | ✓ (Steps offsets) | ✓ (PDC blocks) | ParaView, VTK 9.6, jhdf | strongest standard; custom writer (dolfinx 0.10's is static, single-array); HDF5 not web-native |
| ADIOS2 / VTX | ✓ | ✓ (`VTXMeshPolicy.update`) | ✗ file per segment | ✗ | ParaView only | no Java/JS readers |
| VTU + PVD | mesh every step | ✓ | ✓ | ✗ | everything | bloated |
| **VTU + zarr** | ✓ | ✓ (coords array) | ✓ (segments) | ✓ | Python; tiny Java/JS readers | **chosen** — simplest common case, chunk-per-(var,time) is HTTP- and mid-run-friendly, pyvcell zarr heritage |
| Exodus II | ✓ | ✓ | ✗ | ✓ (blocks) | ParaView, netCDF-Java | not dolfinx-native |

## Results bundle (schema 1)

```
<out>/SimID_<key>_<job>_.fenics/        zarr v2 group directory
  .zattrs     {"vcell_fenics": manifest}   written by us atomically (temp file + os.replace)
  mesh/<domain>.vtu                        VTK XML UnstructuredGrid: inline binary, uncompressed, UInt32 header, little-endian
  <domain>/<var>                           zarr (T, N) '<f8', chunks (1, N), zlib, fill NaN; columns == VTU point order
  stats/<domain>/<var>                     zarr (T, 4) mean, total, min, max (MPI-reduced FEM integrals)
  provenance/{math,geometry}.yaml, summary.json   (convenience, not contract)
```

Manifest (`.zattrs["vcell_fenics"]`): `schema: 1`, `profile: fixed|segmented`,
`status: running|completed|failed`, `message`, `progress`, `times` (rows written so far —
authoritative), `planned_times`, `segments`,
`domains{name: {kind: volume|membrane, dim, gdim, mesh, n_points, n_cells, cell_type}}`,
`variables[{name, domain, assoc: "point", element: "P1", path, stats}]`, `stats_columns`,
`solver{version, dolfinx, options, overrides, mpi_ranks}`,
`source{kind, file, sim_key, job_index, task_id}`, `updated`.

- **Domain names are VCell's own** CompartmentSubDomain / MembraneSubDomain names.
- **Segments are reserved now.** A fixed run has exactly
  `segments = [{index: 0, t0: 0, count: T, motion: "none", prefix: ""}]`. A segmented run adds
  entries with `prefix: "seg0001/"`, each with its own `<prefix>mesh/<domain>.vtu` and
  `<prefix><domain>/<var>`. `motion: "ale"` means `<prefix><domain>/_coords` exists with shape
  `(T_seg, N, 3)`.
- **Reader rules:** reject an unknown `schema`; ignore unknown keys; take row counts from `times` /
  `count`, never from array shapes.
- **Mid-run safety:** a row is written before its time is appended to `times`, so a reader polling
  during the run never sees a partial row.

## CLI and container contract

```
vcell-fenics --simtask SimID_<key>_<job>__<task>.simtask.xml [--out DIR] [--output-prefix P]
             [--vc-print-status | --vc-send-status-config=FILE] [-tid N]
             [--h --dt --t-final --fe-degree --time-integration]     # explicit overrides, logged
mpiexec -n N vcell-fenics …                                          # inside the container; rank 0 writes
```

- **Settings precedence:** explicit flag > `<FEniCSxSolverOptions ElementDegree MaxElementSize
  TimeIntegration/>` (a new element the Java side will add) > generic SolverTaskDescription
  (TimeStep, ErrorTolerance, OutputOptions) > h derived from MeshSpecification > defaults. In
  simtask mode the default integrator is method of lines, driven by ErrorTolerance.
- **Status:**
  - stdout carries only markers: `[[[progress:NN.N%]]]` / `[[[data:<t>]]]` (rank 0, flushed,
    throttled). Library output (Netgen, PETSc) goes to stderr.
  - REST WorkerEvents 999/1001/1000/1003/1002 follow the Langevin query string and properties-format
    config. Messaging failures never abort the solve. COMPLETED is sent only after the bundle is
    finalized.
- **Exit codes:**
  - 0: success.
  - 2: model or user errors — unsupported features, nonlinear backward Euler, moving-boundary
    simtasks, field data, StartTime ≠ 0, steady tasks. These also send FAILURE.
  - 1: crashes (`comm.Abort` under MPI).
- **Solver name:** if the simtask's `Solver` isn't FEniCSx, warn but run, so finite-volume simtasks
  can be cross-validated.

## Work plan — vcell-fenics side

One commit per step. Status: ☐ not started · ◐ in progress · ☑ done.

| # | Step | Status | Notes / commit |
|---|---|---|---|
| 0 | This tracking document | ☑ | |
| 1 | ADR 010 (results bundle) + ADR 011 (VCell solver contract, incl. Java follow-up) | ☑ | ADR 010 §6 awaits the step-2 spike |
| 2 | Spike: zarr v2 via zarr-python 3, VTU encoding vs `VtuGridParser`, `TS.interpolate` output hooks, MPI point-order keys | ☑ | all 13 checks pass; ADR 010 §6 |
| 3 | MPI-correct `realize()` (confirm the suspected mesh duplication with a test first) + `NonlinearTermError` as a user error | ☑ | two real bugs: crash + lost partition-boundary membrane facets; `tests/test_realize_mpi.py` |
| 4 | Output-time hooks in the MOL and interface-coupled integrators | ☑ | `backend/output_times.py`; steps unperturbed |
| 5 | `results/` package: schema, VTU writer/strict reader, P1 gather, bundle writer, recorder, reader | ☑ | byte-identical VTU at n = 1, 2, 3 |
| 6 | Move run logic into `runner.py`; every input kind writes the bundle (XDMF → export) | ☑ | MOL + interface-coupled now write every output time |
| 7 | SimulationTask adapter + MathOverrides/scans + `--simtask` | ☑ | the fvsolver smoke task solves end to end in ~4 s |
| 8 | Status protocol: stdout markers, REST WorkerEvents, exit codes, SIGTERM | ☐ | |
| 9 | Export for ParaView (PVD / XDMF) | ☐ | |
| 10 | Container (writable FFCx cache under Apptainer) + GitHub Actions: multi-arch image, SIF build, ORAS push | ☐ | CI runs only once pushed |

### Step details

1. **Design docs.**
   - `docs/decisions/010-results-bundle-vtu-zarr.md`: the format comparison, the decision, the
     schema, and a "validated by spike" section.
   - `docs/decisions/011-vcell-solver-contract.md`: simtask in, bundle out, status protocol, exit
     codes, container contract, and the Java follow-up list.
2. **Spike.**
   - `pixi add "zarr>=3.1,<4"` and add `vtk` explicitly (already 9.6.1 via pyvista). Check the lock
     solves on osx-arm64, linux-64 and linux-aarch64.
   - `scripts/spike_results_bundle.py` checks:
     - `.zarray` has a zlib compressor, `<f8`, C order and NaN fill;
     - a stdlib decode (json + zlib + numpy) round-trips;
     - pyvcell's zarr 2.18 (`../pyvcell/.venv`) reads the bundle;
     - a Python port of `VtuGridParser` accepts the VTU;
     - `TS.interpolate` on BDF at output times agrees with stepping to each output time
       (MATCHSTEP) within rtol;
     - `input_global_indices` keys agree between serial and `mpiexec -n 2`.
   - Fallback if zarr 3's v2 writing misbehaves: a ~60-line stdlib v2 writer.
3. **MPI realize.**
   - `backend/realize.py:551,804` run Netgen on every rank and pass the full `cells`/`points` to
     `create_mesh(comm, …)`, which probably gives N copies of the mesh.
   - First `tests/test_realize_mpi.py` (`integration`: serial vs `mpiexec -n 2` cell count and
     area). If duplication is confirmed, mesh on rank 0 only (empty arrays elsewhere) and broadcast
     the material tags.
   - Add `NonlinearTermError` to `cli._USER_ERRORS`. The fvsolver smoke simtask is nonlinear, so
     backward Euler can't run it and MOL is the simtask default.
4. **Output-time hooks.**
   - `backend/reaction_diffusion.py`:
     - `_run_time_stepper(..., monitor=)` via `ts.setMonitor`;
     - a shared `_output_monitor` that interpolates (`ts.interpolate`) into a separate snapshot
       Function;
     - `integrate_discrete_problem(..., output_times=, on_output=, on_progress=)` plus rtol/atol.
   - The same for `integrate_interface_coupled` in `backend/interface_coupled.py`.
   - Stepping to each output time stays only as a fallback, because it restarts BDF every interval.
   - Test: `tests/test_backend_output_times.py`.
5. **Results package** (`src/vcell_fenics/results/`):
   - `schema.py`: a pydantic `Manifest` plus a JSON Schema at `docs/results-bundle.schema.json`.
   - `vtu.py`: `write_vtu` with vtk writer settings Binary / NoCompressor / UInt32 / LE, and
     `read_vtu_strict`, which mirrors `VtuGridParser`.
   - `gather.py`: `P1Layout` maps owned dofs to `input_global_indices` keys, `Gatherv`s them to
     rank 0 and canonicalizes cells.
   - `writer.py`: `BundleWriter`.
   - `recorder.py`: replaces `cli._Recorder`.
   - `reader.py`: `Bundle`, plus a `--require-status` CLI.
   - Tests: `tests/test_results_bundle.py`, and `tests/test_results_mpi.py` (`integration`).
6. **Runner.**
   - Move `run_model` and the path functions out of `cli.py` into `src/vcell_fenics/runner.py`.
   - `RunOptions.output_times` replaces `output_dt`.
   - All input kinds write the bundle; XDMF moves to export.
   - Update `tests/test_cli.py`, `docker/README.md` and `CLAUDE.md`.
7. **Simtask.**
   - `pyvcell_bridge/simtask.py`: `read_simtask()` returns a frozen `SimulationTask`.
   - `pyvcell_bridge/overrides.py`: an exact port of the Java scan odometer, `resolve_overrides`,
     `apply_overrides`.
   - `--simtask` becomes a fourth input kind; output naming is `--output-prefix` >
     `SimID_<KeyValue>_<JobIndex>_` > filename.
   - Fixtures in `tests/fixtures/simtask/`. Tests in `tests/test_simtask.py`, including an
     `integration` end-to-end run of the smoke fixture (times `[0, .005, .01]`, status completed).
8. **Status.**
   - `src/vcell_fenics/status.py`:
     - `StatusReporter`;
     - `StdoutMarkers`, which writes to a saved fd with fd 1 redirected to stderr;
     - `MessagingConfig`;
     - `RestWorkerEvents` (urllib, Basic auth, throttled, never raises);
     - `Fanout`.
   - Tests port VCell's stdout parser and use a threaded `http.server` to capture the POSTs.
9. **Export.** `results/export.py` provides `export_bundle(bundle, out, fmt="pvd"|"xdmf")`, plus a
   `vcell-fenics-export` console script.
10. **Container + CI.**
    - `docker/entrypoint.sh`: when the FFCx cache is read-only (a SIF), copy it into a writable
      cache.
    - `docker/Dockerfile`: pre-warm with the simtask fixture.
    - `.github/workflows/container.yml`:
      - native amd64 + arm64 builds and a manifest merge to `ghcr.io/virtualcell/vcell-fenics`;
      - a Docker smoke test (markers, bundle status, n=2 parity);
      - an `apptainer build` + `--containall` smoke test;
      - `apptainer push oras://ghcr.io/virtualcell/vcell-fenics_singularity:<tag>`.

### Verification

- For every step: `pixi run -e dev check`. For MPI and end-to-end tests:
  `pixi run -e dev test-integration`.
- **End-to-end:**
  `pixi run -e dev python -m vcell_fenics.cli --simtask tests/fixtures/simtask/SimID_1585623750_0__0.simtask.xml --out <tmp> --vc-print-status`.
  - stdout parses as VCell markers;
  - `python -m vcell_fenics.results <tmp>/SimID_1585623750_0_.fenics --require-status completed`
    passes;
  - `mpiexec -n 2` gives the same bundle;
  - the export opens in ParaView/pyvista.
- **Numbers:** cross-check against VCell's FV solution of the same simtask with the
  `cross_validation/` harness.
- **Container:** the CI Docker and SIF smoke jobs. Local Docker builds on the development Mac need
  IPv6 disabled in Docker Desktop.

## Follow-up — VCell Java side (`../vcell`, separate plan)

- **Solver definition:**
  - `SolverDescription.FEniCSx` (database name `FEniCSx`; features Spatial and Deterministic;
    uniform and explicit output only) and a `SolverFactory` maker.
  - A `FenicsSolver` class: `initialize` writes `SimID_*_.fenicsMessagingConfig`; the argv is
    `vcell-fenics --simtask …`.
  - `SolverTaskDescription` options, with `XMLTags`, Xmlproducer/XmlReader, and VCML
    `getVCML`/`readVCML`, which the database stores.
- **HPC:**
  - `SlurmProxy`: image property `vcell.htc.vcellfenics.apptainer.image` plus a solver list,
    `ntasks` for MPI.
  - `HtcSimulationWorker` required-property wiring.
  - `vcell-fluxcd`: `submit.env` and prepull-job entries.
- **Local:**
  - Docker detection.
  - `docker run --rm -v <userdir>:<userdir> --user uid:gid ghcr.io/virtualcell/vcell-fenics:<tag> vcell-fenics --simtask …`
    from `ClientSimManager.createQuickRunSolver`.
  - `SimulationListPanel.canQuickRun` allows FEniCSx when Docker is available.
  - UX for the image pull.
- **UI:**
  - A FEniCSx options panel in `SolverTaskDescriptionAdvancedPanel` that hides itself.
  - A FEM mesh-size panel in `MeshTabPanel`.
  - Size estimates in `SimulationWorkspace.checkSimulationParameters`.
- **Viewing:**
  - A FEniCSx data source in `FieldViewerServer` that reads the bundle: fixed profile = `STATIC`
    mode, segmented = `TIME_VARYING`.
  - **Point-data** support in `/info`, `/field`, `/timeseries` and `/stats`, and in the viewer.
  - `VtuGridParser`: `VTK_LINE`, and 3D triangle area.
  - FEniCSx results open the field viewer directly (the Swing `PDEDataViewer` is Cartesian-only),
    which needs viewer packaging (vcell #1851).
  - Remote byte serving of bundle files through the data server or vcell-rest.
  - Sim-data cleanup must remove the `.fenics/` directory.
- **pyvcell:**
  - A `FenicsResult` reader for the bundle, following `sim_results`' zarr conventions.
  - Later, upstream the SimulationTask reader.

## Risks and open questions

- `TS.interpolate` accuracy with BDF and time-dependent Dirichlet data (spike; stepping fallback).
- zarr-python 3 writing `zarr_format=2`, and the three-platform lock (stdlib-writer fallback).
- `(1, N)` chunks mean T files per variable on NFS; revisit if runs output thousands of times.
- Apptainer `/dev/shm` sizing for MPICH under `--containall` (verified in CI).
- No real simtask exercises the interface-coupled or membrane paths; they're covered through the
  YAML-pair input and writer unit tests.
- Local Docker builds on the development Mac fail on IPv6 egress; CI is the reliable build path.

## Progress

Newest first. One entry per landed step or notable finding.

- **2026-09-22** — Step 7: `--simtask`.
  - **`pyvcell_bridge/simtask.py`** reads `<SimulationTask>` by reusing pyvcell's
    `visit_MathDescription` / `visit_Geometry` on a stub `Application`. It parses the
    `<Simulation>` itself: TimeBound, TimeStep, ErrorTolerance, the uniform / explicit / KeepEvery
    output options, NumberProcessors, and a `FEniCSxSolverOptions` block as attributes or children.
    `check_supported` refuses moving-boundary, field-data, steady, `StartTime ≠ 0` and
    non-spatial tasks.
  - **`pyvcell_bridge/overrides.py`** ports VCell's MathOverrides: list and interval (linear/log)
    `ConstantArraySpec`, and the scan odometer (sorted names, first name slowest,
    `jobIndex % scanCount`).
  - **CLI precedence:** flag > FEniCSx options > task > defaults. Overridden settings are recorded
    as `solver.overrides`. The default integrator is MOL; the bundle is named
    `SimID_<key>_<job>_.fenics` and lands next to the task.
  - **Fixtures:** five real VCell tasks in `tests/fixtures/simtask/`. The fvsolver smoke task solves
    end to end in ~4 s: Ran and C totals are equal at every output, and the task identity is in the
    manifest. The moving-boundary, non-spatial (Runge–Kutta scan) and Langevin (particle) tasks exit 2
    with named reasons.
  - **Fixes:** `VcellImportError` was missing from the CLI's user errors, so `--vcml` inputs had the
    same traceback. `python -m vcell_fenics.results` is now the reader command, avoiding a runpy
    double-import warning.
- **2026-09-22** — Step 6: the run logic moved from `cli.py` into `vcell_fenics.runner`
  (`ModelInput`, `RunOptions`, the single-mesh and interface-coupled paths, coupling detection).
  - **Output:** every input kind writes the bundle `<out>/<prefix>.fenics/` (new `--output-prefix`);
    the XDMF output is gone. Provenance (`math.yaml`, `geometry.yaml`, `summary.json`) is under
    `provenance/`.
  - **Output times:** `RunOptions.output_times` replaces `output_dt`, so explicit schedules are
    possible. Backward Euler snaps dt per output interval.
  - **Method of lines:** now records every output time, where it used to record only t = 0 and
    t_final. The interface-coupled path does too, including the IC: `OutputMonitor` now emits an
    output at `t_start` from TS's step-0 monitor call.
- **2026-09-22** — Step 5: the `results/` package.
  - **Schema:** a manifest dataclass plus a pydantic `TypeAdapter`; it ignores unknown keys and
    refuses a newer schema. The published JSON Schema is `docs/results-bundle.schema.json`, kept
    current by a test.
  - **VTU:** written with the VTK writer and a strict `VtuGridParser` mirror to read it back.
  - **`P1Layout`:** a canonical point and cell order, with cells sorted and positively oriented.
  - **`BundleWriter`:** zarr v2, zlib, `(1, N)` chunks, xarray `_ARRAY_DIMENSIONS`, and an atomic
    manifest published after each row lands.
  - **`BundleRecorder`:** P1 interpolation and MPI-reduced statistics for any number of domains.
  - **`Bundle` reader:** with a `--require-status` CLI.
  - **Finding:** `tabulate_dof_coordinates` differs by an ulp between partitions (it pushes a
    reference point through whichever cell it meets last), so point coordinates come from
    `geometry.x` via the dof→node map. With that change, bundles written at n = 2 and 3 have
    byte-identical VTUs and equal fields.
- **2026-09-22** — Step 4: `backend/output_times.py` adds an `OutputMonitor`, a TS monitor that
  records each output time as it is crossed.
  - It uses `TSInterpolate`, or a plain copy when a step lands exactly on the time, into a separate
    snapshot, so the adaptive step sequence is untouched.
  - `integrate_discrete_problem` and `integrate_interface_coupled` take
    `output_times` / `on_output` / `on_progress`.
  - Tests: values match the analytic ODE at every output; the monitored run's steps and final state
    are bitwise identical to an unmonitored run's; the interface-coupled total mass is conserved at
    every recorded time.
- **2026-09-22** — Step 3: the body-fitted (Netgen) realization was **broken under MPI**, worse than
  suspected.
  - **(1) Crash.** Every rank passed its own full Netgen mesh to `create_mesh`, so under
    `mpiexec -n 2` the partition path crashed (`IndexError` realigning material tags). The earlier
    "`mpirun -n 3` matches serial" check had used the box example, which takes the MPI-safe
    `create_rectangle` path.
  - **(2) Lost membrane facets.** Membrane facets on a partition boundary were dropped: without
    ghost cells they have one local cell and looked exterior (a 3D sphere lost 1 of 120 at n=2).
  - **(3) Rank-local decisions.** The per-face majority vote, `face_regions`, and the
    "membrane empty?" checks guarding the collective `create_submesh` were decided per rank.
  - **Fix.** `_mesh_on_rank0` meshes on rank 0 only and broadcasts material tags (and Netgen errors,
    to avoid deadlock); meshes are ghosted with `GhostMode.shared_facet`, which interior-facet (`dS`)
    membrane terms need in parallel anyway; the votes and emptiness checks are global
    (Allreduce / allgather).
  - **Test.** `tests/test_realize_mpi.py` (`integration`) checks disk, sphere and nested
    interface-coupled (box- and background-bounded) at n = 2 and 3 against serial: identical counts
    and measures. It fails on the old code with the original `IndexError`.
  - `NonlinearTermError` is now a clean exit-2 user error naming the fix (MOL).
- **2026-09-22** — Step 2: the spike (`scripts/spike_results_bundle.py`) passes all 13 checks.
  - zarr-python 3.4 writes clean v2 arrays with a zlib compressor, which stdlib and pyvcell's
    zarr 2.18 both read.
  - The vtk writer's binary, uncompressed, UInt32-header VTU parses like `VtuGridParser`.
  - `TS.interpolate` at output times is as accurate as stepping exactly to each one and leaves the
    step sequence unchanged.
  - `input_global_indices` keys give identical points and cells at n = 1, 2, 3, for volume and
    submesh.

  Added `zarr >=3.1,<4` and `vtk 9.6.*`; mypy learned `numcodecs` and `vtkmodules`. Because zarr
  now ships types, `mms/runner_fv.py` needed a `cast` on pyvcell's `Group | Array` return.
- **2026-09-22** — Step 1: ADR 010 (VTU + zarr bundle, schema 1, segments reserved) and ADR 011
  (SimulationTask contract, status protocol verified against `entrypoint.sh` and `LangevinNoVis01`'s
  `VCellMessagingRest` tests, container contract, Java follow-up) written.
- **2026-09-22** — Discovery and design done (three codebase surveys, one design pass). Decisions
  confirmed with the user: VTU + zarr for fixed meshes, segments for moving/remeshed meshes, adapter
  parsing in vcell-fenics, scope = docs + vcell-fenics side. This document created (step 0).

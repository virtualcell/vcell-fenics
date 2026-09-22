# Running vcell-fenics in a container

A single image that solves either kind of model — a **VCell** biomodel or a **native
vcell-fenics** description — and writes the results into a directory you mount.

```bash
docker build -f docker/Dockerfile -t vcell-fenics .
```

The image installs the repository's own `pixi.lock` with `--locked`, so it runs the exact
DOLFINx 0.10 / PETSc / Netgen builds the test suite runs against. It builds natively on
x86-64 and on arm64 (Apple silicon, Graviton); both platforms are in the lock.

### If the build fails part-way through the download

The environment is ~700 conda packages, and a flaky or filtered egress path shows up as a
*late* `failed to fetch <some package>` / `tcp connect error` rather than an immediate error —
a different package each time, which is the tell that the lock is fine and the network is not.
The build already limits itself to 8 concurrent connections and retries four times, resuming
from the BuildKit cache mount each attempt (each retry gets meaningfully further). If it still
fails, raise the retries and rerun — nothing already downloaded is fetched twice:

```bash
docker build -f docker/Dockerfile --build-arg PIXI_DOWNLOAD_RETRIES=10 -t vcell-fenics .
```

If *every* attempt dies that way, the fetch is not reaching the network at all. Check egress
from a container directly:

```bash
docker run --rm alpine wget -O /dev/null https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

Two causes account for nearly all of it, and both live in the VM rather than in this build:

- **No working IPv6 route.** `conda.anaconda.org` publishes A *and* AAAA records
  (`docker run --rm alpine nslookup conda.anaconda.org` shows both). A VM that resolves AAAA
  but cannot route it hangs on each connect until it times out, so downloads die late and on a
  different package every run. The build already prefers IPv4 via `/etc/gai.conf`; if that is
  not enough, disable IPv6 for the VM in Docker Desktop → **Settings → Resources → Network**.
- **A proxy or VPN the host uses but the VM does not** — Docker Desktop →
  **Settings → Resources → Proxies**.

## Run a model

Mount the directory holding your model read-only, and a results directory at `/work/out`
(the runner's default output path):

```bash
mkdir -p results

# 1. A VCell biomodel. Duration, output interval and mesh size come from the .vcml's own
#    simulation, so no other flags are needed.
docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/results:/work/out" \
  vcell-fenics --vcml /models/my_biomodel.vcml

# 2. The native formalism: a MathDescription + GeometryDescription pair.
docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/results:/work/out" \
  vcell-fenics --math /models/my_math.yaml --geometry /models/my_geom.yaml --t-final 1.0

# 3. The VCell YAML pair that `scripts/parse_biomodels_to_yaml.py` produces — same flags;
#    the formalism is detected from the file contents.
docker run --rm -v "$PWD/models:/models:ro" -v "$PWD/results:/work/out" \
  vcell-fenics --math /models/biomodel_123_app_math.yaml --geometry /models/biomodel_123_app_geom.yaml
```

### As a VCell solver: a SimulationTask

VCell hands every solver a `SimID_<key>_<job>__<task>.simtask.xml` document ([ADR 011](../docs/decisions/011-vcell-solver-contract.md)).
Bind the user's data directory **at the same path** inside the container, so the task's own paths
need no translation; the bundle `SimID_<key>_<job>_.fenics` lands next to the task file:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$USERDIR:$USERDIR" \
  vcell-fenics --simtask "$USERDIR/SimID_123_0__0.simtask.xml"
```

End time, output schedule, time step, error tolerances, mesh size and the job's parameter-scan point
all come from the task; method of lines is the default (real VCell kinetics are routinely nonlinear).
A task this solver would mis-solve — a moving boundary, field data, particle/stochastic math — exits 2
with the reason.

Status reporting, the way VCell listens for it (ADR 011 §4):

| flag | for | what |
| --- | --- | --- |
| `--vc-print-status` | a local run (the VCell client) | stdout carries **only** `[[[progress:NN.N%]]]` / `[[[data:t]]]` markers; everything else goes to stderr |
| `--vc-send-status-config=FILE` | a cluster run | REST WorkerEvents (STARTING / PROGRESS / DATA / COMPLETED / FAILURE) to the broker in `FILE` (Langevin properties format); a broker outage never fails the run |
| `-tid N` | the batch system | accepted and checked against the task's `TaskId` |

Exit status: `0` success, `2` a model or usage error, `1` a crash, `143` SIGTERM (the bundle's manifest
is marked `failed`). COMPLETED is sent only after the bundle is finalized.

The image ships a demo model, so you can check an installation with no files of your own:

```bash
docker run --rm -v "$PWD/results:/work/out" vcell-fenics \
  --math /opt/vcell-fenics/examples/diffusion2d_math.yaml \
  --geometry /opt/vcell-fenics/examples/diffusion2d_geom.yaml \
  --t-final 1.0
```

`docker run --rm vcell-fenics --help` lists every flag.

## What lands in the results directory

One **results bundle** per run, `<results dir>/results.fenics/` (`--output-prefix` renames it) —
the format in [ADR 010](../docs/decisions/010-results-bundle-vtu-zarr.md):

| path | what it is |
| --- | --- |
| `mesh/<domain>.vtu` | each VCell domain's mesh (a compartment or a membrane), written once |
| `<domain>/<variable>` | a zarr array, one row per output time, columns in the VTU's point order |
| `stats/<domain>/<variable>` | per output time: mean, ∫u dx, min, max (reduced across MPI ranks) |
| `.zattrs` | the manifest: domains, variables, the output times written so far, run status |
| `provenance/summary.json` | the run's configuration and per-species statistics |
| `provenance/math.yaml`, `geometry.yaml` | the resolved *native* formalism that was actually solved |

`python -m vcell_fenics.results results/results.fenics` summarises a bundle (add
`--require-status completed` to check a run finished). The bundle is readable while the run is still
writing — rows appear in the manifest only once they are complete.

ParaView cannot join the VTU meshes to the zarr fields itself; export the bundle first:

```bash
docker run --rm -v "$PWD/results:/work/out" vcell-fenics \
  vcell-fenics-export /work/out/results.fenics /work/out/paraview        # add --format xdmf for XDMF3
```

— one `<domain>.pvd` time series (a VTU per step, every variable as point data) per domain.

`provenance/math.yaml` / `geometry.yaml` are the ones to read when a VCell import behaves unexpectedly:
they are what the `.vcml` was translated into (doc §2.6), and they can be fed straight back
into the runner with `--math`/`--geometry` to re-run or to edit-and-re-run without VCell.

## File ownership

The container runs as uid 1001, so on a Linux host the results are written as that uid. To
get files owned by you:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/results:/work/out" vcell-fenics --vcml /models/m.vcml
```

Any uid works — the JIT cache lives in a world-writable `/opt/cache`, not in `$HOME`.
On Docker Desktop (macOS/Windows) ownership is mapped for you and the flag is unnecessary.

## Parallel runs

The solver is MPI-parallel; the image ships MPICH. `mpirun` is a command, so pass it directly:

```bash
docker run --rm --shm-size=1g \
  -v "$PWD/models:/models:ro" -v "$PWD/results:/work/out" \
  vcell-fenics mpirun -n 4 python -m vcell_fenics.cli --vcml /models/m.vcml
```

`--shm-size=1g` matters: Docker's default 64 MB of `/dev/shm` is where MPICH puts its
shared-memory transport, and a larger mesh will exhaust it. All reported numbers are reduced
across ranks, and the bundle is written in a rank-count-independent order, so an `-n 4` run's
bundle matches a serial one.

## Choosing the discretisation

Defaults are deliberately conservative rather than accurate, and the run header prints what
was chosen:

- `--h` — element size. From a `.vcml` it defaults to the finest cell spacing of the
  simulation's own finite-volume grid (`extent / mesh_size`); otherwise to 1/32 of the
  narrowest spatial axis. **This is the main cost knob** — halving `h` roughly quadruples the
  2D element count and multiplies 3D by ~8.
- `--dt` — backward-Euler step, defaulting to the output interval. That is one step per
  snapshot, which is coarse for anything stiff; set it explicitly for production runs.
- `--t-final`, `--output-dt` — from the `.vcml` simulation when present, else required.
- `--time-integration method_of_lines` — adaptive PETSc TS instead of fixed-step backward
  Euler. It integrates nonlinear reactions directly and controls its own time error, but has
  no output-time hook, so it writes the final state only.
- `--fe-degree 2` — quadratic elements.

## Scope

The runner drives the two fixed-domain paths:

- **one mesh** — any number of species coupled on a single compartment or membrane;
- **two compartments across a membrane** — the interface-coupled solve (permeability /
  jump-condition models), cross-validated against VCell's finite-volume solver in
  `cross_validation/`.

Moving membranes (ALE), the Stokes/FSI stack, phase field, and bulk-coupled surface PDEs have
their own drivers and are not reachable from this CLI; they report a message naming the
limitation rather than solving something else. Stochastic/particle models are rejected at
import by the VCell bridge, as they are out of scope for the formalism (doc §2.6.3).

## Development shell

```bash
docker run --rm -it -v "$PWD:/repo" vcell-fenics bash
```

Anything after the image name that does not start with `-` is run as a command inside the
activated environment (`bash`, `python`, `mpirun`, …); anything starting with `-` is a flag
for the runner. The image carries the `default` pixi environment and `src/` only — no pytest,
ruff, mypy, or gmsh (those are the `dev` environment). Run the test suite on the host with
`pixi run -e dev test`.

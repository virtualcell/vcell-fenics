# SOLVER-RELEASE — vcell-fenics against VCell's solver release contract

VCell's solver repos share one release contract (§1 of `docs/plan-solver-repos.md` in
[virtualcell/vcell](https://github.com/virtualcell/vcell)). This file records how vcell-fenics meets
each item. How to cut a release: the **Releases** section of `README.md`.

vcell-fenics is **container-only**. VCell never runs it as a local executable: the desktop client
launches the Docker image, and the cluster runs the SIF. So it has no archive assets (items 1–2), and
its images are the release artifacts.

| # | Contract item | vcell-fenics |
|---|---|---|
| 1 | Versioned releases from the default branch | A tag `vX.Y.Z` on `main` runs `.github/workflows/container.yml`. Its setup job fails the run unless the tag equals `v` + `vcell_fenics.__version__` (itself kept equal to pyproject's version by `tests/test_version.py`) and the tagged commit is on `main`. The workflow then creates the GitHub Release with notes naming the images. **No archive assets**: there is no local executable (see above). |
| 2 | Archive layout with VCell's names | Not applicable, since there are no archives. VCell's executable name is `vcell-fenics`, on `PATH` in the image. The `SolverDescription` is `FEniCSx`. |
| 3 | Container | `ghcr.io/virtualcell/vcell-fenics:X.Y.Z` and `:vX.Y.Z`, multi-arch (amd64, plus arm64 when the `ARM64_RUNNER` repo variable is set). `:latest` and `:sha-<short>` come from `main`. The runtime stage is `ubuntu:24.04` plus the locked pixi environment, not the build image. **Messaging ON**: `vcell-fenics` takes `--vc-send-status-config FILE` (REST WorkerEvents to the broker) and the trailing `-tid <n>`, and `--vc-print-status` gives stdout markers (`docker/README.md`). |
| 4 | SIF | `oras://ghcr.io/virtualcell/vcell-fenics_singularity:X.Y.Z` and `:vX.Y.Z` (amd64), built from the tested image and pushed with `apptainer push` by the same workflow. |
| 5 | Standard entrypoint | `docker/entrypoint.sh`, installed as `/usr/local/bin/vcell-solver-entrypoint` (`ENTRYPOINT`, `CMD ["--help"]`). `vcell-fenics-entrypoint` is kept as an alias. It runs `set -euo pipefail`. With no argument or with `--help` it prints the runner's help, headed `vcell-fenics X.Y.Z`, and `--version` prints just that line; both exit 0. A first argument naming a command is run with `exec "$@"`. **Deviation:** any command in the environment is accepted, not only the solver (`mpiexec -n 4 vcell-fenics …`, `python -m vcell_fenics.results …` and `bash` are all supported uses), and an argument starting with `-` goes to the runner. Unknown commands fail with the shell's exit 127, not 2. The entrypoint writes only to the results directory and a writable JIT cache (below), runs as any uid (the image is uid-agnostic; CI runs it as the runner's uid), and works from the read-only SIF under `--containall`. |
| 6 | CI smoke test | Every PR touching the image, every `main` push and every tag runs the same smoke test. (a) Docker: a real VCell SimulationTask, checked for status markers only on stdout and a completed bundle. (b) MPI: two ranks, checked against the serial bundle. (c) Apptainer `--containall`: the same task, plus a P2 model that compiles kernels the image never pre-warmed. (d) The SIF exactly as SlurmProxy writes it: `--bind …:/simdata --bind …:/solvertmp --env TMPDIR=/solvertmp <sif> vcell-fenics --simtask /simdata/… -tid 0`, run as a non-root uid. (e) `<sif> --help` and `--version` exit 0 and report the tag's version. The reference check is the bundle's completed status and the status markers. The FV↔FEniCSx numerical references live in `cross_validation/` and `mms/`. |

## The JIT cache under `--containall`

FFCx compiles forms at run time into `$XDG_CACHE_HOME/fenics`. The image pre-warms `/opt/cache`, but a
SIF is read-only, so the entrypoint moves to a per-user cache, seeded from the pre-warmed one. It uses
the first writable of:

1. `$VCELL_FENICS_CACHE`;
2. `$TMPDIR/vcell-fenics-cache-<uid>`;
3. `/tmp/vcell-fenics-cache-<uid>`.

A candidate that is set but unusable is skipped, with a note on stderr.

VCell's Slurm jobs should pass `--env TMPDIR=/solvertmp`, which is already a bind of the job's scratch
directory, so the cache lands on real disk. Without it the cache falls back to `/tmp`, which under
`--containall` is Apptainer's small in-memory session tmpfs. CI smoke (d) runs exactly that shape and
checks that the cache ends up in the scratch bind.

## What VCell needs to consume a release

- **HPC:** pin `VCELL_HTC_VCELLFENICS_APPTAINER_IMAGE=oras://ghcr.io/virtualcell/vcell-fenics_singularity:X.Y.Z`
  in the site's `submit.env`, keep `FEniCSx` on its solver list, and pass `--env TMPDIR=/solvertmp`.
- **Desktop:** the Docker image `ghcr.io/virtualcell/vcell-fenics:X.Y.Z`.

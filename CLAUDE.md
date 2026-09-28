# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

A working FEniCSx backend and a **VCell solver**. A declarative math model (`formalism/`) plus a
geometry description are compiled into a `DiscreteProblem` IR (ADR 004) and solved by DOLFINx
(`backend/`): fixed-domain reaction–diffusion–advection (single mesh and two compartments across a
membrane), surface PDEs, ALE moving membranes (prescribed and unknown motion, conservative time term),
Stokes/FSI, slip, Cahn–Hilliard phase field, and conservative remeshing/remap (`core/`). VCell models come
in through `pyvcell_bridge/` (VCML, math+geom YAML, SimulationTask XML) and results go out as a VTU +
zarr bundle (`results/`). Verification is standing infrastructure: `mms/` (manufactured-solution suite +
order-regression gate) and `cross_validation/` (numerical comparison against VCell's fvsolver/mbsolver);
each has a README that is the source of truth.

Orientation: `docs/overview.md` (short tour) and `docs/architecture.md` (code structure + data flow).
Live progress on the VCell-solver work: `docs/integration/vcell-solver-integration.md`.

## Project intent

`vcell-fenics` explores **FEniCSx (DOLFINx) as a backend for cell mechanics and cell migration modeling**, in the context of the user's Virtual Cell work. The user intends to implement **multiple finite-element approaches** (ALE explicit membrane, separate-mesh / mixed-dimensional, phase-field, cut/trace FEM) and compare them — not pick one. Full discussion: `docs/modeling/approaches.md`. The scientific North Star is Contri–Massing–Rangamani 2025 (see the research snapshot).

## Relationship to sibling repos

- **`../pyvcell`** — VCell's Python project, the **integration target**. The `vcell_fenics.pyvcell_bridge` package translates pyvcell's lowered math model (`pyvcell.vcml.models_math.MathDescription`) into the formalism (doc §2.6) — the "problem-generation front door" of the template-surface architecture. The coupling is **minimal**: `pyvcell >=0.4.1,<0.5` (a normal PyPI/pixi dependency) ships its pure-Pydantic `vcml` data model with a minimal default dependency set (lxml + numexpr), so none of pyvcell's heavy solver/viz/binary closure (behind optional extras) enters the conda DOLFINx env. The full-stack scripts (BioModel download/parse in `scripts/`, the fvsolver/mbsolver side of `cross_validation/`) instead run in pyvcell's own env, `../pyvcell/.venv`.
- **`../vcell-fvsolver`** and **`../vcell-mbsolver`** — VCell's fixed-grid finite-volume solver and its FronTier-based **moving-boundary** solver, the **comparison baselines** (driven through pyvcell by `cross_validation/` and `mms/`). Keep dimensional and biological conventions documentable so comparisons stay meaningful. (`../vcell-solvers` is the older C++ monorepo that also carries `MBSolver`.)
- **`../vcell`** — the VCell Java platform. The remaining solver-integration work (solver registration, SlurmProxy image wiring, desktop Docker launch, bundle viewing) lands there; ADR 011 §6 lists it.

## docs/

Substantive intellectual content lives in `docs/`, not in this file. Consult these before recommending libraries, designing code, or making environment changes:

- **`docs/modeling/approaches.md`** — the four approaches (A/B/C/D), the canonical surface PDE with the mandatory `ρ ∇_Γ · v_Γ` dilution term, tradeoff analysis, and the architecture sketch for supporting multiple approaches in one package.
- **`docs/modeling/declarative-formalism.md`** — the math formalism (MathDescription): templates, the expression language, the data model, the VCell mapping (§2.6). The VCell import layer (`pyvcell_bridge`) implements it.
- **`docs/modeling/geometric-formalism.md`** — the geometry formalism (GeometryDescription, the **source of truth** for spatial domains per ADR 007): VCell-style subvolumes (analytic/csg/image/compartmental) + surface classes + named faces, the realization layer (Netgen body-fitted / level-set / trivial — gmsh is dev-only, ADR 008), and the increment roadmap.
- **`docs/research/2026-05-21-fenicsx-ecosystem.md`** — May 2026 snapshot of FEniCSx ecosystem libraries (DOLFINx 0.10, CutFEMx, scifem, multiphenicsx, etc.), citations, and the Contri–Massing–Rangamani 2025 paper that is this project's scientific North Star.
- **`docs/integration/vcell-solver-integration.md`** — the living design + progress tracker for running vcell-fenics as a VCell solver (SimulationTask XML in, VTU + zarr results bundle out, VCell status protocol, Docker/Apptainer images, VCell Java follow-up). Update its status table and progress log as steps land.
- **`docs/decisions/`** — ADR-style records: Pixi + pyproject (001), DOLFINx 0.10 pin (002), MPICH-not-OpenMPI (003), DiscreteProblem IR (004), real FEniCSx types (005), namespaced built-ins (006), geometry source of truth (007), Netgen mesher / gmsh-license contingency (008), FSI species transport in the formalism (009, proposed), VTU + zarr results bundle (010), VCell solver contract — SimulationTask in, bundle out, status protocol (011), image-geometry realization — smoothed label field, SurfaceNets, Netgen (012).

When a user asks about libraries or approaches, check the research snapshot for recency before answering — the library state is dated 2026-05-21 and may have moved.

## Environment

Managed by **[Pixi](https://pixi.sh/)** (≥ 0.68) with the manifest embedded in `pyproject.toml` under `[tool.pixi.*]`. Dependencies come from **conda-forge** (`fenics-dolfinx=0.10.*`, `mpich`, `petsc4py`, `scifem`, `dolfinx_mpc`, `netgen`, `zarr`, `vtk`, `pyvista`, etc.). The package is installed editably via `[tool.pixi.pypi-dependencies]` so `import vcell_fenics` works from any Pixi-run process.

Platforms locked: `osx-arm64`, `linux-64`, `linux-aarch64` (the last for the container build on Apple-silicon Docker). Add platforms in `[tool.pixi.workspace].platforms` and re-solve as needed.

**MPI variant: MPICH.** Don't introduce `openmpi` (see `docs/decisions/003-conda-mpich-not-openmpi.md`).

Environments defined:
- `default` — runtime dependencies only
- `dev` — adds `pytest`, `ipython`, `ruff`, `mypy`, `jupyterlab`, and **`gmsh`** (GPL, so dev/test-only — never move it to runtime deps; `src/` must stay gmsh-free, see `LICENSING.md` and ADR 008)

## Commands

```bash
pixi install                  # resolve and install (run after editing deps)
pixi shell                    # activate the default env
pixi shell -e dev             # activate the dev env (pytest, ipython, ruff, mypy)
pixi run -e dev test          # run pytest (excludes the slow `integration`-marked tests)
pixi run -e dev test path/to/test_x.py::test_y   # run a single test
pixi run -e dev test-integration   # run only the slow integration/convergence tests (minutes)
pixi run -e dev lint          # ruff lint
pixi run -e dev format        # ruff format (writes)
pixi run -e dev typecheck     # mypy --strict
pixi run -e dev check         # lint + format-check + typecheck + test
pixi run -e dev notebooks     # JupyterLab on docs/notebooks/
pixi add <pkg>                # add a conda dep (writes to pyproject.toml + pixi.lock)
pixi add --pypi <pkg>         # add a PyPI dep
pixi update                   # upgrade within version specs
```

**Running a model.** `vcell_fenics.cli` is the one-shot runner (argv + loading; the solve lives in
`vcell_fenics.runner`): a VCell **SimulationTask** (`--simtask`, the document VCell hands its solvers — ADR 011),
a VCell `.vcml`, a VCell math+geom YAML pair, or a native formalism pair in —
a **results bundle** out, `<--out>/<--output-prefix>.fenics/` (ADR 010: a VTU mesh per domain, zarr
fields at every output time, per-time statistics, a manifest; provenance + `summary.json` inside).

```bash
pixi run -e dev python -m vcell_fenics.cli --simtask tests/fixtures/simtask/SimID_1585623750_0__0.simtask.xml --out /tmp/r
pixi run -e dev python -m vcell_fenics.cli --vcml model.vcml --out results
pixi run -e dev python -m vcell_fenics.cli --math m_math.yaml --geometry m_geom.yaml --t-final 1.0
pixi run -e dev python -m vcell_fenics.results results/results.fenics   # summarise a bundle
pixi run -e dev vcell-fenics-export results/results.fenics paraview/      # → PVD (or --format xdmf) for ParaView
docker build -f docker/Dockerfile -t vcell-fenics .     # same runner, containerised
```

It drives the fixed-domain paths:
- **a single mesh**, for equations on one subdomain;
- **the multi-compartment solver** (`backend/multi_compartment.py`) for everything whose equations span two or
  more subdomains. That means any number of compartments and membranes (a nucleus in a cytosol in extracellular
  space, touching cells) and any number of species on each, including membrane species (receptor–ligand
  binding), **region variables** (a well-mixed species, a membrane potential — T5 `region_ode`) and box-face
  values per compartment.

It also drives VCell moving boundaries: a prescribed front with species inside it, ALE with remeshing, in 2D
and 3D (the tracker's "Moving boundaries" section). Not yet: membrane species on a moving front. The older
two-compartment integrators (`integrate_interface_coupled`, `integrate_membrane_coupled`) stay for their tests
and cross-validation; the runner no longer routes to them.
Stokes/FSI, phase field and unknown-motion mechanics keep their own drivers — extend the CLI
deliberately rather than routing them through it. `docker/README.md`
is the container reference (results mount, MPI, uid, discretisation defaults).

**Quality enforcement.** Ruff (lint+format) and mypy in `--strict` are required across `src/`, `tests/`, and `examples/` (mypy also covers `mms/` and `scripts/`; the package ships a `py.typed` marker so `examples/` get its real types, not `Any`). `pixi run -e dev check` is the gate. The FEniCSx stack ships `py.typed` but with many unannotated functions, so mypy uses its **real** types (we do *not* `follow_imports = "skip"`); strict mode's `disallow_untyped_calls` is suppressed for that stack via `untyped_calls_exclude` so calls like `grad()`/`dot()` don't flood, while every other real check is kept (ADR 005). When a third-party stub is genuinely wrong (e.g. petsc4py's `PETSc.ScalarType`, some pyvista signatures), use a targeted `# type: ignore[code]` or `cast()` at that exact call site — never a blanket `Any` in our own signatures. The `backend/_typing.py` aliases (`DolfinxFunction`, `UflForm`, …) document which opaque object a field holds where the upstream type is still `Any`.

To run Python directly in the env without going through `pixi shell`:

```bash
.pixi/envs/default/bin/python script.py
```

(Don't define a bare `python = "python"` Pixi task — it hijacks `pixi run python -c '...'` quoting.)

`pixi.lock` is the cross-platform lockfile and **must be committed**.

## Working in this repo

- Frame design discussions in terms of cell-biology modeling concepts (compartments, species, reactions, mechanics) when they map onto FEM constructs — `vcell-fenics` is anchored to the Virtual Cell platform.
- When a user asks "can FEniCSx do X for a moving membrane?", surface the relevant pitfall from `docs/modeling/approaches.md` up front — don't wait for them to hit it.
- The `ρ ∇_Γ · v_Γ` dilution term on a moving surface PDE is mandatory and the most common source of subtle bugs in this problem class. Flag it whenever surface-density code appears.

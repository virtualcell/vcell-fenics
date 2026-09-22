# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early scaffolding. `pyproject.toml` (Pixi + hatchling), `pixi.lock`, and an empty `src/vcell_fenics/` package exist. No models, mechanics, mesh code, or tests yet. Do not infer architecture or conventions from absence — ask before assuming.

## Project intent

`vcell-fenics` explores **FEniCSx (DOLFINx) as a backend for cell mechanics and cell migration modeling**, in the context of the user's Virtual Cell work. The user intends to implement **multiple finite-element approaches** (ALE explicit membrane, separate-mesh / mixed-dimensional, phase-field, cut/trace FEM) and compare them — not pick one. Full discussion: `docs/modeling/approaches.md`.

The first concrete goal is a 2D single-cell prototype with one surface PDE (receptor / activator density: surface diffusion + reaction + dilution from membrane stretch).

## Relationship to sibling repos

Two repos in the same workspace define the broader context.

- **`../pyvcell`** — VCell's Python project, the **integration target**. The `vcell_fenics.pyvcell_bridge` package translates pyvcell's lowered math model (`pyvcell.vcml.models_math.MathDescription`) into the formalism (doc §2.6) — the "problem-generation front door" of the template-surface architecture. The coupling is **minimal**: `pyvcell >= 0.3.0` (a normal PyPI/pixi dependency) ships its pure-Pydantic `vcml` data model with a minimal default dependency set (lxml + numexpr), so none of pyvcell's heavy solver/viz/binary closure (behind optional extras) enters the conda DOLFINx env. The full-stack scripts (BioModel download/parse in `scripts/`) instead run in pyvcell's own env, `../pyvcell/.venv`.
- **`../vcell-solvers`** — contains VCell's existing **moving-boundary solver**, the eventual **comparison baseline** for any moving-membrane work done here. Keep dimensional and biological conventions documentable so the comparison is meaningful when it happens.
- **`../vcell-solvers`** — contains VCell's existing **moving-boundary solver**, the eventual **comparison baseline** for any moving-membrane work done here. Keep dimensional and biological conventions documentable so the comparison is meaningful when it happens.

## docs/

Substantive intellectual content lives in `docs/`, not in this file. Consult these before recommending libraries, designing code, or making environment changes:

- **`docs/modeling/approaches.md`** — the four approaches (A/B/C/D), the canonical surface PDE with the mandatory `ρ ∇_Γ · v_Γ` dilution term, tradeoff analysis, and the architecture sketch for supporting multiple approaches in one package.
- **`docs/modeling/declarative-formalism.md`** — the math formalism (MathDescription): templates, the expression language, the data model, the VCell mapping (§2.6). The VCell import layer (`pyvcell_bridge`) implements it.
- **`docs/modeling/geometric-formalism.md`** — the geometry formalism (GeometryDescription, the **source of truth** for spatial domains per ADR 007): VCell-style subvolumes (analytic/csg/image/compartmental) + surface classes + named faces, the realization layer (gmsh OCC body-fitted / level-set / trivial), and the increment roadmap.
- **`docs/research/2026-05-21-fenicsx-ecosystem.md`** — May 2026 snapshot of FEniCSx ecosystem libraries (DOLFINx 0.10, CutFEMx, scifem, multiphenicsx, etc.), citations, and the Contri–Massing–Rangamani 2025 paper that is this project's scientific North Star.
- **`docs/integration/vcell-solver-integration.md`** — the living design + progress tracker for running vcell-fenics as a VCell solver (SimulationTask XML in, VTU + zarr results bundle out, VCell status protocol, Docker/Apptainer images, VCell Java follow-up). Update its status table and progress log as steps land.
- **`docs/decisions/`** — ADR-style records: Pixi + pyproject (001), DOLFINx 0.10 pin (002), MPICH-not-OpenMPI (003), DiscreteProblem IR (004), real FEniCSx types (005), namespaced built-ins (006), geometry source of truth (007), Netgen mesher / gmsh-license contingency (008), FSI species transport in the formalism (009, proposed), VTU + zarr results bundle (010), VCell solver contract — SimulationTask in, bundle out, status protocol (011).

When a user asks about libraries or approaches, check the research snapshot for recency before answering — the library state is dated 2026-05-21 and may have moved.

## Environment

Managed by **[Pixi](https://pixi.sh/)** (≥ 0.68) with the manifest embedded in `pyproject.toml` under `[tool.pixi.*]`. Dependencies come from **conda-forge** (`fenics-dolfinx=0.10.*`, `mpich`, `petsc4py`, `scifem`, `dolfinx_mpc`, `gmsh`, `pyvista`, etc.). The package is installed editably via `[tool.pixi.pypi-dependencies]` so `import vcell_fenics` works from any Pixi-run process.

Platforms locked: `osx-arm64`, `linux-64`. Add platforms in `[tool.pixi.workspace].platforms` and re-solve as needed.

**MPI variant: MPICH.** Don't introduce `openmpi` (see `docs/decisions/003-conda-mpich-not-openmpi.md`).

Environments defined:
- `default` — runtime dependencies only
- `dev` — adds `pytest`, `ipython`

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
pixi add <pkg>                # add a conda dep (writes to pyproject.toml + pixi.lock)
pixi add --pypi <pkg>         # add a PyPI dep
pixi update                   # upgrade within version specs
```

**Running a model.** `vcell_fenics.cli` is the one-shot runner (argv + loading; the solve lives in
`vcell_fenics.runner`): a VCell `.vcml`, a VCell math+geom YAML pair, or a native formalism pair in —
a **results bundle** out, `<--out>/<--output-prefix>.fenics/` (ADR 010: a VTU mesh per domain, zarr
fields at every output time, per-time statistics, a manifest; provenance + `summary.json` inside).

```bash
pixi run -e dev python -m vcell_fenics.cli --vcml model.vcml --out results
pixi run -e dev python -m vcell_fenics.cli --math m_math.yaml --geometry m_geom.yaml --t-final 1.0
pixi run -e dev python -m vcell_fenics.results.reader results/results.fenics   # summarise a bundle
docker build -f docker/Dockerfile -t vcell-fenics .     # same runner, containerised
```

It drives the two fixed-domain paths only (single mesh; two compartments across a membrane).
ALE/moving membranes, Stokes/FSI, phase field, and bulk-coupled surface PDEs keep their own
drivers — extend the CLI deliberately rather than routing them through it. `docker/README.md`
is the container reference (results mount, MPI, uid, discretisation defaults).

**Quality enforcement.** Ruff (lint+format) and mypy in `--strict` are required across `src/`, `tests/`, and `examples/` (the package ships a `py.typed` marker so `examples/` get its real types, not `Any`). `pixi run -e dev check` is the gate. The FEniCSx stack ships `py.typed` but with many unannotated functions, so mypy uses its **real** types (we do *not* `follow_imports = "skip"`); strict mode's `disallow_untyped_calls` is suppressed for that stack via `untyped_calls_exclude` so calls like `grad()`/`dot()` don't flood, while every other real check is kept (ADR 005). When a third-party stub is genuinely wrong (e.g. petsc4py's `PETSc.ScalarType`, some pyvista signatures), use a targeted `# type: ignore[code]` or `cast()` at that exact call site — never a blanket `Any` in our own signatures. The `backend/_typing.py` aliases (`DolfinxFunction`, `UflForm`, …) document which opaque object a field holds where the upstream type is still `Any`.

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Early scaffolding. `pyproject.toml` (Pixi + hatchling), `pixi.lock`, and an empty `src/vcell_fenics/` package exist. No models, mechanics, mesh code, or tests yet. Do not infer architecture or conventions from absence — ask before assuming.

## Project intent

`vcell-fenics` explores **FEniCSx (DOLFINx) as a backend for cell mechanics and cell migration modeling**, in the context of the user's Virtual Cell work. The user intends to implement **multiple finite-element approaches** (ALE explicit membrane, separate-mesh / mixed-dimensional, phase-field, cut/trace FEM) and compare them — not pick one. Full discussion: `docs/modeling/approaches.md`.

The first concrete goal is a 2D single-cell prototype with one surface PDE (receptor / activator density: surface diffusion + reaction + dilution from membrane stretch).

## Relationship to sibling repos

Two repos in the same workspace define the broader context. **Current scope is FEniCSx-only — do not couple to them yet — but do not architect decisions that foreclose either.**

- **`../pyvcell`** — VCell's Python project, the eventual **integration target**. API and packaging choices here should remain pyvcell-callable (importable as a library, no hard CLI-only assumptions, compatible Python version and core deps).
- **`../vcell-solvers`** — contains VCell's existing **moving-boundary solver**, the eventual **comparison baseline** for any moving-membrane work done here. Keep dimensional and biological conventions documentable so the comparison is meaningful when it happens.

## docs/

Substantive intellectual content lives in `docs/`, not in this file. Consult these before recommending libraries, designing code, or making environment changes:

- **`docs/modeling/approaches.md`** — the four approaches (A/B/C/D), the canonical surface PDE with the mandatory `ρ ∇_Γ · v_Γ` dilution term, tradeoff analysis, and the architecture sketch for supporting multiple approaches in one package.
- **`docs/research/2026-05-21-fenicsx-ecosystem.md`** — May 2026 snapshot of FEniCSx ecosystem libraries (DOLFINx 0.10, CutFEMx, scifem, multiphenicsx, etc.), citations, and the Contri–Massing–Rangamani 2025 paper that is this project's scientific North Star.
- **`docs/decisions/`** — ADR-style records: Pixi + pyproject (001), DOLFINx 0.10 pin (002), MPICH-not-OpenMPI (003).

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
pixi run -e dev test          # run pytest
pixi run -e dev test path/to/test_x.py::test_y   # run a single test
pixi run -e dev lint          # ruff lint
pixi run -e dev format        # ruff format (writes)
pixi run -e dev typecheck     # mypy --strict
pixi run -e dev check         # lint + format-check + typecheck + test
pixi add <pkg>                # add a conda dep (writes to pyproject.toml + pixi.lock)
pixi add --pypi <pkg>         # add a PyPI dep
pixi update                   # upgrade within version specs
```

**Quality enforcement.** Ruff (lint+format) and mypy in `--strict` are required across both `src/` and `tests/`. `pixi run -e dev check` is the gate. Stub gaps in the FEniCSx stack are handled by `follow_imports = "skip"` overrides in `[tool.mypy]`, not blanket `Any` annotations in our code — if a third-party return leaks `Any`, use a targeted `cast()` at the boundary rather than weakening the function signature.

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

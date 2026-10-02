# ADR 001 — Pixi with pyproject.toml as the environment / packaging manifest

**Date:** 2026-05-21
**Status:** Accepted

## Context

`vcell-fenics` depends on FEniCSx (DOLFINx), which is best installed from `conda-forge` because of its tightly coupled native dependency tree (PETSc, MPI, BLAS/LAPACK, basix, ufl, ffcx, etc.). Pure-PyPI installation is brittle. The project will also have:

- An installable Python package (`vcell_fenics`) for eventual integration with `pyvcell`.
- Cross-platform reproducibility needs: developed on macOS arm64, expected to run on Linux clusters.
- A mix of conda-forge dependencies (the FEniCSx stack) and likely PyPI-only ones over time (`multiphenicsx`, `dolfinx-external-operator`).

Candidate environment managers:

- **conda + `environment.yml`** — works, but `environment.yml` is not a real lockfile; no cross-platform locking; awkward to combine with `pyproject.toml` for the package metadata.
- **Docker** — reproducible but heavyweight for everyday development.
- **Pixi** — modern manager built on conda; cross-platform lockfile (`pixi.lock`); supports both conda and PyPI dependencies in one manifest; can host its config inside `pyproject.toml` via `[tool.pixi.*]` tables.

## Decision

Use **Pixi** with the manifest embedded in `pyproject.toml`:

- `pyproject.toml` carries both PEP 621 `[project]` metadata (for the installable `vcell_fenics` package) and `[tool.pixi.*]` tables (workspace, dependencies, features, environments, tasks).
- Build backend: `hatchling`.
- Channels: `conda-forge` only.
- Platforms: `osx-arm64`, `linux-64` (additional platforms can be added as needed).
- Locked via `pixi.lock` (committed).
- The package is registered as an editable PyPI dependency in the workspace via `[tool.pixi.pypi-dependencies]` so `import vcell_fenics` works from any `pixi run` invocation.

## Consequences

**Positive:**
- Single source of truth: `pyproject.toml` carries package metadata and environment spec.
- Real lockfile (`pixi.lock`) — cross-platform, deterministic installs.
- `pixi run <task>` replaces ad-hoc `make`/`invoke` and keeps task definitions in the same file.
- No global env activation — environments are project-local under `.pixi/`.
- Mixed conda + PyPI is first-class — important when adding `multiphenicsx`, `dolfinx-external-operator`.

**Negative:**
- Pixi is younger than conda; manifest schema can change between minor releases (already saw `[tool.pixi.project]` → `[tool.pixi.workspace]` rename between 0.62 and 0.68). Lockfile schema may also change.
- Some HPC sites still default to `conda env create -f environment.yml`. A generated `environment.yml` may need to be produced on demand for those.
- Defining a bare `python = "python"` task in `[tool.pixi.tasks]` would hijack `pixi run python …` invocations and break `-c '…'` quoting; avoid such "shadow" task names.

## Notes

- `.pixi/` is gitignored. `pixi.lock` is committed.
- Pixi version at decision time: 0.68.1 (upgraded from 0.62.2 before the first install).
- The decision to migrate from a hypothetical `environment.yml` flow was made before any environment file was written; this ADR is a forward decision, not a migration.
- **2026-10-01 — VTK comes from PyPI, not conda-forge** (a use of the mixed conda + PyPI support
  above). conda-forge's VTK 9.6 exists only as a Qt build, and with its viskores/mesalib/LLVM and
  ffmpeg dependencies it made up about 3 GB of the 5.5 GB runtime image. The solver needs only
  VTK's data model, SurfaceNets and the VTU writer, so the runtime env takes the PyPI `vtk` wheel
  (glibc + libstdc++ only), Netgen's `occt` is pinned to its `novtk` build, and pyvista (also from
  PyPI, on the same wheel) and full matplotlib move to the `dev` feature. Rule: never put conda
  `vtk`/`vtk-base`/`pyvista` in an environment next to the PyPI wheel. Details: `docker/README.md`,
  "What the image carries".

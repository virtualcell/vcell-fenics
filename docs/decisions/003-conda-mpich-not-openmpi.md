# ADR 003 — Use MPICH (not OpenMPI) as the MPI variant

**Date:** 2026-05-21
**Status:** Accepted

## Context

conda-forge ships `fenics-dolfinx` in two MPI variants: MPICH and OpenMPI. Choosing one is mandatory — they cannot coexist in the same environment, and mixing them silently across packages corrupts MPI communication in ways that are painful to debug.

Both variants are supported on macOS arm64 and Linux. Anecdotal community reports (FEniCS Discourse, conda-forge issues) over the past 2–3 years consistently note fewer issues with MPICH on Apple Silicon. Linux clusters generally have site-specific MPI builds, which means the conda-shipped MPI is replaced by a system MPI at runtime regardless of which conda variant was chosen — so the choice matters mainly for local development.

## Decision

Pin `mpich` explicitly in the default Pixi environment alongside `fenics-dolfinx` and `petsc4py`. Do not introduce `openmpi`.

## Consequences

**Positive:**
- Apple Silicon local development is more reliable.
- All downstream packages (`scifem`, `dolfinx_mpc`, etc.) resolve against the same MPI variant.

**Negative:**
- If a contributor's existing tooling already uses OpenMPI elsewhere, they may need to maintain a separate environment for this project.
- If a future dependency only ships OpenMPI builds on conda-forge, this decision will need to be revisited.

## Notes

- Pinned versions at decision time: `mpich 5.0.1`, `fenics-dolfinx 0.10.0`, `petsc4py 3.25.1`.
- For HPC deployment, the conda-forge MPICH will typically be replaced by the cluster's MPI; verify the ABI / wire-up at deployment time.

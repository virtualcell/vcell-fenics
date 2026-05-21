# ADR 002 — Pin fenics-dolfinx to 0.10.x

**Date:** 2026-05-21
**Status:** Accepted

## Context

DOLFINx is currently at **0.10.0** (released October 2025). The 0.10 release significantly improved native mixed-dimensional assembly support — `dolfinx.mesh.create_submesh`, `EntityMap`, `ufl.MixedFunctionSpace`, `extract_blocks`, `LinearProblem(kind="block"|"nest")`. Approach B (separate-mesh / mixed-dimensional formulations; see `docs/modeling/approaches.md`) is the recommended starting point for this project, and Approach B depends on these 0.10 features.

The 0.10 release also introduced API churn relative to 0.9: many assembly call sites now expect `EntityMap` rather than raw NumPy index arrays.

Approach D (cut / trace FEM) is gated on **CutFEMx**, which currently targets **DOLFINx 0.9** — not 0.10. There is a version-pin conflict between Approach D's library and the rest of the stack.

## Decision

Pin `fenics-dolfinx = "0.10.*"` in the default Pixi environment.

When (and only when) Approach D becomes active, define a **separate Pixi feature / environment** that pins `fenics-dolfinx = "0.9.*"` and adds CutFEMx (built from source). Do not attempt to make a single environment satisfy both pins.

## Consequences

**Positive:**
- Approach B (the recommended starting point) gets the native mixed-dimensional API directly.
- The conda-forge solver finds a clean solution; no version juggling for the default environment.
- The pin is precise enough to avoid accidental jumps to 0.11 (which is on `main` but unreleased) while leaving room for 0.10.x patch releases.

**Negative:**
- Sample code and tutorials written against DOLFINx 0.9 (or earlier) may need porting — particularly around `EntityMap` vs. raw NumPy index arrays at assembly call sites.
- Approach D requires a parallel environment, increasing CI / dev-machine setup complexity.

## Notes

- conda-forge package: `fenics-dolfinx 0.10.0 py312h…_106` (Python 3.14.5 environment).
- See `docs/research/2026-05-21-fenicsx-ecosystem.md` §1 for the full mixed-dimensional API description, and §2 for CutFEMx pinning.

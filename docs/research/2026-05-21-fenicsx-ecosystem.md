# FEniCSx Ecosystem Research Snapshot — May 2026

*Captured: 2026-05-21. Library versions and links are point-in-time; verify before relying on them.*

This is the research pass that informed the architecture choices recorded under `docs/decisions/`. It surveys the FEniCSx ecosystem **as it stands in May 2026** for the purpose of cell mechanics and cell migration modeling — specifically the problem of solving surface PDEs on moving membranes coupled to bulk cytoplasm.

---

## 1. DOLFINx state of play

**Current release:** DOLFINx **0.10.0** (October 2025). Conda-forge package `fenics-dolfinx=0.10.0` is available for `osx-arm64`, `linux-64`, `linux-aarch64`, `win-64`, `osx-64`. The `0.11` branch is in development on `main` but unreleased.

**ALE / moving mesh — verdict: still no first-class API.** This has not meaningfully improved since 2023. The legacy `dolfin.ALE.move()` has **no direct replacement** in DOLFINx as of May 2026. The maintainer-endorsed pattern (per the October 2025 FEniCS Discourse thread *"Mesh moving / ALE in DOLFINx — example or official API"*) is DIY:

1. **Direct manipulation** for Lagrangian / small motion: write displacements into `mesh.geometry.x` (a NumPy ndarray). Breaks down for large deformation.
2. **Robust pattern** for larger motion: solve a separate mesh-displacement PDE each step (harmonic extension, or linear elasticity with Jacobian-based stiffening), then update `mesh.geometry.x`. You write this yourself; there is no `dolfinx.ale` module.

**Mixed-dimensional assembly — verdict: substantial improvement in 0.10.** This is the area with the biggest 2024–2026 progress. Per the v0.10 release notes ("vastly improved" mixed-dimensional support):

- `dolfinx.mesh.create_submesh(mesh, dim, entities)` → returns `(submesh, entity_map, vertex_map, geom_node_map)`.
- New `dolfinx.mesh.EntityMap` class — bidirectional submesh ↔ parent map, replaces the loose NumPy arrays used in older code.
- `dolfinx.fem.form(...)` accepts `entity_maps=[...]` for mixed-dimensional assembly.
- Supports assembly of 0th, 1st, and 2nd-order tensors mixing function spaces over different submeshes + the parent — true bulk + boundary trial/test mixing.
- Use `ufl.MixedFunctionSpace` and `ufl.extract_blocks()` to compose the formulation; `dolfinx.fem.petsc.LinearProblem(..., kind="block"|"nest")` handles solving.

This is the **recommended primary path for separate-mesh / mixed-dimensional formulations** in 2026. Native, supported by the core maintainers, and the subject of Dokken's November 2025 EPFL talk *"Multidimensional modelling in DOLFINx"*.

**0.10 API churn to watch when porting older code:**
- Many assembly call sites now expect `EntityMap` rather than raw NumPy index arrays.
- `LinearProblem` gained `kind="block"`/`kind="nest"` — use for coupled bulk-surface solves.
- VTKHDF mesh I/O is new in 0.10, faster than XDMF for large parallel runs.

---

## 2. Cut FEM / Trace FEM

**CutFEMx** ([github.com/sclaus2/CutFEMx](https://github.com/sclaus2/CutFEMx)) is the only serious option and it is **viable**:

- **First release v0.1.0 on April 27, 2026** — actively developed.
- Targets DOLFINx **0.9** with a customized FFCx for runtime quadrature.
- **Not on conda-forge nor PyPI**; source-only build via CMake + pip. Requires a companion library `CutCells`.
- License: MIT.
- Crucially: includes `demo_moving_poisson.py` — a time-dependent problem with a moving interface where **quadrature rules and integration domains are updated each step**. This is the exact capability needed for evolving-surface cut FEM.
- Other demos: Poisson on circular domains, Stokes around cylinders, level-set reinitialization via fast marching, signed distance from STL.

**Caveats:** v0.1.0 implies the API may change; pinned to DOLFINx 0.9 (not 0.10) creates a version-pin conflict with the rest of the stack. Plan a separate Pixi feature / environment.

**Other unfitted FEM options surveyed:**
- **Burman/Hansbo CutFEM** group: has Firedrake / legacy-FEniCS code but **no maintained FEniCSx port**.
- **Φ-FEM** (Duprez/Lozinski): research code, FEniCS-based; no active FEniCSx port found.
- **Augusto / Olshanskii TraceFEM** code: research-only, not packaged.

**Verdict:** CutFEMx is the only thing to bet on, and it's young but actively developed and targets exactly our use case.

---

## 3. Mixed-dimensional / submesh ecosystem add-ons

Beyond the native DOLFINx 0.10 support already described:

**multiphenicsx** ([github.com/multiphenics/multiphenicsx](https://github.com/multiphenics/multiphenicsx)):
- Active (Francesco Ballarin, Università Cattolica del Sacro Cuore), LGPL-3.
- **PyPI only, not conda-forge** (`pip install --no-build-isolation 'multiphenicsx[tutorials]'`).
- Provides subdomain / boundary restriction of unknowns.
- Project docs explicitly note that DOLFINx ≥ 0.9 now supports mixed-dim assembly natively, "and the two different implementations will co-exist for the foreseeable future."
- **Recommendation:** not strictly necessary in 2026; keep as fallback if native 0.10 hits an expressiveness wall.

**scifem** ([github.com/scientificcomputing/scifem](https://github.com/scientificcomputing/scifem)):
- **v0.18.1 released May 19, 2026.** Extremely active.
- Available via pip, conda, spack. Targets DOLFINx ≥ 0.8.
- Features: `scifem.create_real_functionspace`, `PointSource`, `BlockedNewtonSolver`, simplified MeshTags creation from locator functions, interpolation matrices from UFL expressions, DOF-to-vertex maps, point-evaluation.
- **Not a mixed-dim library itself**, but provides the small utility functions one would otherwise reinvent (point sources for receptor injection, blocked Newton for coupled bulk-surface nonlinear systems). **Include in default dependencies.**

**dolfinx-mpc** ([github.com/jorgensd/dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc)):
- **v0.10.5 April 15, 2026.** Conda-forge: `dolfinx_mpc`.
- Multi-point constraints (periodic BCs, slip conditions). Useful for confluent-cell simulations with periodic boundaries.

**Recommended stack:** DOLFINx 0.10 native `create_submesh` + `EntityMap` + `MixedFunctionSpace` + `scifem` utilities + `dolfinx_mpc` for periodic / constrained problems. Keep `multiphenicsx` as a fallback.

---

## 4. Phase-field for cell shape and migration

**Direct cell-migration phase-field code in FEniCSx — verdict: no good off-the-shelf option.**

- **PhaseFieldX** ([github.com/CastillonMiguel/phasefieldx](https://github.com/CastillonMiguel/phasefieldx)) — well-maintained, **v0.3.0 January 2026**, targets DOLFINx 0.10. JOSS-published 2025. **But: exclusively fracture / damage mechanics** — no cell biology. Useful only as a reference architecture for structuring a phase-field code on DOLFINx.

- **Wenzel / Marth / Voigt (TU Dresden) multi-phase-field collective migration** (one ϕ per cell): the canonical line of work (Phys Rev E 2021; Phys Rev E 110.044403, 2024; Phys Rev Research 2025 on emergent migration via T1 transitions). **None of it is published in FEniCSx**; the Voigt group historically uses their AMDiS C++ framework.

- **Border-cell cluster migration in Drosophila** ([arXiv 2508.21078](https://arxiv.org/abs/2508.21078), August 2025): continuum phase-field with "Tangential Interface Migration" force. Implementation framework not stated in the abstract.

- **Contri–Massing–Rangamani (2025)** (see §5) includes a Cahn–Hilliard phase-segregation test on a deformable surface — surface phase-field on a moving membrane, not bulk cell-shape phase-field, but highly relevant.

**Implication:** for a phase-field approach to cell shape / migration, implement from scratch. Cahn–Hilliard / Allen–Cahn demos in DOLFINx are standard; the cell-mechanics specialization (volume conservation, contractile coupling, multi-ϕ interactions) is custom code.

---

## 5. Cell-mechanics / cell-migration applications already using FEniCSx (2023–2026)

**Strongest match — the scientific North Star for this project:**

> **Contri, Massing, Rangamani (2025).** *"A Finite Element framework for bulk-surface coupled PDEs to solve moving boundary problems in biophysics."* arXiv **2510.23459**; also bioRxiv 2025-10-27. Padmini Rangamani's lab, UC San Diego. Funded by NIH R35 GM158446.

Tests covered:
- Advection–diffusion–reaction on evolving surfaces (the exact use case here).
- Cahn–Hilliard phase separation on deformable membranes.
- Helfrich-energy geometric flows for membranes.
- ALE mesh redistribution with element-quality preservation ("two-step redistribution procedure" driven by surface-tangential velocities, without remeshing).
- Tumor-growth surrogate; phase segregation on deformable membranes.

Software not explicitly confirmed FEniCSx in the abstract, but Massing's prior work and the Rangamani lab's other projects (SMART, see below) use DOLFINx. The associated code repository has not yet been located by name as of May 2026 — check [github.com/RangamaniLabUCSD](https://github.com/RangamaniLabUCSD) and [github.com/Rangamani-Lab](https://github.com/Rangamani-Lab) periodically. The bioRxiv preprint may post a repo link with the journal version. **Email the authors if not located.**

**Other relevant works:**

- **SMART** ([github.com/RangamaniLabUCSD/smart](https://github.com/RangamaniLabUCSD/smart)) — "Spatial Modeling Algorithms for Reactions and Transport". Reaction-transport in biological cells with **mixed-dimensional coupling (3D–2D bulk-surface, 2D–1D)**. Authors include Justin Laughlin, Christopher Lee, **Jørgen Dokken**, Henrik Finsberg, Emmet Francis. LGPL-3. **Important caveat:** the PyPI package `fenics-smart` and the May 2024 bioRxiv preprint state it requires **legacy FEniCS 2019.2.0**, not DOLFINx. Repo updated March 2026 — a migration may be in progress, but as of stated docs it is legacy. Treat as a **design reference** (its model specification language for compartments-and-reactions is the clearest published example of pyvcell-style spatial models in FEM), not as a library to depend on.

- **Mem3DG** ([github.com/RangamaniLabUCSD/Mem3DG](https://github.com/RangamaniLabUCSD/Mem3DG)) — 3D membrane mechanics. **Not FEniCSx**; uses discrete differential geometry (Geometry-Central, libigl, polyscope) via pybind11. Reference for membrane representation choices.

- **Variational system identification for wound healing** — Srivastava, Garikipati et al., [PLOS Comp Bio 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013607). Weak-form PDE inference for cancer-cell migration, advection-diffusion-reaction. Uses **legacy FEniCS**, not DOLFINx.

- **Cell-cortex / active gel in FEniCSx** — no maintained package found. Custom-constitutive papers using `dolfinx-external-operator` provide the pattern for active-stress laws.

**Verdict:** the FEniCSx cell-biology ecosystem is sparse but the Rangamani–Dokken axis is the right one to follow. **Contri–Massing–Rangamani 2025 is the scientific North Star.**

---

## 6. Other libraries worth knowing

- **dolfinx-external-operator** ([github.com/a-latyshev/dolfinx-external-operator](https://github.com/a-latyshev/dolfinx-external-operator)) — **v0.10.1 April 2026**, very active. Lets you embed JAX / Numba / NumPy / neural-net constitutive models inside DOLFINx forms with autodiff and JIT. **Important for cell mechanics**: write the active-stress / contractile law in JAX, plug into a UFL form. PyPI-only (not yet on conda-forge — would need to be added as a Pixi PyPI dependency).

- **dolfiny** ([github.com/fenics-dolfiny/dolfiny](https://github.com/fenics-dolfiny/dolfiny)) — convenience wrappers, nonlinear / SQP / optimization helpers. KLAIM 2025 talk on SQP. Python-only, on PyPI. Less critical but pleasant.

- **fenicsx-pctools** ([gitlab.com/rafinex-external-rifle/fenicsx-pctools](https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)) — JORS September 2025. Block PETSc preconditioning for coupled systems (PCD for Navier-Stokes-like blocks). Useful when coupled solves become ill-conditioned.

- **comet-fenicsx** ([bleyerj.github.io/comet-fenicsx](https://bleyerj.github.io/comet-fenicsx)) — Bleyer's tour book; many ready-to-adapt mechanics demos (hyperelasticity, plasticity, beams, periodic homogenization).

- **FESTIM v2.0** ([arXiv 2509.24760](https://arxiv.org/abs/2509.24760)) — modular multi-species / multi-domain hydrogen transport, migrated to DOLFINx in 2025. Reference for code organization in a multi-domain, multi-species setting.

---

## 7. Platform notes (macOS arm64, May 2026)

- `fenics-dolfinx=0.10.0` on conda-forge officially supports `osx-arm64`.
- **Prefer `mpich` over `openmpi`** on Apple Silicon — anecdotally fewer issues. Pin `mpich` and `petsc` to versions resolved against `fenics-dolfinx=0.10`.
- For source-built add-ons (CutFEMx), Apple Silicon clang must match the conda env's `clangxx_osx-arm64` and the conda env's `include` headers.
- VTKHDF mesh I/O is the recommended large-mesh format in 0.10 (faster than XDMF in parallel).

**petsc4py gotchas:**
- DOLFINx 0.10 changed many assembly call sites to expect `EntityMap` rather than raw NumPy arrays — porting older sample code breaks here.
- `PETSc.Options()` is process-global; be careful in test suites.
- Mixing block / nest solver kinds requires consistent IS / preconditioner setup; use `fenicsx-pctools` for nontrivial cases.

**GPU support** is capability-level in 0.10, not turn-key.

---

## Primary sources

- [DOLFINx releases](https://github.com/FEniCS/dolfinx/releases)
- [conda-forge fenics-dolfinx](https://anaconda.org/conda-forge/fenics-dolfinx)
- [DOLFINx 0.10 release notes](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/release_notes.html)
- [FEniCS Discourse: Mesh moving / ALE in DOLFINx](https://fenicsproject.discourse.group/t/mesh-moving-ale-in-dolfinx-example-or-official-api/18323)
- [CutFEMx](https://github.com/sclaus2/CutFEMx)
- [multiphenicsx](https://github.com/multiphenics/multiphenicsx)
- [scifem](https://github.com/scientificcomputing/scifem)
- [dolfinx-external-operator](https://github.com/a-latyshev/dolfinx-external-operator)
- [dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc)
- [PhaseFieldX](https://github.com/CastillonMiguel/phasefieldx)
- [SMART](https://github.com/RangamaniLabUCSD/smart) and [Mem3DG](https://github.com/RangamaniLabUCSD/Mem3DG)
- **Contri, Massing, Rangamani 2025** — [arXiv 2510.23459](https://arxiv.org/abs/2510.23459)
- [Srivastava et al. PLOS Comp Bio 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013607)
- [Border cell phase-field, arXiv 2508.21078](https://arxiv.org/abs/2508.21078)
- [comet-fenicsx](https://github.com/bleyerj/comet-fenicsx)
- [FESTIM v2.0, arXiv 2509.24760](https://arxiv.org/abs/2509.24760)
- [fenicsx-pctools](https://gitlab.com/rafinex-external-rifle/fenicsx-pctools)
- [Dokken EPFL talk Nov 2025](https://memento.epfl.ch/event/multidimensional-modelling-in-dolfinx/)
- [jsdokken.com](https://jsdokken.com/)

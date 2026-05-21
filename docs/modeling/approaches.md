# Modeling Approaches for Moving Membranes with Surface Densities

This document describes the four finite-element approaches under exploration in `vcell-fenics` for representing a deformable cell membrane carrying surface-resident species (receptors, bound nucleators, polarity markers, etc.), coupled to bulk cytosolic / extracellular fields. The goal is to implement and compare multiple approaches on the same canonical problems, not commit to one.

## The canonical problem

A 2D (initially) single cell with:

- **Cell interior** Ω(t) — a deforming bulk domain carrying cytosolic concentration fields *c(x, t)*.
- **Cell membrane** Γ(t) = ∂Ω(t) — a moving 1D curve (in 2D; 2D surface in 3D).
- **Surface densities** ρ(x, t) living on Γ(t) — e.g., receptors, polarity activators, bound cortical components.

### Governing surface PDE

The receptor / activator density ρ on the moving membrane satisfies:

```
∂ₜρ + ρ ∇_Γ · v_Γ = D_Γ Δ_Γ ρ + R(ρ, c|_Γ, …)
```

where:
- `v_Γ` is the membrane velocity,
- `∇_Γ` is the surface (tangential) gradient: `∇_Γ ρ = (I − n ⊗ n) ∇ρ`,
- `Δ_Γ` is the Laplace–Beltrami operator on Γ,
- `R` is the reaction / binding-exchange term coupling to bulk species *c* at the boundary.

The **`ρ ∇_Γ · v_Γ` term is the dilution from membrane stretch**. It is mandatory whenever the membrane moves and is the single most common source of subtle bugs in moving-membrane codes — easy to omit if you start from a static-boundary code and then "make the boundary move." Any solver in this repo, in any approach, must include it explicitly.

### Bulk and mechanics

The interior carries reaction–diffusion–advection of cytosolic fields, membrane mechanics terms (tension, curvature regularization, area / volume constraint, active contractile stress), and possibly a mesh-displacement PDE depending on the approach. These are largely orthogonal to the membrane representation and live in `core/`.

---

## The four approaches

| Approach | Membrane representation | DOLFINx 0.10 status | Best for |
|---|---|---|---|
| **A. ALE explicit membrane** | Marked outer boundary facets of a deforming bulk mesh; ρ is a boundary trace | No first-class API; write your own harmonic-extension mesh motion. Reference: Contri-Massing-Rangamani 2025. | Single / few cells, moderate deformation |
| **B. Separate-mesh / mixed-dimensional** | Submesh with independent surface DOFs, coupled to bulk via mixed-dim assembly | **Native and best-supported.** `create_submesh`, `EntityMap`, `MixedFunctionSpace`, `extract_blocks`, `LinearProblem(kind="block")`. | True membrane-only variables; cleanest semantics. **Start here.** |
| **C. Phase-field / diffuse membrane** | Smooth order parameter ϕ(x, t) per cell; membrane is the implicit transition layer | No FEniCSx-native cell-migration package. Build from Cahn–Hilliard / Allen–Cahn demos. | Multi-cell migration; topology changes; division / contact |
| **D. Trace FEM / cut surface FEM** | Surface moves through a fixed background mesh | **CutFEMx** v0.1.0 (April 2026, MIT). Pinned to DOLFINx 0.9, source-only build. Separate Pixi feature / env. | Strong topology changes; evolving surfaces without remeshing |

See `docs/research/2026-05-21-fenicsx-ecosystem.md` for full library-state details and citations.

### Recommended starting order: B → A → C → D

- **B** is the most-supported in 2026 and produces cleanest semantics for membrane-only variables. Native DOLFINx 0.10 support, no source-built dependencies.
- **A** is the natural follow-up — needed if finite-deformation FSI-style problems become the focus, or for the most direct comparison with VCell's existing moving-boundary solver.
- **C** for multi-cell collective migration; build from Cahn–Hilliard demos.
- **D** requires a separate Pixi environment pinned to DOLFINx 0.9 and a source-built CutFEMx + CutCells stack. Defer until A or B has a working prototype to compare against.

---

## Approach details and tradeoffs

### A. ALE explicit membrane

**Representation.** The cell interior Ω(t) is a deforming bulk mesh. The membrane Γ(t) is the set of outer boundary facets marked at mesh-generation time. ρ is represented as a boundary trace — degrees of freedom on the volumetric function space restricted to the boundary facets.

**Mesh motion.** Solve a mesh-displacement PDE each step (harmonic extension, or linear elasticity with Jacobian-based stiffening to resist tangling). Write the displacement into `mesh.geometry.x`. The Contri–Massing–Rangamani 2025 paper describes a two-step ALE redistribution scheme driven by surface-tangential velocities that maintains element quality without remeshing — use as the reference pattern.

**Strengths.**
- Conceptually closest to a "physical" representation of a cell with a membrane.
- Direct comparison with VCell's existing moving-boundary solver is most natural.
- Bulk and surface fields share a mesh — no inter-mesh coupling code.

**Weaknesses.**
- No `ALE.move()` equivalent in DOLFINx 0.10 — write the mesh-motion solver yourself (~150–300 lines).
- ρ as a boundary trace inherits DOFs from the bulk mesh; for genuinely membrane-resident species, this couples membrane resolution to bulk resolution. Independent surface DOFs (Approach B) avoid this.
- Large deformations break ALE: protrusions, blebs, contact, division need remeshing or a different representation.

### B. Separate-mesh / mixed-dimensional

**Representation.** The bulk Ω(t) is one mesh; the membrane Γ(t) is a codimension-1 submesh of the bulk boundary, with its own independent function space and DOFs for ρ. Bulk and surface are assembled together via DOLFINx 0.10's mixed-dimensional machinery.

**Key API surface (DOLFINx 0.10):**
- `dolfinx.mesh.create_submesh(bulk_mesh, dim=tdim-1, entities=boundary_facet_indices)` returns `(submesh, entity_map, vertex_map, geom_node_map)`.
- `dolfinx.mesh.EntityMap` carries the bidirectional submesh ↔ parent map.
- `ufl.MixedFunctionSpace` to declare a function space mixing bulk and surface spaces.
- `ufl.extract_blocks()` to compose the block form.
- `dolfinx.fem.form(..., entity_maps=[em])` to assemble mixed-dim forms.
- `dolfinx.fem.petsc.LinearProblem(..., kind="block"|"nest")` to solve.

**Strengths.**
- Native, supported by core maintainers (Dokken et al.).
- Independent membrane DOFs — membrane resolution decoupled from bulk.
- Cleanest semantics for genuine membrane-only species.
- Mesh motion is still possible (the submesh moves with the bulk boundary), but the surface PDE is no longer a "trace" — it has its own equations.

**Weaknesses.**
- Newer code paths in DOLFINx 0.10 — expect occasional sharp edges, especially around MPI partitioning of submeshes crossing rank boundaries.
- Coupled bulk-surface solves may need block preconditioning (`fenicsx-pctools` if ill-conditioned).
- The conceptual cost of carrying two related meshes coherent through time integration.

### C. Phase-field / diffuse membrane

**Representation.** Each cell is a smooth order parameter ϕᵢ(x, t) ∈ [0, 1] on a fixed background mesh. The "membrane" is the implicit transition layer where ϕᵢ varies steeply. Membrane-localized quantities are represented either as fields concentrated near the interface or as separate variables weighted by an interfacial delta approximation.

**Strengths.**
- Robust against topology changes — cell division, fusion, contact, separation handled natively.
- Best for collective migration: one ϕᵢ per cell, interactions encoded through free-energy terms.
- No mesh motion, no remeshing.

**Weaknesses.**
- No FEniCSx-native cell-migration package exists in 2026. The canonical Wenzel / Marth / Voigt work is in AMDiS C++. PhaseFieldX (the only maintained FEniCSx phase-field package) is fracture-only.
- Diffuse membrane is not a sharp interface; "surface density ρ" requires care in how it's defined (delta-function weighted bulk field vs. genuine surface DOFs via additional mechanism).
- Membrane mechanics (tension, bending, active stress) require careful asymptotic matching to recover the sharp-interface physics in the thin-interface limit.

### D. Trace FEM / cut surface FEM

**Representation.** The membrane Γ(t) is described implicitly by a level-set function φ(x, t) ≤ 0 = Γ. The bulk and surface PDEs are integrated over the parts of the *fixed background mesh* that are cut by the level set. Quadrature rules and integration domains are regenerated each time step.

**Strengths.**
- No mesh motion — the background mesh is fixed.
- Topology changes are handled naturally (level set can split / merge).
- Independent surface DOFs are realized via a "ghost penalty" or stabilization mechanism on the cut cells.

**Weaknesses.**
- CutFEMx is young (v0.1.0, April 2026); API may change.
- Pinned to DOLFINx 0.9 — version-pin conflict with the rest of the stack; needs a separate Pixi environment.
- Source-only build, depends on a companion library `CutCells`.
- Conditioning of the cut-cell system is delicate; stabilization choices matter.

---

## Architecture for supporting multiple approaches

Per the research, a sensible code separation is:

```
src/vcell_fenics/
  core/                       # approach-agnostic
    biochemistry.py           # surface ρ RHS: surface Laplacian + reaction + dilution
    mechanics/                # constitutive laws
    time_integrators.py
    geometry.py
    io.py
  approaches/
    ale/                      # A
    submesh/                  # B  <- start here
    phase_field/              # C
    cut_fem/                  # D  <- separate Pixi feature, DOLFINx 0.9
  benchmarks/                 # identical canonical problems run across approaches
  pyvcell_bridge/             # eventual integration with VCell's pyvcell project
  tests/
```

**Shared abstractions worth investing in:**

- `BiochemistryRHS` — a callable returning a UFL form for the surface PDE RHS, taking ρ and the surface measure. All four approaches use the same expression for the receptor-density physics.
- `MechanicsModel` — returns a strain-energy density. Used by A and B; phase-field and cut-FEM have different mechanics coupling but the constitutive laws should be shareable.
- `Geometry` — adapter exposing "the membrane" in each approach's native form (marked facets / submesh / level set / ϕ level set), so the biochemistry code doesn't need to know which approach it's running in.
- `TimeStepper` — exposes `step(state, dt)` regardless of approach.
- **Common benchmark suite** — identical initial geometry, run all four. This is also how divergences between approaches are discovered.

Precedent codebases: **FESTIM v2.0** (DOLFINx-migrated 2025) is the closest example of modular multi-domain / multi-species code structure. **SMART** (legacy FEniCS) has the clearest model DSL for compartments-and-reactions, but is not a code-architecture reference for FEniCSx.

---

## Key external references

- **Contri, Massing, Rangamani (2025)** — *"A Finite Element framework for bulk-surface coupled PDEs to solve moving boundary problems in biophysics"* (arXiv 2510.23459). The scientific North Star.
- **SMART** ([RangamaniLabUCSD/smart](https://github.com/RangamaniLabUCSD/smart)) — design reference for the model DSL only (still legacy FEniCS 2019).
- **DOLFINx 0.10 release notes** ([docs.fenicsproject.org/dolfinx/v0.10.0/python/release_notes.html](https://docs.fenicsproject.org/dolfinx/v0.10.0/python/release_notes.html)) — mixed-dimensional API.
- **CutFEMx** ([github.com/sclaus2/CutFEMx](https://github.com/sclaus2/CutFEMx)) — for Approach D.
- **FEniCS Discourse — Mesh moving / ALE in DOLFINx** ([fenicsproject.discourse.group/t/.../18323](https://fenicsproject.discourse.group/t/mesh-moving-ale-in-dolfinx-example-or-official-api/18323)) — canonical answer for Approach A's mesh-motion pattern.

Full library state and additional references in `docs/research/2026-05-21-fenicsx-ecosystem.md`.

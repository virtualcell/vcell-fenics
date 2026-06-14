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

**Mesh motion.** Solve a mesh-displacement PDE each step (harmonic extension, or linear elasticity with Jacobian-based stiffening to resist tangling). Write the displacement into `mesh.geometry.x`. The Contri–Massing–Rangamani 2025 paper describes a two-step ALE redistribution scheme driven by surface-tangential velocities that maintains element quality without remeshing — use as the reference pattern. *Implemented:* the harmonic-extension variant lives in `backend/discrete.py` (`_HarmonicExtension`); `_MeshMotion` applies it to any moving codim-0 (bulk) subdomain — prescribed velocity sets the boundary displacement, ∇²d=0 fills the interior, `geometry.x += d` — so interior nodes follow the boundary smoothly and interior singularities of the velocity expression (e.g. `x/r(x)` at the centre) are sidestepped. Linear-elasticity / Jacobian-stiffened variants and the Contri two-step redistribution remain future options.

**Strengths.**
- Conceptually closest to a "physical" representation of a cell with a membrane.
- Direct comparison with VCell's existing moving-boundary solver is most natural.
- Bulk and surface fields share a mesh — no inter-mesh coupling code.

**Weaknesses.**
- No `ALE.move()` equivalent in DOLFINx 0.10 — write the mesh-motion solver yourself (~150–300 lines).
- ρ as a boundary trace inherits DOFs from the bulk mesh; for genuinely membrane-resident species, this couples membrane resolution to bulk resolution. Independent surface DOFs (Approach B) avoid this.
- Large deformations break ALE: protrusions, blebs, contact, division need remeshing or a different representation.
- **Conservative field transfer on remeshing is the hard part, not the remeshing itself.** Node displacement preserves the material identity of surface elements, so ρ rides along for free — *until* you remesh, at which point ρ must be mapped from the old surface mesh to the new one without creating or destroying mass (∫_Γ ρ must be preserved to the dilution-balance tolerance). The front-tracking / FV literature treats this as a first-class problem with a standard menu of strategies — Lagrangian material-element tracking, space-time swept-volume control volumes, overlap-based conservative remapping (geometric intersection → sparse transfer matrix), or Eulerian narrow-band transport with a mass-correction step. The current backend sidesteps it by refusing to remesh (raises `MeshQualityError`); when remeshing lands this is the design decision to get right. The algorithm for ρ specifically is sketched in `docs/modeling/conservative-surface-remap.md` (including why a conservative *bulk* remap does not conserve *surface* mass — an Approach-A–specific catch), and the loop that would drive remesh-and-continue is sketched in `docs/modeling/ale-remesh-driver.md`. Background: `docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md`.

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
- **Cut geometry is regenerated per step, but the cut library gives you no temporal lineage.** CutFEMx / CutCells produce the cut subcells and interface facets at *each* time independently — they do not maintain a map from an old interface fragment to the new fragments it became (overlap / sweep / split / merge). For a conserved surface density that map is exactly what a conservative transfer needs, so as with Approach A's remeshing, the conservative-remap layer is yours to build (overlap intersection, space-time swept volumes, or Eulerian transport with correction). The implicit-interface representation removes mesh *motion*, not the conservation bookkeeping. See `docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md`.

---

## Architecture for supporting multiple approaches

Per the research, a sensible code separation is:

```
src/vcell_fenics/
  core/                       # approach-agnostic  [exists]
    surface_remap.py          #   conservative remap kernel (pure NumPy)        [done]
    surface_remap_mesh.py     #   DOLFINx Function bridge for membranes         [done]
    surface_remap_trace.py    #   Approach-A bulk-trace correction              [done]
    bulk_remap.py             #   conservative bulk (2D area) remap kernel      [done]
    bulk_remap_mesh.py        #   DOLFINx Function bridge for bulk fields       [done]
    region_remesh.py          #   gmsh region remesher (polyline -> fresh mesh) [done]
    bgn_curve.py              #   BGN tangential redistribution (curvature flow) [done]
    bgn_curve_mesh.py         #   DOLFINx Mesh bridge for BGN redistribution    [done]
    biochemistry.py           #   surface ρ RHS: surface Laplacian + reaction + dilution  [planned]
    mechanics/                #   constitutive laws                            [planned]
    time_integrators.py       #   [planned]
    geometry.py               #   [planned]
    io.py                     #   [planned]
  approaches/
    ale/                      # A
    submesh/                  # B  <- start here
    phase_field/              # C
    cut_fem/                  # D  <- separate Pixi feature, DOLFINx 0.9
  benchmarks/                 # identical canonical problems run across approaches
  pyvcell_bridge/             # eventual integration with VCell's pyvcell project
  tests/
```

`core/` now exists; its first occupants are the conservative surface-density remap (below), not the `biochemistry.py` the original sketch imagined — that primitive was forced first by the remesh problem, and is orthogonal to the formalism backend that holds the receptor-density physics today.

**The conservative surface remap (implemented).** Three composable pieces realizing `docs/modeling/conservative-surface-remap.md`; together they carry a surface density ρ from one membrane discretization to another while preserving ∫_Γ ρ ds. Used by Approach A on remesh (and the eventual B/D remeshing paths):

- `surface_remap.py` — the **kernel**. Pure NumPy, no DOLFINx: `arclength_parameterization` lifts an ordered polyline onto its arc-length coordinate; `supermesh_remap_1d` merges old+new node sets into a supermesh and returns `ρ_new = M⁻¹ B ρ_old`, conserving total mass by the partition-of-unity argument; `project_points_to_polyline_arclength` is the closest-point step that puts two distinct discretizations in a common frame. Isolated from DOLFINx so conservation is provable and testable on its own.
- `surface_remap_mesh.py` — the **DOLFINx bridge**. `ordered_membrane_loop(V)` orders a closed P1 membrane's dofs into a loop by walking the cell→dof edge list; `remap_surface_function(u_old, V_new, conserve=True)` reads ρ off `u_old`, remaps in the old mesh's arc-length frame, writes ρ_new into a new `Function`, and (with `conserve=True`) rescales so the surface integral on the new mesh equals the old exactly.
- `surface_remap_trace.py` — the **Approach-A trace correction**. Because ρ in A is a bulk *trace*, a conservative bulk (volume) remap does not conserve the surface integral. `BulkBoundaryTrace` maps bulk-boundary DOFs ↔ a boundary surface space; `correct_surface_trace(u_old, u_new)` gathers the old trace, surface-remaps it, and scatters the result over the new bulk function's boundary DOFs (interior untouched). With independent surface DOFs (Approach B) this step is unnecessary.

  *Scope:* serial, P1, single closed 2D membrane. Deferred: MPI/multi-rank, higher-order spaces, open arcs, P0 variant, 3D triangle-surface supermesh, and the ALE remesh *driver* that would call `correct_surface_trace` (depends on Approach A mesh-motion-with-remeshing, not yet built).

**The conservative bulk remap (implemented).** The area sibling of the surface remap — it carries a P1 cytosolic field *c* from one 2D triangulation to another while preserving ∫_Ω c dx. This is the `transfer_bulk` prerequisite the ALE remesh driver sketch (`docs/modeling/ale-remesh-driver.md`) names on its critical path:

- `bulk_remap.py` — the **kernel**. Pure NumPy + scipy.sparse, no DOLFINx: `supermesh_project_2d(old_verts, old_tris, c_old, new_verts, new_tris)` builds the supermesh by clipping each new triangle against bbox-overlapping old triangles (Sutherland–Hodgman), integrates the P1×P1 products with a degree-2 edge-midpoint rule, and returns `c_new = M⁻¹ B c_old` (M = true new-mesh mass matrix, B = mixed mass matrix). Conservation is structural (partition of unity), exact to round-off when the two meshes triangulate the same polygon. Isolated from DOLFINx so it can be tested on plain arrays.
- `bulk_remap_mesh.py` — the **DOLFINx bridge**. Much simpler than the surface bridge: no loop-ordering, because for a P1 space on a triangle mesh the dof index *is* the vertex-array row and `V.dofmap.list` *is* the (n_cells, 3) triangle list. `remap_bulk_function(u_old, V_new, conserve=True)` reads `c_old` and `(verts, tris)` straight off `u_old`'s mesh, runs the kernel, writes `c_new` into a new `Function` on `V_new`, and (with `conserve=True`) rescales so the volume integral on the new mesh equals the old exactly — closing the geometric gap when the two meshes approximate the same domain (e.g. a disk) at different resolutions.

  *Scope:* serial, P1, 2D triangle mesh. Deferred: MPI/multi-rank, higher-order spaces, 3D tetrahedra, broad-phase acceleration for large meshes.

**The region remesher (implemented).** `region_remesh.py` — `mesh_region(loop, h)` drives gmsh (geo kernel: one point per loop vertex, straight segments, one plane surface) to produce a fresh uniform-quality 2D mesh of the region a closed polyline encloses. This is step (b) of the ALE remesh routine (`docs/modeling/ale-remesh-driver.md`) — meshing an *arbitrary deformed* boundary, not just the analytic disk the geometry helpers build. Because the boundary segments are straight, any nodes gmsh inserts along them stay on the polyline, so the meshed region is exactly the input polygon and its area is preserved to round-off. `fix_boundary_nodes=True` forces exactly the input vertices onto the boundary (Γ_new ⊂ Γ_old) — the interior-only fast path that lets `correct_surface_trace` be skipped (subtlety 3 of the driver sketch). `h` is the authoritative uniform size (gmsh's extend-from-boundary / from-points / from-curvature sizing is disabled, so a deformed boundary's non-uniform spacing is not inherited). The deformed loop itself is recovered from a live mesh via `BulkBoundaryTrace.boundary_loop()`.

  *Scope:* serial, 2D, a single simple closed loop; the caller owns the self-intersection / pinch-off guard. Deferred: holes / multiple loops, 3D, MPI.

**BGN tangential mesh redistribution (implemented).** `bgn_curve.py` — `bgn_curvature_flow_step(points, mobility=m, dt=dt)` advances a closed membrane polyline by one semi-implicit **Barrett–Garcke–Nürnberg** step of mean-curvature flow (V = −m κ from the surface-tension force balance η v + σ H n = 0, mobility m = σ/η). The defining feature is *intrinsic tangential redistribution*: the new positions and the curvature are solved **together**, with only each node's *normal* velocity pinned to the physical law, so the *tangential* node motion is free and is set by the discrete curvature-vector identity (the same ∫κ·φ = ∫∇_Γ X : ∇_Γ φ projection the curvature-force path uses) to keep nodes asymptotically equidistributed. Because the geometry (normals, the 1D Laplace–Beltrami stiffness, the lumped mass) is taken from the old polyline, the (position, curvature) system is **linear** — semi-implicit and unconditionally stable. This is the missing piece that lets curvature flow run at usable `dt`: pure normal motion crowds nodes at high-curvature tips until the mesh tangles (the `MeshQualityError` the unknown-motion curvature path hits), whereas BGN holds the edge-length ratio bounded. Verified on geometry alone (pure NumPy): a circle shrinks per the exact `r² = r₀² − 2 m t` and stays round; *any* convex curve loses area at the constant curve-shortening rate `dA/dt = −2π m`; and on an ellipse the BGN edge ratio improves where a naive normal-only step bunches it an order of magnitude worse.

  *Scope:* serial, P1, a single closed 2D membrane polyline; mean-curvature (surface-tension) flow specifically.

  - `bgn_curve_mesh.py` — the **DOLFINx Mesh bridge** (implemented). `bgn_redistribute_membrane(mesh, mobility=m, dt=dt)` orders the membrane loop (reusing `ordered_membrane_loop`), runs the kernel, and writes the new positions back into `mesh.geometry.x` in place — scattering the per-dof displacement onto geometry rows by the same closest-point (`cKDTree`) match `_MeshMotion` uses, since dof order ≠ geometry-node order. The **backend wiring** lives in `unknown_motion.assemble_unknown_motion(..., redistribute=True)`: for a *motion-only* curvature membrane it replaces the force-balance velocity solve + normal mesh move with a BGN step, running curvature flow at a `dt` several× past where the velocity path tangles. The BGN mobility `m = σ/η` is **calibrated from the force balance** (`v = −m κ⃗` nodally, so `m = −(v·κ⃗)/(κ⃗·κ⃗)` off one solve) rather than parsed out of the force expression — and because redistribution is a pure discretization choice (the continuous solution is unchanged), it is a backend flag, not a MathDescription field.

  **Redistribution with a co-moving receptor (implemented).** Once the mesh slides tangentially, mesh velocity ≠ material velocity, so a surface species picks up the ALE advection term `−(v_mesh − v_material)·∇_Γ ρ`. Rather than assemble that term — whose discrete mass conservation is delicate — `unknown_motion._BGNReceptorMotion` **decomposes** each step into the two motions BGN superposes, each carried by the machinery that conserves it: the **normal** displacement (the physical curvature flow) drives the existing dilution scheme (mass conserved under area change), and the **tangential** displacement (the re-noding) carries ρ by a conservative surface remap on the now-fixed curve (mass conserved under re-noding — the remap *is* the discrete ALE advection). Critically, the normal substep uses BGN's *normal* part, not the explicit force-balance velocity, so it stays stable at the large `dt` BGN allows. Verified: ∫_Γ ρ ds invariant (rel 5e-3) under a redistributing flow with a non-uniform ρ at a `dt` where the non-redistribute path tangles; a negative control (skip the remap) leaks ~3% of the mass.

  **Deferred:** multiple surface species and bulk-coupled species on the redistributing membrane (the general reaction-diffusion-advection mix — the decomposition generalises, but the multi-species vector-space plumbing and the bulk ALE terms are separate increments); general (non-curvature) redistribution; open arcs; 3D surfaces.

**Lagrangian surface vs Eulerian volume (a design note for moving multi-compartment coupling).** The surface and the volume need *different* ALE treatments, and conflating them is the central trap when extending to "arbitrary volume + membrane species on a moving membrane":

- **The membrane is Lagrangian.** It has material points (the lipids/receptors that *are* the membrane), so a membrane node has a real material velocity and "carrying ρ with the node" is physical. This is why the surface dilution `ρ ∇_Γ·v_Γ` and the conservative re-noding remap (above) are correct: re-noding transfers a density between two discretizations of the same *material* curve.
- **The volume is Eulerian — there are no material points.** A diffusing cytosolic/extracellular species lives in a lab-frame field; when the boundary moves, the mesh moves too, but the **mesh velocity `w` is geometric bookkeeping** (the harmonic extension of the boundary motion, `_HarmonicExtension`), *not* a material velocity. An interior mesh node tracks no material point. The general ALE volume law is `∂c/∂t|_mesh + (u − w)·∇c = D∇²c + R`, where `u` is the medium's *own* advective velocity (`0` for a still fluid, or a real flow field) — the convective term uses the **relative** velocity `u − w`.
- **`coupled.py`'s moving-membrane bulk dilution `L ∇·v_mesh` is the special case `u = w`** (the medium co-moves with the mesh — valid for an incompressible co-moving cytoplasm or a closed translating/deforming cell, where the impermeable membrane forces the interior to co-move). For a genuinely Eulerian volume (`u ≠ w`, e.g. a cell migrating through a static extracellular medium) the `(u − w)·∇c` term is live and the co-moving dilution is wrong. The conservative *bulk* remap (`bulk_remap`) enters only on a discrete *remesh*, not during smooth motion.
- **A rigid-translation test discriminates the two regimes** and is the cheapest known-answer check: a closed cell translating at constant velocity has `u = w` (the co-moving baseline — both the membrane and the co-moving bulk transport rigidly with `∇·v = 0`, no spurious dilution, profile carried exactly; pinned by `test_translation_transports_rigidly_with_no_spurious_dilution` and `test_translation_transports_a_comoving_bulk_rigidly`), whereas a lab-frame field that must stay put while the mesh sweeps through it exercises the `(u − w)·∇c` term (the Eulerian case, not yet implemented).

**Shared abstractions worth investing in:**

- `SurfaceRemap` — **implemented** as the three modules above; the approach-agnostic primitive every remeshing path calls.
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

### Related finite-volume / front-tracking work (cross-method reference)

These are **FV / OpenFOAM / C++** codes, not FEniCSx — not integration candidates, but conceptual references for moving-surface transport and comparison baselines. Surfactant transport on a moving interface is mathematically the same object as this project's canonical surface PDE (surface advection–diffusion + dilution from area change + bulk–surface exchange), so the FV treatment of the geometric conservation law is directly relevant to getting `ρ ∇_Γ · v_Γ` right. Full notes and provenance in `docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md`.

- **`../vcell-mbsolver`** — VCell's moving-boundary solver, recently extracted into its own repo (was in the monorepo). It **bundles FronTier** (`FronTierLib/`) — i.e. the comparison baseline named throughout this doc is a FronTier-based front-tracking FV code — and now builds a `pybind11` Python module (`vcellmbsolver_py`), so it is callable from Python for side-by-side comparison. It implements an **overlap-based conservative remap** (Clipper polygon clipping) on a Voronoi mesh; see the subsection below.
- **FronTier / FronTier++** — the classic Stony Brook front-tracking library (the lineage of the baseline above). Historically important; no modern standalone release. Recent derivative activity is mainly AMReX-coupled AMR around the existing front-tracking core.
- **twoPhaseInterTrackFoam** — OpenFOAM ALE interface-tracking module *with surfactants*; the closest FV cousin to the moving-surface conservative-transport problem here (FV analogue of Approach A). Strongest external reference for the dilution / GCL treatment.
- **cfdmfFTFoam** — OpenFOAM front-tracking solver (explicit Lagrangian front mesh + Eulerian grid, remeshing, volume correction); closest architectural cousin to classic FronTier-style front tracking.
- **LENT / lentFoam**, **PARIS** — hybrid level-set/front-tracking and structured-grid FT/VOF reference implementations, respectively.

Note: explicit front tracking (a Lagrangian surface mesh advected and remeshed independently of any bulk) is effectively a fifth representation, not in the A–D taxonomy above — closest to Approach B but FV-flavored and reliant on conservative remap rather than FEM trace/mixed-dim coupling.

#### How the vcell-mbsolver baseline conserves mass (verified by reading the source, 2026-06-06)

Worth understanding in detail, because it is a *working reference implementation* of the overlap-based conservative remap that this project's Approach A (on remesh) and Approach D (per step) will eventually need. **Correction to an earlier note:** `matlab/clipperLink/` is only a MATLAB MEX binding to the Clipper library — the remap itself lives in the C++ solver core (`Solver/src/intersection.cpp`, `Solver/src/clipper.cpp`, `Solver/src/MeshElementNode.cpp`).

- **Geometry.** A Voronoi mesh of control volumes (`VoronoiMesh.cpp`); the cell boundary is a FronTier-tracked front (`VCellFront.cpp`). A boundary control volume is the Voronoi cell *clipped to the inside of the front* — `formBoundaryPolygon()` computes `vol = voronoiVolume.intersection(front)` (`MeshElementNode.cpp`). Each element stores `amtMass` (the conserved quantity), `amtMassTransient` (working copy), and `concValue = amtMass / volume`.
- **Per-step sequence** (`MovingBoundaryParabolicProblem.cpp`): front propagates → diffusion–advection + reaction solve on the *old* geometry → `voronoiMesh.adjustNodes()` rebuilds control volumes against the *new* front → `collectMassFromNeighbors` + `distributeMassToNeighbors` do the remap → `EndCycle` sets `conc = mass / new_volume`.
- **The remap is overlap-area-weighted, not area-ratio rescaling.** Only the boundary band — cells that change state (inside↔boundary↔outside) as the front sweeps — is remapped; interior cells keep fixed Voronoi geometry and transfer nothing. A new inside cell collects from each old neighbor at the neighbor's concentration: `mu = nb.amtMass / donorVolume; m = mu * intersectVolume` (`MeshElementNode.cpp`, `collectMassFromNeighbors`). A cell going outside distributes its mass by overlap fraction: `m = iVol * amtMassTransient / volumeValue` (`distributeMassToNeighbors`). Both directions debit the donor and credit the recipient by the same `m`, so mass is *moved*, not scaled. The overlap area comes straight from `ourVolume.intersection(nb.getControlVolume())` (Clipper `ctIntersection`). In matrix terms this is the `M_new[j] = Σ_i T[j,i] M_old[i]` remap with `T[j,i] = overlap_area(j,i) / vol_old(i)`, assembled cell-by-cell rather than as an explicit sparse matrix. A runtime assertion (`ReportClient.cpp`) throws if total mass drifts more than `1e-3` relative.

**The design contrast that matters for us.** The mbsolver has **no dilution term at all** — because it re-clips control volumes against the moved front and conservatively remaps, the area change *is* the geometry update and dilution falls out for free. This project's FEniCSx backend takes the opposite, mathematically-equivalent stance: keep the mesh topology fixed and put the stretch into an explicit `ρ ∇_Γ · v_Γ` term in the weak form (discrete-exact-by-construction vs. O(dt) backward-Euler balance). The two stances coincide only while no remeshing happens; **the moment a FEniCSx approach must remesh, the explicit term no longer suffices and the overlap remap above is what moves ρ from the old mesh to the new.** One caveat porting it: the mbsolver remaps *bulk* concentration on a 2D area mesh, whereas this project's hard case is *surface* density on a codim-1 (1D) membrane — the overlap idea ports directly (overlap of old vs new membrane *segments*), but on a curve, so Clipper itself is not the tool; the algorithm is. The 1D surface version is sketched in `docs/modeling/conservative-surface-remap.md`.

Full library state and additional references in `docs/research/2026-05-21-fenicsx-ecosystem.md`.

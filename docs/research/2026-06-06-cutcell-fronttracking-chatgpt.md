# Cut-cell & front-tracking methods for moving interfaces — ChatGPT session

> **Provenance.** Verbatim capture of a ChatGPT conversation, lightly reformatted
> into Markdown (speaker labels, fenced blocks). Captured **2026-06-06**. This is
> an *external source*, not project analysis — it records claims and library
> pointers as ChatGPT stated them, including ones we have **not** independently
> verified. Treat library names, dates, and feature claims as leads to check, not
> facts. For how these ideas relate to this project's approach taxonomy, see
> `docs/modeling/approaches.md` and any follow-up notes we write.
>
> The session opens mid-conversation: the user had asked whether the **CutCells**
> library maintains temporal interface tracking (intergenerational facet lineage).
> The first block is ChatGPT's answer to that; later turns pivot to **FronTier**
> and modern front-tracking alternatives.

---

## 1. Does CutCells do temporal interface tracking?

*(ChatGPT)*

> I'll verify the library's stated data model rather than infer too much from the
> phrase "cut-cell," because temporal tracking is a very specific capability.

Probably no, not in the sense you mean.

From the README, CutCells is primarily a per-time/per-geometry cut decomposition
tool. Given a parent cell or mesh and vertex level-set values, it builds a local
submesh for φ < 0, φ > 0, and/or the interface φ = 0, with connectivity, element
types, parent IDs, and provenance tags. Its advertised purpose is runtime
quadrature / interface extraction for CutFEM or unfitted FEM, not temporal surface
tracking.

So it likely gives you things like:

```
at time n:
  background cell K_i
  level set φⁿ
  cut pieces K_iⁿ, interface facets Γ_iⁿ

at time n+1:
  background cell K_i
  level set φⁿ⁺¹
  cut pieces K_iⁿ⁺¹, interface facets Γ_iⁿ⁺¹
```

But it does not appear to maintain a lineage graph such as:

```
old interface facet A at tⁿ
    overlaps / sweeps / maps to
new interface facets B, C, D at tⁿ⁺¹
```

That distinction matters for conservative transport. For finite-volume-style mass
reassignment on an evolving membrane/surface, you usually need one of these
additional layers:

1. **ALE / Lagrangian surface mesh tracking** — keep explicit surface mesh
   nodes/elements and move them. Then mass can stay attached to material surface
   elements, with remeshing plus conservative interpolation when needed.
2. **Space-time cut-cell / swept-volume construction** — treat the interface
   evolution from tⁿ to tⁿ⁺¹ as a geometry in d+1 dimensions. This is closer to a
   conservative finite-volume treatment because you can compute fluxes through
   space-time control-volume faces.
3. **Overlap-based conservative remapping** — independently cut the mesh at tⁿ and
   tⁿ⁺¹, then compute geometric intersections/overlaps between old and new surface
   or volume fragments. This produces a sparse transfer matrix:

   ```
   M_new[j] = Σ_i T[j,i] M_old[i]
   ```
4. **Eulerian level-set transport with correction** — store surface density in an
   Eulerian/narrow-band field and evolve it with a PDE, with reinitialization and
   possible mass correction. This avoids explicit facet lineage but needs careful
   conservation control.

CutCells could help with one piece of this: generating the cut subcells/interface
facets at each time step. But the conservative transfer machinery would be your
responsibility.

The related CutFEMx project does mention a "Moving Domain" demo that updates
quadrature rules and integration domains dynamically at each time step, but that is
still described as updating cut geometry for a time-dependent problem, not as
maintaining intergenerational facet relationships or conservative mass remapping.

For your "surface densities on moving membranes" use case, I would treat CutCells as
a possible geometry/quadrature backend, not a full evolving-surface finite-volume
framework.

A reasonable architecture would be:

```
level set φⁿ, φⁿ⁺¹
    ↓
CutCells: cut cells and interface facets at each time
    ↓
custom mapper:
    old surface fragments ↔ new surface fragments
    overlap / closest-point / normal-flow / space-time swept relation
    ↓
conservative transfer matrix
    ↓
surface PDE / membrane mechanics / bulk coupling
```

The hard part is the middle: defining "same material patch" when the surface moves,
splits, merges, or passes through background cells. That is beyond what CutCells
appears to provide.

---

## 2. FronTier — recent updates, derivatives, similar packages?

**User:** There was a c++ package FronTier which we used in our 2D FV code. Has
there been any recent updates or derivative work or similar packages?

*(ChatGPT)*

> I'll look for the current status of FronTier/front tracking work, then compare it
> with modern C++/C/Python-adjacent options for conservative moving-interface
> finite-volume workflows.

Yes — but the answer splits into FronTier itself, recent derivative work, and modern
alternatives.

### FronTier itself

The original Stony Brook FronTier / FronTier++ code still has public GitHub mirrors,
but it does not look like a modern actively released package. The
`frontier-sunysb/FronTier` repo shows 476 commits, GPL-2.0, no published releases,
and the language mix is mostly C, with only a little C++.

The Stony Brook GitHub org also has a small `frontier-apps` repo described as
applications using the FronTier library, but again this looks more like a
research-code archive than an actively productized C++ library.

So: I would not expect recent polished FronTier++ releases, package-manager support,
CMake modernization, or a clean library API unless a particular group has a private
branch.

### Recent FronTier-related work

There is evidence of recent **AMReX + FronTier** work. A 2023 University of Arkansas
seminar abstract describes using the front-tracking FronTier library for
Rayleigh–Taylor / Richtmyer–Meshkov-type instability simulations, and specifically
says they were implementing AMReX block-structured AMR to reduce cost and improve
resolution in complex mixing regions.

A 2024 ICERM poster abstract says essentially the same thing: FronTier has been used
for validation/verification of turbulence mixing from hydrodynamic instabilities, and
a Rayleigh–Taylor test case was being used to combine FronTier with AMReX for
adaptive refinement.

That sounds like the most relevant "derivative" line if you used FronTier before: not
a replacement API, but modernizing the Eulerian grid/AMR side around the old
front-tracking machinery.

### Similar packages / methods worth looking at

1. **cfdmfFTFoam — new OpenFOAM front-tracking solver.** Probably the most directly
   relevant recent open-source item. A 2026 front-tracking method package for
   OpenFOAM-9, written in C++, for two-phase and three-phase flow problems. It tracks
   interfaces as surface triangulated meshes interacting with a general unstructured
   Eulerian grid. The README says it was extended from or inspired by Ftc3D and
   OpenFOAM v9. The 2026 arXiv abstract says it supports general unstructured grids,
   serial and parallel runs, front-mesh-to-Eulerian-grid communication, front node
   advection, volume correction, remeshing, surface tension computation, and
   indicator-function construction. Closer to old 2D FV usage than CutCells, because
   it is explicitly about front tracking + finite volume multiphase flow, not just
   per-cell cut geometry.

2. **twoPhaseInterTrackFoam — ALE interface tracking with surfactants.** A 2024 paper
   describes it as an open-source C++ OpenFOAM module implementing unstructured
   finite-volume ALE interface tracking for incompressible immiscible two-phase flow,
   with surfactant transport and subgrid-scale modeling. Especially interesting for
   "surface mass on moving interfaces": the paper explicitly includes finite-area
   discretization of surfactant transport on a moving surface, bulk-interface
   transfer, moving control volumes, and geometric conservation-law concerns. More an
   ALE moving-mesh sharp-interface FV method than FronTier-style Lagrangian front
   tracking, but conceptually very close to "conservative transport on an evolving
   interface."

3. **LENT / lentFoam — hybrid Level Set / Front Tracking in OpenFOAM.** The
   `CRC-1194/lent` repository describes itself as an actively developed research code
   implementing a hybrid Level Set / Front Tracking method in OpenFOAM for DNS of
   two-phase flows using collocated unstructured finite-volume discretization.
   Attractive for some geometric sharpness of front tracking with level-set-style
   robustness.

4. **PARIS — robust FT/VOF multiphase simulator, but Fortran.** A finite-volume code
   for immiscible multiphase flow using either Front Tracking or Volume of Fluid,
   written in Fortran95/2002, MPI-parallel. Based on earlier Ftc3D, SURFER, and Gerris
   code lines. Worth knowing for algorithms/validation; less convenient to integrate
   into a C++ codebase.

### Practical read

| Need | Best candidate |
|---|---|
| Closest modern FronTier-like front tracking in C++ FV | cfdmfFTFoam |
| Conservative moving-surface transport / surfactant-like surface PDE | twoPhaseInterTrackFoam |
| Hybrid level-set/front-tracking research code | LENT / lentFoam |
| Structured-grid FT/VOF reference implementation | PARIS |
| Legacy front tracking with topology handling | FronTier, possibly with AMReX forks |

For successive generations of an evolving surface and conservative reassignment of
surface mass, look hardest at **twoPhaseInterTrackFoam** and **cfdmfFTFoam**. FronTier
remains historically important, but the most active modern ecosystem appears to have
shifted toward OpenFOAM-based C++ modules and AMReX-coupled research efforts, rather
than a refreshed standalone FronTier++ library.

---

## 3. Expand on the two suggestions

**User:** Please expand upon your two suggestions.

*(ChatGPT)*

> I'll expand the two most relevant options as engineering choices: what each method
> represents, what it likely gives you out of the box, and where it would or would not
> fit your old 2D finite-volume/front-tracking needs.

The two to expand:

- **cfdmfFTFoam** — closest to a modern FronTier-like front-tracking solver in
  C++/OpenFOAM.
- **twoPhaseInterTrackFoam** — more interesting if your central problem is
  conservative transport of mass/surfactant on a moving interface.

They are related but philosophically different.

### 3.1 cfdmfFTFoam: closer to classic front tracking

A 2026 OpenFOAM-based front-tracking method package for multiphase flow on general
unstructured grids. The author describes it as integrating the older Ftc3D
front-tracking code into OpenFOAM, adding front-mesh-to-Eulerian-grid communication,
front-node advection, parallel support, volume correction, remeshing, surface-tension
computation, and indicator-function construction.

Conceptually closest to FronTier:

```
Eulerian finite-volume grid
        +
explicit Lagrangian front mesh
        ↓
front moves through background grid
        ↓
interface geometry is reconstructed / maintained explicitly
        ↓
fluid equations solved on Eulerian grid with interface coupling
```

The key feature: the interface is a **separate tracked mesh**, not merely a level-set
contour regenerated from scratch each timestep. Naturally suited to questions like:
where did this interface element go? which front nodes moved where? how should
interface geometry be remeshed? how do we correct enclosed volume? how do we compute
surface tension from the front?

**What it likely gives you:** explicit front mesh; front-node advection; coupling
between front mesh and Eulerian OpenFOAM grid; remeshing; volume correction;
surface-tension computation; indicator-function construction; serial and parallel
support; use on general unstructured OpenFOAM grids. The "volume correction" and
"remeshing" parts are especially relevant because classic front tracking suffers from
exactly those geometric-maintenance issues.

**What not to assume:** that it gives a general-purpose conservative surface-density
remapping framework. It probably maintains enough front connectivity to advect and
remesh the interface, but that is not the same as exposing a clean API like:

```
old_front_elements -> new_front_elements
conservative_transfer_matrix(old_surface_density, new_surface_density)
```

For cell-biological membrane transport this distinction matters. A front-tracking
solver may preserve interface mesh identity over many steps, but after remeshing you
still need a conservative remap of surface quantities. Questions to ask: does it
support scalar fields living on front nodes/faces? when remeshing happens, are front
fields conservatively transferred? does it expose old/new interface correspondence?
can I attach arbitrary surface species to the interface? can I replace the
multiphase-flow physics with my own membrane/bulk coupling?

**Good fit if** your problem is: moving explicit interface; bulk FV equations; surface
tension/curvature/membrane force; need front remeshing; need Eulerian-grid coupling;
want C++ infrastructure. **Less ideal if:** surface reaction-diffusion on a cell
membrane; bulk-surface biochemical coupling; many membrane-bound species; complex
membrane mechanics; topological changes with conservative species transfer. Adaptable,
but research-code work rather than "use package."

### 3.2 twoPhaseInterTrackFoam: stronger for conservative moving-surface transport

An OpenFOAM module for unstructured finite-volume arbitrary Lagrangian/Eulerian
interface tracking with surfactants and subgrid-scale modeling. Implements an ALE
interface-tracking method for incompressible, immiscible two-phase flows, including
surfactant transport and subgrid-scale modeling for sharp boundary layers.

Less "classic FronTier front-tracking," more:

```
moving mesh / ALE finite volume
        +
sharp interface
        +
surface finite-area equations
        +
bulk-interface transfer
```

**Why surfactants matter here.** Surfactant transport is mathematically close to the
thing you care about: a scalar density living on a moving interface. A typical
insoluble surface species equation has the flavor:

```
dΓ/dt + surface divergence(Γ u_surface)
    = surface diffusion + reactions
```

For soluble surfactant there is also exchange with the bulk:

```
bulk concentration near interface
        ↔
surface concentration on moving interface
```

That maps well to computational-biology analogues:

```
bulk cytosolic species
        ↔
membrane-bound species
        ↔
surface diffusion / reaction / advection
        ↔
moving cell boundary
```

Because it is explicitly "interface tracking with surfactants," it has to confront
moving-surface transport, not just move a geometric front.

**What it likely gives you:** unstructured finite-volume ALE interface tracking; sharp
interface between two phases; finite-volume OpenFOAM implementation; surfactant
transport on the interface; bulk-interface coupling; subgrid-scale modeling for
boundary layers; moving mesh / ALE handling. The key phrase is not just "interface
tracking" — it is "with surfactants," which strongly suggests built-in machinery for
scalar quantities on moving interfaces (surface control areas, surface concentration,
surface fluxes, surface advection/diffusion, bulk-interface exchange, mesh-motion
effects).

**Why this may be better than cfdmfFTFoam for surface mass.** If your central concern
is: "I have material/chemical mass on a moving surface; as the surface moves,
stretches, and changes mesh representation, I want the surface amount to remain
conservative" — then this is closer conceptually than a pure front-tracking package
that mainly cares about geometry, curvature, surface tension, and indicator functions.

**What not to assume:** that it handles arbitrary cell-biological membranes out of the
box. It is built for two-phase incompressible flow, not cell mechanics. Inspect
whether the code separates the *interface transport machinery* from the *two-phase
Navier-Stokes / surfactant constitutive model* — the more modular that separation, the
more useful as a starting point. Also check: topology changes; cell-like deformable
closed membranes; large curvature changes; surface reaction networks; multiple
membrane species; coupling to cytoskeletal mechanics. Those may not be primary goals.

### 3.3 Practical comparison

| Question | cfdmfFTFoam | twoPhaseInterTrackFoam |
|---|---|---|
| Closest to old FronTier/front tracking? | Yes | Less so |
| Explicit front mesh? | Yes | Interface-tracking/ALE mesh |
| OpenFOAM/C++ finite-volume ecosystem? | Yes | Yes |
| Surface tension / front geometry emphasis? | Strong | Present, not the only focus |
| Surface scalar conservation emphasis? | Maybe, but check | Much stronger (surfactants) |
| Good for membrane species transport ideas? | Possible | More directly relevant |
| Good for classic multiphase front-tracking CFD? | Very relevant | Relevant, different method |
| Likely easy to extract as a standalone library? | Probably not | Probably not |
| Best role for you | FronTier-like geometry/front-tracking reference | Moving-surface FV transport reference |

### 3.4 How to evaluate them

Don't start by running full CFD benchmarks — look for the **surface-data lifecycle**.

For **cfdmfFTFoam**, inspect: front mesh data structures; front node/face
connectivity; remeshing algorithm; volume correction algorithm; surface field
support, if any; front-to-Eulerian interpolation/spreading; Eulerian-to-front
interpolation. The question: *can you attach a scalar field to the front and preserve
it through advection/remeshing?*

For **twoPhaseInterTrackFoam**, inspect: surface concentration field representation;
finite-area / surface finite-volume operators; mesh-motion correction terms;
bulk-interface flux implementation; surfactant remapping during mesh motion; treatment
of surface divergence and area change. The question: *can you replace "surfactant
concentration" with your membrane-bound species and extend it to reaction networks?*

### 3.5 Recommendation

- To revive/modernize an old FronTier-style 2D FV moving-front code: start by studying
  **cfdmfFTFoam** — the closer architectural cousin.
- For conservative mass transport on moving biological membranes: study
  **twoPhaseInterTrackFoam** first. Even if you never use OpenFOAM directly, its
  treatment of surfactant transport on an ALE-tracked interface is likely to contain
  the ideas you want.

Most promising hybrid path:

```
Use cfdmfFTFoam ideas for:
  explicit front tracking
  front remeshing
  Eulerian-grid communication

Use twoPhaseInterTrackFoam ideas for:
  moving-surface control volumes
  surface concentration transport
  bulk-surface exchange
  conservative ALE terms
```

For this background, treat both less as turnkey packages and more as modern reference
implementations that can inform the design of a computational-cell-biology membrane
transport framework.

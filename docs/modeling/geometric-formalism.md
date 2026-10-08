# The geometric formalism

A declarative description of a spatial domain — the **Geometry** artifact of the three-artifact
model (`MathDescription`, **Geometry**, `SolverConfiguration`; declarative-formalism §3.4). Per
ADR 007 this declarative description is the *source of truth* for spatial domains; the concrete
mesh is a derived realization. This note specifies the formalism, its mapping to/from VCell, how a
description is realized into something a backend can solve on, and the increment roadmap.

It deliberately parallels the math formalism (`declarative-formalism.md`): a typed, named,
round-trippable artifact with a faithful VCell importer, decoupled from the backend that realizes
it. Where the math formalism names *species, reactions, equations*, this names *regions, membranes,
and faces* — and the two bind by name (`cross_validate`): a MathDescription's `subdomain: cytosol`
resolves to a Geometry subvolume named `cytosol`, its membrane to a surface class, its boundary to
a named face.

## 1. The formalism

### 1.1 `GeometryDescription`

A bounding domain partitioned into named regions, mirroring VCell's `Geometry`
(`pyvcell.vcml.models_geometry`):

| field | meaning |
|---|---|
| `name` | the identifier the MathDescription's `geometry:` references |
| `dim` | embedding dimension: `0` (non-spatial / well-mixed), `1`, `2`, or `3` |
| `extent` | bounding-box side lengths `(lx, ly, lz)` |
| `origin` | bounding-box origin `(ox, oy, oz)` |
| `subvolumes` | the volume regions (§1.2) — the **volume subdomains** |
| `surfaces` | the membranes between region pairs (§1.3) — the **surface subdomains** |
| `image` | optional segmented image backing `image`-typed subvolumes (§1.2) |

`dim = 0` is the non-spatial case (a single well-mixed compartment, no geometry to mesh) — half the
corpus. It carries exactly one `compartmental` subvolume and no surfaces.

### 1.2 Subvolumes (volume subdomains)

A subvolume is a named volume region. Its membership is defined by one of four **types** (VCell's
`SubVolumeType`):

| type | defined by | realization |
|---|---|---|
| `compartmental` | nothing — the whole (non-spatial) domain | a trivial well-mixed cell (§3.2) |
| `analytic` | `expression` — a **boolean predicate** over `geom.x` (the region where it is *true*), e.g. `geom.x[0]**2 + geom.x[1]**2 < 1` | the Rvachev lowering to an implicit field, its boundary extracted and meshed body-fitted with Netgen (§3.2, §3.5); the same field is the input for a future unfitted path |
| `csg` | a constructive-solid-geometry tree of primitives + booleans | imports and validates; not realized yet (§3.2) |
| `image` | `pixel_value` — the voxels of `image` carrying that class value | smoothed label field → conforming mesh (§3.2, ADR 012) |

Fields: `name`, `type`, and the type-specific payload (`expression` for analytic, a `csg` tree for
csg, `pixel_value` for image). The `analytic` expression is in the §1.8 expression language of the
math formalism — it may use `geom.x[0..2]` and parameters, so the two formalisms share one parser
and one set of namespaced built-ins (ADR 006).

A subvolume with `dim` volume becomes a **volume subdomain** of kind `volume`; the MathDescription
puts bulk species and reactions on it.

### 1.3 Surface classes (membranes)

A surface class is the interface between an *ordered pair* of subvolumes — `(inside, outside)` —
exactly VCell's `SurfaceClass(subvolume_ref_1, subvolume_ref_2)`:

| field | meaning |
|---|---|
| `name` | the membrane's name (a **surface subdomain**) |
| `inside` / `outside` | the two subvolumes it separates |

It realizes to a codim-1 region — the facets between the two subvolume meshes — and becomes a
**surface subdomain** of kind `surface`, the home for membrane species and the surface PDE with its
mandatory `ρ ∇_Γ·v_Γ` dilution term. The ordered pair fixes the outward normal and the `inside` /
`outside` trace directions for cross-membrane coupling (jump conditions, §4).

### 1.4 External faces

The bounding box's outer boundary, named by face so per-face boundary conditions can attach. For an
axis-aligned box the convention mirrors VCell's `Xm/Xp/Ym/Yp/Zm/Zp`:

```
x_minus  x_plus   y_minus  y_plus   z_minus  z_plus
```

Each is a named **external boundary** (an ` external` `BoundaryGeometry`, incident to the one
subvolume it bounds). A non-box outer boundary (e.g. the circle bounding a disk) is a single named
external boundary instead (today's `boundary="wall"` convention).

### 1.5 The naming contract

The whole point of the artifact is the set of names the MathDescription binds to, checked by
`cross_validate`:

- **subvolume name** ↔ a `volume` subdomain in the MathDescription;
- **surface name** ↔ a `surface` subdomain;
- **face / external-boundary name** ↔ the `boundary:` of a boundary condition;
- an **internal surface** between two subvolumes ↔ an `interface_*` boundary condition's two-sided
  incidence.

These names are stable across realizations (§3.3): the same description meshed body-fitted or
carried as a level-set exposes the same names, so a model is not re-specified when the approach
changes.

### 1.6 Carriers

YAML (primary), JSON (canonical), Python dataclasses (in-memory) — round-tripping without loss, as
for the math formalism. A geometry references nothing by file path; it is registered by name
(`register_geometry` / `load_geometry`), so a MathDescription + GeometryDescription pair can be
assembled entirely in memory (e.g. by a VCell import) and handed to the backend.

## 2. Mapping to / from VCell `Geometry`

The importer is the geometry analogue of the math import layer (`pyvcell_bridge`, declarative-
formalism §2.6): `pyvcell.vcml.models_geometry.Geometry` → `GeometryDescription`, duck-typed over
the pydantic object. The correspondence is direct:

| VCell `models_geometry` | geometric formalism |
|---|---|
| `Geometry.dim / extent / origin` | `GeometryDescription.dim / extent / origin` |
| `SubVolume` (`analytic` / `csg` / `image` / `compartmental`) | a subvolume of the same type |
| `SubVolume.analytic_expr` | `subvolume.expression` (translated through the §2.6 expression rules) |
| `SubVolume.image_pixel_value` + `Geometry.image` | an `image` subvolume + the carried image |
| `SurfaceClass(subvolume_ref_1, subvolume_ref_2)` | a surface class `(inside, outside)` |
| per-face `BoundaryType` on `Xm/Xp/…` | external faces `x_minus/x_plus/…` (the per-face BC targets, §4) |

Like the math importer, the geometry importer **loudly rejects** what it cannot represent and is
validated against the 5615 parsed corpus geometries (`vcml_biomodels/parsed/*_geom.yaml`), with a
survey script tracking coverage — the same workflow that drove the math side from 0.4% → 45% clean
import.

## 3. Realization

### 3.1 The realization contract

A `GeometryDescription` plus realization options (mesh size `h`, target approach) produces a
concrete artifact the backend consumes. For body-fitted approaches that is today's `Geometry`
(`backend/geometry.py`): subdomain name → submesh + kind, surface name → labelled facets, face name
→ external `BoundaryGeometry`, optionally over a shared `parent_mesh` with `cell_tags` / `facet_tags`
for multi-compartment domains. For unfitted approaches it is a background mesh plus a level-set
(§3.3). The realization is where dimension, mesh resolution, and approach enter — never the spec.

### 3.2 Backends by subvolume type

- **`compartmental` / `dim = 0`** → a trivial single-cell domain (a well-mixed "point"): no real
  mesh, the `lumped_ode` / well-mixed templates need none. Covers ~50% of the corpus immediately.
- **`analytic`** (any boolean predicate, primitive or not) → lower it to a single real-valued
  **implicit function** `φ` via the Rvachev lowering below (`formalism/rvachev.py`), extract its
  boundary — marched with scikit-image in 2D, marching cubes in 3D, or, for shapes that touch the
  bounding box, rasterized by VCell's priority rule and projected onto the exact implicit functions
  — and embed it as a conforming internal boundary in a **Netgen** model that partitions the box
  (`backend/realize.py`; nested shapes nest, disjoint shapes sit side by side, junctions go through
  the label route of ADR 012). Each region is tagged with its subvolume name, each shared interface
  with its surface name, and each outer face with its face name → a conforming `Geometry`. *(As
  originally planned this was gmsh OCC booleans for the primitive/CSG subset and a level-set for the
  rest; ADR 008 replaced gmsh with Netgen, and the implicit-field route turned out to serve every
  analytic shape, so there is no separate primitive path.)* The implicit function is an
  approach-independent **pre-mesh intermediate**: sampled onto a background mesh it is also the
  natural input to **cut / trace FEM** (Approach D, CutFEMx — unfitted), which is not built.
- **`csg`** (a constructive-solid-geometry tree) → imports and validates (VCell csg geometries round-
  trip through `pyvcell_bridge/geometry.py`), but is **not realized yet**: `realize()` refuses it
  with a `RealizationError` (only `analytic`, `image` and `compartmental` subvolumes are meshed). The intended route is the same implicit-field lowering (a CSG tree is
  a boolean of primitives).
- **`image`** → a **smoothed label field** on an ≈ h lattice (`backend/labels.py`: per-subvolume
  Gaussian-smoothed indicators, argmax, speck and pinch clean-up), its **conforming multi-label
  boundaries** (`backend/label_surfaces.py`: VTK SurfaceNets with one sentinel label per box face,
  constrained smoothing, projection onto the smooth interfaces), and a **Netgen** mesh with those
  boundaries embedded — any topology: nested regions, regions cut by the box, junctions where three
  subvolumes meet. The voxels travel in `GeometryImage.compressed_content` (VCell's hex-zlib encoding;
  the lattice is vertex-centred). Analytic subvolumes mixed in are rasterized over the image, the
  earliest winning. See [ADR 012](../decisions/012-image-geometry-realization.md).

**Rvachev lowering (predicate → implicit function).** A VCell `analytic` subvolume is a boolean
predicate; the realization first lowers it to an implicit function `φ` whose **sign** encodes
membership — `φ < 0` inside, `= 0` on the boundary, `> 0` outside (the **inside-negative**
convention). The lowering is a faithful port of VCell's (`RvachevFunctionUtils`,
`FiniteVolumeFileWriter.convertAnalyticGeometryToRvachevFunction`), using plain `min`/`max`
R-functions: `a < b → a − b`, `a > b → b − a`, `&&` (and all-boolean `*`) → `max`, `||` (and
all-boolean `+`) → `min`, `!` → negation; `==`/`!=` and boolean/numeric-mixed products are rejected.
It is the same field VCell's embedded-boundary **fvsolver** consumes (§3.5), so it doubles as the
apples-to-apples comparison input. The result is sign-exact but not a true distance — reinitialise
before marching or smoothing, and use a non-shrink smoother to match fvsolver. Implemented in
`formalism/rvachev.py` (`lower_predicate`, `subvolume_implicit_functions`).

**Subvolume priority.** The subvolumes of a `GeometryDescription` are **ordered**, and that order
*is* their priority (index 0 highest). Each region is its own predicate **minus** every
higher-priority region, and the **last** subvolume is the *background* — the complement of the union
of all the others (its own expression, if any, is ignored). This painter's-algorithm partition
matches VCell and is preserved by the importer; it is carried implicitly by list order, not an
explicit `priority` field.

**Tooling — VTK / pyvista for the implicit-field pipeline.** The 2D realizer marches with
scikit-image (`find_contours`) and meshes the box body-fitted with Netgen (originally gmsh OCC; ADR 008). **VTK** is an alternative
with a deeper geometry-processing toolbox — contouring / marching cubes, distance fields, implicit
modelling, surface extraction and reconstruction — and is **already in the environment via pyvista**
(a dependency), so it needs no new package. Worth evaluating as the pipeline grows to 3D, non-shrink
smoothing, and surface reconstruction, where VTK's filters may beat the hand-rolled steps.

**Related formalism — SBML Spatial, and interior points.** The SBML Level 3 **Spatial** package is
another variation on this geometry handling alongside VCell (analytic / CSG / sampled-field domains
over a bounded space). One idea it adds is worth borrowing: it disambiguates domain membership with an
explicit **list of interior points** — coordinates known to lie inside each domain — rather than
inferring inside/outside from an inequality's sign or a winding rule. That makes region identification
*computationally* unambiguous: after fragmenting the box, each region is simply the one containing its
seed point. It is the principled fix for the classification problem the realizer hits — v1 assigns
regions by **area** (smallest fragment = the interior cell), robust only for the interior + background
topology, whereas per-domain interior points generalize cleanly to multi-region partitions (and sidestep
the centroid-in-a-hole failure of a naive centroid sign-test). A future increment can carry optional
interior-point seeds on `SubVolume` — importable from both VCell and SBML Spatial — and classify
fragments by seed containment.

### 3.3 Approach-dependent realization

The same description realizes differently per FE approach, but exposes the same names:

| approach | realization of the geometry |
|---|---|
| ALE / body-fitted (Approach A) | conforming Netgen mesh; the boundary moves and is remeshed |
| separate-mesh / mixed-dim (Approach B) | conforming mesh + extracted submeshes per region/surface |
| phase-field (Approach C) | a fixed background mesh; the interface is a field, no body-fitting |
| cut / trace FEM (Approach D) | a fixed background mesh + the subvolume's level-set (from `analytic`) |

A model written against subvolume/surface/face *names* is therefore portable across approaches; the
realization layer chooses how to honour those names.

### 3.4 Moving geometry

For a moving boundary the **spec defines the initial configuration and the names; the realization is
the live SOT during simulation**. In the body-fitted path the deformed mesh is authoritative and the
existing remesh machinery (`backend/ale.py`, `core/region_remesh_netgen.py`, `backend/remesh_3d.py`)
regenerates it from the deformed boundary with Netgen — it does *not* re-evaluate the analytic spec. In the unfitted
path the level-set evolves on the fixed background mesh. The geometric formalism is not a live
description of a moving domain; it is the initial-and-naming SOT.

### 3.5 The two VCell solvers — comparison baselines (not the recipe)

VCell has **two** solvers, with different geometry treatments; each is a baseline for a different
class of our approaches.

**`../vcell-fvsolver` — fixed-grid, implicit-surface FV** (the general-purpose finite-volume PDE
solver; wrapped by pyvcell via the `fvsolver` extra). It does *not* solve on the staircase mesh or a
body-fitted mesh. Pipeline:

1. choose a Cartesian solver mesh `(nx, ny, nz)`;
2. sample whatever representation (analytic / csg / image) uniformly onto that grid → an indicator;
3. build staircase quad boundary surfaces (voxel-face patches);
4. (optionally smooth those for *visualization*);
5. the **solver** uses an *implicit* surface from a different, volume-preserving smoothing of the
   surface grid — Gaussian-like, **without shrinkage**, on a tangent Voronoi grid.

So fvsolver is effectively an **embedded-boundary FV** method on a structured grid with a *smoothed
implicit interface* — every geometry type is first reduced to a sampled indicator, and the interface
the solver sees is a smoothed level-set, not the voxel staircase. Its closest analogue here is
**cut / trace FEM (Approach D)** on a fixed background mesh (and, loosely, phase-field).

**`../vcell-mbsolver` — FronTier front-tracking, explicit moving geometry** (the moving-boundary
solver; a FronTier-based cut-cell method with an *explicit* tracked interface). Its closest analogue
here is **ALE / explicit-membrane moving-boundary work (Approach A)** — an explicit, conforming-ish
moving interface rather than a fixed-grid implicit one.

Framing:

- **We are not bound to either pipeline, and can do better.** What we *share* with VCell is the
  declarative geometric *formalism* (§1–§2) — imported faithfully. The *realization* is ours to
  improve: from the same spec, FEniCSx can build a **body-fitted conforming mesh** (no staircase;
  the boundary lies on the analytic shape to second order in `h`, and image boundaries are smoothed
  and projected, ADR 012), or a **clean level-set** for cut/trace FEM. The SOT is the formalism, not
  VCell's realized grid.
- **But match the baseline to the comparison.** For a *fixed-grid, implicit-geometry* problem, the
  apples-to-apples baseline is **fvsolver**, and our nearest realization is the **structured-grid +
  implicit-surface (cut/embedded) path** — discrepancies there may come from fvsolver's
  volume-preserving interface smoothing and grid sampling, not the method. For a *moving-boundary*
  problem, the baseline is **mbsolver** (FronTier explicit front), against our ALE/explicit-membrane
  realization. Either way, document the interface/conservation conventions on both sides so the
  difference being measured is the method, not the geometry treatment.

## 4. Boundary conditions on a realized geometry

Two kinds of boundary, both named by the formalism (§1.4, §1.3):

- **External (per-face) BCs** — Dirichlet / Neumann / Robin on a named external face. VCell's
  per-face `Value`/`Flux` on `Xm/Xp/…` (§2.6.2) maps to a `BCDirichlet`/`BCNeumann` on
  `x_minus/x_plus/…`. The default no-flux face needs no BC (it is the natural zero-Neumann boundary),
  matching how the math importer already treats default faces.
- **Internal (interface) BCs** — value-equality / single-sided flux across a surface class, between
  its `inside` and `outside` subvolumes (VCell's `JumpCondition` → a pair of `interface_flux`, one per
  side). The ordered surface pair fixes which side each single-sided flux applies to.

Once the realization names its external faces (realization v1, §3.2), per-face BC import is the small
step it was always meant to be — the §2.6.2 mapping onto labelled boundaries the existing
`BoundaryGeometry` already supports.

## 5. Corpus coverage and priorities

Survey of the 5615 parsed corpus geometries (`scripts/`, the geometry analogue of
`survey_parsed_math.py`):

- **dim 0 / compartmental — 50%.** Realizable now (trivial domain). The whole non-spatial,
  well-mixed-ODE half of the corpus.
- **analytic, spatial — the dominant spatial type** (4139 subvolume occurrences). Every analytic
  predicate realizes body-fitted through the implicit-field route with Netgen (2D and 3D, any nesting,
  shapes touching the box); the unfitted level-set/cut-FEM consumption of the same field is not built.
- **image — 16%** (926 geometries). Imports losslessly (voxels included); realized body-fitted (ADR 012).
- **topology is simple** — 1–2 subvolumes in 86% of geometries, 0–1 membranes in 89%. The common
  realization targets are: a single region (no membrane), and one cell inside extracellular space
  with one membrane — both already prototyped by `make_disk_geometry` /
  `make_cell_extracellular_geometry`.

So the realization priority was: **non-spatial (free) → primitive/CSG analytic → arbitrary
analytic → image (mesh).** As built (2026-10-08): non-spatial, analytic (all of it, body-fitted
Netgen) and image are realized; `csg` trees are not.

## 6. Roadmap

1. **Formalism + schema + carriers + VCell-geometry importer** (pure data, no realization) —
   `GeometryDescription` dataclasses mirroring VCell, YAML/JSON loaders, a `models_geometry →
   GeometryDescription` importer, and a coverage survey over the corpus. This is the geometry
   counterpart of the math import layer and is independently useful (a validated geometry pool)
   before anything is meshed. **Done** (`formalism/geometry_*.py`, `pyvcell_bridge/geometry.py`,
   `scripts/survey_parsed_geom.py`): **98.6% of the 5615 corpus geometries import + validate clean.**
2. **Realization v1 — Done** (`backend/realize.py`): `compartmental`/dim-0 (trivial) + every
   `analytic` subvolume via the implicit-field route and Netgen (ADR 008; originally planned as gmsh
   OCC for the primitive/CSG subset) → a `Geometry` with tagged regions, surfaces, and **named
   external faces**. Subsumes the imperative `make_*` helpers as recipes over the formalism. `csg`
   trees are still refused (`RealizationError`).
3. **Per-face boundary conditions** — the §2.6.2 `Xm/Xp/…` → `x_minus/x_plus/…` mapping on the named
   faces from step 2, lifting the import layer's `reject_not_implemented` bucket (44.7%).
4. **Interface (jump-condition) BCs** — `SurfaceClass` + `JumpCondition` → single-sided `interface_flux`
   (a pair, one per side), the cross-membrane coupling.
5. **`image` → conforming mesh — Done** (ADR 012: label field → SurfaceNets boundaries → Netgen, 2D and 3D,
   any topology; cross-validated against fvsolver on VCell's tutorial image).
6. **Later** — arbitrary-analytic → level-set realization (cut/trace FEM).

## 7. Non-goals (for now)

- **A full CSG-tree authoring surface** beyond what VCell's csg trees need.
- **1D geometries** as a first-class realization (39 in the corpus; revisit if a real model needs
  one).
- **Live re-evaluation of a moving analytic boundary** — moving domains are handled by the
  realization (deformed mesh / evolving level-set), per §3.4 and ADR 007.

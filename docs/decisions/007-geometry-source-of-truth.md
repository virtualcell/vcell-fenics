# ADR 007 — The source of truth for spatial domains is a declarative geometric formalism

**Date:** 2026-06-18
**Status:** Proposed

## Context

The declarative formalism separates a model into three artifacts that reference each other by
*name* — `MathDescription`, **Geometry**, `SolverConfiguration` (§3.4). The MathDescription never
contains a shape; it names subdomains and boundaries, and the Geometry artifact says what those
names *mean*. `cross_validate` checks that every name resolves.

Today the **Geometry** artifact (`backend/geometry.py`) is a thin adapter over a *concrete* DOLFINx
mesh: `subdomains` (name → submesh + kind) and `boundaries` (name → labelled facets), built by
imperative gmsh helpers (`make_disk_geometry`, `make_cell_extracellular_geometry`, …). That was
fine while geometries were hand-written for tests, but two forces now press on it:

1. **The VCell import layer.** VCell models carry a rich, *declarative* geometry — a bounding box
   partitioned into **subvolumes** defined by `analytic` expressions, `csg` trees, `image`
   segmentation, or `compartmental` (non-spatial), with **surface classes** naming the membrane
   between a pair of subvolumes (`pyvcell.vcml.models_geometry`). To import and *run* a VCell model
   we must represent and realize that geometry. A survey of the 5615 parsed corpus geometries:

   | dimension | share | | subvolume type | occurrences |
   |---|---|---|---|---|
   | 0 (non-spatial) | 50% | | analytic | 4139 |
   | 3D | 26% | | compartmental | 2806 |
   | 2D | 23% | | image | 2245 (926 geoms, 16%) |
   | 1D | <1% | | | |

   Half the corpus is non-spatial; the spatial half is analytic-dominated; images are a minority;
   topology is simple (1–2 subvolumes, 0–1 membranes in the vast majority).

2. **Per-face boundary conditions (§2.6.2)** and **multiple FE approaches.** VCell's per-face BCs
   live on the named bounding-box faces (`Xm/Xp/Ym/Yp/Zm/Zp`), so the geometry must *name* its
   external faces. And the project's approaches need *different realizations of the same domain*:
   body-fitted (ALE / separate-mesh) wants a conforming mesh with tagged regions; cut/trace FEM
   (Approach D) wants an **implicit** level-set on a fixed background mesh; phase-field wants no
   explicit geometry at all. No single concrete representation is the source of truth for all of
   them — and while a boundary *moves*, the SOT is the evolving realization (a deformed mesh, or an
   evolving level-set), not a re-evaluated analytic spec.

The question this ADR settles: **is the source of truth for spatial domains an abstract geometry
specification or a concrete mesh, and do we build a VCell-grade geometric formalism (analytic / CSG
/ image) or lean on a meshing package's primitives?**

## Decision

**The source of truth is a declarative geometric formalism — a `GeometryDescription` — modelled on
VCell's geometry.** The concrete mesh is a *derived realization*, not the SOT.

1. **A declarative `GeometryDescription`** (design: `docs/modeling/geometric-formalism.md`) mirrors
   VCell: a bounding domain (`dim`, `extent`, `origin`), **subvolumes** (volume subdomains) typed
   `compartmental | analytic | csg | image`, **surface classes** (a membrane between a subvolume
   pair), and named **external faces**. It carries the same way as the MathDescription
   (YAML / JSON / dataclasses) and is referenced by name. Its names — subvolumes → subdomains,
   surfaces → surface subdomains, faces → external boundaries — are the contract the MathDescription
   binds to.

2. **A realization layer** turns a `GeometryDescription` into a concrete artifact, with a backend
   per subvolume type: `compartmental` / dim-0 → a trivial well-mixed domain; `csg` and analytic
   shapes expressible as primitives → a **gmsh OCC** body-fitted mesh with tagged
   regions/surfaces/faces; arbitrary `analytic` → a sampled **level-set / indicator** (the input for
   cut/trace FEM); `image` → segmentation-to-mesh. The existing `Geometry` (mesh + tags) is the
   body-fitted realization type and is kept.

3. **gmsh (with the OCC kernel) is the meshing engine — we never write a mesher.** CSG comes for
   free from OCC booleans on primitives. Image-to-mesh, when it lands, uses an existing tool or
   libvcell, not bespoke code.

   *Implementation note (2026-10-08):* the engine became **Netgen** (LGPL) under ADR 008; the
   decision above stands with that substitution. Analytic and CSG subvolumes are realized not by
   OCC booleans but by lowering the predicate to a Rvachev implicit field, extracting its boundary
   (marched in 2D, marching cubes in 3D, or rasterized by VCell's priority rule and projected onto
   the exact implicit functions when a shape touches the box) and meshing body-fitted with Netgen;
   image subvolumes go through the smoothed label field of ADR 012. `src/` is gmsh-free.

4. **The formalism is the committed SOT now; capabilities grow incrementally.** We do not build
   every realization backend at once (image meshing in particular is deferred), but the *spec* and
   its VCell importer are built first, so the geometry side mirrors the math side: a validated,
   declarative artifact with a faithful VCell import path, decoupled from how it is meshed.

## Consequences

**Positive:**

- **VCell import becomes faithful and end-to-end.** The geometry maps construct-for-construct
  (analytic/csg/image/compartmental → subvolume types; membranes → surface classes), the same way
  the math model does — and the 5615 parsed corpus geometries become a validated pool.
- **One named spec, many realizations.** The same `GeometryDescription` realizes as a body-fitted
  mesh *or* an implicit level-set *or* a trivial domain, so the geometry layer serves all four FE
  approaches without re-specifying the domain. The names the MathDescription binds to are stable
  across realizations.
- **Portability and round-tripping.** A declarative geometry round-trips to/from VCell and is
  independent of the mesher; swapping gmsh for another realization does not touch models.
- **Per-face BCs fall out.** Named external faces are part of the spec, so the §2.6.2 per-face BC
  mapping is a small step on top.

**Negative / costs:**

- **A larger up-front build** than keeping the concrete mesh as SOT — a formalism + schema +
  importer + a realization layer, not just imperative builders.
- **Realization is approach- and type-dependent**, and not every backend is cheap: arbitrary
  analytic → body-fitted conforming mesh is hard (hence the level-set/cut-FEM route), and image →
  mesh is a real pipeline (deferred). Some geometries will only be realizable for some approaches at
  first.
- **A moving boundary's SOT is the evolving realization, not the spec.** The spec defines the
  *initial* configuration and the *names*; during simulation the deformed mesh (ALE) or evolving
  level-set (cut-FEM) is authoritative. The remesher already treats the mesh boundary as the
  evolving object — consistent with this, but worth stating so the spec is not mistaken for a live
  description of a moving domain.

## Alternatives considered

- **Concrete mesh as the source of truth** (status quo: a built/loaded DOLFINx mesh with tagged
  regions, constructed imperatively per case; no declarative layer). Simplest now, and the existing
  `make_*` helpers already work. Rejected as the *SOT* because it does not capture VCell's
  declarative geometry (so import is lossy/manual), does not round-trip, and pins us to a single
  body-fitted realization — foreclosing cut/trace FEM and clean moving-geometry regeneration. It
  remains, correctly, the *realization* layer.
- **gmsh primitives only, no geometric formalism** (express geometries directly as gmsh OCC calls).
  We use gmsh OCC anyway as the engine, but making it the user/import surface loses the
  analytic/image cases VCell relies on and gives no named, validatable, round-trippable artifact for
  the MathDescription to bind to. Rejected as the surface; adopted as the engine.
- **Build the full formalism *including* image-based meshing up front.** Most faithful to VCell, but
  images are 16% of the corpus and image→conforming-mesh is the heaviest piece; building it before
  per-face BCs and before the analytic-dominated majority would invert the priority. Deferred, not
  rejected — the formalism reserves the `image` subvolume type so the data imports losslessly even
  before it can be meshed.

## Notes

- The existing `Geometry`/`BoundaryGeometry`/registry (`backend/geometry.py`) is unchanged in role:
  it is the *body-fitted realization* the realization layer produces and the backend consumes.
- The geometry side deliberately parallels the math side (ADR 004 / the import layer): a declarative
  spec + schema + carriers + a faithful VCell importer, validated against the corpus, with the
  backend-specific realization kept separate.
- Roadmap and the full formalism design live in `docs/modeling/geometric-formalism.md`.

## Amendment (2026-09-23) — the image carries its voxels

The `image` type no longer carries metadata only: `GeometryImage.compressed_content` holds VCell's own
voxel encoding (hex of a zlib stream of uint8, x-fastest), so an image geometry imports losslessly *and*
realizes — see [ADR 012](012-image-geometry-realization.md). The parsed corpus YAMLs still strip the blob
(they stay metadata-only); a model imported from VCML, a SimulationTask or a lowered YAML dump keeps it.

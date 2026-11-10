# ADR 015 — Remeshing a moving partition

**Date:** 2026-11-09
**Status:** Proposed. Revised after an independent review (12 findings, all addressed below).
**Builds on:** ADR 014 decision 6 (accepted 2026-11-09: "Remeshing comes second"; its first stage's stop on a
degraded mesh was built in #228, its stop on contact was not), ADR 013 (the single-mesh moving path and its
remesh), ADR 010 §2–3 (a remesh starts a bundle segment), ADR 012 (the labelled-curve Netgen mesher this
reuses), ADR 008 (Netgen, not gmsh).
**Would amend:** `backend/multi_compartment.py` (the solver runs in segments), `backend/moving_partition.py`
(it undoes a tangling move and stops on contact), `runner._run_multi_compartment` (it drives the segments),
`backend/realize.py` (its partition tail is shared with the remesh).

## Context

**What happens today** (*measured in runs, 2026-11-09*). The moving partition (ADR 014) moves every compartment
and membrane of a multi-compartment geometry with one membrane's prescribed velocity, by a harmonic extension
with the box and every other membrane held fixed. It never remeshes. moving_bleb (BioModel 294599465, a cell
with 35 small teeth) at VCell's mesh (h = 0.04) tangled its outside compartment at t ≈ 1.5 of 5: 4 inverted
triangles, 79 by t = 2.6, the membrane species' total (which only converts between X and B) down 2 %. The same
cell drawn analytically tangled at t ≈ 1.4; with 8 teeth instead of 35, at t = 2.0. At h = 0.2 it ran to t = 5
untangled and conserving. Since #228 such a run stops with `MeshQualityError` instead of continuing, so these
models now fail at VCell's default mesh rather than give wrong results.

**What exists** (*checked in the code on `main` at `7344a8d`*):
- **The single-mesh path remeshes** (`ale.step_with_remeshing`, `stride_with_remeshing`): before each step or
  stride, if the cell-size-ratio growth has reached `_REMESH_QUALITY_LIMIT = 4.0` (`runner.py` 347), it meshes
  the current boundary afresh and transfers the field; since #228 a move that would invert a cell is undone
  and retried once on a fresh mesh (`MeshTangled`, `discrete.py` 411–418, `ale.py` 143–156, 204–217).
- **A multi-region mesher from labelled curves**: `realize._netgen_from_curves` (ADR 012) takes a
  `LabelBoundary` — segments, each with the regions on its two sides (a box face counting as a side, encoded
  `-(k + 1)`), its left normal pointing into `pairs[:, 1]` — and meshes the box with every segment as a
  conforming internal boundary; Netgen's domain is the region, numbered as `_tag_facets` expects (subvolume
  index + 1). Junctions and curves ending on the box are what it was built for.
- **Netgen keeps every input vertex** (*checked by experiment, 2026-11-09, for one closed loop in a box*): a
  100-vertex loop with segments from 0.0006 to 0.19 long, meshed at h = 0.1, kept all 100 vertices exactly
  (added points lie on the input segments), and the enclosed domain's area equalled the input polygon's to 15
  digits. That junctions and several domains behave alike is *inferred*; decision 4's first test checks it.
- **Tagging from a region-tagged mesh**: `realize._tag_facets(parent, description, cell_values, …)` names
  membrane facets by the declared surface classes and box facets by position; `realize_multi_compartment`
  then builds the compartment and membrane submeshes and their `EntityMap`s (`multi_compartment.py` 158 on).
- **Transfer**: `core.bulk_remap_mesh.remap_bulk_function_3d` evaluates the old P1 field at the new nodes by
  DOLFINx's non-matching interpolation (a convex combination per old cell: no new extrema) and optionally
  rescales to the old total; nothing in it is specific to 3D. The 2D supermesh projection
  `remap_bulk_function` is an L2 projection (conservative in total, not cell by cell, no positivity bound)
  whose broad phase tests every new triangle against every old one in Python (`core/bulk_remap.py` 90–100).
- **The bundle**: `BundleRecorder.start_segment` moves any set of domains onto new meshes and opens a new
  segment (`results/recorder.py` 97); the writer rebuilds the membrane adjacency maps per segment.
- **The solver** builds its spaces, forms and PETSc objects once on the geometry and runs every stride inside
  one call, with a fresh TS per stride (`integrate_multi_compartment`, `multi_compartment.py` 282; the stride
  loop 723–738). `MovingPartition.advance` moves every mesh, updates the swept measures, then checks and
  raises a plain `MeshQualityError` (`moving_partition.py` 225–236): it has no undo.

## Decisions

1. **Scope: 2D, serial.** In 3D, and under MPI, a moving partition that needs a remesh keeps #228's stop,
   with the message saying why.
   - 3D needs the moving membrane's *surface* rebuilt (it degrades, and 3D has no slide), across junction
     curves with fixed membranes; `_netgen_from_surfaces` keeps its input triangles and the single-region 3D
     remesher (`remesh_3d.py`) rebuilds one region implicitly. That is a design of its own.
   - The slide is serial only (ADR 014 decision 4), and the mesher runs on rank 0 from a boundary that would
     first have to be gathered. VCell's FEniCSx runs use the simtask's `NumberProcessors` (`simtask.py` 193),
     one by default (*inferred from VCell's default, not surveyed*).

   *Recommendation: yes.* Every corpus moving model is 2D (VCell's own MovingBoundary solver is 2D only).

2. **When: before a stride whose mesh has degraded, and when a stride's move would tangle it.**
   - Before each stride's move, if the parent's growth (`CellOrientation`, #228) has reached
     `_REMESH_QUALITY_LIMIT` (4.0), remesh first.
   - If the move would invert a cell anyway, `MovingPartition.advance` undoes it — every mesh's coordinates
     restored, then `MeshTangled` raised (new code; the swept measures need no restoring, since the remesh
     builds a new partition) — and the stride is retried once on a fresh mesh. A retry that tangles the fresh
     mesh stops the run with both possible causes named: "the stride is too large (`--dt`), or the moving
     membrane folds on itself".

   *Recommendation: yes*, the same limit and the same undo-and-retry as the single-mesh path. Both measure the
   max/min cell-size ratio against the mesh as built. *A known limit of that measure:* pure shear (the outside
   of a rotating membrane) changes no cell's size, so it never triggers a remesh; only the undo-and-retry
   catches it, once cells invert. An angle-based measure is a possible refinement, not part of this ADR.
   Option rejected: remeshing every stride (the mesher and the transfer cost more than a stride's solve).

3. **Contact is a stop, not a remesh.** A remesh keeps the topology, so it cannot let the moving membrane meet
   the box, another membrane, or itself. After each move (so a stride larger than the margin is caught), if the
   moving membrane has come within `δ` of a box facet or of another membrane's facet, or two of its own points
   are within `δ` in space while more than `4δ` apart along it, the move is undone and the run stops with a
   named error and the time ("the moving membrane has reached the box at t = …").
   - `δ = min(h/4, g₀/2)`, with `g₀` the smallest such gap in the mesh as built, so a geometry built with a
     narrower gap does not stop at once.

   *Recommendation: yes.* ADR 014 decision 6's first stage promised this stop and #228 did not build it;
   without it a membrane pushed into the box surfaces as tangling, with the wrong cause. *`h/4` is a judgement,
   not a measurement* (it leaves about a quarter of a cell between the membrane and what it approaches); the
   contact tests check that the stop comes before any cell inverts, and the margin is tuned there.

4. **What is meshed: the moved parent's own boundary skeleton, unchanged.** Every parent facet between two
   regions and every box facet becomes a `LabelBoundary` segment, its sides the regions of the cells beside it
   (a box face for the outside of a box facet), oriented so that its left normal points into `pairs[:, 1]`;
   `_netgen_from_curves` meshes it at the run's `h`. Netgen keeps every vertex and adds points only on the
   segments, so every compartment's polygon and every membrane's polyline are the old ones (Context).
   - **Every compartment is remeshed**, including those the motion leaves unchanged (a fixed nucleus): Netgen
     meshes the whole box. Their fields then pass through decision 6's interpolation too.
   - **The moving membrane's node count only grows**: the slide (ADR 014 decision 4) keeps the count between
     remeshes, and a remesh adds nodes where segments have stretched past `h`.
   - **Option (b), resampling the moving loop** at spacing `h` before meshing, would also coarsen a membrane
     whose nodes have crowded, at the price of slightly different regions. Not now.
   - A Netgen failure on the skeleton stops the run (no smoothing fallback, unlike ADR 012's image path: the
     skeleton is the solver's own geometry, and altering it would break decision 6's exactness); its messages,
     written for image geometries, are reworded for the remesh.

   *Recommendation: (a), unchanged.* The regions stay exactly the same, it reuses ADR 012's mesher as is, and
   it is what the single-mesh path does (`ale._remesh` hands its loop to Netgen unchanged). Its cost: a remesh
   never coarsens, so a geometry already fine at its membrane (moving_bleb's teeth, segments of 0.01 against
   h = 0.04) remeshes more often; the verification measures how often.

5. **The new geometry: the realizer's own tail.** The cells take their region from Netgen's domain;
   `_tag_facets` tags the membranes and box faces; the compartment and membrane submeshes and maps are built as
   at realization, for the same compartments. `realize_multi_compartment` is split so that realization and the
   remesh share that tail. Membrane names, region tags and box-face tags are unchanged, so everything keyed by
   them (the math's subdomains, the bundle's domains, the membrane adjacency) carries over.

   *Recommendation: yes.* Option rejected: classifying the new cells by the description's implicit functions,
   as realization does — they describe the *initial* shape, not the moved one.

6. **The transfer: interpolation, rescaled per compartment; exact on membranes; checked.**
   - **A compartment's species**: DOLFINx's non-matching interpolation from the old compartment submesh to the
     new one (`remap_bulk_function_3d`'s method), then one rescale of that species to its old total in that
     compartment. Interpolation makes no new extrema and keeps a non-negative field non-negative; the
     rescale makes the total exact (the regions being unchanged, it corrects only the interpolation error,
     O(h²) where the field is smooth). It is not conservative cell by cell.
   - **A membrane's species**: every new membrane node lies on an old membrane segment (decision 4), so the
     same interpolation reproduces the old piecewise-linear density exactly, on fixed, open and junction
     membranes alike. No rescale.
   - **The checks**: a membrane species' total must agree to `1e-10` relative plus `1e-14` absolute (a species
     with total zero passes on the absolute floor); a compartment species' rescale factor is logged, and one
     off by more than 1 % stops the run ("the remesh moved X's total by …; the field is under-resolved at this
     `h`"), since interpolation should miss by O(h²).
   - Region variables are refused on a moving partition (ADR 014 decision 5), so there are none to carry.

   *Recommendation: yes.* Option rejected: the 2D supermesh L2 projection (`remap_bulk_function`). It is more
   accurate for smooth fields but not cell-by-cell conservative either, it can undershoot below zero near a
   steep front, and its all-pairs Python broad phase is far too slow at h = 0.04 (~10⁵ triangles a side; an
   R-tree and vectorized clipping would be needed first).

7. **The solver runs in segments.** `integrate_multi_compartment` gains a starting state — `t_start` and the
   fields as functions, which it takes as they are instead of interpolating initial conditions — and returns at
   the stride boundary where the partition asks for a remesh, with its last step size (a new field of
   `MultiCompartmentResult`). The runner loops: integrate → remesh (decisions 4–5) → transfer (decision 6) →
   `recorder.start_segment` for every domain → integrate from that time on the new geometry, with a new
   `MovingPartition` (its slide fractions taken afresh from the new mesh), starting at the carried step size.
   - A segment records its start time only when it is the run's start (t = 0): an output time that falls on a
     remesh is recorded once, at the end of the segment that reaches it, on the old mesh.
   - Each stride already starts a fresh TS, so no integrator history crosses a remesh.

   *Recommendation: yes*, the shape the single-mesh path has (`ALEState.remesh` rebuilds the problem rather
   than mutating it, ADR 004). Option rejected: rebuilding the forms inside the solver's stride loop, which
   would tangle the PETSc object lifetimes with the geometry's.

8. **Output: a new bundle segment per remesh** (ADR 010 §2–3), for every compartment and membrane domain, and a
   log line "remeshed the partition at t = … (mesh quality); continuing on N cells", as the single-mesh path
   logs. The result summary counts the remeshes.

   *Recommendation: yes.* VCell's viewer already reads segmented bundles: vcell#2093 tested "a remesh across
   two segments" and the viewer "shows the remeshed shape" (integration tracker, row M5; *not re-checked*).

## Verification

- **A remesh of an unmoved partition**: every compartment's area and membrane's length equal to round-off; a
  linear field in each compartment reproduced exactly (interpolation of a linear is exact); membrane values
  identical at the old nodes; each total equal. Cases: nested loops, a membrane ending on the box, and a
  three-region junction (checks the vertex-keeping inference beyond one loop).
- **Across a remesh, against discretization error**: an expanding and a translating cell run with remeshes
  forced every few strides, against the same runs at a finer stride and mesh without remeshing (while they
  stay valid); each species' total conserved to the checks' tolerance; no field below zero that started
  non-negative.
- **The undo-and-retry path**: a stride whose move would invert a cell remeshes and continues; one too large
  for a fresh mesh stops with the decision 2 message.
- **moving_bleb at h = 0.04 to t = 5**: it remeshes instead of tangling, X + B conserved, no inverted cell; the
  number of remeshes, their cost and the largest rescale factor reported. Its analytic 8-tooth and 35-tooth
  recasts likewise.
- **Agreement across meshes**: the analytic recast at h = 0.2, 0.1 and 0.04 converging (the image geometry's
  smoothing scales with `h`, ADR 012, so the analytic recast is the clean case).
- **Contact**: a cell expanding toward the box, toward a fixed inner membrane, and a membrane folding toward
  itself each stop with the contact error before any cell inverts; a geometry built with a gap under `h/4`
  runs.
- **Output**: a remeshed run's segments read back (`results.reader`), each domain's mesh per segment; an
  output time that coincides with a remesh recorded exactly once.

## Staging

1. This ADR, accepted.
2. The contact stop and the undo in `MovingPartition.advance` (decisions 2–3): useful without remeshing.
3. The split realizer tail, the boundary skeleton and the remesh of a partition (decisions 4–5), with the
   unmoved-partition tests.
4. The transfer and its checks (decision 6); the segmented solver and the runner loop (decisions 2, 7, 8);
   the remaining verification.

## Not decided here

- **3D** remeshing of a moving partition (decision 1): a later ADR.
- **MPI**: a gathered boundary for the rank-0 mesher; the slide under MPI.
- **Coarsening** a crowded membrane (decision 4 option (b)); a cell-by-cell conservative 2D transfer (a fast
  supermesh).
- **An angle-based quality trigger** for shear (decision 2).
- **Topology changes**: a membrane meeting another, pinching off, or reaching the box (decision 3 stops them).

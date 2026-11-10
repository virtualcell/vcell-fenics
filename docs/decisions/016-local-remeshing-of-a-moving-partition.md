# ADR 016 — Local remeshing of a moving partition

**Date:** 2026-11-10
**Status:** Proposed. Revised after an independent review (8 findings, addressed below); decision 3 is put to
Jim as a choice.
**Builds on:**
- ADR 014 decision 6: remeshing comes second; its first stage, the named stop, is built (#228).
- ADR 015: each mesh move carries every species' nodal moments exactly; relaxation was measured and not
  adopted; a whole-box remesh conserving only totals was rejected.
- ADR 012: the labelled-curve Netgen mesher.
- ADR 010: a remesh starts a new bundle segment.
- ADR 008: Netgen (LGPL) in-process.

**Amends:** ADR 014 decision 6's second stage (a box-wide multi-region remesher remapping the membrane species):
this replaces it with a local one that leaves membranes untouched.
**Would amend:** `backend/multi_compartment.py` (the solver runs in segments), `backend/moving_partition.py` (it
reports the cells that need a remesh), `runner._run_multi_compartment` (it drives the segments) and
`backend/realize.py` (its partition tail is shared).

## Context

**Why a remesh is needed** (*measured, ADR 015 decision 4*):
- moving_bleb (BioModel 294599465) at VCell's mesh (h = 0.04) tangles its outside compartment at t ≈ 1.4–1.5.
  This happens in the image geometry, in an analytic recast with 35 teeth, and in one with 8 teeth (at t = 2.0).
- The cells that fail lie in notches of the outside compartment, between two flanks of the front that close in
  on each other. The image run's neck closes from 0.25 to 0.09 by t = 2.6 and heads for about 0.05. The tooth
  valleys have radii of curvature of 0.002–0.004 and close from their tips.
- The notch's area shrinks, so its cells must be removed. No fixed-topology mesh motion can do that: three
  extension variants all tangled at the same time.
- The front itself does not fold: no membrane segment turns over (*measured to t = 1.2*).

**What Jim requires of the transfer.** ADR 015's first draft remeshed the whole box and rescaled each species'
interpolated field to its old total. Jim rejected it: it changes every field everywhere and conserves only
totals. So a remesh must:
- leave every value outside the remeshed region exactly as it was;
- conserve mass locally, not only in total.

**What exists** (*checked in the code on `main` at `f15df39`*):
- **Netgen meshes a polygon given as labelled segments and keeps every input vertex** (`_netgen_from_curves`,
  ADR 012). *Checked by experiment 2026-11-09* for a closed loop in a box: all 100 vertices kept, and the area
  equal to 15 digits. That a hole bounded by a patch's edges behaves alike is *inferred*.
- **A conservative 2D supermesh projection** (`core.bulk_remap.supermesh_project_2d`): an L2 projection,
  conservative in total when both meshes triangulate the same polygon. It solves `M⁻¹B c` on a whole target
  mesh and returns only the result (`bulk_remap.py` 69–71); a constrained or local variant needs its private
  `_new_mass_matrix` and `_mixed_mass_matrix`. Its broad phase is a NumPy bounding-box filter inside a Python
  loop over the new triangles (`bulk_remap.py` 97–108): fine for a patch of hundreds of cells, not a whole mesh.
- **Netgen splits boundary segments longer than its `maxh`** (*checked by the reviewer, 2026-11-10*): a square
  with 0.1-long edges meshed at `maxh` = 0.04 came back with 12 boundary segments, while edges of 0.03–0.05
  were kept. So a patch whose boundary edge the motion has stretched to about 2h would gain a vertex that
  hangs off the neighbouring cell outside the patch, and `_mesh_on_rank0` (`realize.py` 535–580) neither
  merges coincident points nor checks conformity.
- **Rebuilding the geometry from a region-tagged mesh** (`_tag_facets`, the submesh tail of
  `realize_multi_compartment`), and a bundle that starts a segment for new meshes
  (`BundleRecorder.start_segment`).
- **The solver** builds its spaces, forms and PETSc objects once per geometry, and runs every stride in one
  call (`integrate_multi_compartment`).
- **Prior art in this ecosystem:** vcell-mbsolver remaps its 2D bulk by overlap-area-weighted control volumes
  (`docs/modeling/conservative-surface-remap.md`, background section). That is locally conservative in the
  finite-volume sense.

## Decisions

1. **Remesh a patch, not the box.**
   - The patch is the cells that need it (decision 2) and two rings of their neighbours, within **one
     compartment**.
   - The patch's boundary edges are kept exactly: the edges it shares with the rest of the mesh, and any
     membrane or box edges on it. Netgen meshes only the hole inside them, at the run's `h`.
   - Every node and value outside the patch is untouched, and so is every membrane: its nodes, its species, its
     slide. (Rebuilding the mesh renumbers the dofs, so membrane values are re-assigned to the rebuilt
     submesh's nodes by coordinate, as `moving_partition._matching` already does: identical values.)
   - **No hanging vertices.** A patch boundary edge longer than `h` would be split by Netgen. So the patch grows
     by the cells beside any such edge until every boundary edge is at most `h`. The stitch then merges points by
     exact coordinate and checks that the patch boundary has the same segments before and after, and that the
     rebuilt mesh's exterior facets are exactly the box's. A failure stops the run.

   *Recommendation: yes.*
   - A patch never crosses a membrane, so a membrane species is never transferred, and a membrane's polyline
     is never re-sampled.
   - The notches that fail in moving_bleb lie inside one compartment, against the membrane.
   - Rejected options:
     - **The whole box:** Jim's objection.
     - **A patch across a membrane:** it would need a membrane remap too. It is a later stage, only if a
       measurement shows it is needed.

2. **When: when a move would invert a cell; and, proactively, when a cell has shrunk far since its mesh was
   built.**
   - **On inversion:** the moving partition has no undo today (`moving_partition.py` 219–228 checks and raises;
     only the single-mesh path undoes, `discrete.MeshTangled`). It gains one: the check raises before the
     moments are carried (`multi_compartment.py`'s `move()`), so the states and mass matrices are still the old
     ones, and the undo subtracts the stored displacement from every mesh. The patch around the inverted cells
     is then remeshed and the move retried once; a retry that tangles a fresh patch stops the run.
   - **Within a stride:** each stride has two half-moves (ADR 015). A tangle on the first half-move remeshes
     before the stride's integration. A tangle on the second, after it, undoes that half-move, remeshes on the
     midpoint mesh at the stride's end time, and redoes the half-move on the new mesh.
   - **Proactively:** before a move, a cell whose area has fallen below a quarter of its area when its current
     mesh was built (the original mesh, or the remesh that made it) starts a patch remesh. Measured against
     the cell's own build, so a freshly remeshed patch does not re-trigger; and at most one remesh per move.
     There is no angle threshold: image meshes can have small angles as built. The quarter is a *judgement*;
     the spike tunes it.
   - #228's hard stops stay (inversion that a retry cannot fix; growth past 100×).

   *Recommendation: yes.*

3. **The transfer in the patch: what can be conserved, and the choice.** *This is Jim's decision.*

   With P1 nodal fields, no transfer across a local topology change keeps every control volume's mass exact
   while leaving the field outside the patch unchanged:
   - Leaving the field outside unchanged requires the patch-boundary nodes to keep their values, because
     their basis functions reach into the cells outside.
   - With those fixed, the interior nodes can make their own control volumes exact.
   - But a patch-boundary node's control volume is partly re-triangulated, so the patch total then misses
     slightly, and closing it spoils exactness somewhere.
   - Freeing the boundary values instead changes the field in the ring of cells just outside, and the same
     problem moves out one ring.

   (*Derived; the reviewer derived the same for option (a).*) The options:

   - **(a) Constrained projection.** The patch's interior node values are the L2 projection of the old field
     (by supermesh, on the patch only), and one Lagrange multiplier makes the patch total exact.
     - **Conserves:** the patch total exactly; everything outside untouched.
     - **Does not keep each node's moment:** the multiplier shifts them all, `∫φᵢ(u⁺ − u⁻) = λ ∫φᵢ`.
     - **Accuracy:** reproduces linear fields exactly (λ = 0); the most accurate option.
     - **Positivity:** can undershoot next to a steep gradient.
     - **Scale:** "local" here means the patch, the failing cells plus two rings, at least about 5 cells
       across.
   - **(b) Control-volume averages.** Each interior node's new value is the old field's average over its new
     median-dual cell, integrated exactly by clipping. The patch total's small shortfall (from the
     boundary nodes' re-triangulated control volumes) is closed by one multiplicative correction over the
     interior nodes.
     - **Conserves:** each interior control volume up to that correction (finite-volume-style, at one node's
       scale); the patch total exactly; everything outside untouched.
     - **Positivity:** non-negative.
     - **Accuracy:** first order inside the patch (an average is not a nodal value).
   - **(c) A cell-based finite-volume remap** (vcell-mbsolver's kind). Exact per cell, but it needs cell-based
     fields, not this solver's P1 nodal ones.

   **When no non-negative conservative solution exists** (with large boundary values and new cells larger than
   the old, the boundary nodes alone can carry more than the patch's total): grow the patch by a ring and try
   again, at most twice; then stop the run with a named error.

   *Recommendation: (a), with (b) as the fallback wherever (a) undershoots.* Remeshes are rare and patches
   small, so (a)'s accuracy matters more than (b)'s per-volume bookkeeping. Each of (a) and (b) conserves the
   patch total exactly and leaves everything else untouched. Neither keeps every control volume's mass exactly.

4. **The rebuild: the whole geometry, in segments.** DOLFINx meshes are
   immutable, so each patch remesh builds a new parent mesh:
   - the old mesh's cells outside the patch, plus Netgen's cells inside it, with the same nodes on the
     patch's boundary;
   - region and facet tags carried over (inside the patch, its single region);
   - the submeshes and maps rebuilt with the realizer's tail (`realize_multi_compartment` split so the remesh
     shares it).

   The solver runs in segments. It returns at the stride where a remesh is needed, then the runner:
   1. remeshes and transfers;
   2. starts a new bundle segment (ADR 010);
   3. builds a new `MovingPartition`, whose slide fractions and orientation reference are taken afresh;
   4. continues from that time, carrying over the step size.

   *Recommendation: yes.* Option rejected: mutating the mesh in place, which DOLFINx does not support.

5. **Contact is a stop.** A patch remesh keeps the topology between regions, so it cannot let a notch vanish.
   - If, after a move, the gap between two parts of the front (points more than `h` apart along it), or between
     the front and the box or another membrane, falls below `δ = min(h/4, g₀/2)`, the run stops with a named
     error and the time. `g₀` is the smallest such gap as built, measured with the same `h` exclusion along the
     front, so the definition is not circular.
   - A geometry built with a narrow gap (a sharp valley) has a small `g₀`, so its stop fires only close to true
     contact.
   - This is ADR 014 decision 6's promised stop, built here because a remesh makes runs reach it.

   *Recommendation: yes.*

   **The risk, plainly:** a local remesh may not carry moving_bleb. The image recast's neck heads for about
   0.05, a little over `h` = 0.04, so it may stay above `δ` = 0.01. The 35-tooth recast is worse.
   - Its valleys (radii 0.002–0.004) are much sharper than the front's displacement (0.03 by t ≈ 1.4), the
     note's own condition for a cusp.
   - Its notches' area shrinks toward zero (ADR 015 decision 4).
   - With the membrane's nodes kept (decision 1), a notch narrower than `h` can be filled only with slivers.
   - "No membrane segment turns over" was measured only to t = 1.2.

   So for the sharp recast, a remesh may only postpone the stop. If the spike confirms that, moving_bleb as
   authored (sharp teeth moving along their normals) cannot run to t = 5 at h = 0.04 in any front-tracking
   solver without changing the front itself (a curvature-limited velocity, say), which is a modelling question,
   not a meshing one.

6. **Scope: 2D, serial.**
   - In 3D a patch is a tetrahedral hole, which Netgen fills while keeping its surface triangles
     (`_netgen_from_surfaces`). But the transfer needs a 3D supermesh (tetrahedron intersections), which does
     not exist here. That is a later ADR.
   - Under MPI a patch would have to be gathered to rank 0.
   - Both keep #228's stop.

   *Recommendation: yes.*

**An alternative recorded: MMG.** MMG (mmgtools) adapts a mesh locally, with edge collapses, swaps and splits,
while keeping required edges such as the membrane. It is the standard tool for this job, and it could replace
decision 1's patch-and-Netgen with true local adaptation in 2D and 3D. It would add a dependency.
- *Not checked:* its licence (LGPL-3 to my knowledge) against ADR 008, and its conda-forge packaging.
- The transfer (decision 3) would be needed either way.

*Recommendation:* Netgen patches first, since Netgen is already in the stack; MMG if the spike shows patch meshing
fragile.

## Verification

- **The stitch is conforming:** the rebuilt mesh's exterior facets are exactly the box's (no crack), the patch
  boundary has the same segments before and after, and a field diffuses across the patch boundary as across
  any interior edge.
- **An unmoved patch remesh changes nothing physical:**
  - the patch's area to round-off;
  - every value outside the patch identical, bit for bit;
  - a linear field reproduced exactly inside the patch;
  - each species' patch total to `1e-12`.
- **Conservation across remeshes:** a cell expanding toward the box with remeshes forced, each species' total
  to `1e-12` throughout, and the exact-solution test (`tests/test_moving_partition_exact.py`) with forced
  remeshes still converging at its rates.
- **Positivity:** a steep, non-negative front crossing a patch remesh stays non-negative; a case where no
  non-negative conservative solution exists on the first patch grows the patch, or stops with the named error.
- **A moved patch remesh:** values outside the patch identical bit for bit after a remesh mid-run, not only
  when nothing moved.
- **moving_bleb's analytic recasts (35 and 8 teeth) at h = 0.04 to t = 3.5:** they run past t = 1.4 by
  remeshing, or stop with the contact error. Measure the number of remeshes, the patch sizes, their cost and the
  smallest gap.
- **Bundle:** each remesh starts a segment; the bundle reads back.

## Staging

1. This ADR, accepted.
2. **A spike** (not merged): patch remeshing with a simple transfer on moving_bleb's analytic recast. It
   answers whether patches carry the run, or whether it reaches contact or needs a membrane remap. The numbers
   go into this ADR before code lands.
3. The contact stop (decision 5): useful on its own.
4. The patch remesh and the rebuilt geometry (decisions 1, 4), with the unmoved-patch tests.
5. The constrained transfer and its positivity fallback (decision 3), the trigger (decision 2), the segmented
   solver, and the remaining verification.

## Not decided here

- **Patches across a membrane**, with a membrane remap (decision 1), if the spike needs them.
- **3D and MPI** (decision 6).
- **MMG** as the adapter (above).
- **A front that develops a cusp** (a normal-velocity front entering a concave corner sharper than its
  displacement): a geometric question about the front, not the mesh.

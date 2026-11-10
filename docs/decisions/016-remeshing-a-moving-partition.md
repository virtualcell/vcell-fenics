# ADR 016 — Remeshing a moving partition

**Date:** 2026-11-10
**Status:** Proposed. Jim chose (2026-11-10) to remesh the whole partition, not a patch, after the first draft's
independent review (8 findings) showed what a patch transfer can conserve. Revised accordingly, then after a
second independent review (12 findings, addressed below).
**Builds on:**
- ADR 014 decision 6: remeshing comes second; its first stage, the named stop, is built (#228).
- ADR 015: each mesh move carries every species' nodal moments exactly; relaxation was measured and not adopted.
- ADR 012: the labelled-curve Netgen mesher.
- ADR 010: a remesh starts a new bundle segment.
- ADR 008: Netgen (LGPL) in-process.

**Amends:** ADR 014 decision 6's second stage (a box-wide multi-region remesher remapping the membrane species).
This specifies it: the whole partition meshed from its own boundary skeleton, each region unchanged.

**Would amend:** `backend/multi_compartment.py` (the solver runs in segments), `backend/moving_partition.py` (an
undo, and the trigger), `runner._run_multi_compartment` (it drives the segments) and `backend/realize.py` (its
partition tail is shared).

## Context

**Why a remesh is needed** (*measured, ADR 015 decision 4*):
- moving_bleb (BioModel 294599465) at VCell's mesh (h = 0.04) tangles its outside compartment at t ≈ 1.4–1.5,
  in the image geometry and in analytic recasts with 35 teeth (and with 8 teeth, at t = 2.0).
- The cells that fail lie in notches of the outside compartment, between flanks of the front that close in on
  each other. The notch's area shrinks, so cells must be removed, which no fixed-topology mesh motion can do.

**What a remesh can conserve.** With P1 nodal fields, no transfer across a topology change keeps every control
volume's mass exact (*derived in the first draft's review*):
- The new mesh's basis functions are not the old ones, so no node's moment survives in general.
- What can be exact is a region's total, when the region's polygon is unchanged.

A patch remesh would keep every value outside the patch exact and the patch's total exact. Jim chose the whole
partition instead, for simplicity and robustness, given that remeshes are rare outside pathological models.

**What exists** (*checked in the code on `main` at `f15df39`*):
- **Netgen meshes labelled segments, keeping every input vertex and splitting segments longer than `maxh`:**
  - *Checked by experiment 2026-11-09:* a closed 100-vertex loop in a box, every vertex kept and the area
    equal to 15 digits.
  - *Checked by the reviewer 2026-11-10:* a square with 0.1-long edges at `maxh` = 0.04 came back with
    12 segments; edges of 0.03–0.05 were kept.
  - So Netgen may add points on any segment, by its own rule (a 0.05 edge was kept at `maxh` = 0.04), but every
    added point lies on an old segment, to round-off: the new boundaries lie on the old ones. That the same
    holds for a full partition skeleton with junctions is *inferred*; the first verification checks it.
- **The labelled-curve mesher** (`realize._netgen_from_curves`, ADR 012) takes a `LabelBoundary`: segments,
  each with the regions on its two sides, a box face encoded negative, the left normal into `pairs[:, 1]`
  (`label_surfaces.py` 47–49). It returns cells tagged by region, which `_tag_facets` and the tail of
  `realize_multi_compartment` (`multi_compartment.py` 181–247) turn into tags, submeshes and `EntityMap`s.
- **The supermesh projection** (`core.bulk_remap`):
  - It is the L2 projection of a P1 field onto another triangulation, `c_new = M⁻¹ B c_old`.
  - It conserves the total exactly when both triangulate the same polygon, and reproduces linear fields.
  - `M` and `B` depend only on the two meshes (`_new_mass_matrix`, `_mixed_mass_matrix`).
  - Pure Python: 1.3 s, 4.4 s and 12.7 s for 1.8k, 7.2k and 20k cells (*measured 2026-11-10*, unit squares).
    Its broad phase compares each new triangle against every old one, so the cost grows with the product: the
    reviewer measured 0.19 ms per new triangle against 138k old ones, against 0.03 ms at 20k.
  - Its clipping uses an absolute area `_EPS = 1e-12`, both to discard tiny pieces and as its in/out
    tolerance (`bulk_remap.py` 42, 115, 148–150). That does not scale with the mesh.
- **The bundle** starts a segment for new meshes (`BundleRecorder.start_segment`, ADR 010).
- **The solver** builds its spaces, forms and PETSc objects once per geometry.

## Decisions

1. **Remesh the whole partition from its own boundary skeleton.**
   - Every parent facet between two regions, and every box facet, becomes a `LabelBoundary` segment, built from
     the parent's own vertex array, so junction vertices are shared.
     - **Sides:** each side is the region of the cell beside it, as `tag − 1` (`_domains` adds 1 back). A box
       facet's outside is any negative value: every negative side maps to domain 0, and `_tag_facets`
       re-tags the faces by position.
     - **Orientation:** a geometric side test, since cells are not all oriented alike (`discrete.py`
       179–186). Each segment is ordered so that the region in `pairs[:, 1]` lies on its left
       (`realize.py` 1198).
     - `_netgen_from_curves` meshes it at the run's `h`.
   - Every region keeps exactly its polygon: Netgen keeps every vertex, and any point it adds lies on an old
     segment.
   - A Netgen failure on the skeleton stops the run, with no smoothing fallback, unlike ADR 012's image path:
     altering the skeleton would change the regions. Its messages, written for image geometries, are reworded.

   *Recommendation and Jim's choice: yes.*
   - Simpler than a patch: no patch growth, no stitching, no hanging vertices, no patch-boundary
     constraints.
   - It resets the quality of the interior cells, including cells compressed against the box away from the
     notch. Cells beside short skeleton segments stay small, since no boundary vertex is ever removed, and the
     boundary's vertex count only grows across remeshes.
   - Cost: every field is re-projected everywhere (decision 3).

2. **When: before a move, if the mesh has degraded; and when a move would invert a cell.**
   - **Proactively, per cell:** if any cell's area has fallen below a quarter of its area when its mesh was built
     (the original mesh, or the remesh that made it), remesh before the move.
     - Not `CellOrientation`'s global growth, `(max/min)/(max₀/min₀)` (`discrete.py` 196–215). A geometry
       whose smallest as-built cells are much smaller than the notch's (the analytic recast's valleys) would let
       a notch cell shrink almost to inversion before that ratio moved (*mechanism checked; magnitude
       inferred*).
     - The quarter mirrors the single-mesh path's 4× (`_REMESH_QUALITY_LIMIT`, `runner.py` 347). It is a
       *judgement*; the spike records both measures at each tangle and tunes it.
   - **On inversion:** the moving partition has no undo today (`moving_partition.py` checks and raises). It
     gains one. The check raises before the moments are carried (`multi_compartment.py`'s `move()`), so the
     states and mass matrices are still the old ones.
     - The undo restores a stored copy of every mesh's coordinates (not `x + d − d`, which is not bitwise `x`),
       and releases the moment vectors on that path.
     - Then remesh and retry the move once; a retry that tangles a fresh mesh stops the run.
     - Growth past #228's 100× limit raises the same error and is treated the same way.
   - **Within a stride** (two half-moves, ADR 015):
     - A tangle on the first half remeshes before the stride's integration.
     - A tangle on the second, after it, undoes that half-move, remeshes on the midpoint mesh at the stride's
       end time, and redoes the half-move on the new mesh.
     - So a segment may begin with a pending second half-move: the solver gains an entry point for it,
       besides its usual stride loop, which always starts with the first half (`multi_compartment.py`
       780–781).
   - At most one remesh per move, so a fresh mesh cannot loop. #228's hard stops stay.

   *Recommendation: yes*, the single-mesh path's limit and undo-and-retry.
   - *A known gap:* pure shear changes no cell's size, so only the undo catches it.

3. **The transfer: each compartment's total exact, by projection; each membrane exact.**
   - **A compartment's species:** the supermesh L2 projection from the old compartment submesh to the new one.
     - Both triangulate the same polygon (decision 1), so the compartment's total is kept with no rescale, exact
       to the kernel's clipping tolerance and the solve's round-off, and a linear field is reproduced exactly.
     - The clipping tolerance becomes relative to the cells' area (it is an absolute 1e-12 today). The spike
       measures the loss at 138k cells before the check below is set.
     - Each node's moment is not kept, and the field is re-projected everywhere at each remesh. The error is
       about the fields' interpolation error at that mesh.
     - `M` and `B` are built once per compartment per remesh and applied to every species on it.
   - **Where the projection would add undershoot:** a field whose new minimum falls below its old minimum by
     more than `10⁻⁸` of its maximum takes the lumped variant, per field. The comparison is with the old
     minimum, since Galerkin fields already undershoot by about −1e-2 on steep fronts (ADR 015 decision 3).
     - The lumped variant is `uᵢ = (B u_old)ᵢ / mᵢ`, with `mᵢ` the new lumped node masses.
     - `B` is entrywise non-negative, so it is non-negative where the old field is.
     - Its total is exact too, since `Σᵢ (B u_old)ᵢ = ∫ u_old`.
     - It is first order rather than second.
   - **A membrane's species:** the new membrane's nodes are the old ones plus points on old segments (decision 1).
     So P1 interpolation along the old segments reproduces the old piecewise-linear density exactly: values
     and totals, on fixed and moving membranes, open or closed. A kept node takes its value by coordinate (as
     `moving_partition._matching` does). An added node has no twin, so it takes the old segment it lies on
     and interpolates along it. The slide and the riding velocity are rebuilt with fresh fractions.
   - **Vector blocks** (one block per subdomain, a component per species, on the matrix-free path) are transferred
     component by component. Region variables are refused on a moving partition (ADR 014 decision 5).
   - **The check:** each compartment's and membrane's total must agree, or the run stops ("a remesh lost mass").
     The tolerance, relative with an absolute floor for a zero total, is set from the spike's measured loss
     (expected near round-off, `1e-12`): with identical regions, a larger miss is a bug.

   *Recommendation and Jim's choice: yes.* Rejected options:
   - **Interpolation plus a rescale:** it needs a rescale to hit the total, and it is less accurate.
   - **A patch with its boundary fixed:** more exact outside the patch, but more machinery; it is kept as a later
     optimization (Not decided here).

4. **The rebuild: the whole geometry, in segments.** DOLFINx meshes are immutable, so a remesh builds a new
   parent mesh, tags, submeshes and maps (`realize_multi_compartment`'s tail, split so the remesh shares it).
   The solver runs in segments: it returns at the stride where a remesh is needed, and the runner:
   1. remeshes and transfers;
   2. starts a new bundle segment (ADR 010);
   3. builds a new `MovingPartition`, whose slide fractions and orientation reference are taken afresh;
   4. continues from that time, carrying over the step size.

   An output time that falls on a remesh is recorded once, after the remesh and any redone half-move, on the
   new mesh: only then does the geometry match that time.

   *Recommendation: yes.*

5. **Contact is a stop.** A remesh keeps every region's topology, so it cannot let a notch vanish.
   - If, after a move, the gap between two parts of the front falls below `δ = min(h/4, g₀/2)`, the run stops
     with a named error and the time. "Two parts" means points more than `h` apart along the front.
   - The same applies to the gap between the front and the box or another membrane.
   - `g₀` is the smallest such gap as built, with the same `h` exclusion, so the definition is not circular. A
     geometry built with a narrow gap stops only close to true contact.
   - This is ADR 014 decision 6's promised stop, needed here because a remesh lets runs reach it.

   *Recommendation: yes.*

   **The spike's answer: moving_bleb reaches contact at t ≈ 1.75** (Spike results). The remesh carries its mesh
   to that point; the front's own geometry stops it there.

6. **A front that cannot advance is a stop.** If a fresh mesh tangles on its first move (the retry of
   decision 2 fails), the cause is the front's own geometry, not the mesh: a valley that a front moving along
   its normal folds into a cusp once the displacement exceeds the valley's radius of curvature. The run then
   stops with a named error ("the moving membrane cannot advance at t = …: it folds or closes on itself"),
   rather than remeshing again. Real membranes resist this focusing (tension and bending give the motion a
   curvature-dependent part); a prescribed uniform normal speed does not, so it is a property of the model's
   kinematics.

   *Recommendation: yes.* The spike remeshed 20–40 times in place without taking a step before hitting its
   limit (Spike results).

7. **Realization: no traced membrane segment far below the mesh's resolution.** Marching squares on an `h`-grid
   gives arbitrarily short segments wherever a contour passes near a grid node. The simplification
   (`approximate_polygon` at `0.01 h`) removes them on a circle, but keeps them where the shape is finer than the
   grid: the 35-tooth cell's crests (radius 34 nm) and valleys (59 nm) at h = 0.04 become near-corners with 1–4 nm
   segments, which force cells of about 5e-7 next to the membrane. After the simplification, drop each traced
   vertex closer than `h/4` to the last one kept.

   *Recommendation: yes, as its own change to `realize._netgen_box_with_contours`*, with the realization tests
   and area fidelity on curved shapes.
   - Measured on the 35-tooth cell at h = 0.04: area changed by +0.0009 %, membrane length by −0.06 %,
     the shortest membrane segment went from 9.1e-4 to 8.5e-3, and the smallest cell from 4.8e-7 to 4.1e-5
     (85× larger).
   - It helps every moving run on a sharply featured analytic geometry, remeshed or not.

8. **Scope: 2D, serial.**
   - 3D needs a 3D supermesh (tetrahedron intersections), which does not exist here; interpolation plus a rescale
     is the rejected option. A later ADR.
   - Under MPI the skeleton would have to be gathered to rank 0, and the supermesh kernel is serial.
   - Both keep #228's stop.

   *Recommendation: yes.*

**Cost** (*estimates*; the spike measures them). For moving_bleb at h = 0.04 (138k cells), one remesh is about
2 minutes:
- Netgen on the whole partition: a few seconds (*an estimate*: realizing ADR 015's 85k-cell sphere-in-a-box
  benchmark took 2.2 s).
- Rebuilding the geometry and the solver: *an estimate*, about 10–30 s.
- The supermesh transfer: about 110 s for a compartment holding most of the cells, at the reviewer's measured
  0.19 ms per new triangle against 138k old ones. Once per compartment, shared by its species.

That is one or two strides' worth (a stride took about 100 s there in ADR 015's 4-stride profile), so a handful of
remeshes per run costs a few percent. Vectorizing the kernel's clipping would cut the transfer's share; it is not needed to start.

## Spike results (2026-11-10)

A spike (branch `spike/partition-remesh`, not merged) implemented decisions 1–4 without the contact stop or
decision 6, and ran moving_bleb's geometries.
- **The 35-tooth analytic recast, h = 0.1** (23k cells):
  - Without decision 7, the first tangle came at t = 1.05. The remesh worked (totals within 6e-15, 11 s), but
    the fresh mesh tangled on its first move, at cells beside 1 nm membrane segments.
  - With decision 7, the first tangle came at t = 2.15. One remesh carried the run to t = 2.55.
  - There every fresh mesh tangled on its first move, 20+ times. The front's narrowest gap stayed at 0.041,
    above the contact threshold, so the front folds (decision 6).
- **The image geometry, h = 0.04** (132k cells, without decision 7):
  - The first tangle came at t = 1.6; the earlier run without remeshing inverted cells at t ≈ 1.5.
  - Remeshes at t = 1.6 and 1.7 carried it to t = 1.75.
  - The front's narrowest gap fell from 0.021 to 0.011 to 0.008, below `δ` = 0.01 at t = 1.75: contact
    (decision 5).
  - Each remesh took about 78 s (Netgen 3.9 s, transfer 72–75 s), about 1.6 strides' worth.
- **What the 35-tooth and image runs show:** both stop at the time the shape predicts. A front moving along its normal at
  `0.05 (1 − cos t)` reaches the 59 nm valleys' radius at t ≈ 1.75. The remesh carries the mesh to that limit;
  the model's kinematics stop it.
- **The 8-tooth recast, h = 0.04** (133k cells, with decision 7): **carried to t = 3.5.**
  - One remesh, at t = 1.75; the run without remeshing had tangled near t = 2.0. The front's narrowest gap
    was 0.040, far from contact.
  - Totals were kept to 2.4e-13 across the remesh; the membrane pair X + B held at 1846.73 from t = 1.75 to 3.5.
  - The remesh took 79 s (Netgen 3.8 s, transfer 75 s), against about 36 minutes for the whole run.
  - A first attempt without decision 7 failed in the time integrator (`DIVERGED_STEP_REJECTED`) before any
    remesh. It also overlapped a laptop sleep, so the failure is *not attributed*; the rerun with decision 7
    completed.
- **Conservation across remeshes:** most totals were kept to 1e-13–1e-10, the worst to 3.8e-7 (132k cells) and
  3.7e-8 (23k). This is the supermesh kernel's absolute clipping cutoff, as decision 3 anticipates, growing
  with the mesh. Its cutoff must become relative to the cells' area before the check can be `1e-12`.
- **The lumped fallback** switched on in about half the remeshes, for the species with a sharp initial step.

## Verification

- **A remesh of an unmoved partition:**
  - every compartment's area and every membrane's length equal to round-off;
  - a linear field reproduced exactly in every compartment;
  - membrane values identical at the old nodes;
  - each total equal to `1e-12`.

  Cases: nested loops, a membrane ending on the box, a three-region junction.
- **A remesh of a moved, distorted partition**, with skeleton segments forced longer than `h` and cells of both
  orientations: this exercises Netgen's added points and the orientation rule, which an unmoved remesh at the
  same `h` does not.
- **The rebuilt mesh is conforming, geometrically:** its region-boundary facets cover the old skeleton's
  segments exactly, its exterior facets are exactly the box's, and a field diffuses across every former segment
  as across any interior edge.
- **Across remeshes:**
  - an expanding cell with remeshes forced keeps each species' total to `1e-12` throughout;
  - the exact-solution test (`tests/test_moving_partition_exact.py`) with a fixed number of forced remeshes,
    the same at every refinement, still converges at its rates (a remesh every stride would add `N·h²` and
    stall it);
  - a steep non-negative front stays non-negative.
- **moving_bleb's analytic recasts (35 and 8 teeth) and the image geometry at h = 0.04 to t = 3.5:** they run past
  t = 1.4 by remeshing, or stop with the contact error. Measure the number of remeshes, their cost and the
  smallest gap.
- **The undo and the second half-move:** a forced tangle on each half-move remeshes and continues, and the
  output at that stride's end is recorded once, on the new mesh.
- **Bundle:** each remesh starts a segment; the bundle reads back.

## Staging

1. This ADR, accepted.
2. **A spike** (not merged): the whole-partition remesh with the projection on moving_bleb's recasts. It answers
   whether remeshing carries the run or reaches contact, how often it remeshes, and at what cost. The numbers go
   into this ADR before code lands.
3. The contact stop (decision 5) and the realizer fix (decision 7): useful on their own.
4. The skeleton remesh, the rebuilt geometry and the transfer (decisions 1, 3, 4), with the unmoved-remesh tests.
5. The undo and the trigger (decision 2), the stop for a front that cannot advance (decision 6), the
   segmented solver, and the remaining verification.

## Not decided here

- **Patch remeshing**, as an optimization for runs that remesh often: it keeps the field outside the patch exact.
- **3D and MPI** (decision 8).
- **A vectorized supermesh kernel with a spatial index** (the cost paragraph): the broad phase is all-pairs.
- **MMG** (mmgtools, local adaptation in 2D and 3D) as an alternative mesher. Its licence and packaging are
  unchecked against ADR 008.
- **A front that develops a cusp:** a geometric question about the front, not the mesh.

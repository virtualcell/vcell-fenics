# ADR 015 — Relaxing the mesh of a moving partition

**Date:** 2026-11-09
**Status:** Proposed. Rewritten after Jim's review of the first draft (remeshing the whole partition with a
transfer that conserved only each species' total): **relax the mesh within ALE first, with its topology fixed
and no transfer; a true remesh comes later, local and locally conservative, only where relaxation is not
enough.** The rewrite was then revised after an independent review (11 findings, all addressed below).
**Builds on:** ADR 014 decisions 3–6 (accepted 2026-11-09: the bulk on both sides of the moving membrane is at
rest and transported relative to the mesh; the membrane's mesh may slide along it; remeshing "comes second",
its first stage's stop on a degraded mesh built in #228, its stop on contact not yet), ADR 013 (the single-mesh
moving path).
**Would amend:** ADR 014 decision 5 ("the parent mesh moves by a harmonic extension of the membrane's
displacement"), `backend/moving_partition.py`, and later `backend/discrete._MeshMotion` (decision 6).

## Context

**The failure** (*measured in runs, 2026-11-09*). moving_bleb (BioModel 294599465: a cell with 35 small teeth,
its membrane pulsing along its normal by `0.05 sin t`, so at most about 0.1 µm) at VCell's mesh (h = 0.04)
tangled its outside compartment at t ≈ 1.5 of 5: 4 inverted triangles, 79 by t = 2.6. An analytic recast of
the same cell tangled at t ≈ 1.4; one with 8 teeth at t = 2.0. At h = 0.2 it ran to t = 5 untangled. The
smallest cells sit at the teeth: area 6e-5 in the image mesh, 5e-7 in the analytic one, against about 7e-4
for a cell of size h. Since #228 such a run stops with `MeshQualityError`.

**Where it tangles** (*measured in the stalled h = 0.04 run's output, 2026-11-09*). Every inverted cell lies
against the membrane (within 0.008 at t = 1.5, 0.034 at t = 2.6), in a neck of the outside compartment between
two parts of the front: the narrowest gap between points of the membrane more than 0.4 apart along it closes
from 0.253 (t = 0) to 0.226 (t = 1.5), 0.157 (t = 2.0) and 0.094 (t = 2.6). The front's displacement is at most
0.1, so the gap bottoms out near 0.05 at t = π: the front does not touch itself, but squeezes the neck's cells
by about 80 %.

**Why the mesh fails there** (*inferred from the code, not yet measured*). The partition's interior moves by a
plain harmonic extension of the membrane's displacement over each stride (`moving_partition.py` 163–172:
`∇u·∇q` with the front's displacement on the membrane and zero on the box and every other membrane), solved
afresh on the moved mesh every stride. A plain Laplacian moves a tiny cell as much as a large one beside it, and
reassembling it on the moved mesh accumulates distortion even when the front comes back. A squeeze of 80 % is
survivable with the topology fixed only if it is spread evenly across the neck's cells; whether a better
extension does that is what the spike (Staging 2) measures first.

**The interior's motion is free** (*checked in the code on `main` at `7344a8d`*). Each compartment's material
is at rest and its species are transported relative to the mesh: the bulk form has `−u (c − w)·∇q` with `w` the
parent's mesh velocity (`multi_compartment.py` 476), and the time term the swept rate `ln(|Kⁿ⁺¹|/|Kⁿ|)/h` from
each cell's actual measure before and after the move (`_SweptMeasures`, `interface_coupled.py` 296–306). Only
the membrane's normal motion is physical (ADR 014 decision 4 already uses this for the membrane's tangential
slide). So any interior node motion is accounted for by the same terms — no interpolation, no rescale — **to
first order in the stride**: the mesh jumps at a stride's start and the stride is then solved on the moved mesh
with `w` the displacement over the stride (`multi_compartment.py` 720–727), the splitting error of one stride
that ADR 014 decision 4 names, and `test_the_conservation_error_is_first_order_in_the_stride` measures. So a
different interior motion changes results within discretization error, not exactly; the partition tests'
conservation tolerances (1e-5 to 1e-2 relative, `tests/test_moving_partition.py` 200–308) bound how much. The
moving membrane's species, its normal, the traces its velocity reads and the fluxes on it depend only on the
membrane's own nodes, so interior relaxation does not touch them; nor species on fixed membranes, whose nodes
stay put (*checked in the code*).

**Remeshing is the other remedy, and the costlier one.** A remesh changes the topology and must transfer every
field; the first draft of this ADR remeshed the whole box and conserved each species' total only, by a
rescale. Relaxation keeps the topology, so it needs neither.

## Decisions

1. **The interior mesh motion of a moving partition is a mesh choice.** The parent's interior nodes, and the
   nodes of every submesh with them, may move however keeps the mesh good, provided the moving membrane's
   nodes follow the front (with ADR 014 decision 4's slide) and the box's and the fixed membranes' nodes stay
   put. The solver already transports every compartment species relative to that motion and dilutes it by the
   actual swept measures, so the choice needs no transfer, and changes results only within the scheme's
   first-order splitting error in the stride (Context).

   *Recommendation: yes.* It is ADR 014 decision 3's reading (the bulk at rest, transport relative to the
   mesh) made explicit for the interior, as decision 4 made it for the membrane's tangential motion.
   *Costs:* the splitting error grows with how far a cell moves in a stride against its own size (a Courant
   number `|w|·Δt / size`), and with how unevenly neighbouring cells change; stiffening (decision 2) moves small
   cells by nearly the front's full displacement on purpose. For moving_bleb the front moves at most 0.0025 per
   stride (speed 0.05, strides ≤ 0.05), a Courant number of about 0.25 at the image mesh's smallest cells
   (~0.01 across) and about 2.5 at the analytic recast's (~0.001). The bulk Péclet number `|w| h / D` is small
   (D = 10 for its bulk species, so about 2e-4). *Inferred:* an extension with a maximum principle keeps the
   interior's `|w|` no larger than the front's. The verification records both numbers.

2. **Stiffen the extension by cell size.** The interior displacement solves `∇·(k ∇u) = 0`, with `k = 1/|K|`
   per cell (`|K|` the cell's area, or volume in 3D), instead of `∇·∇u = 0`. Small cells become stiff and move
   almost rigidly with what is beside them; large cells absorb the deformation. This is the standard
   Jacobian-based stiffening of ALE mesh motion (Tezduyar et al.; Stein, Tezduyar & Benney 2003), with
   exponent 1.
   - Options: exponent 0 (today's plain Laplacian), 1, or 2 (stiffer still; the literature's range).

   *Recommendation: exponent 1*, to be confirmed by the spike (Staging 2), which measures 0, 1 and 2.

3. **Freeze the extension on the mesh as built: extend the total displacement, not each stride's increment.** The interior's
   position at any time is the as-built position plus the stiffened extension, assembled on the **as-built**
   mesh, of the boundary's total displacement since the build (the membrane's nodes where the front and slide
   have put them; the box and fixed membranes at zero). The mesh velocity of a stride is the change of that
   position over it.
   - **(a) Incremental** (today's scheme, stiffened on the current mesh): adapts its stiffness as cells change,
     but accumulates distortion stride by stride; an oscillating front does not return the mesh to its start.
   - **(b) Total from the as-built mesh**: path-independent — the interior is a function of the boundary's
     current position only, so moving_bleb's pulsing front returns its mesh exactly; and cheaper, since the
     operator is assembled and factorized once. It is a linear (small-deformation) map of the as-built mesh, so
     a boundary displacement large against the cells near it can still fold them; decision 4 then stops the
     run.

   - The real difference between (a) and (b) is the operator: with one linear operator frozen on the as-built
     mesh, extending the total displacement equals summing the increments' extensions. Today's accumulation
     comes from reassembling the operator on the moved mesh. (b) is built by assembling and factorizing the
     operator once at construction and solving with it every stride (`LinearProblem` reassembles on each
     `solve`, so it is replaced by a kept factorization).
   - A node shared by the front and the box or a fixed membrane (a front ending on one) takes the front's
     value today (`moving_partition.py` 156); stiffening concentrates the squeeze in the cells at that junction.
     Every corpus moving front is a closed loop, so it does not arise there.

   *Recommendation: (b)*, to be confirmed by the spike against (a) on moving_bleb, an expanding cell and a
   rotating membrane. *Inferred, not measured:* that (b) handles moving_bleb, whose displacement is small and
   oscillating, and that (a) is better for large monotone expansions.

4. **When relaxation is not enough, the run stops — no remesh in this ADR.** #228's checks stay: an inverted
   cell, or growth past the hard limit, stops the run with the time. The message now says that the mesh could
   not be kept valid by relaxation and suggests a coarser mesh, a smaller motion or a shorter run. A growth
   past 4× (the single-mesh path's remesh limit, `_REMESH_QUALITY_LIMIT`, `runner.py` 347; new wiring for the
   partition) is logged once as a warning, so a run approaching the stop says so.
   - **A true remesh is a later ADR**, written with this one's measurements: **local** (only the patch that
     relaxation could not keep, every node and value elsewhere kept exactly) and with a **locally conservative
     transfer** in the patch. A whole-box remesh with an interpolation rescaled to each species' total — the
     first draft — is rejected: it changes every field everywhere and conserves only totals.

   *Recommendation: yes.*

5. **Contact is a stop.** Relaxation keeps the topology, so it cannot let the moving membrane meet the box,
   another membrane or itself. After each move, if the moving membrane has come within `δ` of a box facet or of
   another membrane's facet, or two of its own points are within `δ` in space while more than `4δ` apart along
   it, the run stops with a named error and the time ("the moving membrane has reached the box at t = …").
   `δ = min(h/4, g₀/2)`, with `g₀` the smallest such gap in the mesh as built, so a geometry built with a
   narrower gap does not stop at once.
   - A front that touches the box or another membrane as built (`g₀ = 0`: it ends on it) is not checked
     against that boundary, only against the others.
   - The self-contact test ("apart along it") is 2D only, where the membrane is an ordered loop; in 3D the
     stop covers the box and other membranes. Under MPI the membrane's nodes are gathered for the test.

   *Recommendation: yes.* ADR 014 decision 6's first stage promised this stop and #228 did not build it;
   without it a membrane pushed into the box surfaces as tangling, with the wrong cause. *`h/4` is a judgement,
   not a measurement* (about a quarter of a cell between the membrane and what it approaches); the contact
   tests check that the stop comes before any cell inverts.

6. **The single-mesh path takes the same mesh motion, later.** Its interior is equally free (its volume is at
   rest and transported relative to the mesh, ADR 013 decision 3), and it remeshes when its mesh degrades; a
   better motion makes those remeshes rarer, and with them its own transfers. A separate stage, with its own
   comparison against today's results, since that path is accepted and its results would change within
   discretization error.

   *Recommendation: yes, after the partition.*

**Scope.** The stiffened extension and the frozen map work alike in 2D and 3D and under MPI (one linear solve on
the parent mesh; under MPI the kept factorization needs a parallel direct solver such as MUMPS, or an iterative
one — *not checked in this build*). The membrane's slide stays 2D and serial (ADR 014 decision 4), so in 3D
and under MPI the membrane's nodes follow `v` exactly, as today.

## Verification

- **The central claim, directly**: a static front (`v = 0`) with an imposed, strongly non-uniform interior mesh
  motion (in the style of ADR 013's lab-frame advection test): every compartment field converges to the
  static solution as the stride and `h` shrink, and each species' total holds, for compartment species,
  moving-membrane species and species on a fixed membrane.
- **Agreement with today's scheme**: the partition tests pass within their tolerances; a translating cell
  agrees with today's scheme to discretization error (the outside compartment deforms differently under the
  two, since the box stays put).
- **moving_bleb at h = 0.04 to t = 5**: no inverted cell, X + B conserved, the worst cell-size growth and the
  largest Courant and Péclet numbers reported, and the neck's cells tracked; its analytic 35- and 8-tooth
  recasts likewise; with decision 3(b) the interior at any time equals the as-built map of the membrane's
  position then (the membrane itself returns at t = 2π only to within its own time error).
- **How far relaxation reaches**: an expanding cell and a translating cell run until the stop, for exponents
  0, 1, 2 and decision 3 (a) and (b) — the displacement each reaches before stopping, against today's scheme.
- **Shear**: a rotating membrane (ADR 014's own case), which a size-based measure does not see; checked by
  orientation and the smallest angle.
- **Agreement across meshes**: the analytic recast at h = 0.2, 0.1 and 0.04 converging.
- **3D**: the moving-nucleus pulse on the segmented-cell image (run 2026-11-09 with today's scheme, conserving
  to 2e-4) repeated, with a larger amplitude, to compare how far each scheme reaches.
- **Contact**: a cell expanding toward the box, toward a fixed inner membrane, and a membrane folding toward
  itself each stop with the contact error before any cell inverts; a geometry built with a gap under `h/4`
  runs.

## Staging

1. This ADR, accepted.
2. **A spike** (not merged): first, where the cells fail in the neck under each option; then decisions 2 and
   3's options on moving_bleb, an expanding cell and a rotating
   membrane; the numbers go into this ADR, and decisions 2–3 are confirmed or revised before any code lands.
3. The relaxation in `MovingPartition` (decisions 1–4), with the verification above.
4. The contact stop (decision 5).
5. The single-mesh path (decision 6).
6. A later ADR for local remeshing, if stages 2–3 show where relaxation stops.

## Not decided here

- **The true remesh** (decision 4): where relaxation is not enough — large expansion needing more cells,
  compression needing fewer — local, with a locally conservative transfer.
- **Sliding the box's and fixed membranes' nodes along themselves** (more freedom for relaxation near the walls).
- **Optimization-based smoothing** (moving nodes to maximize a quality measure), if the stiffened extension is
  not enough.
- **The membrane's slide in 3D and under MPI** (ADR 014 decision 4).

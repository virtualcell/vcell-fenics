# ADR 015 — Conservative mesh moves on a moving partition

**Date:** 2026-11-10
**Status:** Accepted (Jim, 2026-11-10) and built. It began as the second stage of ADR 014 decision 6
(remeshing). Jim rejected remeshing the whole partition with a transfer that conserved only each species'
total. The relaxation that replaced it was measured and not adopted (decision 4). Two independent reviews of
the earlier drafts (12 and 11 findings) shaped what was measured.
**Amends:** ADR 014 decision 5 ("the swept-measure rates supply the GCL-consistent time terms"): the moving
partition no longer uses a swept rate. ADR 014 decision 6's remesh stays open (decision 4 below).

## Context

**The scheme it replaces** (`multi_compartment.py` and `moving_partition.py` on `main` at `7344a8d`). Each stride
moved the mesh at its start by the stride's whole displacement, then integrated the stride on the moved mesh
with a dilution rate `ln(|Kⁿ⁺¹|/|Kⁿ|)/h` per cell (`_SweptMeasures`). That rate cancels a cell's volume change
only while the field in it stays uniform. Diffusion and exchange move mass between cells that dilute at
different rates, so totals drifted at first order in the stride. The test
`test_the_conservation_error_is_first_order_in_the_stride` measured exactly that.

**Measured on the partition tests** (2026-11-10; `tests/test_moving_partition.py`'s cases):
- Ligand exchanged with a membrane receptor across a moving membrane: total off by −7.2e-4, −3.6e-4 and
  −1.8e-4 at 10, 20 and 40 strides.
- An inside species under a stretch, with the mesh moved non-uniformly (the stiffened extension of decision 4):
  −4.7e-4 at 10 strides, halving with the stride.

**Literature** (*checked 2026-11-10*):
- Arbitrary Lagrangian–Eulerian (ALE) methods split a step into a mesh move that carries the material, then
  transport relative to the mesh: Hirt, Amsden & Cook, *J. Comput. Phys.* 14 (1974) 227–253 (Lagrangian
  phase, then rezone and remap).
- On moving meshes, stability and conservation are tied to the discrete geometric conservation law (GCL):
  - Nobile & Formaggia, *East-West J. Numer. Math.* 7 (1999) 105, for ALE finite elements on parabolic
    problems;
  - Farhat, Geuzaine & Grandmont, *J. Comput. Phys.* 174 (2001) 669–694, for finite volumes.
- None of these papers states the mass-matrix form used below. It is our reading of the conservative ALE
  finite-element form: the P1 basis functions move with the mesh, and the time derivative is taken of the
  moments `∫φᵢu`.

## Decisions

1. **Each mesh move carries every species' nodal moments exactly.** Across a move, a species' P1 values are
   replaced by the solution of `M⁺u⁺ = M⁻u⁻`, where `M⁻` and `M⁺` are its mass matrices before and after the move.
   - The topology is unchanged, so `φᵢ` is the same basis function on both meshes, and every node's moment
     `∫φᵢu` is kept.
   - The P1 basis sums to one, so the species' total is kept exactly.
   - The move carries the material with the mesh. The stride's transport relative to the mesh,
     `−u (c − w)·∇q`, then moves it back. That term is in conservative form, so the stride keeps each total
     too, and an exchange between species keeps their sum.
   - The swept-rate dilution terms are removed, from compartment and moving-membrane species alike.

   *Recommendation and decision: yes.* Every total is now kept to round-off whatever the mesh does. Measured:
   about 1e-15 for the stretch, for the exchange across the moving membrane at every stride count, and for a
   membrane species (1e-7 before).

2. **Symmetric split: half the move, the stride on the midpoint mesh, the other half.** Moving the whole
   stride at its start and then carrying the moments is first order in the stride, and about 1.7× less
   accurate than the scheme it replaces (decision 1's measurements). Splitting the move in two halves around
   the stride's integration (Strang's symmetric split) recovers the accuracy. Each half carries the moments
   (decision 1), so conservation is unchanged. The membrane's velocity is read at the end time of each half
   (the stride's midpoint, then its end).

   *Recommendation and decision: yes.* Measured against an exact solution (Verification): about 3× more
   accurate than the old scheme at the finest level, and second order in time for the membrane species.

3. **The consistent mass matrix, not the lumped one.** The lumped variant (`uᵢ⁺ = mᵢ⁻uᵢ⁻ / mᵢ⁺`) is
   positivity-preserving in principle. In the measurements it was no more accurate, and it showed no less
   undershoot on a steep front (about −1.1e-2 in every scheme, from unstabilized Galerkin on an under-resolved
   front with D = 0.001, not from the move).

   *Decision: consistent.* A direct solve (LU) of each mass matrix per move: two per species per stride.

4. **Mesh relaxation is not adopted. moving_bleb needs a local remesh.**
   - Variants measured on the analytic 35-tooth recast of moving_bleb (BioModel 294599465) at h = 0.04:
     - an extension stiffened by cell size (`k = 1/|K|`), frozen on the as-built mesh;
     - the same, assembled on the current mesh every stride;
     - the plain Laplacian.

     Every one tangled at t = 1.35–1.4, as the scheme before them did.
   - The cells that fail sit in notches of the outside compartment between a tooth valley's two flanks. The
     valleys' radii of curvature are 0.002–0.004; the front has moved 0.03 by then. The image recast fails the
     same way in a wider neck, closing from 0.25 to 0.09 by t = 2.6.
   - The notch's area shrinks toward zero, so its cells must be removed, which no fixed-topology mesh can do.
     The front itself never folds: no membrane segment turns over.
   - So decision 6 of ADR 014 (remeshing) stays open. A later ADR designs a **local** remesh that removes
     cells where a region closes, with a locally conservative transfer in the patch. A whole-box remesh with an
     interpolation rescaled to each species' total stays rejected.

   *Decision: as stated.* The extension stays today's plain Laplacian, assembled on the current mesh.

5. **The single-mesh moving path (ADR 013) keeps its scheme for now.** Its method-of-lines path uses the same
   swept rate (`discrete._MeshMotion.effective_dilution_rate`), and its backward-Euler path a per-cell ratio.
   Moving it to decision 1–2's scheme is a follow-up, measured against its own tests and the RacRho study.

## Verification

**Exact solution** (`tests/test_moving_partition_exact.py`, and the study behind it):
- The membrane, a circle, expands along its normal, `v = 0.2 n`, so `R(t) = 0.5 + 0.2 t`.
- Inside it a bulk species at rest, `u = e⁻ᵗ(2 + cos π r²)`, with D = 0.5.
- On it a membrane species `ρ = 1 + t²` that rides and dilutes, with D_Γ = 0.1.
- They are coupled by a flux that depends on the computed `trace(u)`. The total flux relative to the moving
  membrane is `D ∂ₙu + a u`.
- The geometry is traced at 8× the mesh's density, so the error measured is the solver's. The realizer's
  contour grid has at least 51 points per axis, and below h ≈ extent/50 it caps convergence: the traced circle
  then starts at radius 0.4996, not 0.5.

Errors at t = 1 (relative L2 of u / largest error of ρ), h halved with the strides quadrupled:

| h, strides | before (swept rate) | decision 1 alone | decisions 1–2 (built) |
|---|---|---|---|
| 0.2, 5 | 6.2e-2 / 3.5e-2 | 1.09e-1 / 5.3e-2 | 1.14e-2 / 7.8e-3 |
| 0.1, 20 | 1.46e-2 / 7.6e-3 | 2.5e-2 / 1.17e-2 | 4.8e-3 / 2.4e-3 |
| 0.05, 80 | 3.6e-3 / 2.0e-3 | 6.1e-3 / 3.1e-3 | 1.34e-3 / 6.1e-4 |

- Every scheme converges to the exact solution, second order in space.
- In time, the old scheme and decision 1 alone are first order.
- With decisions 1–2, against a 160-stride reference on one mesh (h = 0.05, 5 → 40 strides):
  - ρ is second order (rates 2.05, 2.02, 1.98);
  - u falls between first and second order (1.15–1.45, inflated by the finite reference), at 1/8 of the old
    scheme's error at every stride count.
  - *Inferred, not tested:* u's remaining first-order part comes from sources and coefficients evaluated at
    the stride's times on the midpoint mesh.

**Tests:**
- The partition tests' conservation checks are tightened to round-off (`rel=1e-12`; a nonlinear reaction's pool
  `1e-10`).
- The first-order-drift test is replaced by `test_ligand_is_conserved_to_round_off_at_any_stride` (3, 10 and
  30 strides).
- The exact-solution tests check:
  - the error bounds and their fall under refinement;
  - ρ's second order in time;
  - the exchange's total to round-off.

## Not decided here

- **The local remesh** (decision 4): ADR 014 decision 6's open stage.
- **The single-mesh path's scheme** (decision 5).
- **The realizer's contour-grid floor** (Verification): a fine `h` does not trace an analytic shape more
  finely than about extent/50.
- **The contact stop** ADR 014 decision 6 promised (the moving membrane meeting the box, another membrane or
  itself): still not built.
- **Positivity** on steep, under-resolved fronts: a stabilization question, independent of the move.

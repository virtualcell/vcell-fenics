# ADR 022 — Error tolerances that mean the same thing for every species

**Date:** 2026-11-18
**Status:** Proposed. Asked for by Jim on 2026-11-18 ("start on per-species tolerances"), after he pointed out
that a looser tolerance is unsafe because many models are not well scaled (ADR 021 decision 1's notes). Jim's
scope: every solver that uses the method of lines (decision 3). Reviewed once (16 findings, 1 blocking; "Review"
at the end).

**Builds on:**
- ADR 011: the SimulationTask's `ErrorTolerance` (one absolute and one relative tolerance) reaches the runner as
  `RunOptions.atol` and `RunOptions.rtol` (`cli.py` 302–303, 424–425, *checked*).
- ADR 021 decision 1's notes (2026-11-18): the persistent integrator, and the observation that the stagger's
  error, not the integrator's, dominates on the myosin model.
- PR #262: the fix to the order of the tolerances ("the swap" below).

## Context

### The integrators' error control (PETSc 3.25.1, *checked* in its source)

Every method-of-lines path runs a PETSc TS: adaptive BDF by default, and on the single mesh any `ts_type`
(`reaction_diffusion.py` 76, 210). The adaptive controller accepts a step when a weighted norm of its local error
estimate is at most 1. For each entry `i` of the state:
- the weight is `atolᵢ + rtolᵢ · max(|uᵢ|, |yᵢ|)`, with `u` and `y` the two solutions the estimate compares
  (`vector.c` 2539–2544);
- an entry whose weight is 0 is skipped: it is in neither the sum nor the count (`vector.c` 2545–2558);
- **the default norm is a root mean square:** the sum of squared weighted errors over the entries with a positive
  weight, summed across ranks with an `Allreduce`, divided by their global count, square-rooted (`vector.c`
  2567–2583; `ts.c` 5253–5258). `-ts_adapt_wnormtype infinity` takes the maximum instead.

The C prototype is `TSSetTolerances(TS ts, PetscReal atol, Vec vatol, PetscReal rtol, Vec vrtol)` (`ts.c` 5159):
either tolerance may be a vector with one entry per unknown, used in preference to the scalar. PETSc keeps a
*reference* to the vector (`ts.c` 5169–5173), so changing its entries in place changes the tolerances without
another call. petsc4py's `TS.setTolerances(rtol, atol)` takes either as a float or a `Vec` at run time (*checked*:
both, by the reviewer), though its type stub says float only, so mypy needs a cast at that call.

All four of our call sites pass two scalars (`multi_compartment.py` `make_ts` 845–862, `reaction_diffusion.py`
222, `interface_coupled.py` 1147 and 2047, *checked*). VCell's own PDE solver does the same: CVODE with scalar
tolerances, `CV_SS` (`vcell-fvsolver` `SundialsPdeScheduler.cpp` 57 and 520, *checked*). VCell's defaults for its
Sundials solvers are `atol = 1e-9`, `rtol = 1e-7` (`ErrorTolerance.java` 58–60, *checked*).

### The swap (fixed separately, PR #262)

From the first method-of-lines integrator (2026-06-16, `1c75601`) until 2026-11-18, every call passed
`setTolerances(atol, rtol)` positionally, but petsc4py's order is `(rtol, atol)` (*checked*: the signature, and
`ts.view()` printing "relative error tolerance of 1e-08, absolute error tolerance of 1e-06" for a call
`setTolerances(1e-8, 1e-6)`). Every run used the absolute tolerance as the relative one and vice versa.
- At `|u| ≈ 1` the two weights nearly coincide (`1e-6 + 1e-8·|u|` against `1e-8 + 1e-6·|u|`); they part by up to
  100× as `|u|` moves away from 1. The MMS suite and cross-validation are O(1), which is why they did not see it
  (*inferred*).
- For a species of size 1e-3 at `rtol = 1e-6`, `atol = 1e-12`, the relative error was 2.2e-2 swapped, 1.4e-4
  fixed (`tests/test_ts_tolerances.py`, measured).
- ADR 021 decision 1's persistent-integrator table was measured swapped. Its runs passed `atol = rtol·1e-2` on the
  O(1) myosin, so the effective weight was close to the labelled `rtol` either way, and its comparisons stand
  (*inferred*).

### The problem

With the order fixed, two scaling problems remain. Measured on 2026-11-18 at `rtol = 1e-6`, `atol = 1e-8`; the
scripts are listed in the References.

**1. One absolute tolerance for every species.** A uniform species decaying as `c' = −c` to `t = 2` (single mesh,
h = 0.3), its relative error at `t = 2`:

| scale of `c` | `atol = 1e-8` (scalar): steps, error | `atol = min(1e-8, rtol·scale)`: steps, error |
|---|---|---|
| 1e3 | 141, 1.4e-4 | 141, 1.4e-4 |
| 1 | 140, 1.5e-4 | 140, 1.5e-4 |
| 1e-3 | 53, 1.3e-3 | 94, 3.6e-4 |
| 1e-6 | 15, 9.3e-2 | 94, 3.6e-4 |
| 1e-9 | 14, 1.6e-1 | 94, 3.6e-4 |
| 1e-12 | 14, 1.6e-1 | 94, 3.6e-4 |

With the scaled `atol` the error is the same from 1e-3 down; above `atol/rtol = 1e-2` the scalar `atol` is
negligible either way and the error is 2.4× smaller (the scaled `atol` adds its own term to the weight).

The same with two species twelve decades apart in one multi-compartment solve (nested circles, h = 0.08): `c`
(size 1, slow decay) and `e` (size 1e-12, a non-uniform start diffusing with D = 1), relative L² errors at `t = 1`
against a tight run:

| `e`'s source | scalar `atol`: steps, `c`, `e` | per-species `atol`: steps, `c`, `e` | the same, linear solve at `rtol` 1e-12 |
|---|---|---|---|
| none (`e` in the medium) | 14, 1.8e-6, 1.5e-2 | 194, 3.2e-8, 4.8e-6 | 194, 3.2e-8, 5.4e-6 |
| none (`e` in the cytosol) | 14, 1.8e-6, 1.0e-1 | 207, 7.7e-8, 3.4e-6 | 207, 7.7e-8, 2.4e-7 |
| `1e-12·c − 2e` (coupled) | 14, 1.8e-6, 3.0e-2 | 223, 1.1e-8, 2.4e-5 | 223, 1.1e-8, 7.1e-5 |
| `1e-12·c² − 2ec` (nonlinear) | 14, 1.8e-6, 3.1e-2 | 223, 1.1e-8, 2.5e-5 | 223, 1.1e-8, 7.3e-5 |

Under the scalar `atol` the integrator takes 14 steps paced by `c` and ignores `e` (1.5–10 % error). Per species,
`e` is controlled. A tighter linear solve does not change that consistently, so on these models the linear solve
does not pollute the small species (decision 6 keeps a test for it).

**2. The norm averages over every unknown.** A well-mixed nucleus `N` (a region variable, one unknown, `N' = −3N`)
beside two slowly decaying bulk species, nested circles, its relative error at `t = 2` (reproduced exactly by the
reviewer):

| h | unknowns | 2-norm (default): steps, `N` error | infinity norm: steps, `N` error |
|---|---|---|---|
| 0.12 | 803 | 126, 5.4e-3 | 365, 5.9e-4 |
| 0.06 | 1861 | 111, 7.0e-3 | 365, 5.9e-4 |
| 0.03 | 4620 | 97, 9.4e-3 | 365, 5.9e-4 |

`N` is one entry among `n` in the mean square, so its weight is `1/n`: as the mesh is refined the integrator
controls it less, its error grows and the steps fall. A membrane species beside bulk species is diluted the same
way, less severely (*inferred*: fewer unknowns on a curve than in an area).

## Decisions

1. **Each species gets its own absolute tolerance, from its own scale.**
   - `atol_s = min(atol, rtol · scale_s)`, where `scale_s` is the largest `|u|` of species `s` so far over the whole
     domain (its initial values to start with, raised after each accepted step).
   - **A floor for noise.** `scale_s` counts only above `ε · max_s' scale_s'`, with `ε` the machine epsilon
     (2.2e-16): a species no larger than the round-off of the largest one cannot be resolved in a shared solve
     (*inferred*), so below that it keeps `atol` (as does a species that is exactly zero). This stops an initial
     condition of 1e-30 next to O(1) species, or P1 undershoot near a front, from collapsing the step.
   - **Never less control than the scalar `atol`:** every `atol_s ≤ atol`. A large species is unchanged (its
     `rtol·|u|` dominates either way); a small one gets the control its size needs. Tightening has a cost: the
     steps above (14 → 194–223) are what controlling `e` takes, and decision 5 measures it across the suites.
   - **The limitation:** `scale_s` never falls. A species that decays by decades after its peak is controlled at
     `rtol · peak` absolute, no worse than the scalar `atol` but not relative to its present size.
   - **Where `scale_s` lives.** In an object the integrator's caller owns, so it persists across a stride loop's
     fresh TS per stride, the persistent integrator (ADR 021), and a restart after a remesh (`initial=`). It is
     updated in a TS post-step, collectively: each rank's owned maximum, then an `Allreduce(MAX)`.
   - **The options:**
     - **(a) Initial values only.** A species that starts at zero, or grows by decades, keeps the scalar `atol`.
     - **(b) The running maximum,** as above.
     - **(c) The present maximum,** raised and lowered. Relative to the present size throughout, including after
       a decay, at the cost of tightening as a species falls (more steps for a species that may no longer
       matter).
     - **(d) A model-supplied scale per species.** VCell has no field for it today; it could be added later and
       would override the others.
   - *Recommendation: (b), with the floor.*

2. **The error norm weighs each species equally, whatever its number of unknowns.**
   - **The options:**
     - **(a) The 2-norm over all unknowns** (today): a region variable's control weakens as the mesh is refined.
     - **(b) The infinity norm:** every unknown meets the tolerance, at every node.
     - **(c) Balanced vector tolerances under the 2-norm.** Species `s`'s `atol` and `rtol` entries are multiplied
       by `√(n_s / n)` (`n_s` its counted unknowns, `n` all counted unknowns, both global). Its share of the mean
       square becomes its own mean square, so the norm becomes `√(Σ_s RMS_s²)`, and a step is accepted only if
       the species' mean squares *together* are at most 1 (*checked* against the norm above). The budget is
       shared: with `S` species erring alike, each meets `1/√S` of its tolerance. A species with one unknown is
       held at every step to its weighted error, exactly as under (b).
   - **Measured.** When the region variable drives the step (the nucleus model above), (c) and (b) are identical:
     365 steps and 5.9e-4 on every mesh (the reviewer's measurement). When bulk species drive it (a non-uniform
     cytosol and medium diffusing with D = 0.05, exchanging across the membrane, beside a slowly decaying nucleus;
     `t = 1`; errors against a tight run on the same mesh):

     | h | unknowns | (a) 2-norm: steps | (b) infinity: steps | (c) balanced: steps |
     |---|---|---|---|---|
     | 0.08 | 1160 | 192 | 313 | 214 |
     | 0.04 | 3123 | 205 | 340 | 227 |

     All three hold `N` to 2–4e-7 and the bulk species to 0.6–1.8e-5 here. (b) costs 63–66 % more steps than (a),
     (c) 11 %: (b) also turns each bulk species' RMS into its worst node.
   - *Recommendation: (c).* It is never costlier than (b) on these models, much cheaper when bulk species drive
     the step, and mesh-independent where (a) is not.

3. **Where it applies: every method-of-lines integrator** (Jim, 2026-11-18).
   - A PETSc TS is created in three modules only (*checked*: `PETSc.TS().create` appears in no other file under
     `src/`), and every method-of-lines path runs through them. One helper builds the tolerance vectors from the
     state's layout and is called at all four `setTolerances` sites:
     - **the multi-compartment solver** (`make_ts`), every route through it (fixed, moving partition, force
       balance). Its layout: one scalar P1 block per species; or, when membrane species make the Newton
       matrix-free, one block per subdomain with a component per species, interleaved (`multi_compartment.py`
       467–480); then one Real per region variable, owned by one rank;
     - **the single-mesh method of lines** (`reaction_diffusion.py`): `integrate_discrete_problem` and its moving
       and stride variants, which the ALE paths and the runner call. The problem may carry several species as
       interleaved components of one vector space (`runner.py` 477, `discrete.py` 154), so decisions 1 and 2 apply
       per component there too;
     - **the two-compartment integrators** (`interface_coupled.py` 1147 and 2047): the runner no longer routes to
       them, but their tests and cross-validation do.
   - **Dirichlet-held unknowns** get `atol = rtol = 0`: PETSc then leaves them out of the sum and the count, so
     they neither dilute a species' mean square nor count towards `n_s` (`vector.c` 2545–2558).
   - **Not affected:** backward Euler (no error control), and the drivers with their own fixed stepping (Stokes and
     FSI, phase field, unknown-motion mechanics), which create no TS.

4. **What the user's tolerances mean, and what changes.**
   - **`rtol`** stays the relative tolerance each species is held to, in its own RMS. Under decision 2 (c) the
     entries PETSc sees are `rtol · √(n_s/n)`; that is how the per-species meaning is implemented, not a different
     tolerance.
   - **`atol` changes meaning:** from the absolute tolerance of every unknown to a ceiling, the largest absolute
     tolerance any species gets. A SimulationTask's `ErrorTolerance`, a true absolute tolerance in VCell's CVODE
     (`CV_SS`), is read this way from now on. Every species is controlled at least as tightly as before.
   - **The summary records** `rtol`, `atol`, `tolerance_scaling`, the norm, and each species' final `atol_s` and
     `scale_s`, before balancing.

5. **On by default, decided by measurement.** `--tolerance-scaling species|scalar`, and in `RunOptions`.
   - Before the default is set, the implementation PR measures steps and wall time before and after on the MMS
     suite, the cross-validation cases, the integration tests, and the coverage survey's runnable BioModels.
   - *Recommendation:* default `species`, unless that measurement shows a cost the accuracy gain does not justify,
     in which case the PR brings it back to Jim.
   - **Cross-validation keeps the default.** It measures agreement with fvsolver's solution; a tighter
     integrator on our side can only shrink our part of the difference, so there is no need to pin `scalar`
     (*inferred*).

6. **Tests.**
   - The scale table: errors at scales 1, 1e-6 and 1e-12 within a factor of 3 of each other.
   - The two-species table: `e` within 1e-4 per species, and failing under `scalar`. The coupled and nonlinear
     rows, with the linear solve's default tolerance, guard against solve pollution.
   - The nucleus model: the region variable's error the same on two meshes to within a factor of 2 under (c).
   - The floor: a species starting at 1e-30 beside an O(1) one takes no more steps than at `scalar`.
   - The running scale: a species starting at zero that grows to 1e-6 is controlled at its own scale, and the
     scale survives a stride boundary and the persistent integrator.
   - A Dirichlet-held unknown does not change the step count.
   - The matrix-free layout (membrane species as components) and the single mesh with several components.
   - MPI: the same steps and errors on 1 and 2 ranks.

## Not decided here

- **Loosening `rtol`** for the force balance's strides (ADR 021 decision 7's cost measurement). This note is its
  precondition.
- **A model-supplied scale** (decision 1 (d)), and whether VCell should send one.
- **Scaling the nonlinear and linear solves** (a diagonal scaling of the state), if a model shows solve pollution
  that these tests do not.
- **Whether VCell's scalar control should change** to match; a question for VCell, raised only with Jim's word.

## Review

An independent reviewer read the first draft and its sources, read PETSc 3.25.1's `ts.c` and `vector.c`,
reproduced the nucleus measurement, and implemented 2 (c) on it. It found 16 issues: 1 blocking, 10 major, 4
minor, 1 nit.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | blocking | 2 (c) costs exactly what (b) does on the nucleus model (365 steps, 5.94e-4), so the recommendation's condition fails there. | Measured on a model where bulk species drive the step: (c) +11 %, (b) +63–66 %. Decision 2 gives both cases; (c) is never costlier than (b). |
| 2 | major | The claim that (b) costs more by tightening the bulk species was contradicted on the nucleus model, where the region variable drives the step. | Restated with the bulk-driven measurement, where it holds. |
| 3 | major | "As if it were alone" overstates (c): the species share one budget, `Σ RMS_s² ≤ 1`. | Decision 2 (c) says so. |
| 4 | major | The C prototype was cited in the wrong order; it is `(ts, atol, vatol, rtol, vrtol)`. | Corrected, citing `ts.c` 5159. |
| 5 | major | The single mesh can carry several species as components; the multi-compartment solver interleaves membrane species. | Decision 3 gives both layouts. |
| 6 | major | The RMS is checkable, and the weight uses `max(|u|, |y|)`. | Read in source; the Context marks it checked. |
| 7 | major | No floor: 1e-30 or undershoot noise collapses the step. | Decision 1's floor at machine epsilon of the largest species; a test. |
| 8 | major | The nonlinear and linear solves are not scaled; a small species may be polluted. | Measured on uncoupled, coupled and nonlinear pairs twelve decades apart: no pollution. Kept as a test; scaling the solves is listed as not decided. |
| 9 | major | A species that decays after its peak is not controlled relative to its present size. | Stated as decision 1's limitation; option (c) offered. |
| 10 | major | Decision 4 said the tolerances keep their meaning while `atol` became a ceiling. | Decision 4 states the change of meaning. |
| 11 | major | The default changes every result and baseline, with its cost unmeasured. | Decision 5 makes the default depend on the measurement. Cross-validation is not pinned to `scalar` (answered there). |
| 12 | minor | How the running scale is kept, updated and shared under MPI is unstated. | Decision 1, "where `scale_s` lives". |
| 13 | minor | Dirichlet-held unknowns dilute the RMS. | Decision 3: zero tolerances, which PETSc skips. An infinite tolerance, as suggested, would still count. |
| 14 | minor | Not every path is BDF. | The Context says so. |
| 15 | minor | petsc4py accepts a `Vec` for both tolerances; its stub needs a cast. | The Context says so. |
| 16 | nit | "Coincide" holds only at `|u| ≈ 1`; the scale-1e-3 error differed from scale 1; the PETSc version. | All corrected. |

## References

- **This repository** at `a74a853` plus PR #262: `src/vcell_fenics/backend/multi_compartment.py` 467–480 and
  845–862, `reaction_diffusion.py` 76, 210 and 222, `interface_coupled.py` 1147 and 2047, `discrete.py` 154,
  `runner.py` 477, `cli.py` 302–303 and 424–425; `tests/test_ts_tolerances.py`.
- **The measurements** (scratch scripts, not committed; the implementation PR turns them into tests): the scale
  table (`scale_measure.py`), the nucleus (`norm_measure.py`), the bulk-driven norm comparison
  (`adr022/e1_bulk_driven.py`) and the two-species tables (`adr022/e2_scales.py`, `adr022/e3_coupled.py`). The
  last three inject a TS subclass with vector tolerances into `multi_compartment`.
- **PETSc 3.25.1**, the version in this environment: `src/ts/interface/ts.c` 5133–5173 (`TSSetTolerances`, its
  notes and its reference to the vectors) and 5241–5258 (`TSErrorWeightedNorm`); `src/vec/vec/interface/vector.c`
  2518–2583 (`VecErrorWeightedNorms_Basic`), read at gitlab.com/petsc/petsc tag v3.25.1; `-ts_adapt_wnormtype`
  and its default, read from `-help`. Abhyankar et al., "PETSc/TS: A modern scalable ODE/DAE solver library",
  arXiv:1806.01437 (2018) (cited second hand, as in ADR 021).
- **VCell:** `vcell-fvsolver` `VCell/src/SundialsPdeScheduler.cpp` 57 and 520 (scalar CVODE tolerances, read);
  `vcell` `vcell-core/src/main/java/cbit/vcell/solver/ErrorTolerance.java` 45–69 (the defaults and the
  constructor's order, read).
- **SUNDIALS CVODE** user guide: the weighted RMS norm, and the advice to set a vector absolute tolerance when
  components differ in scale (cited second hand, not read for this note). Hindmarsh et al., "SUNDIALS: Suite of
  nonlinear and differential/algebraic equation solvers", ACM TOMS 31(3), 363–396 (2005).
- Hairer, Nørsett and Wanner, *Solving Ordinary Differential Equations I*, 2nd ed., Springer (1993), §II.4, the
  componentwise scaled error norm (cited second hand, not read for this note).

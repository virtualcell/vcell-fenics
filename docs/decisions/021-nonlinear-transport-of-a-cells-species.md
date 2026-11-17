# ADR 021 — Nonlinear transport of the species a cell force balance carries

**Date:** 2026-11-17
**Status:** Proposed. Answers issue #255, asked from vcell-core's E4. Jim asked for this note on 2026-11-17 and
decided then that a nonlinear model switches to the method of lines automatically (decision 2). Revised after
an independent review (19 findings: none blocking, 7 major), each answered in "Review" at the end.
**Builds on:**
- ADR 020 (accepted 2026-11-15):
  - decision 2 (the conservative lab-frame form `−∫ m (v + w − v_mesh)·∇q`; "how a staggered force-balance
    step fits ADR 015's half-move stride stays open");
  - decision 3 (staggered, one pass per step);
  - decision 4 (the fixed skeleton, with linear transport only: "the crowding cut-offs are nonlinear in `m`.
    Backward Euler refuses them (`NonlinearTermError`), and the method of lines is stage 5's");
  - decision 5 (`v` and `m` written to the bundle, fixed skeleton only);
  - decision 6 (tests; `min m` reported).
- ADR 019 decision 6 (accepted 2026-11-14): the staging. 4a is one cell moving, on the moving partition, with
  no species. 5 is "chemistry on the moving cells, with crowding cut-offs; UC-A's phase-diagram points".
- ADRs 014–015: the moving partition.
  - Each stride moves the mesh by half a stride, integrates the whole stride on that midpoint mesh with a fresh
    method-of-lines integrator, then moves the other half (`multi_compartment.py` 782–786, *checked*).
  - ADR 015 decision 2 reads the front's velocity at the end of each half, a symmetric split, measured second
    order in time.
- ADR 010: the results bundle, with moving domains as segments and per-row coordinates.
- vcell-core's `spec/semantics/myosin-on-moving-cell.md`, accepted 2026-11-16, at vcell-core `1eff3f7`
  (*checked*):
  - decisions 1–3 (the cut-offs; a front speed reading `m`);
  - decision 6 (the pass criteria);
  - decision 7 (the four questions).
- vcell-core's `conformance/math/myosin-on-moving-cell.jsonl` at `1eff3f7`: the six E4 documents
  (*checked*; E4a read in full).
- `mms/cases/member_transport_carried.yaml`: ADR 020's manufactured solution for a member species under a
  prescribed velocity, run by backward Euler only (`applicable_solvers: [fenics-be]`, *checked*).

**Would amend:**
- ADR 020:
  - decision 3: the step advances `m` by backward Euler *or* by a method-of-lines stride (decision 1);
  - decision 4: on a fixed skeleton, terms nonlinear in the species are admitted (decision 3);
  - "Not decided here": nonlinear transport.
- ADR 010: a moving volume domain gains a per-row centroid array (decision 6).
- ADR 019 decision 6: stage 5 split into 5a and 5b (decision 3).
- `backend/cell_force_balance_run.py`, `backend/reaction_diffusion.py` (`integrate_discrete_problem`) and
  `runner.py`; the test `test_nonlinear_transport_waits_for_stage_5`; the formalism's T9 status lines (§1.4.2,
  §3.6.1); the status paragraph of `CLAUDE.md`.

*Checked* marks what was read in a source (file and line, or section), or run here. *Inferred* marks
reasoning. This repository was read at `7807a14`.

## Context

**E4.** It is Nickaeen et al.'s full model ([Nickaeen 2017]):
- myosin `m` on the cell, carried by the solved `v`;
- with the paper's two crowding cut-offs, written by vcell-core as
  - `relative_advection: (max(0, 1 − m/mmax_u) − 1) * v_cyt`, a drift `w` that makes the flux `m (v + w)`
    equal to `f m v`, with `f = max(0, 1 − m/m_max,u)`;
  - `diffusion: D (1 − m/mmax_d)`.

Both are nonlinear in `m`. Of its four cases, E4a is E2's fixed circle with the cut-offs. The other three
(ZV, ZS, ZS0) are growing fronts, so they move.

**Why backward Euler refuses them** (*checked*):
- `assemble_cell_force_balance_run` composes the species' backward-Euler forms at assembly
  (`problem.bilinear_form`), with the species' slots compiled against the trial functions.
- With `m` in a coefficient, the composition fails, either the arity check or UFL's `lhs`/`rhs` split. For
  E4a's drift it is the split: UFL raises a `ValueError` for an argument inside `max_value` (the review's run).
  `_compose_backward_euler` reports either as `NonlinearTermError` (`discrete.py` 630–645).
- The runner turns that into a refusal naming stage 5 (`runner.py` 366–370).
- The runner also forces backward Euler for every T9 model, overriding `--time-integration`
  (`runner.py` 152–153).

**The method of lines already takes a nonlinear residual** (*checked*).
- `integrate_discrete_problem` builds its residual from any assembled `DiscreteProblem`'s tagged terms, with
  `trial → unknown`, and hands it to PETSc TS (adaptive BDF), whose Newton solves it (`reaction_diffusion.py`
  314–392).
- The force-balance run's species are such a `DiscreteProblem`, with `v` a solved `Function` in the advection
  term. So the integrator can advance them over an interval with `v` held.
- **But it always starts at `t = 0`.** It does not pass a start time on to the time stepper
  (`reaction_diffusion.py` 361–373). A stride over `[tⁿ, tⁿ + Δt]` must be called with `t_final = Δt`, and
  the time Constant it sets, shared with the force balance, then runs over `[0, Δt]`, not the run's time (the
  review's run).
- E4a reads no `sim.t`, so the experiment below is unaffected. A model with a time-dependent slot would not
  be.
- Each call also rebuilds the forms and the Jacobian, and restarts BDF at order 1 from a step of `Δt/10⁴`.
  That came to about 24 TS steps per stride, even after the myosin had settled.

**The experiment** (*checked*: run here at `7807a14`, and the first row reproduced by the review):
- **The procedure:**
  - assemble E4a with the backward-Euler composition bypassed;
  - geometry: E2's disk in a 5 × 4 box, as in `tests/test_runner_cell_force_balance.py`, with the membrane
    named `cell_ec_membrane`;
  - repeat to `t = 10`:
    - solve `v` from `mⁿ`;
    - call `integrate_discrete_problem` with `t_final = Δt`, `v` held, `rtol = 1e-6` and `atol = 1e-8`.
- **The quantities:**
  - `∫m`, as a drift relative to `∫m₀ = 4.707`;
  - `min m` and `max m`;
  - vcell-core's band share: myosin in `r ≥ 0.9`, `|θ − π| ≤ 45°`, divided by its share at `t = 0`.

At `t = 10`:

| `h` | `Δt` | relative `∫m` drift | `min m` | `max m` | share ratio | wall time |
|---|---|---|---|---|---|---|
| 0.1 | 0.05 | −1.4e-12 | 0.073 | 13.30 | 6.07 | 23 s |
| 0.1 | 0.0125 | −6.3e-12 | 0.073 | 13.30 | 6.07 | 91 s |
| 0.05 | 0.05 | −1.5e-12 | 0.076 | 13.26 | 6.23 | 39 s |

- **Every one of vcell-core's proposed E4a criteria is met:**
  - `∫m` to `1e-8`;
  - `m ≥ −0.01 μ_tot/π` and `m ≤ m_max,d = 125`, *throughout*: the review checked every stride of the first
    row (min 0.0726, max 13.30), and here they were checked at each `t = 1, 2, …`;
  - a share ratio of at least 3.
- The myosin settles by `t ≈ 5`, with its peak at 13.3.
- **The cut-off never switches in E4a.** The peak is 11% below `m_max,u = 15`, so `f ≥ 0.11` everywhere and
  `max(0, ·)` is never at its kink. E4a tests `f` as a linear function of `m`; the moving cases, where ZV's
  rear band is meant to reach `m_max,u`, will not be (decision 7 adds a test that does).
- **The steady state does not depend on `Δt`** (*inferred*). A steady state of the staggered step is a
  steady state of the coupled system, since `v` lags `m` by a step in which `m` no longer changes. This holds
  for a given steady state. A problem with several could reach a different one at a different `Δt`.

**The transient does depend on `Δt`.** The share ratio at `t = 2`, at `h = 0.1`, is:

| `Δt` | 0.1 | 0.05 | 0.025 | 0.0125 |
|---|---|---|---|---|
| share ratio | 2.72 | 3.70 | 4.23 | 4.47 |

- The differences fall by about half as `Δt` halves (0.98, 0.53, 0.24; ratios 0.54 and 0.45): first order.
- That it is ADR 020 decision 3's splitting error is *inferred*. BDF restarting at order 1 each stride
  depends on `Δt` too.
- **The Richardson extrapolation.** Extrapolating the last two values as first order gives a limit of about
  4.71. At `Δt = 0.05` the error is then about 1.0, some 20% of the ratio at `t = 2`.
- On a moving run the transient is the result (a centroid speed, a turning radius), so 5b needs a rule for
  `Δt` ("Not decided here").
- The paper's `Δt = c h`, with `c ≤ 0.025` and `h ≤ 0.16` (as E4's criteria quote it), is at most 0.004,
  below the smallest `Δt` here.

## Decisions

1. **The scheme on a fixed skeleton: a staggered method-of-lines stride.**
   - **Each step:**
     - solve `v` on the current configuration from `mⁿ`;
     - integrate the species over `[tⁿ, tⁿ + Δt]` with PETSc TS (adaptive BDF), with `v` held;
     - the transport form, its natural condition and its conservation are ADR 020 decision 2's.
   - **The new code is small, but it is needed** (the Context):
     - the TS and its residual are built once per run, not once per stride;
     - each stride is solved from `tⁿ`, so the shared time Constant carries the run's time;
     - each stride starts from the previous stride's last step size, as the moving partition's stride loop
       does.
     - Restarting BDF at order 1 each stride stays: `v` changes between strides, so the history would be of
       a different operator.
   - **Conservation is limited by the solver's tolerance, not by round-off.** With `q = 1` the transport and
     diffusion terms vanish. There are no Dirichlet rows on a member species, and constants are in P1. So `∫m`
     changes only by what the Newton and GMRES solves leave unconverged. The drift was systematic, about
     −7e-15 relative per stride (the review's run), so it grows with the number of strides. At the paper's
     `Δt` to `t = 30` that is about 5e-11, still well inside the criterion's `1e-8`.
   - **The options:**
     - **(a) A method-of-lines stride,** as above.
     - **(b) Newton backward Euler:** one nonlinear solve per step, `v` held. It is the same code as (a) with
       `ts_type = "beuler"` and a fixed step, so it stays available; (a)'s adaptive steps control the error
       within a stride for free.
     - **(c) Lagged coefficients:** the cut-offs evaluated at `mⁿ`, so the step stays linear. It is the
       cheapest, and a consistent first-order scheme for the same model. But it adds a second `O(Δt)` lag on
       top of the splitting's, in the term that limits aggregation, and its positivity and boundedness near
       the cut-off are untested (*inferred*).
     - **(d) Monolithic:** `(v, m)` in one nonlinear solve. It removes the splitting error, and it is the most
       work. ADR 020 decision 3 keeps it, as fixed-point iteration (3 (b)), for when a measurement shows the
       splitting error matters.

   *Recommendation: (a).*

2. **The runner switches a nonlinear model to the method of lines by itself.** *Decided by Jim, 2026-11-17.*
   - **How it detects a nonlinear model: structurally, before composing.**
     - A species' `diffusion` or `relative_advection` is nonlinear if it reads a species.
     - A `source` is nonlinear if its compiled form is not affine in the trial functions: UFL's arity check
       on that term alone.
     - The catch-all is not used. Today `_compose_backward_euler` turns *any* `ValueError` into
       `NonlinearTermError` (`discrete.py` 640–643), so an unrelated defect would switch the scheme instead of
       failing.
   - **Nonlinear sources are admitted too,** on the same route, since TS's Newton solves them the same way.
     ADR 020 decision 4's linear-only limit is lifted for every slot, not only the transport.
   - **A nonlinear model** runs by method-of-lines strides whatever `--time-integration` says. A request for
     backward Euler is overridden with a logged note naming the slots that are nonlinear, as the runner
     already notes an ignored option (`runner.py` 333–337).
   - **A linear model** keeps backward Euler by default, so step 3b's runs and tests are unchanged.
     `--time-integration method_of_lines` (or the SimulationTask's `TimeIntegration=MethodOfLines`) is
     honoured for it, no longer ignored.
     - Whether VCell writes `TimeIntegration` for T9 tasks was not checked. If it writes `MethodOfLines`, a
       linear T9 task from VCell changes scheme with this decision. It stays a correct run, with a different
       first-order error.
   - **What the manifest and `summary.json` record.** They already record `h`, `dt`, `dt_requested` and
     `time_integration` (`runner.py` 186–206), the last hard-coded to backward Euler for T9. Under the method
     of lines:
     - `time_integration` is `method_of_lines`;
     - `dt` is the stride, the length `v` is held for;
     - `steps` is the number of TS steps, and `strides` is added;
     - `rtol` and `atol` are added, since the result depends on them.

     vcell-core's criteria ask for `Δt` and `h` with every result.

   *Recommendation: yes.*

3. **Nonlinear terms on a fixed skeleton come first: stage 5 is split into 5a and 5b.**
   - **5a:** decisions 1, 2, 4 and 7 on ADR 020 decision 4's fixed skeleton. Every other limit of that scope
     stays: one member, species on it only, no BC on them. E4a is its test.
   - **5b:** species on the moving cell. It needs 4a (one cell moving, on the moving partition), and decisions
     5 and 6. E4-ZV, E4-ZS and E4-ZS0 are its tests.
   - **What 5a does not settle for 5b.** The moving partition integrates its species with its own TS inside
     `integrate_multi_compartment` (`multi_compartment.py` 765–778, 890–900), not through
     `integrate_discrete_problem`. How a solved `v` enters ADR 015's half-move stride is open (ADR 020 decision
     2). That covers the configuration `v` is solved on (the start mesh, or the midpoint mesh the stride
     integrates on) and when it is solved. 5b decides it ("Not decided here"). 5a fixes the scheme's kind (method-of-lines
     strides, `v` held across each), not 5b's code.
   - **Why 5a now** (*inferred*):
     - it needs no moving mesh;
     - the experiment shows the integrator already runs it;
     - it tests the nonlinear terms on a fixed mesh, so a defect found in 5b is less likely to be one of the
       transport.
   - **5a is independent of 4a.** Either can come first. This note recommends 5a first because it is small.

   *Recommendation: yes, 5a before 4a.*

4. **A diffusion coefficient that turns negative stops the run.**
   - **Why.** `D (1 − m/m_max,d)` is negative above `m_max,d`. Diffusion backwards in time is ill-posed. A run
     that crosses it would fail inside TS with a generic `SolveError`, or would carry growing mesh-scale noise
     (*inferred*, untested). Either way the cause should be named.
     - The drift's cut-off keeps `m` near 13, far from 125, in every run here (*checked*). A model with other
       parameters could cross it.
   - **The check.** After every stride, each species' diffusion coefficient is evaluated at the mesh's nodes,
     with a collective minimum over the ranks. If it is negative anywhere, the run stops with an error naming
     the species, the time and the least value.
     - It reads the model's own expression, so it needs no knowledge of `m_max,d`.
     - It is exact for a coefficient affine in a P1 `m`, such as this one. For another coefficient a node check
       can miss a negative value between nodes (*inferred*).
   - **`min m` and `max m`** are already written per output row by the recorder (`results/recorder.py`
     148–152, *checked* by the review); nothing is added for them.
   - **The options:**
     - **(a) Stop,** as above.
     - **(b) Warn and continue.**
     - **(c) Clamp the coefficient at 0.** That changes the model the document wrote.

   *Recommendation: (a).*

5. **A front speed that reads `trace(m)`: nothing is missing in the grammar; stage 5b wires it.**
   - **The grammar admits it.** vcell-core's six documents validate with no diagnostics (*checked*, run here
     at `7807a14`).
   - **The moving partition already evaluates such a speed,** for a prescribed front.
     - `MovingMembrane` copies each bulk species named in a `trace(...)` onto the membrane's nodes before it
       evaluates the velocity (`moving_partition.py` 219–247 and 359–362, *checked*).
     - The stride loop scatters the current state into those species before each half-move
       (`multi_compartment.py` 805–808, *checked*). The first half-move comes before the integration, so it
       reads `mⁿ`. The second comes after it, so it reads `mⁿ⁺¹` (890–901, *checked*). That is ADR 015
       decision 2's symmetric split.
   - **What 5b adds:**
     - the front of a cell whose force balance is solved (4a's) reads the same traces;
     - the species it reads must be the one integrated on that cell.
   - **The front keeps ADR 015's split.** Whether the solved `v` should be lagged the same way, and whether
     that costs the split its second order, is part of 5b's open question (decision 3).

   *Recommendation: yes.*

6. **The outputs a moving run needs.**
   - **What exists** (*checked*):
     - The bundle stores a moving domain's per-row point coordinates in segments (ADR 010 §2–3;
       `results/writer.py`).
     - A moving multi-compartment run, with a prescribed membrane, registers each compartment and each
       membrane as a moving domain (`runner.py` 757–776). Its membrane's rows are the boundary over time.
   - **What 5b's route must do** (*inferred*):
     - Today the T9 route registers one fixed volume domain on the members' mesh (`runner.py` 397) and never
       moves (`runner.py` 154).
     - 4a's and 5b's route must register the member and its membrane as moving domains, as the
       multi-compartment route does.
     - Then `m` on the current mesh and the boundary are written, with no new bundle machinery.
   - **The centroid over time: added.**
     - The recorder already re-assembles a moving domain's measure at every capture (`results/recorder.py`
       135–136, *checked*). It will also assemble `∫x dx / |Ω|` there, which is exact for the mesh's own
       geometry (*inferred*).
     - **Where it is stored.** ADR 010's statistics are per variable (mean, total, min, max), so the centroid
       does not fit them. It gets an array of its own per moving volume domain, `<prefix>centroid/<domain>`,
       of shape `(T_seg, gdim)`, listed in the manifest's domain entry.
     - It is an additive minor version of the manifest: a reader that does not know the array ignores it.
   - **`v` on a moving member: written,** by component, as ADR 020 decision 5 writes it on a fixed skeleton,
     on the member's moving domain.
   - **The options:**
     - **(a) The bundle carries the centroid.**
     - **(b) Readers compute it from the coordinates.** Every reader (VCell's, vcell-core's tests, ours) then
       repeats the same integral.

   *Recommendation: (a).*

7. **Stage 5a's tests.**
   - **E4a from vcell-core's sibling checkout,** as #253 reads E2's (skipped when it is absent), against its
     proposed criteria (#255), checked at every output. vcell-core proposes those criteria; the tests here
     decide what is asserted.
   - **The cut-off at its kink.** E4a never reaches it (the Context), so a separate test drives `m` past
     `m_max,u`. It prescribes a velocity converging on a point with the drift cut-off on.
     - It checks that the strides converge, that `∫m` is conserved, and that `m` saturates near `m_max,u`.
     - The last is not a bound of the continuous model. At `m = m_max,u`, `∂m/∂t = ∇m·v`, which can be
       positive (the review's algebra). So a band is asserted, set from a measurement.
   - **Structural and exact:**
     - `∫m` conserved to `1e-10` relative over a run of a few hundred strides. The bound is set by the
       solver's tolerances (decision 1), so a longer test tightens the Newton tolerance rather than the bound.
     - The runner's switch (decision 2):
       - a nonlinear model runs by the method of lines whatever was asked, and the manifest says so;
       - a linear model keeps backward Euler;
       - an explicit method-of-lines request is honoured;
       - an unrelated `ValueError` is not taken for nonlinearity.
     - A slot that reads `sim.t` sees the run's time in every stride (decision 1).
     - The stop on a negative diffusion coefficient (decision 4).
   - **Against a known answer:**
     - **A manufactured solution:** a new case of the `member_transport` harness, with a diffusion and a drift
       that read the species, under a prescribed velocity, with `fenics-mol` among its solvers, and its spatial
       order.
       - Like `member_transport_carried`, its source reads `sim.t`, so it also checks the stride's time.
       - It keeps `m` below `m_max,u`, away from the kink, which the test above covers.
     - **The two schemes on a linear model:** E2 run both ways at `Δt` and `Δt/2`. Their difference shrinks
       with a ratio between 0.4 and 0.6.
   - **Measured, not pass/fail:**
     - the transient's dependence on `Δt` (the Context's table), reported with E4a's result;
     - the cost per stride;
     - a run on 2 MPI ranks, which must match 1 rank to the solver's tolerance. ILU becomes block Jacobi in
       parallel, so the iterations differ, not the answer (*inferred*).

   *Recommendation: yes.*

## For vcell-core (the reply to #255, posted on Jim's say)

1. **The scheme.**
   - Method-of-lines strides, staggered: `v` from `mⁿ`, then `m` over the stride by adaptive BDF with `v`
     held.
   - The runner chooses it by itself for a model with a nonlinear term.
   - On the fixed skeleton first (5a). On the moving partition (5b), the scheme's kind is the same, but where
     the solved `v` enters its half-move stride is decided with 5b.
2. **E4a comes first,** as stage 5a, independent of 4a.
   - The experiment above meets every E4a criterion at `h = 0.1`, `Δt = 0.05`, throughout the run.
   - E4a never reaches `m_max,u`: its peak is 13.3, so the drift's cut-off never switches. ZV's rear band is
     where the kink is first exercised, and 5a adds its own test of it.
3. **`trace(m)` in a front speed.** Nothing is missing in the grammar.
   - The moving partition evaluates it today for a prescribed front.
   - It reads `mⁿ` at the stride's first half-move and `mⁿ⁺¹` at its second, ADR 015's symmetric split.
   - 5b wires it for a solved cell.
4. **The outputs.**
   - The bundle already carries `m` on a moving mesh and the boundary over time, for a moving multi-compartment
     run.
   - 5b's route will register the cell and its membrane the same way.
   - The centroid per row and `v` on the moving cell are added (decision 6).
5. **The approximations.**
   - The one-step lag is measured: first order in `Δt` in the transient, absent at a steady state.
   - `Δt` (the stride), `h`, `rtol` and `atol` are reported with each result.
   - The diffusion coefficient's sign is checked after every stride.

## Not decided here

- **How the solved `v` enters ADR 015's half-move stride** (5b; ADR 020 decision 2's open question):
  - on which mesh it is solved;
  - whether it is solved once or twice per stride;
  - what that does to the split's order.
- **The time-step rule for 5b:** a fixed `Δt`, or one tied to the front's speed and `h`. The transient's
  first-order dependence on `Δt` is the reason it matters (the Context).
- **Fixed-point iteration** of the stagger (ADR 020 decision 3 (b)), if the moving cases' transient needs it.
- **Stabilisation of the transport** (SUPG, a limiter), if `min m` fails on the moving cases.
- **Species beside the cell, or on its membrane**, on a moving run.
- **A drift together with a carried fraction** (vcell-core's note, decision 1).
- **Cost.**
  - At the paper's `Δt ≤ 0.004` and `h = 0.1`, a run to `t = 30` is about 7500 strides.
  - That is about 12 minutes at the review's 0.1 s per stride, before any mesh moves (*inferred*). Building
    the TS once (decision 1) should lower it.
  - Whether 5b needs the paper's `Δt` is the time-step rule above.

## Review

An independent reviewer read the draft and its sources. It re-ran the experiment's first row, traced the
integrator's calls, checked the line references (all hold; `runner.py` 760–773 is 757–776), and checked the
issue's numbers. It found 19 issues: none blocking, 7 major, 9 minor and 3 nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | major | "Needs no new code" is false. `integrate_discrete_problem` always starts at `t = 0`, so `sim.t` is wrong in every stride after the first, and each call rebuilds forms and restarts BDF. | The Context states it. Decision 1 lists the new code. Decision 7 tests `sim.t`. |
| 2 | major | Which `m` the front reads is checkable now: the first half-move reads `mⁿ`, the second `mⁿ⁺¹`, ADR 015's symmetric split. A "fix" to `mⁿ` would contradict ADR 015. | Decision 5 states what the code does and keeps ADR 015's split. |
| 3 | major | "Needs no second scheme" glosses over ADR 020's open question. The moving partition integrates with its own TS, not `integrate_discrete_problem`. | Decision 3 says what 5a does not settle. "Not decided here" lists it. |
| 4 | major | `NonlinearTermError` is raised for any `ValueError`, and for nonlinear sources too, so the switch would admit both silently. | Decision 2: structural detection per slot; nonlinear sources admitted, stated; an unrelated `ValueError` tested. |
| 5 | major | E4a's peak is 13.3, so `max(0, ·)` never switches; nothing tests the kink. | The Context says so. Decision 7 adds a test that drives `m` past `m_max,u`. The peak is given as 13.3. |
| 6 | major | Option (c) is a consistent scheme for the same model, `m ≤ m_max,u` is not a bound of the PDE, and (b) is the same code with `ts_type = "beuler"`. | Decision 1's options corrected. |
| 7 | major | "Written already, for any moving run" covers the multi-compartment route only. The T9 route registers a fixed domain. | Decision 6 separates what exists from what 5b's route must do. The reply to vcell-core matches. |
| 8 | minor | The manifest already records `h` and `dt`. Under the method of lines `steps` and `dt` change meaning, and `rtol`/`atol` matter. | Decision 2 lists what changes. |
| 9 | minor | `min`/`max` are already written. A check only at outputs is late. It must be collective, and is exact only for an affine coefficient. | Decision 4: after every stride, collective, its exactness stated, "noise" marked inferred. |
| 10 | minor | The criteria hold "throughout", but only `t = 10` was shown. The drift column does not say relative. | The Context says throughout, and labels the drift relative. |
| 11 | minor | Attributing the first order to the splitting is inferred, and the 20% is an unshown extrapolation. | Both marked and shown. |
| 12 | minor | "Decision 7" sets no `Δt` rule. | Points to "Not decided here". |
| 13 | minor | The `member_transport` harness exists and is not cited. "Agrees to `O(Δt)`" is not assertable. | Cited, and extended. A ratio band is given. |
| 14 | minor | Conservation is limited by the solvers' tolerance and drifts systematically; `1e-10` is close at the paper's `Δt`. | Decision 1 says so. Decision 7 ties the bound to the run's length. |
| 15 | minor | Whether VCell writes `TimeIntegration=MethodOfLines` for T9 tasks is unchecked. | Decision 2 states the consequence; not checked here. |
| 16 | minor | No MPI statement and no cost. | Decision 7 adds a 2-rank run; "Not decided here" has the cost. |
| 17 | nit | For E4a's drift, the failure is UFL's split, not the arity check. | Corrected. |
| 18 | nit | The steady state's independence of `Δt` holds for a given steady state. | Qualified. |
| 19 | nit | A per-domain centroid has no slot in ADR 010's per-variable statistics. | Decision 6 names the array and the manifest change. |

## References

- **This repository** at `7807a14`:
  - ADRs 010, 014, 015, 019, 020;
  - `src/vcell_fenics/backend/cell_force_balance_run.py`, `discrete.py` 630–645, `reaction_diffusion.py`
    314–392, `multi_compartment.py` 765–808 and 890–901, `moving_partition.py` 219–247 and 359–362;
  - `src/vcell_fenics/runner.py` 152–154, 186–206, 333–337, 366–370, 397 and 757–776;
  - `src/vcell_fenics/results/recorder.py` 135–152;
  - `mms/cases/member_transport_carried.yaml`;
  - issues #251, #253, #255.
- **vcell-core** at `1eff3f7`: `spec/semantics/myosin-on-moving-cell.md` (decisions 1–3, 6, 7) and
  `conformance/math/myosin-on-moving-cell.jsonl`.
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A
  free-boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13:e1005862 (2017).
  doi:10.1371/journal.pcbi.1005862.
  - *Cited through ADR 020,* which read pp. 3–8 and 12. Not re-read for this note.
  - *Use:* the cut-offs, the fixed-point iteration, the ranges of `Δt` and `h`.
- **PETSc TS.** Abhyankar S, Brown J, Constantinescu EM, Ghosh D, Smith BF, Zhang H. PETSc/TS: a modern
  scalable ODE/DAE solver library. arXiv:1806.01437 (2018).
  - *Cited second hand.* Not read for this note.
  - *Use:* the adaptive BDF integrator of decision 1. It is already listed in `ACKNOWLEDGEMENTS.md` as PETSc.

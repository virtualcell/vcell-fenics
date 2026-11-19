# ADR 021 — Transport that reads the species, and the species a cell force balance carries

**Date:** 2026-11-17
**Status:** Accepted (Jim, 2026-11-17): decisions 1–8 as recommended, with decision 8 at stage 5a and 5a's
scope widened as decision 3 states. Answers issue #255, asked from vcell-core's E4.
- Jim asked for this note on 2026-11-17, and decided then that a nonlinear model switches to the method of
  lines automatically (decision 2).
- **Amended the same day, at Jim's request.** Jim asked whether the work overfits E4, or makes
  improvements that are useful everywhere. The amendment does three things:
  - it widens decisions 2 and 4 to every route;
  - it makes a transport slot that reads the species compile on every route (decision 2; until now only the
    T9 driver could compile one);
  - it adds decision 8: a cell's species run through the multi-compartment solver, so there is one species
    path.
- **Reviewed twice.** The first independent review found 19 issues (none blocking, 7 major). The second
  reviewed the amended note and found 16 (1 blocking, 4 major). Both are answered in "Review" at the end.

**Builds on:**
- ADR 018 (accepted 2026-11-14): T9 and its staging. Decision 8: species on a member ride the solved `v`.
- ADR 020 (accepted 2026-11-15):
  - decision 2: the conservative lab-frame form `−∫ m (v + w − v_mesh)·∇q`. It also says: "how a staggered
    force-balance step fits ADR 015's half-move stride stays open".
  - decision 3: staggered, one pass per step, with `m` advanced by backward Euler. Slots that read the lagged
    `v` are linear.
  - decision 4: the fixed skeleton, with one member and species on it only, and linear transport. "The
    crowding cut-offs are nonlinear in `m`. Backward Euler refuses them (`NonlinearTermError`), and the method
    of lines is stage 5's".
  - decision 5: `v` and `m` written to the bundle, on the fixed skeleton only, with the `v` solved from `mⁿ`
    written at `tⁿ`.
  - decision 6: the tests, with `min m` reported.
  - Its Context table lists four transport forms in the code. The T9 driver now assembles a fifth.
- ADR 019 decision 6 (accepted 2026-11-14): the staging. 4a is one cell moving, on the moving partition, with
  no species. 5 is "chemistry on the moving cells, with crowding cut-offs; UC-A's phase-diagram points".
- ADRs 014–016: the moving partition. Each stride moves the mesh by half a stride, integrates the whole stride
  on that midpoint mesh with a fresh method-of-lines integrator, then moves the other half
  (`multi_compartment.py` 782–786, *checked*). ADR 015 decision 2 reads the front's velocity at the end of each
  half: a symmetric split, measured as second order in time.
- ADR 010: the results bundle, with moving domains as segments and per-row coordinates.
- vcell-core's `spec/semantics/myosin-on-moving-cell.md`, accepted 2026-11-16, at vcell-core `1eff3f7`
  (*checked*): decisions 1–3 (the cut-offs; a front speed reading `m`), decision 6 (the pass criteria), and
  decision 7 (the four questions).
- vcell-core's `conformance/math/myosin-on-moving-cell.jsonl` at `1eff3f7`: the six E4 documents. *Checked*;
  E4a read in full.
- `mms/cases/member_transport_carried.yaml` and `mms/runner.py`'s `_run_member_transport` (321–369): ADR 020's
  manufactured solution for a member species under a prescribed velocity. It runs by backward Euler, through
  `assemble_cell_force_balance_run` (*checked*).

**Would amend:**
- **ADRs:**
  - ADR 020:
    - decision 3: a member's species are advanced by method-of-lines strides, in the multi-compartment solver
      (decisions 1 and 8);
    - decision 4: terms nonlinear in the species are admitted (decision 2). Its sentence that the species "are
      assembled on the members' mesh … the same mesh, and for P1 the same space" now names the member
      compartment's mesh (decision 8). Its limit to species on the member is lifted on a held membrane
      (decision 3);
    - decision 5: the T9 route writes its domains as the multi-compartment route does (decision 6);
    - "Not decided here": nonlinear transport.
  - ADR 019 decision 6: stage 5 split into 5a and 5b, with 5a allowed before 4a (decision 3).
  - ADR 010: a moving volume domain gains a per-row centroid array (decision 6).
- **The code:**
  - `backend/assemble.py` and `backend/multi_compartment.py`: transport slots compiled with the species bound
    (decision 2);
  - `backend/multi_compartment.py` also gains:
    - the member carrier;
    - strides on a fixed partition, with a hook before each;
    - T9 handed to the force balance, where it is refused today (decision 8);
  - `backend/cell_force_balance.py`: the force balance built on the member compartment's mesh (decision 8);
  - `backend/cell_force_balance_run.py`: its species problem retired (decision 8);
  - `backend/discrete.py`: the catch-all in `_compose_backward_euler` (decision 2);
  - the TS builders in `reaction_diffusion.py` and `multi_compartment.py`: the diffusion check (decision 4);
  - the formalism: a linearity check on a model (decision 2);
  - `runner.py`:
    - the switch, on every route (decision 2);
    - T9 routed into the multi-compartment route, with `v` kept out of its species and written by component
      (decisions 6 and 8);
  - `mms/runner.py`: `_run_member_transport` moved to the new path, and `member_transport_carried` run by it.
- **The tests:** `test_nonlinear_transport_waits_for_stage_5`.
- **The docs:**
  - the formalism's T9 status lines (§1.4.2, §3.6.1);
  - the status paragraph of `CLAUDE.md`.

*Checked* marks what was read in a source (file and line, or section), or run here. *Inferred* marks
reasoning. This repository was read at `7807a14`.

## Context

**E4.** It is Nickaeen et al.'s full model ([Nickaeen 2017]): myosin `m` on the cell, carried by the solved
`v`, with the paper's two crowding cut-offs. vcell-core writes them as:
- `relative_advection: (max(0, 1 − m/mmax_u) − 1) * v_cyt`. This is a drift `w` that makes the flux
  `m (v + w)` equal to `f m v`, with `f = max(0, 1 − m/m_max,u)`;
- `diffusion: D (1 − m/mmax_d)`.

Both are nonlinear in `m`. Of E4's four cases, E4a is E2's fixed circle with the cut-offs added. The other
three (ZV, ZS, ZS0) are growing fronts, so they move.

**A transport slot that reads the species is general, and is missing on the general routes** (*checked*):
- **The formalism admits it.**
  - The validator passes all six E4 documents (run here).
  - Parameters cannot read a variable: "parameter expression may not reference the variable … (§1.8.3)",
    checked by the second review. So a slot reads a species only directly, never through a parameter.
- **The single-mesh assembler compiles it without the species bound.** It compiles `diffusion`,
  `relative_advection` and `advection` in the parameters' context only (`assemble.py` 231, 238, 249, 265).
  `source` alone gets the species (277), and the boundary fluxes get them too (343).
- **The multi-compartment solver does the same.** Its transport slots compile in `local = ctx`, the
  parameters only (`multi_compartment.py` 514, 520–525). `source` alone gets `bulk_ctx`, the compartment's
  species and the region variables (482–485, 541).
- **So a diffusion or a drift that reads a species fails with `CompileError: unresolved name` on both
  routes.** That holds under either scheme. The second review ran it on both.
- **Only the T9 driver compiles one.** It binds the species' trial functions in every slot
  (`cell_force_balance_run.py` 159–176). That is why the experiment below could run.
- **Nonlinear sources and fluxes compile everywhere.** Backward Euler refuses them, and the method of lines
  runs them. The second review ran a Neumann flux `−m/(1+m)` both ways.

**Where a model nonlinear in its species is refused today** (*checked*):

| Route | Integrator | Nonlinear source or flux | Transport slot reading a species |
|---|---|---|---|
| single mesh, fixed | backward Euler by default; whole-run TS on request (`runner.py` 489) | refused under backward Euler (`cli.py` 93) | `CompileError` |
| single mesh, moving (ALE) | backward Euler by default; strides by `integrate_discrete_problem_stride` on request (`runner.py` 559–600; `ale.py` 218–219) | refused under backward Euler (`runner.py` 624–629) | `CompileError` |
| multi-compartment, fixed or moving partition | the method of lines only (`runner.py` 726–727) | runs | `CompileError` |
| T9 with member species (`cell_force_balance_run.py`) | backward Euler only, overriding any request (`runner.py` 152–153) | refused (`runner.py` 366–370) | refused (the same) |

- **The refusal is raised in shared code.** `_compose_backward_euler` reports any `ArityMismatch`, and *any*
  `ValueError`, as `NonlinearTermError` (`discrete.py` 630–645). For E4a's drift it is UFL's `lhs`/`rhs`
  split that fails, on an argument inside `max_value` (the first review's run). An unrelated `ValueError`
  would be reported the same way.
- **The multi-compartment solver refuses T9** (`multi_compartment.py` 345–349).
- **The members' mesh and a compartment's submesh are built alike.**
  - Both are `create_submesh(parent, tdim, cells)` over the cells tagged with the region
    (`cell_force_balance.py` 552–557; `multi_compartment.py` 233–237; *checked*).
  - Their DOF coordinates are identical, rank by rank, on 1 and 2 ranks (the second review's run).
  - But they are two mesh objects, so a `Function` on one cannot enter a form on the other.

**The method of lines, and its strides** (*checked*):
- **`integrate_discrete_problem`** builds its residual from any `DiscreteProblem`'s tagged terms, with
  `trial → unknown`, and hands it to PETSc TS (adaptive BDF), whose Newton solves it (`reaction_diffusion.py`
  314–392). It integrates a whole run, from `t = 0`.
- **`integrate_discrete_problem_stride`** integrates one stride from its own `t_start`, setting the time
  Constant from there (`reaction_diffusion.py` 461–500; 219 and 239). ALE's method-of-lines route strides with
  it (`ale.py` 218–219). It rebuilds its forms on every call.
- **The multi-compartment solver** strides on a moving partition. Each stride gets a fresh TS starting at its
  own time, and carries the last step size (`multi_compartment.py` 850–901). Its Jacobian is
  `ufl.derivative` of the residual (636–638), so a coefficient that reads a species enters it with no further
  code (*inferred*).

**The experiment: a prototype on the T9 driver** (*checked*: run here at `7807a14`; the first review
reproduced the first row).
- **The procedure:**
  - assemble E4a with the backward-Euler composition bypassed. The geometry is E2's disk in a 5 × 4 box, as in
    `tests/test_runner_cell_force_balance.py`, with the membrane named `cell_ec_membrane`;
  - repeat to `t = 10`: solve `v` from `mⁿ`, then call `integrate_discrete_problem` with `t_final = Δt`,
    `v` held, `rtol = 1e-6` and `atol = 1e-8`.
  - E4a reads no `sim.t`, so starting each call at 0 does not affect it.
  - This is not decision 8's path. It shows what the scheme gives, not the code that will run it.
- **The quantities:**
  - `∫m`, as a drift relative to `∫m₀ = 4.707`;
  - `min m` and `max m`;
  - vcell-core's band share: the myosin in `r ≥ 0.9`, `|θ − π| ≤ 45°`, over its share at `t = 0`.

At `t = 10`:

| `h` | `Δt` | relative `∫m` drift | `min m` | `max m` | share ratio | wall time |
|---|---|---|---|---|---|---|
| 0.1 | 0.05 | −1.4e-12 | 0.073 | 13.30 | 6.07 | 23 s |
| 0.1 | 0.0125 | −6.3e-12 | 0.073 | 13.30 | 6.07 | 91 s |
| 0.05 | 0.05 | −1.5e-12 | 0.076 | 13.26 | 6.23 | 39 s |

- **Every one of vcell-core's proposed E4a criteria is met:**
  - `∫m` is conserved to `1e-8`;
  - `m ≥ −0.01 μ_tot/π` and `m ≤ m_max,d = 125` hold *throughout*. The first review checked every stride of
    the first row (min 0.0726, max 13.30); here they were checked at each `t = 1, 2, …`;
  - the share ratio is at least 3.
- The myosin settles by `t ≈ 5`, with its peak at 13.3.
- **The cut-off never switches in E4a.** The peak is 11% below `m_max,u = 15`, so `f ≥ 0.11` everywhere and
  `max(0, ·)` is never at its kink.
  - E4a therefore tests `f` only as a linear function of `m`.
  - The moving cases will reach the kink: ZV's rear band is meant to reach `m_max,u`. Decision 7 adds a test
    that does.
- **The steady state does not depend on `Δt`** (*inferred*). A steady state of the staggered step is one of
  the coupled system, since `v` lags `m` by a step in which `m` no longer changes. That holds for a given
  steady state; a problem with several could reach a different one at a different `Δt`.

**The transient does depend on `Δt`.** The share ratio at `t = 2`, at `h = 0.1`, is:

| `Δt` | 0.1 | 0.05 | 0.025 | 0.0125 |
|---|---|---|---|---|
| share ratio | 2.72 | 3.70 | 4.23 | 4.47 |

- The differences fall by about half as `Δt` halves (0.98, 0.53, 0.24; ratios 0.54 and 0.45): first order.
- That this is ADR 020 decision 3's splitting error is *inferred*. BDF restarting at order 1 at every stride
  also depends on `Δt`.
- **The Richardson extrapolation.** Extrapolating the last two values at first order gives a limit of about
  4.71. At `Δt = 0.05` the error is then about 1.0, some 20% of the ratio at `t = 2`.
- On a moving run the transient is the result (a centroid speed, a turning radius). So 5b needs a rule for
  `Δt` ("Not decided here").
- The paper's `Δt = c h`, with `c ≤ 0.025` and `h ≤ 0.16` (as E4's criteria quote it), is at most 0.004. That
  is below the smallest `Δt` here.

## Decisions

1. **The scheme for a cell's species: a staggered method-of-lines stride.**
   - **Each stride:**
     - solve `v` on the current configuration from `mⁿ`;
     - integrate the species over `[tⁿ, tⁿ + Δt]` with PETSc TS (adaptive BDF), with `v` held;
     - the transport form, its natural condition and its conservation are those of ADR 020 decision 2.
   - **Where it runs.** In the multi-compartment solver's stride loop (decision 8), which starts each stride at
     its own time and carries the step size. BDF restarts at order 1 each stride, since `v` changes between
     strides, so the history would belong to a different operator.
   - **Conservation is limited by the solver's tolerance, not by round-off.**
     - With `q = 1` the transport and diffusion terms vanish.
     - A member species has no Dirichlet rows, and constants are in P1.
     - So `∫m` changes only by what the Newton and GMRES solves leave unconverged.
     - The drift was systematic, about −7e-15 relative per stride (the first review's run), so it grows with
       the number of strides. At the paper's `Δt` to `t = 30` that is about 5e-11, well inside the
       criterion's `1e-8`.
   - **The options:**
     - **(a) A method-of-lines stride,** as above.
     - **(b) Newton backward Euler:** one nonlinear solve per step, with `v` held. It needs the stride's TS to
       take a type: `make_ts` sets `bdf` today (`multi_compartment.py` 765). (a)'s adaptive steps control the
       error within a stride at no extra cost.
     - **(c) Lagged coefficients:** the cut-offs evaluated at `mⁿ`, so the step stays linear. It is the
       cheapest, and a consistent first-order scheme for the same model. But it adds a second `O(Δt)` lag on
       top of the splitting's, in the term that limits aggregation, and its positivity near the cut-off is
       untested (*inferred*).
     - **(d) Monolithic:** `(v, m)` in one nonlinear solve. It removes the splitting error, and it is the
       most work. ADR 020 decision 3 keeps it, as fixed-point iteration (its option (b)), for when a
       measurement shows the splitting error matters.

   *Recommendation: (a).*

   - *Recorded in the implementation, 2026-11-18: one integrator across the strides, a solver option.*
     - **What it is.** `integrate_multi_compartment(..., persistent_integrator=True)` (CLI
       `--persistent-integrator`, `RunOptions.persistent_integrator`) keeps one TS across a force balance's
       strides. Between strides `v` is re-solved and the TS continues, its BDF history and order carried over the
       new `v`. Off by default. Refused with a moving membrane, whose moves remap the species, so the history
       would describe another mesh. A remesh changes the unknowns, so it always needs a fresh integrator. The
       summary records `integrator` ("fresh per stride" or "persistent").
     - **Checked first:** PETSc's TSBDF keeps its history across repeated `TSSolve` calls with a raised max time.
       On `u' = −u` in 50 solves it took 193 steps, against 171 for one solve and 338 restarting each time.
     - **Measured on this ADR's myosin model** (h = 0.15, β = 5). "Integration error" is the relative L² error
       against the same stagger integrated tightly; "total" is against a run with 4× shorter strides.

       | run | rtol | fresh per stride: steps, integration / total error | persistent: steps, integration / total error |
       |---|---|---|---|
       | t = 0.5, Δt = 0.025 | 1e-6 | 105, 9.1e-6 / 7.5e-4 | 140, 3.4e-5 / 7.7e-4 |
       | t = 0.5, Δt = 0.025 | 1e-4 | 26, 1.2e-4 / 8.6e-4 | 26, 5.4e-4 / 1.3e-3 |
       | t = 4, Δt = 0.05 | 1e-6 | 3902, 1.2e-5 / 3.0e-2 | 4058, 2.1e-4 / 3.0e-2 |
       | t = 4, Δt = 0.05 | 1e-4 | 1011, 3.3e-3 / 3.3e-2 | 1016, 1.7e-2 / 4.7e-2 |
       | t = 4, Δt = 0.05 | 1e-3 | 532, 8.1e-3 / 3.8e-2 | 403, 6.7e-2 / 9.5e-2 |

     - **What it shows.**
       - Keeping the integrator saves no steps, except at rtol 1e-3, and saves 0–30 % of the wall time (the TS
         set-up per stride).
       - Its integration error is 4–17× larger. Its BDF history spans the kink each new `v` puts in the species'
         rate, and its error estimate, built from that history, under-reports it (*inferred* from the error
         ratios).
       - The steps per stride (about 50 at rtol 1e-6, t = 4) are set by the dynamics inside a stride, not by
         BDF's restart at order 1. The restart was not the cost.
     - **On this model, a looser tolerance saved more.** The stagger's error dominates (total ≫ integration). At
       t = 4, rtol 1e-4 instead of 1e-6 takes 3.9× fewer steps and 2.9× less wall time, and raises the total error
       from 3.0e-2 to 3.3e-2.
     - **This is not a default to loosen** (Jim, 2026-11-18: many models are not well scaled). This model is O(1)
       in one species. The integrator weighs each unknown's error by `atol + rtol·|uᵢ|`, with one scalar `atol` for
       every species (`make_ts`), so (*inferred* from that weight):
       - a species far below `atol / rtol` is controlled by `atol` alone, and loosening `rtol` changes nothing
         for it. If `atol` is already above its scale, it is effectively uncontrolled;
       - a species far above that is controlled by `rtol`;
       - a quantity that matters through a difference or a small fraction of a large total can lose its
         significant digits at a looser `rtol`, whatever its scale.
     - **What would make a looser tolerance safe** is a question for decision 7, not settled here:
       - per-species absolute tolerances from each species' own scale, as a PETSc vector `atol`;
       - a measured indicator of the stagger's error (the change in `v` over a stride), so that `rtol` follows
         the error the stride commits rather than a fixed number.

2. **A slot may read the species on every route, and a model nonlinear in its species runs by the method of
   lines.** The switch was decided by Jim on 2026-11-17, for the T9 route. The amendment widens it to every
   route.
   - **Both assemblers bind the species in every slot.** This is the general capability the amendment adds.
     - **The single-mesh assembler** compiles `diffusion`, `relative_advection` and `advection` as it compiles
       `source`: with the species' trial functions bound. Its method of lines then replaces `trial → unknown`,
       as it already does for a source.
     - **The multi-compartment solver** compiles them in `bulk_ctx` (a compartment's species and the region
       variables). Its Jacobian follows by `ufl.derivative` (the Context).
     - **A membrane species' transport** reads only the membrane's own species. Its forms are on the
       membrane's own mesh, where neither the bulk traces nor the region variables (Reals on the parent) live.
       A membrane diffusion or drift that reads one is refused by name (*recorded in the implementation,
       2026-11-17*).
     - **On a member,** the T9 velocity is bound as a lagged coefficient (decision 8).
     - A tensor `diffusion` is bound the same way.
   - **The linearity check.** A pure function of the model decides which expressions are nonlinear in its
     unknowns, before anything is assembled. It is the same on every route. Its rules follow what backward
     Euler can split, not affinity alone: `u**1` and a literal such as `[u, 0]` are affine, but UFL cannot split
     them, so they count as nonlinear (*recorded in the implementation, 2026-11-17*). A Dirichlet value is
     imposed rather than assembled, so it is not checked.
     - **The unknowns** are the model's variables: species and region variables. The velocity of a T9 force
       balance is excluded, because it is lagged (ADR 020 decision 3). Parameters cannot read a variable (the
       Context), so no closure through parameters is needed.
     - **Every expression is checked:** each template's slots (T1's and T2's transport and `source`, T4's
       `rate`, T5's `uniform_rate` and `region_rate`) and every boundary condition (`flux`, `neumann`,
       `robin`, interface fluxes).
     - **An expression is nonlinear** in any of these cases:
       - a transport slot (`diffusion`, `advection`, `relative_advection`) reads an unknown: its coefficient
         multiplies `∇u` or `u`;
       - any other expression is not affine in the unknowns: a product of two factors that read unknowns, an
         unknown inside a function (`exp`, `max`, `min`, a conditional), an unknown in a denominator, or an
         unknown raised to a power other than 1.
     - `trace(X)` counts as reading `X`.
   - **The catch-all is narrowed.** `_compose_backward_euler` will report only an `ArityMismatch` and UFL's
     error for an argument inside a nonlinear operator as `NonlinearTermError`. Any other `ValueError`
     propagates as the defect it is.
   - **The switch, by route:**
     - **Single mesh, fixed or moving:** a nonlinear model runs by the method of lines, whatever
       `--time-integration` says.
       - A request for backward Euler is overridden with a logged note naming the nonlinear expressions, as
         the runner already notes an ignored option (`runner.py` 333–337).
       - A linear model keeps the requested scheme, backward Euler by default.
     - **Multi-compartment:** already method-of-lines only.
     - **T9 with species:** goes through the multi-compartment solver (decision 8), so it always runs by
       method-of-lines strides.
   - **VCell's SimulationTask.** Whether VCell writes `TimeIntegration` for its tasks was not checked here. A
     `BackwardEuler` it writes is overridden for a nonlinear model, with the note. Today such a task fails.
   - **What the manifest and `summary.json` record.** They already record `h`, `dt`, `dt_requested` and
     `time_integration` (`runner.py` 186–206). Under the method of lines:
     - `time_integration` is `method_of_lines`;
     - `rtol` and `atol` are added, since the result depends on them;
     - `steps` counts TS steps;
     - on a route that strides (a moving mesh, or a T9 velocity), `dt` is the stride (the length for which a
       mesh move or a solved `v` is held) and `strides` is added;
     - on a fixed single mesh, one TS covers the run, so `dt` is the initial step and there is no `strides`.
     - vcell-core's criteria ask for `Δt` and `h` with every result.

   *Recommendation: yes.*

3. **Stage 5 is split into 5a and 5b, and 5a comes first.**
   - **5a: a cell's species on the fixed skeleton.**
     - It covers decisions 1, 2, 4, 7 and 8.
     - **The scope widens where decision 8 removes the reason for a limit.** ADR 020 decision 4 refused
       species beside the cell, and fluxes across its membrane, because they "would need the multi-compartment
       solver's coupling". With decision 8 the species are in that solver. So 5a admits:
       - species in the other compartments, coupled across a **held** membrane by its flux conditions. On a
         held membrane `v = 0`, so a flux there is relative to the material, as ADR 020 decision 2 already
         allows ("unless a flux BC is given there").
     - **Still refused in 5a:**
       - a flux across a stationary front, where the material crosses the edge, so the meaning of a flux there
         needs its own decision;
       - membrane species on the member's membrane (ADR 018 decision 8's `ρ∇_Γ·v_Γ`, with 5b);
       - several members.
     - E4a is its test, with a coupled two-compartment case for the widened scope (decision 7).
   - **5b: a cell's species on the moving cell.**
     - It needs 4a (one cell moving, on the moving partition), and decisions 5 and 6.
     - E4-ZV, E4-ZS and E4-ZS0 are its tests.
     - How the solved `v` enters ADR 015's half-move stride is decided with 5b ("Not decided here").
   - **Why 5a now** (*inferred*):
     - it needs no moving mesh;
     - decision 2's capability is general and is wanted on its own;
     - decision 8's member carrier and fixed-partition strides are tested before anything moves. 5b then adds
       the moves, which the partition already makes for a prescribed front.
   - **5a is independent of 4a.** ADR 019 decision 6 orders 4 before 5. This amends it so that 5a may come
     first; 5b still follows 4a.

   *Recommendation: yes, 5a before 4a.*

4. **A diffusion that loses its sign stops the run, with the cause named, on every route.**
   - **Why.** A coefficient that reads a species can turn negative: `D (1 − m/m_max,d)` does above `m_max,d`.
     Diffusion backwards in time is ill-posed. A run that crossed it would fail inside TS with a generic
     `SolveError`, or carry growing mesh-scale noise (*inferred*, untested).
     - In every run here, the drift's cut-off keeps `m` near 13, far from 125 (*checked*).
     - A model with other parameters could cross it.
   - **One check function, called from every place a TS runs.** There is no single monitor shared by every
     route today.
     - `_run_time_stepper` installs a monitor only when there is an output or progress callback
       (`reaction_diffusion.py` 135–311).
     - `make_ts` installs one only on the fixed, unstrided path (`multi_compartment.py` 762–779, 924–945).
     - The stride loops set none.
     - So the check is a function both TS builders install as a monitor, on every path, strides included.
   - **What it checks.** Only an expression that reads an unknown (decision 2's check) can change sign during a
     run, so only those are checked:
     - a scalar diffusion, bulk or surface, must be positive at the mesh's nodes;
     - a tensor diffusion must have a positive smallest eigenvalue at the nodes;
     - the minimum is collective over the ranks;
     - if the check fails, the run stops with an error naming the species, the time and the least value.
   - **A failure inside a step.** Newton's iterates can cross zero before any step is accepted, so the TS
     fails first. Its failure path (`ts_failure_message`, `diagnostics.py` 54) runs the same check on the last
     iterate. If the check fails there, it names the diffusion rather than giving the generic message.
   - **How exact it is.** It reads the model's own expression, so it needs no knowledge of `m_max,d`. It is
     exact for a coefficient affine in a P1 field, such as E4's. For another coefficient, a check at the nodes
     can miss a negative value between them (*inferred*).
   - **Its cost** is one interpolation per accepted step, small beside a Newton solve (*inferred*; decision 7
     measures it).
   - The recorder already writes `min` and `max` of every variable at every output (`results/recorder.py`
     148–152). Nothing is added for them.
   - **The options:**
     - **(a) Stop,** as above.
     - **(b) Warn and continue.**
     - **(c) Clamp the coefficient at 0.** That changes the model the document wrote.

   *Recommendation: (a).*

5. **A front speed that reads `trace(m)`: nothing is missing in the grammar, and stage 5b wires it.**
   - **The grammar admits it.** vcell-core's six documents validate with no diagnostics (*checked*, run here
     at `7807a14`).
   - **The moving partition already evaluates such a speed for a prescribed front** (*checked*):
     - `MovingMembrane` copies each bulk species named in a `trace(...)` onto the membrane's nodes before it
       evaluates the velocity (`moving_partition.py` 219–247 and 359–362);
     - the stride loop scatters the current state into those species before each half-move
       (`multi_compartment.py` 805–808);
     - the first half-move comes before the integration, so it reads `mⁿ`. The second comes after it, so it
       reads `mⁿ⁺¹` (890–901). That is ADR 015 decision 2's symmetric split.
   - **What 5b adds.** The front of a cell whose force balance is solved (4a's) reads the same traces. With
     decision 8 the species it reads are the multi-compartment solver's own, so nothing is copied between
     solvers.
   - **The front keeps ADR 015's split.** Whether the solved `v` should be lagged the same way, and whether
     that costs the split its second order, belong to 5b's open question (decision 3).

   *Recommendation: yes.*

6. **The outputs a run with a force balance needs.**
   - **What exists** (*checked*):
     - The bundle stores a moving domain's per-row point coordinates in segments (ADR 010 §2–3;
       `results/writer.py`).
     - A moving multi-compartment run with a prescribed membrane registers each compartment and each membrane
       as a moving domain (`runner.py` 757–776). Its membrane's rows are the boundary over time.
   - **What decision 8 changes.**
     - Today the T9 route registers one fixed volume domain on the members' mesh (`runner.py` 397), and it
       never moves (154).
     - Routed into the multi-compartment route, T9 registers its domains as that route does.
     - `v` is kept out of the species and written by component on the member, as ADR 020 decision 5 writes it.
     - At 5b they are moving domains, so `m` on the current mesh and the boundary are written with no new
       bundle machinery (*inferred*: 5b's route is not built).
   - **`v` at an output.** ADR 020 decision 5 writes, at `tⁿ`, the `v` solved from `mⁿ`. The stride loop
     records an output at the stride's end, before the next stride's hook solves `v`. So the route solves `v`
     once more at each output time, as the T9 route does today with `solve_velocity()`. This costs one extra
     solve per output, not per stride.
   - **The centroid over time: added, for every moving volume domain on any route.**
     - The recorder already re-assembles a moving domain's measure at every capture (`results/recorder.py`
       135–136). It will also assemble `∫x dx / |Ω|` there, which is exact for the mesh's own geometry
       (*inferred*).
     - ADR 010's statistics are per variable, so the centroid gets an array of its own per moving volume
       domain: `<prefix>centroid/<domain>`, of shape `(T_seg, gdim)`, listed in the manifest's domain entry.
     - It is an additive minor version of the manifest. A reader that does not know the array ignores it.
   - **The options:**
     - **(a) The bundle carries the centroid.**
     - **(b) Readers compute it from the coordinates.** Then every reader (VCell's, vcell-core's tests, ours)
       repeats the same integral.

   *Recommendation: (a).*

7. **Stage 5a's tests.**
   - **E4a, from vcell-core's sibling checkout,** read as #253 reads E2's (skipped when the checkout is absent).
     It is checked against vcell-core's proposed criteria (#255) at every output. vcell-core proposes the
     criteria; the tests here decide what is asserted.
   - **The two species paths agree** (decision 8). Before the T9 driver's species problem is retired:
     - step 3b's tests pass on the new path;
     - E2 is run both ways, over `t ∈ [0, 0.5]`, before the instability grows, at four `Δt` halving from
       0.05, with the method of lines at `rtol = 1e-9`.
     - Both paths carry the same stagger, so their difference is backward Euler's own error. It must fall at
       a ratio between 0.4 and 0.6 per halving.
     - `_run_member_transport` (`mms/runner.py`) moves to the new path, and `member_transport_carried` keeps
       its order there.
     - *Recorded in the implementation, 2026-11-17:*
       - The difference's ratio per halving, measured at h = 0.15 on the step 3b model (E2 as written there): 0.652
         from Δt = 0.05, then 0.577, 0.539 and 0.52 down to 0.003125. The first halving is pre-asymptotic, so the
         test (`tests/test_member_species.py`) starts at Δt = 0.025 and asserts the next three.
       - `member_transport_carried` runs by the method of lines in one stride (its velocity is steady), at
         `rtol = 1e-9`: order 1.83 in L∞ for h 0.1 → 0.025 (1.8 by backward Euler). With the carrier's sign
         flipped the error stays at 2.0.
   - **A transport slot reading a species, on every route** (decision 2): a diffusion and a drift that read
     the species compile and run on the single mesh (fixed and moving) and on the multi-compartment solver.
     They are checked against a manufactured solution with their spatial order: a new case of the
     `member_transport` harness, and the matching single-mesh case.
     - Their sources read `sim.t`, so the time is checked in every stride.
     - The manufactured solution keeps `m` below `m_max,u`, away from the kink, which the next test covers.
   - **The cut-off at its kink.** E4a never reaches it (the Context), so a separate test drives `m` past
     `m_max,u`. It prescribes a velocity that converges on a point, with the drift cut-off on.
     - It checks that the strides converge, that `∫m` is conserved, and that `m` saturates near `m_max,u`.
     - Saturation is not a bound of the continuous model. At `m = m_max,u` the transport's contribution to
       `∂m/∂t` is `∇m·v`, which can be positive (the first review's algebra). So a band is asserted, set from a
       measurement.
   - **The widened scope** (decision 3): a species in the medium coupled to the cell's across the held
     membrane by a flux conserves the total of the two.
   - **Structural and exact:**
     - `∫m` is conserved to `1e-10` relative over a run of a few hundred strides. The bound is set by the
       solvers' tolerances (decision 1), so a longer test tightens the Newton tolerance rather than the bound.
     - The linearity check (decision 2), expression by expression. It covers:
       - `trace`;
       - a species in a denominator and in a conditional;
       - a region variable;
       - a slot that reads only the lagged `v` (linear);
       - a boundary flux `−m/(1+m)` (nonlinear).
     - The switch (decision 2), on each route:
       - a nonlinear model runs by the method of lines whatever was asked, and the manifest says so;
       - a linear model keeps its scheme;
       - an unrelated `ValueError` is not taken for nonlinearity.
     - The diffusion check (decision 4):
       - on a single-mesh run and on a member, at an accepted step;
       - in a TS failure.
     - `v` at an output is the one solved from that output's species (decision 6).
   - **Measured, not pass/fail:**
     - the transient's dependence on `Δt` (the Context's table), reported with E4a's result;
     - the cost per stride, and the cost of decision 4's check;
     - a run on 2 MPI ranks, which must match 1 rank to the solver's tolerance. ILU becomes block Jacobi in
       parallel, so the iterations differ, not the answer (*inferred*).

   *Recommendation: yes.*

8. **One species path: a cell's species run through the multi-compartment solver.**
   - **Why.** At 5b the species must move with the cell. The moving partition already does what that needs:
     - it integrates species stride by stride;
     - it carries them across each move exactly (ADR 015);
     - it remeshes and transfers them (ADR 016);
     - it writes them as moving domains.

     A T9 driver that kept its own species problem would need either a second moving-species path, with its
     own conservation and remesh transfer, or a copy of the species between two solvers at every half-move.
     The T9 driver's species assembly is already a fifth transport form, beside ADR 020's four.
   - **One mesh.** The force balance is built on the member compartment's own submesh: a `MembersMesh` made
     from `geometry.compartments[member]`, passed as `prebuilt`. So `v` and the species live on the same mesh
     object, and each is a coefficient in the other's forms with no copy.
     - The members' mesh is built today as a second submesh of the same cells (the Context). Reusing the
       compartment's removes the second object.
     - With several members (not in 5a) the force balance spans several compartments, and the question of how
       `v` enters each compartment's forms returns ("Not decided here").
   - **What the multi-compartment solver gains:**
     - **The member's carrier.** On a compartment whose motion is a T9 velocity, the form is ADR 020 decision
       2's: `−∫ u (v + w − v_mesh)·∇q`. Here `v_mesh = 0` on a fixed partition, and the parent's mesh
       velocity on a moving one.
       - Today it adds `−u (w − v_mesh)·∇q` on a moving partition, and the non-conservative `(w·∇u) q` on a
         fixed one (`multi_compartment.py` 528–538).
       - Compartments that are not members keep their forms and natural conditions (ADR 020's Context).
       - The T9 velocity is bound in the member's compile context, so a slot such as E4a's drift can read
         `v_cyt`.
     - **Strides on a fixed partition,** with a hook before each that solves `v` from the current species.
       Today strides exist only with a moving membrane (`stride_ends`, `multi_compartment.py` 334).
     - **T9 is handed to the force balance,** instead of being refused (345–349). The force balance's held
       membranes (`dirichlet v = 0`) and ADR 020 decision 4's refusals stay in `cell_force_balance.py`.
   - **What retires.** The species problem in `cell_force_balance_run.py`, once decision 7's agreement tests
     pass.
     - Step 3b's backward-Euler run of a member's species becomes a method-of-lines run.
     - Its conservation goes from exact for backward Euler with a direct solve to solver-tolerance-limited
       (decision 1; −1.4e-12 here). Both are far inside every criterion.
   - **What is not moved.** The single-mesh routes keep their own assembly, which decision 2 extends. Merging
     them into the multi-compartment solver is not this note's question.
   - *Recorded in the implementation, 2026-11-17:*
     - **The scope check** is `member_species` in `cell_force_balance.py`. The solver runs it, and the runner runs
       it before a bundle starts, with the force balance assembled once on placeholders for the species, so a
       refusal still writes nothing.
     - **The velocity is a coefficient** of the member's species forms, which are on the parent mesh with the
       compartment's entity map, as the species' own functions are. Its Jacobian has no `v` column: `v` is held.
     - **Strides** reuse the moving partition's loop, with no move when no membrane moves. The `v` solved for an
       output carries the next stride, so an output costs no second solve there.
     - **`velocity_override(t, v)`** fills `v` in place of the solve, for verification (the MMS case, the known
       steady state).
     - **The runner's summary** gives `backend: multi_compartment`, `coupling: staggered`, `dt` the longest stride,
       `strides`, `steps` (TS steps), `rtol` and `atol`. The bundle gains the membrane as a domain without
       species, as on every multi-compartment run. As there, `geometry.cells` counts the whole partition (it was
       the member's cells) and `geometry.measure` sums every domain's, the membrane's length included.
     - **At a stationary front** any species' BC and any region variable are refused, not only a member
       species' BC: a flux there, on either side, is decision 3's open question. A front also held at `v = 0`
       counts as held.
     - **The solver refuses** a T9 run with a restart (`t_start`, `initial`), a pending move or a remesh stop, an
       output time that is not a stride's end, and a `velocity_override` without T9.
     - **Parameters cannot read the species** on this path, since the solver validates the model (the Context).
       The T9 driver, which did not validate, compiled such a parameter; its test of that is not ported.
   - *Recorded 2026-11-18:* `cell_force_balance_run.py` and its tests are deleted, with the agreement test that
     compared it with the new path (its ratios are recorded in decision 7). Its test that a parameter may read the
     species was not ported (above); the rest are in `tests/test_member_species.py`.
   - **The options:**
     - **(a) At 5a,** as above.
     - **(b) At 5b.**
       - 5a runs E4a in the T9 driver, striding with `integrate_discrete_problem_stride`, which already exists
         (the Context).
       - The cost is not a second stride implementation. It is a member carrier and a species path that 5b
         replaces anyway, plus the widened scope of decision 3 waiting until 5b.
     - **(c) Never.** The T9 driver grows its own moving species, a second moving-species path to keep in step
       with ADRs 015 and 016.
   - **Why (a)** (*inferred*). The deciding reason is 5b: a cell's species must end up in the moving
     partition. Doing it at 5a costs more than (b) at first, but:
     - it builds the member carrier once;
     - it tests the carrier on a fixed partition before anything moves;
     - it brings decision 3's widened scope (species beside the cell) now.

   *Recommendation: (a).*

## For vcell-core (the reply to #255, posted on Jim's say)

1. **The scheme.**
   - Method-of-lines strides, staggered: `v` from `mⁿ`, then `m` over the stride by adaptive BDF, with `v`
     held.
   - It runs in our multi-compartment solver, which then carries a cell's species on a fixed partition (5a)
     and on a moving one (5b).
   - On every route, a model nonlinear in its species switches to the method of lines by itself.
   - Diffusions and drifts that read the species compile on every route, not only with a force balance.
   - Where the solved `v` enters the moving partition's half-move stride is decided with 5b.
2. **E4a comes first,** as stage 5a, independent of 4a.
   - A prototype of the scheme, the T9 driver with the method of lines bolted on, meets every E4a criterion
     throughout a run at `h = 0.1`, `Δt = 0.05`. 5a builds the real path and re-runs E4a on it.
   - E4a never reaches `m_max,u`: its peak is 13.3, so the drift's cut-off never switches. ZV's rear band is
     where the kink is first exercised, and 5a adds its own test of it.
3. **`trace(m)` in a front speed.** Nothing is missing in the grammar.
   - The moving partition evaluates it today for a prescribed front.
   - It reads `mⁿ` at the stride's first half-move and `mⁿ⁺¹` at its second: ADR 015's symmetric split.
   - 5b wires it for a cell whose force balance is solved.
4. **The outputs.**
   - The bundle already carries the species on a moving mesh, and the boundary over time, for a moving
     multi-compartment run. With a cell's species in that solver, 5b writes them the same way.
   - The centroid per row and `v` on the moving cell are added (decision 6).
5. **The approximations.**
   - The results depend on `Δt` at first order in the transient, and not at a steady state. We attribute this
     to the one-step lag; the BDF restart at each stride may contribute.
   - `Δt` (the stride), `h`, `rtol` and `atol` are reported with each result.
   - A diffusion coefficient's sign is checked at every step.

## Not decided here

- **How the solved `v` enters ADR 015's half-move stride** (5b, and ADR 020 decision 2's open question):
  - on which mesh it is solved;
  - whether it is solved once or twice per stride;
  - what that does to the split's order.
- **The time-step rule for 5b:** a fixed `Δt`, or one tied to the front's speed and `h`. It matters because the
  transient depends on `Δt` at first order (the Context).
- **Several members:** how `v` enters each member compartment's forms.
- **Fixed-point iteration** of the stagger (ADR 020 decision 3 (b)), if the moving cases' transient needs it.
- **Stabilisation of the transport** (SUPG, a limiter), if `min m` fails on the moving cases.
- **A flux across a stationary front;** membrane species on the member's membrane; species beside a moving cell.
- **A drift together with a carried fraction** (vcell-core's note, decision 1).
- **Merging the single-mesh routes into the multi-compartment solver.**
- **Cost.** At the paper's `Δt ≤ 0.004` and `h = 0.1`, a run to `t = 30` is about 7500 strides. At the first
  review's 0.1 s per stride that is about 12 minutes, before any mesh moves (*inferred*). Whether 5b needs the
  paper's `Δt` is the time-step rule above.

## Review

### First review (the note before the amendment)

An independent reviewer read the draft and its sources. It re-ran the experiment's first row, traced the
integrator's calls, checked the line references and the issue's numbers. It found 19 issues: none blocking, 7
major, 9 minor and 3 nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | major | "Needs no new code" is false. `integrate_discrete_problem` always starts at `t = 0`, so `sim.t` is wrong in every stride after the first, and each call rebuilds forms and restarts BDF. | The Context states it, and names the stride primitive that starts at its own time (second review, 4). Decision 1 strides in the multi-compartment solver. Decision 7 tests `sim.t`. |
| 2 | major | Which `m` the front reads is checkable now: the first half-move reads `mⁿ`, the second `mⁿ⁺¹`, ADR 015's symmetric split. | Decision 5 states what the code does and keeps ADR 015's split. |
| 3 | major | "Needs no second scheme" glosses over ADR 020's open question. The moving partition integrates with its own TS. | Decision 8 puts a cell's species in that TS from 5a. The open question is in "Not decided here". |
| 4 | major | `NonlinearTermError` is raised for any `ValueError`, and for nonlinear sources too. | Decision 2: a linearity check on every expression; nonlinear sources admitted, stated; the catch-all narrowed. |
| 5 | major | E4a's peak is 13.3, so `max(0, ·)` never switches; nothing tests the kink. | The Context says so. Decision 7 adds a test past `m_max,u`. |
| 6 | major | Option (c) is a consistent scheme for the same model; `m ≤ m_max,u` is not a bound of the PDE; (b) is the same code. | Decision 1's options corrected. |
| 7 | major | "Written already, for any moving run" covers the multi-compartment route only. | Decision 6 separates what exists from what decision 8 gives 5b. |
| 8 | minor | `steps` and `dt` change meaning; `rtol`/`atol` matter. | Decision 2, per route. |
| 9 | minor | A check only at outputs is late; it must be collective; it is exact only for an affine coefficient. | Decision 4. |
| 10 | minor | "Throughout" not shown; the drift column's units. | The Context. |
| 11 | minor | Attribution to the splitting is inferred; the 20% is unshown. | Marked and shown. |
| 12 | minor | No `Δt` rule in decision 7. | "Not decided here". |
| 13 | minor | The `member_transport` harness is not cited; "agrees to `O(Δt)`" is not assertable. | Cited and extended; decision 7 gives the window, the `Δt`s and the band. |
| 14 | minor | Conservation is limited by the solvers' tolerance. | Decisions 1 and 7. |
| 15 | minor | Whether VCell writes `TimeIntegration` is unchecked. | Decision 2 states the consequence. |
| 16 | minor | No MPI statement and no cost. | Decision 7; "Not decided here". |
| 17 | nit | The failure is UFL's split, not the arity check. | Corrected. |
| 18 | nit | The steady state's independence of `Δt` holds for a given steady state. | Qualified. |
| 19 | nit | The centroid has no slot in ADR 010's statistics. | Decision 6 names the array. |

### Second review (the amended note)

A second independent reviewer read the amended note. It was asked in particular whether the amendment answers
the overfitting question, and whether decision 8 is feasible.
- It ran the multi-compartment and single-mesh routes on species-reading slots.
- It compared the two submeshes on 1 and 2 ranks.
- It checked every new line reference; all hold.

It found 16 issues: 1 blocking, 4 major, 7 minor and 4 nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | blocking | No general route can compile a diffusion or drift that reads a species, under either scheme: both assemblers compile transport slots without the species (`CompileError`). The switch alone fixes nothing, and "every route" was a promise. | The Context's table corrected. Decision 2 binds the species in every slot of both assemblers. "Would amend" lists both. Decision 7 tests it on each route. |
| 2 | major | Decision 8 does not say how `m` and `v` cross between the force balance's mesh and the solver's. They are two mesh objects; `v_cyt` is bound nowhere in the solver. | Decision 8: the force balance is built on the member compartment's own submesh, and `v` is bound in the member's context. |
| 3 | major | No TS monitor is shared by every route; the stride loops set none. | Decision 4: one check function, installed by both TS builders on every path. |
| 4 | major | `integrate_discrete_problem_stride` already starts at its own time, so option (b)'s cost was overstated. | The Context names it. Decision 8 re-weighs the options on 5b's moving species. The "Not decided here" entry is removed. |
| 5 | major | The linearity rule omits boundary conditions, region variables, other templates' slots and tensors, and does not exclude the lagged `v`. | Decision 2: every expression, the unknowns defined, `v` excluded, parameters noted. Decision 7 tests a nonlinear flux. |
| 6 | minor | "Would amend" omits `assemble.py`, `cell_force_balance.py`, the runner's routing, `mms/runner.py`, ADR 020 decision 4's sentence, and ADR 019's order. | Added. |
| 7 | minor | 5a keeps "species on the member only" though decision 8 removes the reason. | Decision 3 widens 5a to species coupled across a held membrane; what stays refused is stated. |
| 8 | minor | Decision 4 misses tensors, surface diffusion, and a failure inside a Newton iteration. | Decision 4 covers all three. |
| 9 | minor | With a hook before each stride, an output would get `v` from `mⁿ⁻¹`. | Decision 6: one extra solve per output. |
| 10 | minor | The agreement test's ratio band needs a window, enough `Δt`s and a tight `rtol`; E2 diverges later. | Decision 7 gives them. |
| 11 | minor | The reply calls the `Δt` dependence the lag's (inferred), and does not say the experiment was a prototype. | The reply corrected. |
| 12 | minor | "`dt` is the stride" does not fit a fixed single mesh. | Decision 2 defines it per route. |
| 13 | nit | `make_ts` hard-codes BDF. | Decision 1 (b) says so. |
| 14 | nit | The node order can be marked checked. | The Context marks it checked. |
| 15 | nit | `∂m/∂t = ∇m·v` omits diffusion. | "The transport's contribution". |
| 16 | nit | ADR 018 is missing from the references. | Added. |

## References

- **This repository** at `7807a14`:
  - ADRs 010, 014, 015, 016, 018, 019, 020;
  - `src/vcell_fenics/backend/assemble.py` 228–278 and 343;
  - `cell_force_balance.py` 552–557;
  - `cell_force_balance_run.py`;
  - `discrete.py` 630–645;
  - `reaction_diffusion.py` 135–311, 314–392 and 461–500;
  - `multi_compartment.py` 233–237, 309–349, 482–541, 636–638, 762–808 and 850–945;
  - `moving_partition.py` 219–247 and 359–362;
  - `ale.py` 218–219;
  - `diagnostics.py` 54;
  - `src/vcell_fenics/runner.py` 152–154, 186–206, 333–337, 366–370, 397, 489, 559–629, 726–727 and 757–776;
  - `src/vcell_fenics/cli.py` 93;
  - `src/vcell_fenics/results/recorder.py` 135–152;
  - `mms/runner.py` 321–369 and `mms/cases/member_transport_carried.yaml`;
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

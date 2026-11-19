# ADR 023 — Stage 4a: one cell moving in 2D

**Date:** 2026-11-19
**Status:** Proposed. Asked for by Jim on 2026-11-19 ("start on stage 4a"). Reviewed once (16 findings, 1
blocking; "Review" at the end).

**Builds on:**
- ADR 019 decision 6, item 4 (the scope, *checked*): one closed membrane in 2D, solved rather than prescribed, with
  no nucleus and not touching the box. First a material membrane with a tension, then a growing front (ZS, and ZV
  with `v_p` from the area and constants). No species. It reuses the closed-loop slide and contact check, and
  runs on the **moving partition** (ADRs 014–016), where `region_size` is re-measured after every move (Jim,
  2026-11-15, #240).
- ADR 019 decision 5 (the growing front, *checked*): its velocity is a surface `prescribed` motion reading
  `trace(v)`; the member keeps `motion: unknown`; "The domain of a member follows its bounding surfaces, not `v`";
  no tension on a front; the solve order is "solve `v` on the current geometry, evaluate the front velocity from
  it, move".
- ADR 018 decisions 5 and 7 (*checked*): semi-implicit tension, where "`Δt` is the run's, as on every other path,
  and is fixed in the first stages"; membranes move with `v`, the interior by a harmonic extension of the
  skeleton.
- ADR 015 decisions 1–2 (*checked*): each stride moves the mesh by half a stride, integrates on the midpoint mesh,
  moves the other half, and carries every species' moments exactly across each move; the symmetric split was
  chosen for its second order in time (015 lines 54–62, 137–144).
- ADR 021, "Not decided here" (*checked*, lines 676–679): how the solved `v` enters ADR 015's half-move stride —
  on which mesh it is solved, how often, and "what that does to the split's order". This note decides it for 4a,
  which has no species; 5b inherits the motion and decides the species' side.

## Context

### What exists (*checked* unless marked)

- **The moving partition** (`backend/moving_partition.py`, `MovingPartition`). One membrane moves with a
  prescribed velocity, a string compiled once into a `fem.Expression` on the membrane mesh and re-interpolated at
  every move (186–253, 358–366). It may read the parameters, `region_size` (Constants the solver re-measures after
  each move, 240–241), `geom.x`, `sim.t` (the move's end time, set by `advance`, 360), `geom.normal` (out of the
  inside compartment) and `trace(X)` for **bulk scalar species**, copied per scalar dof by coincident node (215–225,
  362). A membrane without a prescribed velocity is refused (199–201).
  - The move: the front's displacement `h·v`; in 2D serial the slide back to the nodes' arc-length fractions
    (258–264, 370–372); a harmonic extension into the parent with the box and every other membrane held (266–298);
    every submesh takes the parent's displacement at its coincident nodes (300–308).
  - The checks: `FrontContact` (2D serial, 114–170, 386–388), `CellOrientation` and a growth limit (389–394); a
    failed move is undone (341–356). Under MPI the slide and the contact check are skipped silently (258, 311). A
    remesh is triggered when a cell falls below a quarter of its as-built area (72, 330–339), in 2D serial only
    (`runner.py` 909; `partition_remesh.py` 73, 150).
- **The stride loop** (`multi_compartment.py` 913–1080): first half-move, then (with a force balance) the velocity
  solve, then a fresh TS over the whole stride on the midpoint mesh, then the second half-move (1054–1076). Each
  move carries the species' moments (941–978). On a fixed partition the velocity solved for an output carries the
  next stride (`output_fields`, 1000–1008, reused at 1060–1062). Stride lengths may differ per output interval
  (`runner.py` `_stride_ends`, 730–739).
- **The force balance on the moving partition is refused**: `multi_compartment.py` 395–399 ("a cell force balance
  … beside a moving membrane …: species on a moving cell are stage 5b") and 402–405 (no restart, pending move or
  remesh stop). A force balance with no species on the member is refused (`cell_force_balance.py` 360–363), also
  through the runner's `_prepare_cell_force_balance` (`runner.py` 391), and the runner refuses any prescribed
  motion with a T9 model (379–384).
- **The runner's routing.** A moving partition is found from the subdomains with a *prescribed* velocity
  (`_moving_subdomains`, `runner.py` 546–551), so a material membrane (`motion: unknown`) is not seen as moving. A
  force balance reaches `integrate_multi_compartment` only when no membrane moves (854–869); the moving branch,
  `_run_moving_partition` (872–875), records no velocity. With no species the member's domain is not registered
  (796–815), and a remesh's `start_segment` sizes each domain by its species (around 1004 in `_run_moving_partition`).
- **The semi-implicit tension** (`force_balance.py` 176–178, `TENSION_SEMI_IMPLICIT`, `Δt γ ∇_Γv:∇_Γφ`, its `Δt` a
  Constant, 274) is added only when `semi_implicit_dt` is passed. The multi-compartment solver passes none (552),
  since its membranes are held.
- **What is bound to the mesh at assembly** (`cell_force_balance.py`): the checks `_CellMinimum` (822–825; its
  docstring says a moving mesh "must rebuild it") and `_FacetMinimum` (841–869), the facet tags (694–717), and the
  force balance's own `region_size` Constants (`compile_parameters`, 269–295). The solver's re-measured set is built
  only from its own parameters, the surface contexts and the partition's `size_constants` (`multi_compartment.py`
  470–473, 542), so the force balance's are not in it.
- **With one member, the members' mesh is the member compartment's own submesh** (654–664), the same object the
  partition moves in place (300–308). `LinearProblem.solve` reassembles its matrix and vector on every call
  (DOLFINx `fem/petsc.py` 905–916), `Expression.eval` reads the current coordinates, and tags and dofs are
  topological. So after a move the force balance and its checks see the moved geometry with no rebuild (the
  review read these; decision 6 tests it). The `_CellMinimum` docstring is wrong about a move and right about a
  remesh.

### What vcell-core's E3 asks (`conformance/math/moving-resolved-cell.jsonl`, vcell-core `4e1d672`, *checked*)

- **Two solvable cases**, `zs` and `zv`. vcell-fenics' document is each line's `fenics.math_description` (not
  `description`, whose front is `v_p` alone). Both are a growing front on a unit disk, `cell` in `ec`, T9
  `laplacian` with `η = 0.5`, `η_b = 0`, `ζ = 1`, `σ_a = 1`, and the front velocity

  `trace(v_cyt) + (v0·V0/(A·d) − kp·(A·d − V0³/(A·d)²))·geom.normal`, with `A = region_size(cell)`, `d = 0.5`,

  `V0 = π/2`, `kp = 3`, `v0 = 2.5` (so `k = kp·d = 1.5` in area units). ZS has no condition on the edge (zero
  traction); ZV holds `v_cyt = 0` there. The other seven lines are vcell-core's lowering refusals, with no document
  to solve. The geometry is a name only; vcell-core's oracle builds a unit disk in the box `[−2, 3] × [−2, 2]`
  (`tools/oracle/motility_solve.py` 68–79), which the test builds too.
- **No `criteria` field.** Each solvable line carries `exact`: an ODE for the area, `da/dt = 2√(πa)(v_p + u_r(R))`
  for ZS (with `u_r` from `I₁`, `ℓ = √(η/ζ)`) and `da/dt = 2√(πa) v_p` for ZV, and the areas at `t = 1`
  (3.36667386 and 3.69234754, reproduced by the review). vcell-core's spec asks for "the area against the ODE solved
  to tolerance, up to a stated time (`t = 1`), not the shape" (`moving-resolved-cell.md` 235).
- **Both ODEs settle fast.** Linearized at their equilibria, their rates are −30.6 (ZS) and −26.7 (ZV); they are
  within 0.1% of equilibrium by `t ≈ 0.14` and `0.19`, and `a(1)` equals `a(10)` to 1e-12 (the review's script,
  re-run here). So the area at `t = 1` carries almost no time error: it measures the equilibrium, in space only.
  The transient is in `t ≲ 0.2` (`a(0.05)` = 3.3192 for ZS, 3.5539 for ZV).
- **Rates.** `motility-end-to-end.md` 242–248 sets the rates in `h` for E1, capped by the realized boundary of
  #249, which is closed (PR #257): E1's criterion is now an observed rate ≥ 1.8 in `h`. It also warns that
  refining `h` and `Δt` together lets the time error cancel the spatial one.
- **E3's ZV `v_p` is the ZS law** (line 2), not UC-A's ZV law `v₀a₀/(a(1 + m)) − k(a − a₀)` (spec 23–24). The line
  does not say why (*inferred*: a myosin-free ZV). It is reported to vcell-core; the test runs the line as written.
- **A material membrane** has no vcell-core vector: "E3 does not need it" (spec 138–144).

## Decisions

1. **4a runs in the multi-compartment solver's stride loop, with no species.**
   - Each stride is the existing loop with no TS when there are no species: half-move, half-move (decision 3). The
     refusals of a force balance on a moving partition (`multi_compartment.py` 395–405) and the "needs a species"
     refusal (`cell_force_balance.py` 360–363, `runner.py` 391) are lifted for 4a's scope (decision 7); species on
     a moving cell stay refused until 5b.
   - **Why the solver, not a new loop** (*inferred*): 5b puts the species into this loop, with their moments
     carried across each move (ADR 015). A separate 4a loop would be replaced at 5b.
   - *Recommendation: yes.*

2. **The front reads the solved velocity through `trace(v)`, as it reads a species.**
   - `MovingPartition`'s compile context binds `trace(<T9 variable>)` to the velocity's values at the membrane's
     nodes, copied by coincident node as a species' trace is, vector-valued (block-size indexing in the copy).
   - **A material membrane** (`motion: {kind: unknown, variable: v}`, ADR 018 decision 2) moves with `trace(v)`:
     `MovingPartition` builds that velocity itself for such a membrane. A growing front writes its law, which reads
     `trace(v)` (ADR 019 decision 5; E3's line above).
   - The force balance is then assembled before the partition is built (today the order is the reverse,
     `multi_compartment.py` 539 against 552), so the partition can read its velocity Function.
   - **The options:** (a) bind the trace, as above; (b) a new hook that hands the solver's `v` Function to the
     partition. (a) keeps one path for every symbol a front may read, and E3's documents compile unchanged.
   - *Recommendation: (a).*

3. **The motion is the explicit midpoint rule: `v` solved at the stride's start and at its midpoint, two solves per
   stride, second order in `Δt`.**
   - Write `F(x)` for the front's velocity evaluated on the geometry `x` (solve `v` on `x`, then the front law, with
     `geom.normal` and `region_size` on `x`). A stride from `xⁿ`:
     - **first half-move:** solve on `xⁿ`; move by `(Δt/2)·F(xⁿ)` to the midpoint mesh `x^{n+½}`;
     - **second half-move:** solve on `x^{n+½}`; move by `(Δt/2)·(2F(x^{n+½}) − F(xⁿ))`.
     - In total `x^{n+1} = xⁿ + Δt·F(x^{n+½})`: the explicit midpoint rule, second order in `Δt` (*inferred*, a
       Taylor expansion; decision 8 measures it). The front law's `sim.t` is each move's end time, as today
       (`advance`); `v` comes from the mesh the half-move starts on.
   - **Its stability** is half of option (a)'s: on a linear ODE `a' = λa`, `|λ|Δt ≤ 2` against `4` (*inferred*). E3's
     `λ ≈ −31` gives `Δt < 0.065` for the midpoint rule. A time-step rule is not decided here; the runner takes
     the user's `Δt` (the stride), and decision 8's runs stay below the limit.
   - **The options:**
     - **(a) Forward Euler per half-move:** solve on the current mesh before each half-move and move by
       `(Δt/2)·F`. Two solves per stride, first order in `Δt`. With no species it is exactly (b) at a stride of
       `Δt/2`; the two differ only once species enter, at 5b.
     - **(b) Forward Euler per stride:** one solve at `xⁿ`, both half-moves using it (today's order on the fixed
       partition). First order; the species of 5b would be integrated on the midpoint mesh with `v` from `xⁿ`.
     - **(c) The explicit midpoint rule,** as above: the same two solves as (a), second order, which keeps ADR 015's
       reason for the symmetric split.
     - **(d) A fixed-point iteration** per step, as [Nickaeen 2017] did ("up to 35 iterations, tolerance 1e-10",
       p. 8, as ADR 020 decision 3 cites it). It removes the lag; kept as a later option.
   - **At 5b** the midpoint solve is the `v` held over the species' integration on the midpoint mesh (ADR 015).
     The species' transport relative to the mesh uses the mesh velocity over the stride, which today is the first
     half-move's (`multi_compartment.py` 632); under (c) the second half-move's differs, which 5b must reconcile
     (*inferred*; 5b decides).
   - **The output solve:** today the `v` solved for an output carries the next stride. Under (c) it is that stride's
     `F(xⁿ)` solve, so an output costs no extra solve.
   - *Recommendation: (c).*

4. **The semi-implicit tension's `Δt` is the length of the move the solve drives: the half-move's actual length.**
   - ADR 018 decision 5 linearizes the tension at `x + Δt v`. Each solve here drives a half-move, so the force
     balance's `Δt` Constant (`force_balance.py` 274) is set before each solve to that half-move's length,
     `(t_end − t)/2` for the current stride. Stride lengths can differ per output interval, so it is not one fixed
     value; ADR 018's "fixed in the first stages" is amended to "the length of the move the solve drives".
   - Under decision 3 (c) the second move is `(Δt/2)(2F^{n+½} − Fⁿ)`, not `(Δt/2)F^{n+½}`; the linearization is
     still taken at `x + (Δt/2) v`, the move the solved `v` alone would make (*inferred*: the stabilization's
     scale, not an exact match).
   - **It makes a membrane with a tension first order in `Δt`.** The term perturbs the force balance by `O(Δt)`: on
     a circle it adds `−(Δt/2)γ u(R)/R²` to the radial condition (the review's derivation). So decision 3's second
     order holds for a growing front, which has no tension, and not for a material membrane with one.
   - A held membrane, and a growing front, pass none.
   - **The options:** (a) the half-move's length; (b) the stride `Δt`; (c) none (the explicit tension, which ADR 018
     decision 5 rejected for its grid-scale instability).
   - *Recommendation: (a).*

5. **The force balance's own `region_size` is re-measured with the partition's.**
   - Its Constants (`compile_parameters`) join the solver's re-measured set, so a T9 slot that reads
     `region_size(cell)` (a pressure set by the area) sees the moved area. Today they would keep the as-built value
     (Context).
   - *Recommendation: yes.*

6. **What is rebuilt when.**
   - **After a move:** nothing (Context). A test compares a solve after moves with a fresh assembly on the same
     moved mesh, to the solver's tolerance, and the `_CellMinimum` docstring is corrected.
   - **After a remesh** (2D serial, ADR 016): the run continues in a new segment on the remeshed partition, and the
     force balance is assembled afresh there; its first stride starts with a solve on the new mesh. The restart
     refusal (`multi_compartment.py` 402–405) is lifted for 4a's scope.
   - *Recommendation: yes.*

7. **The routing and the outputs.**
   - **The predicate.** A T9 model in 2D with one member, whose one bounding membrane either moves with the
     member's material (`motion: unknown` naming the T9 variable: a material membrane) or has a prescribed velocity
     (a growing front), runs on the moving partition whatever its equation count (ADR 019 decision 6). The runner's
     `_moving_subdomains` and its refusal of a prescribed motion with T9 (`runner.py` 379–384) change for that
     scope; `_run_moving_partition` gains the force balance.
   - **The scope** (refused by name outside it): one closed membrane bounding the member, not touching the box, no
     nucleus or other membrane; a material membrane with a tension, an applied traction, or neither; a growing
     front with zero traction or `dirichlet v = 0`; no species, no region variables. Also refused, with no use
     case yet: a growing front with an applied traction (ADR 019 decision 4 allows it) and a material membrane
     with a given `dirichlet g`.
   - **MPI:** refused by name for more than one rank, since the slide, the contact check and the remesh are serial
     (ADR 014 decision 4, ADR 016 decision 8) and the partition skips the first two silently today.
   - **The outputs.** The member's domain is registered even with no species, holding `v` by component (ADR 010
     §3, "Vector fields"); a remesh's new segment includes those components. The moving domains' coordinates and
     centroids follow (#268). The summary gives the member's `region_size` at each output, the strides, the solves
     and the remeshes.
   - *Recommendation: yes.*

8. **The tests and the measurements.**
   - **E3, from vcell-core's sibling checkout,** `zs` and `zv` as written (`fenics.math_description`, the oracle's
     geometry):
     - **the transient, for the time rate:** `a(0.05)` at a fixed fine `h`, with `Δt` halving three times below the
       stability limit; the rate from Richardson extrapolation in `Δt`, asserted ≥ 1.8 under decision 3 (c).
     - **the equilibrium, for the space rate:** `a(1)` against `exact` at three `h` halving, at a `Δt` small enough
       that its error is negligible (the equilibrium carries almost none, Context); the rate asserted against the
       measurement, expected near 2 since #257, and the area at the finest `h` within a tolerance set from it.
     - `h` and `Δt` are never refined together (`motility-end-to-end.md`'s warning).
   - **A material disk with a tension** (no vcell-core vector): step 3a's exact disk under a pressure, a tension `γ`
     and a drag has `u_r(R)` in closed form (`tests/test_backend_cell_force_balance.py` 92–120). With a pressure that
     reads the area, `P = k a₀ / region_size(cell)` (decision 5), the disk has a stable radius `R* = k a₀/(π γ)`, and
     its area follows `da/dt = 2√(πa) u_r(R(a))` to it (*inferred*; the review checked the closed form's
     derivation for `Δt = 0`). Checked from a disk larger and one smaller than `R*`, to `t` past the transient:
     the area against the ODE, at first order in `Δt` (decision 4), and the departure from a circle (the largest
     against the smallest centroid-to-edge distance), measured, since a P1 mesh is not symmetric.
   - **Structural:** a solve after moves equals a fresh assembly on the moved mesh (decision 6); a T9 slot reads
     the re-measured `region_size` (decision 5, the disk above); a remesh mid-run continues with the area on its
     ODE; the refusals of decision 7, MPI included.
   - **Measured, not pass/fail** (ADR 019 decision 4: whether zero-traction runs "stay resolved is measured in stage
     4a, not assumed"): ZS and ZV from an elliptical cell (semi-axes 1.2 and 0.8), to `t = 2`: the shape over
     time (the radius ratio, any corner or concavity), the remeshes, the worst cell quality and the contact gap.
     And the cost per stride of decision 3's two solves.
   - *Recommendation: yes.*

## Not decided here

- **Species on the moving cell** (5b), and how their transport's mesh velocity fits decision 3 (c)'s second
  half-move.
- **A time-step rule** tied to the front's speed and stability (decision 3), and a fixed-point iteration of the
  stagger (decision 3 (d)).
- **3D, several loops, junctions** (stages 4b and 4c), and MPI for the moving partition.
- **A tension on a growing front** (ADR 019 decision 5 keeps it out), and the refused combinations of decision 7.

## Review

An independent reviewer read the first draft and its sources, re-solved E3's ODEs, and read DOLFINx's
`LinearProblem.solve`. It found 16 issues: 1 blocking, 3 major, 8 minor, 4 nits. Each is answered in the note.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | blocking | E3's area at `t = 1` is at equilibrium (rates −31, −27; settled by `t ≈ 0.14–0.19`), so it cannot show a rate in `Δt`. | Decision 8: the transient `a(0.05)` with `Δt` refined at fixed `h` for the time rate; `t = 1` for the space rate. Re-run here. |
| 2 | major | "Capped at first order" misreads vcell-core: the cap was spatial, from #249, now closed; refining `h` and `Δt` together is warned against. | Context and decision 8 corrected; `h` and `Δt` refined separately. |
| 3 | major | Decision 3 omitted the split's order (ADR 021's third question) and a second-order option with the same two solves; (b) was compared at unequal accuracy. | Decision 3: the explicit midpoint rule, now recommended; (a) restated as (b) at half the stride. |
| 4 | major | The routing and outputs were not designed: material membranes are not seen as moving, the moving branch records no `v`, no domain without species, segments sized by species, `member_species` refuses. | Context and decision 7. |
| 5 | minor | The output solve is reused by the next stride, and stride lengths vary, so a fixed `Δt/2` is wrong. | Decisions 3 and 4: the output solve is the next stride's first; `Δt` set per half-move. |
| 6 | minor | No step size or stability limit. | Decision 3: `Δt < 0.065` for E3 under (c); the rule left open. |
| 7 | minor | The disk test: the semi-implicit term changes the condition by `O(Δt)`; a constant pressure never settles; circularity is not guaranteed. | Decision 4 states the `O(Δt)`; decision 8 uses an area-dependent pressure with a stable radius and measures the shape. |
| 8 | minor | The measured ZS run had no stop and stays circular. | Decision 8: from an ellipse, to `t = 2`. |
| 9 | minor | `_CellMinimum`'s docstring contradicts decision 6. | Context and decision 6: the docstring is corrected. |
| 10 | minor | MPI is not refused; the partition skips the slide and contact silently. | Decision 7: refused by name. |
| 11 | minor | The Nickaeen citation (p. 6, "35 iterations"). | "up to 35 iterations", p. 8, through ADR 020 decision 3. |
| 12 | minor | E3's source key and geometry are not stated. | Context: `fenics.math_description`, the oracle's box. |
| 13 | nit | "The run's fixed `Δt`" was not verbatim. | Quoted verbatim. |
| 14 | nit | The force balance must be assembled before the partition; the trace copy needs block-size indexing. | Decision 2. |
| 15 | nit | A front with a traction and a material membrane with `dirichlet g` are not mentioned. | Decision 7: refused, no use case. |
| 16 | nit | Which time `sim.t` and `v` belong to in a half-move. | Decision 3. |

## References

- **This repository** at `c5de8ff`: `backend/moving_partition.py`, `backend/multi_compartment.py`,
  `backend/cell_force_balance.py`, `backend/force_balance.py`, `runner.py`, `tests/test_backend_cell_force_balance.py`;
  ADRs 010, 013–016, 018–021; issues #240, #249 (PR #257).
- **DOLFINx 0.10** `python/dolfinx/fem/petsc.py` 905–916 (`LinearProblem.solve`), read by the review.
- **vcell-core** at `4e1d672`: `conformance/math/moving-resolved-cell.jsonl`; `spec/semantics/moving-resolved-cell.md`;
  `spec/semantics/motility-end-to-end.md`; `tools/oracle/motility_solve.py`.
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A
  free-boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13:e1005862 (2017).
  doi:10.1371/journal.pcbi.1005862.
  - *Cited second hand* here, through ADRs 019 and 020 (which read the Model section, Numerical methods and Fig 4).
  - *Use:* the ZS and ZV front laws E3 lowers, and the fixed-point iteration of decision 3 (d).
- **Hairer E, Nørsett SP, Wanner G.** *Solving Ordinary Differential Equations I*, 2nd ed. Springer (1993).
  - *Not read here.*
  - *Use:* the explicit midpoint rule's order and stability region (decision 3).

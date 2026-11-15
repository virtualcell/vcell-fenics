# ADR 020 — Species carried by the velocity of a cell force balance

**Date:** 2026-11-15
**Status:** Proposed. Answers issue #251, asked from vcell-core's E2. Jim asked for this note (2026-11-15).
Revised after an independent review: 20 findings, 1 blocking, 6 major. Each is answered below and listed in
"Review" at the end.
**Builds on:**
- ADR 018 decision 8: a species on a member rides the solved `v`, and the first implementation refuses one.
  The decision's assembly is *inferred* there.
- ADR 019 decision 5 (member species at a growing front, *inferred*) and decision 6 (step 3b, the
  fixed-circle instability, brings decision 8 forward on a fixed mesh).
- Formalism §1.10.1, motion. It says: "When `kind: prescribed` or `kind: unknown`, the substrate velocity at
  every point on the subdomain is the value of the relevant motion field at that point. All variables on the
  subdomain experience the same substrate velocity" (line 1224, *checked*).
- Formalism §1.4.2, T1's transport slots. `relative_advection` is **w**, "the species' drift relative to the
  substrate". `advection` is **c**, "the species' lab-frame (Eulerian) velocity" (lines 315–316, *checked*).
- ADR 013 decision 3 (transport relative to a moving boundary) and ADR 015 (conservative mesh moves on the
  moving partition).
- vcell-core's E2 document, `conformance/math/resolved-cell.jsonl` at vcell-core `8ca02eb` (*checked*):
  - myosin `m` on the member `cell`, with `bulk_radv_diff` terms `diffusion` and `source` only;
  - the force balance with `active_stress: (beta * m)`, `viscous_operator: laplacian` and a drag;
  - `dirichlet v_cyt = 0` on `cell_ec_membrane`;
  - no BC for `m`;
  - `β = 5`, `α = 0.5`, `D = 1`, `m₀ = 1.5 + 0.01x` on the unit disk.

  It validates with no diagnostics at `b02bf19` (*checked*, run here). The runner refuses it at
  `_refuse_cell_force_balance` (`runner.py`). `solve_cell_force_balance` refuses it at `FieldsNotYetSupported`.

**Would amend:**
- ADR 018: decision 8, made concrete; "Not decided here", the line "Exposing T9 through the CLI: the runner
  routes it only once stage 4 works", for decision 5's scope.
- ADR 010 (the results bundle): a vector field per member (decision 5).
- Formalism:
  - §1.4.2: T9, member species; and T1, the meaning of `relative_advection` on a member;
  - §1.6.4: the internal-boundary rule, for a member species;
  - §1.10.1;
  - the T9 status lines that say "the runner refuses it" (§1.4.2 and §3.6.1).
- The validator: `advection` refused on a member species (decision 1).
- `docs/modeling/cell-mechanics/use-cases.md` UC-A: the row that maps myosin to the `advection` slot.
- `backend/cell_force_balance.py` and `runner.py`, for decision 4's scope.

*Checked* marks what was read in a source (file and line, or section, or page) or run here. *Inferred* marks
reasoning. This repository was read at `b02bf19`.

## Context

The three questions of #251:
1. On step 3b's fixed mesh, is `m` on a member carried by `v` in the conservative form, with the
   compressible dilution `m∇·v`, and with nothing written by vcell-core? Or should vcell-core write the
   carrier, e.g. `advection: v_cyt`?
2. On the moving partition (step 5), is it still no slot, with the carrier supplied by the assembly?
3. What is the condition on the held edge? Is it `n·∇m = 0` where `v = 0`?

**The transport forms in the code today** (*checked* at `b02bf19`). Here `q` is the test function, `w` the
drift, `c` a lab-frame carrier and `v_mesh` the mesh velocity:

| Path | Where | Form added to the residual | Natural boundary condition |
|---|---|---|---|
| single mesh, surface-driven (ADR 013) | `assemble.py` 233–242 | `−∫ u (w − v_mesh)·∇q` | zero total flux relative to the boundary |
| single mesh, `relative_advection` | `assemble.py` 243–257 | `∫ (w·∇u + (∇·w) u) q` | zero diffusive flux; mass changes at rate `−∮ u w·n` |
| single mesh, `advection` | `assemble.py` 258–267 | `−∫ u (c − v_mesh)·∇q` | zero total flux relative to the boundary |
| multi-compartment, fixed mesh, a drift | `multi_compartment.py` ~536 | `∫ (w·∇u) q`, not in conservation form | zero diffusive flux |

None of these paths reads a T9 velocity. The single-mesh assembler refuses unknown motion (`assemble.py`
615–618, *checked*).

**What the physics asks for** ([Nickaeen 2017]; Eq 1 p. 3, the boundary conditions pp. 5 and 7, Eq 2 p. 7,
*checked*):
- Myosin obeys `∂_t m = ∇·(D_eff∇m − U_eff m)` in the lab frame. At the edge,
  `n·(−D_eff∇m + (U_eff − V_f)m) = 0`: zero flux relative to the moving edge.
- `U_eff = U(1 − M/M_max,u)` where `M < M_max,u`, and 0 above it. `D_eff = D(1 − M/M_max,d)`.
- **The limit without the cut-offs is singular:** "using U as an advection velocity and constant diffusion
  coefficients D results in singular solutions, in which M and U develop Dirac-delta singularities"
  (p. 4, *checked*).
- The paper's runs fix `m_max,u = 15` and `m_max,d = 125` (p. 8, *checked*).
- **E2 is past the singular threshold, unless the cut-offs are added** (*inferred*, the review's
  derivation). Eliminating `u` from `αΔu − u = −β∇m` makes the system parabolic–elliptic Keller–Segel, with
  sensitivity `β/α = 10`. Its critical mass is then at most `8πα/β = 0.8π`, while E2's mass is about `1.5π`.

## Decisions

1. **The assembly supplies the carrier. vcell-core writes no slot for it.**
   - **Why.** Formalism §1.10.1 already says that every variable on a subdomain whose motion is `unknown`
     takes that motion as its substrate velocity, "without per-equation user input" (*checked*). E2's
     document follows it. The answer to question 1: nothing is written.
   - **`relative_advection`** (the drift `w`) is the only transport slot a member species takes. It holds a
     drift relative to the material.
   - **`advection` is refused on a member species,** by a new validator rule. A lab-frame carrier on top of
     the material's would be a second carrier. The review checked that no such rule exists today.
     `use-cases.md` UC-A's row that maps myosin to `advection` is amended: on a member, myosin rides `v` and a
     crowding correction is a drift.
   - **The options:**
     - **(a) The assembly supplies `v`** (as above).
     - **(b) vcell-core writes `advection: v_cyt`.** It contradicts §1.10.1, and it lets a document carry a
       member species by something other than its material's velocity.

   *Recommendation: (a).*

2. **The form is the conservative lab-frame one, `−∫ m (v + w − v_mesh)·∇q`. Its natural condition is zero
   total flux relative to the boundary.**
   - **The form.** It is the single-mesh `advection` path's form, with the carrier `v + w`: the material's
     velocity plus the drift.
     - Integrated by parts, it is `∫ ∇·(m(v + w − v_mesh)) q` minus a boundary term, so it contains the
       compressible dilution `m∇·v` (*inferred*, the divergence theorem). The answer to question 1 is yes.
     - With `q = 1` the term vanishes. So `∫m` is conserved exactly by the transport, when `m` has no
       Dirichlet condition and no flux BC (*inferred*; the review confirmed it for backward Euler with `v`
       lagged).
   - **On a member, `relative_advection` changes form.** Elsewhere its form is not integrated by parts, and its
     natural condition is zero *diffusive* flux. On a member the drift joins the carrier in the integrated
     form, with zero *total* flux. Formalism §1.4.2's T1 slot text says so. Without this, myosin would leak
     out of the cell at rate `∮ m w·n` wherever its drift points out.
   - **The boundary condition.** A member species takes the natural condition
     `n·(−D∇m + (v + w − v_mesh)m) = 0` on each surface bounding its member, unless a flux BC is given there
     (*inferred*, from the integration by parts):
     - **on a material membrane,** which moves with `v` (`v_mesh·n = v·n`): `n·(−D∇m + w m) = 0`. No species
       crosses it relative to the material;
     - **on a held material edge** (`v = 0` and no growth, so `v_mesh = 0`): `n·(−D∇m + w m) = 0`. With no
       drift that is `n·∇m = 0`, which answers question 3;
     - **at a growing front** (`v_mesh·n = v_f·n`), including a *held* front (`v = 0` with `v_f = v_p n`,
       Nickaeen's ZV edge): `n·(−D∇m + (v + w − v_f)m) = 0`, the Rankine–Hugoniot condition (ADR 019
       decision 5). On a held front that is `n·(−D∇m + (w − v_p n)m) = 0`, not `n·∇m = 0`;
     - **at a stationary front** (a prescribed velocity of zero, on a fixed mesh): `n·(−D∇m + (v + w)m) = 0`.
       The material flows through the edge, as in Nickaeen's ZS model in a fixed circle (p. 12);
     - **on a box wall** (`v = 0`, `v_mesh = 0`): as on a held material edge.
   - **This is an exception to §1.6.4,** which requires an explicit BC on every internal boundary a variable
     touches. A member species' surfaces need none, as the T9 velocity's do (ADR 018 decision 2). The
     validator does not check §1.6.4 today: E2 has no BC for `m` and validates (*checked*). So the exception
     changes the document's rule, not a check.
   - **On the moving partition (question 2): yes, the same form, with `v_mesh` the mesh velocity.** ADR 015
     carries each species' nodal moments exactly across each mesh move (its decision 1) and integrates the
     stride on the midpoint mesh (decision 2). A member's species then take `v + w − v_mesh` in that
     integration, in place of `w − v_mesh` (*inferred*; built in stage 5). How a staggered force-balance step
     fits ADR 015's half-move stride stays open (ADR 018, "Not decided here").
   - **Membrane species** on a member's membrane keep ADR 018 decision 8's rule (ride `trace(v)`, with
     `ρ∇_Γ·v_Γ`) and ADR 019 decision 5's at a growing front. They come with stage 5.

   *Recommendation: yes.*

3. **The coupling in time: staggered, one pass per step.**
   - **The step.** Solve `v` on the current configuration from `mⁿ`. Then advance `m` to `tⁿ⁺¹` by backward
     Euler, with `v` held at that solution. This is the staggering of `unknown_motion.py` (formalism §3.6.1,
     *checked*).
   - **Slots of `m`'s equation that read `v`** (a drift such as `k * v_cyt`, a source reading `div(v_cyt)`) read
     the same lagged `v`, so they stay linear in `mⁿ⁺¹`. They are admitted.
   - **A source that reads `div(v)` gets a warning,** naming the slot: the dilution `m∇·v` is already in the
     transport, so a hand-written `−m * div(v)` would count it twice (*inferred*).
   - **The options:**
     - **(a) One pass.**
     - **(b) Fixed-point iteration to a tolerance in each step,** as Nickaeen et al. did ("up to 35
       iterations, tolerance 1e-10", p. 8, *checked*).
     - **(c) Monolithic:** one nonlinear solve in `(v, m)`.
   - **What (a) costs** (*inferred*). Backward Euler is first order in `Δt` either way. The splitting adds an
     `O(Δt)` error constant, which (b) removes; it does not change the order.
     - Below the instability threshold a perturbation decays under (a) for any `Δt`. The review derived this
       from the linear amplification factor `(1 + ΔtA)/(1 + Δtk²)`, `A = βm₀k²/(1 + αk²)`.
     - Above the threshold, the growth rate carries the splitting error.

   *Recommendation: (a) now; (b) as an option once a measurement shows the splitting error matters.*

4. **Step 3b's scope: a fixed skeleton, one kind of species, linear transport.**
   - **A fixed skeleton.** Each membrane bounding a member is either held (`dirichlet v = 0`, with no growth)
     or a stationary front (a prescribed velocity of zero, with no tension). Box faces a member touches are
     walls. Then no boundary moves, though the material inside does, so a fixed mesh is consistent
     (*inferred*).
     - This excludes every model with more than one member, such as a nucleus or touching cells, because
       a membrane between two members cannot be held (`_check_geometry`, *checked*). Those come with stage 4.
   - **Species on the members only.** They are assembled on the members' mesh (step 3a's), with `v` from the
     same solve: the same mesh, and for `P1` the same space. Their sources may couple them to one another.
     - **Refused by name:**
       - a species on a non-member volume, a membrane or a region (non-member species would need the
         multi-compartment solver's coupling across the members' membranes);
       - a flux BC that couples a member species across its membrane.
   - **Force-balance slots that may read the species:** `viscosity`, `bulk_viscosity`, `drag` and
     `active_stress`. Not `pressure`, which stays uniform (`NonUniformPressure`). Not `tension` or
     `traction`: a held membrane carries neither, and a front carries no tension.
   - **Linear transport in `mⁿ⁺¹` only:** the crowding cut-offs are nonlinear in `m`. Backward Euler refuses
     them (`NonlinearTermError`), and the method of lines is stage 5's.
   - **Consequence for E2:** without the cut-offs, E2 is past the singular threshold (Context). So 3b can
     test the onset of the instability, but not the relocation of myosin in the paper's Fig 4A. That needs
     the cut-offs, so stage 5, with vcell-core adding them to E2's document as a drift
     `w = −v·min(m/m_max,u, 1)` and a diffusion `D(1 − m/m_max,d)`. This agrees with ADR 019 decision 6, which
     has 3b measure the onset and "not the saturated state".

   *Recommendation: yes.*

5. **Step 3b runs through the runner, writing `m` and `v` to the results bundle.**
   - **The options:**
     - **(a) A Python driver only,** as step 3a.
     - **(b) The runner routes a T9 model in decision 4's scope to a fixed-configuration driver.**
   - **Why (b).** vcell-core ran E1 through the CLI (#249, its comment of 2026-11-15, *checked*), and E2 is
     posed for the same route. ADR 018's "Not decided here" routes T9 "only once stage 4 works". This scope
     cannot move, so routing it now does not anticipate stage 4.
   - **The runner still refuses a T9 model outside the scope,** by name, pointing to stage 4.
   - **The bundle needs a vector field,** which it has never had (ADR 010; `src/vcell_fenics/results`,
     *checked* by the review). `v` is written per member, by component, beside the species. At each output
     time `tⁿ` it writes the `v` solved from `mⁿ`, the velocity that carries the next step. This brings
     forward the held step 3c, for this scope only. ADR 010 and the VCell-side reader gain the vector field.

   *Recommendation: (b).*

6. **Step 3b's tests.**
   - **Known answers for the transport,** with a prescribed velocity in place of the solved one:
     - with `v = ∇Φ` and `v·n = 0` on the edge, the steady state is `m ∝ exp(Φ/D)` (zero total flux
       everywhere). This catches a wrong sign or direction of the carrier, which conservation cannot;
     - a manufactured solution in `mms/` with its order (ADR 018 step 3b's original test, held with 3c until
       now).
   - **Structural and exact:**
     - `∫m` is conserved to round-off over a run with the edge held (the check #251 offers);
     - a uniform `m` gives `v = 0` and stays uniform. A uniform active stress exerts no force on a velocity
       held at the edge, since `∫σ_a∇·φ = σ_a∮φ·n = 0`; the review confirmed it;
     - the transport term's form;
     - the warning on a source that reads `div(v)`.
   - **The instability's onset on E2's model,** before any collapse:
     - below the threshold (a small `βm₀`), a perturbation decays;
     - above it (E2's own `βm₀`), it grows.

     A wrong-signed carrier would suppress the growth. The linear growth rate against a mode analysis is a
     further step, since a disk with mixed conditions has no closed-form modes.
   - **Positivity.** Galerkin `P1` transport can undershoot under aggregation. The run reports `min m` at
     each output, and the tests assert `m ≥ 0` in the onset runs. Stabilisation (SUPG, or a limiter) is
     decided if this fails.
   - **The splitting error** of decision 3, measured at two `Δt`: a measurement, not a pass/fail test.
   - **Fig 4A's relocation** moves to stage 5, with the cut-offs (decision 4).

   *Recommendation: yes.*

## For vcell-core (to send to the vcell-core session, not changed from here)

- E2's document is right as written. It needs no carrier slot and no BC for `m` (decisions 1–2).
- A drift goes in `relative_advection`. `advection` on a member species will be refused (decision 1).
- **E2 without the cut-offs is past the singular threshold** (*inferred*, Context). It can test the
  instability's onset, but not Fig 4A's relocation. For that, E2 needs the cut-offs:
  `relative_advection: -v_cyt * min(m / m_max_u, 1)` and `diffusion: D * (1 - m / m_max_d)`, with the paper's
  `m_max,u = 15` and `m_max,d = 125`. They run in stage 5, when nonlinear transport lands.
- E2 will run through the CLI once step 3b lands (decision 5).

## Not decided here

- Fixed-point iteration (decision 3 (b)), nonlinear transport, and the cut-offs (stage 5).
- Stabilisation of the transport, if positivity fails (decision 6).
- Member species on several members, and species across a member's membrane (stage 4 and the
  multi-compartment coupling).
- Membrane species on a member's membrane; region variables on the cells.
- How the staggered step fits ADR 015's half-move stride (ADR 018, open).
- Which moving path step 4a uses is not decided here. Jim decided it on 2026-11-15 (#240), and it is
  recorded in ADR 019 decision 6: the moving partition, with `region_size` re-measured after every move.

## Review

An independent reviewer read the draft and its sources. It ran the validator on E2 and on three variants
(`advection: v_cyt`, a drift `0.1*v_cyt`, a source `0.1*div(v_cyt)*m`), all of which validate today. It derived
the linear amplification factor, and checked the paper's text and the line numbers. It found 20 issues: 1
blocking, 6 major, 7 minor, 6 nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | blocking | The uncut limit is singular (paper p. 4), Fig 4A used the cut-offs (p. 8), and E2 is supercritical by a Keller–Segel argument. So "relocation by t ≈ 10" is not a valid 3b test, and it conflicts with ADR 019 decision 6. | Context states it. Decision 4: 3b tests the onset; Fig 4A moves to stage 5 with the cut-offs. "For vcell-core" asks for them in E2. |
| 2 | major | ZV's held edge is a growing front, so its condition is Rankine–Hugoniot, not `n·∇m = 0`. | Decision 2 separates a held material edge from a held front. |
| 3 | major | A stationary front (prescribed velocity 0) has `v·n ≠ 0` on a fixed mesh (ZS in a fixed circle). | Decision 2 gives its condition; decision 4 admits it. |
| 4 | major | It does not say which assembler and mesh carry `m`; the multi-compartment drift is a fourth form. | Context's table gains the row. Decision 4: member species on the members' mesh, `v` in the same space; non-member species refused in 3b. |
| 5 | major | No validator rule refuses `advection` today; UC-A maps myosin to `advection`. | Decision 1 adds the rule; both are in "Would amend". |
| 6 | major | T1 slots that read `v` are undecided; a hand-written `−m div(v)` double-counts. | Decision 3: admitted with `v` lagged; a warning on a source reading `div(v)`. |
| 7 | major | The bundle has no vector fields; amendments to ADR 018, ADR 010 and the formalism are unlisted; which `v` pairs with which `m` is unstated. | Decision 5; "Would amend". `v(mⁿ)` is written at `tⁿ`. |
| 8 | minor | The scope silently excludes multi-member models. | Decision 4 says so. |
| 9 | minor | "The other volume slots" is loose: `pressure` cannot read a field. | Decision 4 lists the slots. |
| 10 | minor | `relative_advection` changes meaning on a member; the symbols clash with the formalism's. | Decision 2 says so, and the formalism's symbols (`w` drift, `c` carrier) are used. |
| 11 | minor | ADR 015 carries moments; it does not use the single-mesh conservative time term. The stride question was dropped. | Decision 2 corrected; the stride is in "Not decided here". |
| 12 | minor | Conservation and a uniform `m` cannot catch a wrong carrier sign; there is no manufactured solution. | Decision 6: `m ∝ exp(Φ/D)` with a prescribed `v`, an MMS order test, and the onset bracket. |
| 13 | minor | Option (b) removes a splitting constant, not an order. | Decision 3 corrected. |
| 14 | minor | Positivity is not discussed. | Decision 6: `min m` reported and asserted; stabilisation if it fails. |
| 15 | nit | Page numbers. | Corrected (Eq 1 p. 3, BCs pp. 5 and 7, Eq 2 p. 7). |
| 16 | nit | "t ≈ 10" is not in the paper's text; E2's initial maximum is at the edge. | The relocation test moved to stage 5 (finding 1). |
| 17 | nit | The cut-off is defined piecewise, and `d_eff` is missing. | Context states both. |
| 18 | nit | The sign of the `relative_advection` mass change. | Corrected to `−∮ u w·n`. |
| 19 | nit | The material moves; only the skeleton is still. | Decision 4 says "skeleton". |
| 20 | nit | Through the runner, E2 stops earlier, at `_refuse_cell_force_balance`. | "Builds on" names both routes. |

## References

- **This repository** at `b02bf19`:
  - ADRs 010, 013, 015, 018, 019;
  - `docs/modeling/declarative-formalism.md` §1.4.2 (lines 315–316), §1.6.4, §1.10.1 (line 1224), §3.6.1;
  - `docs/modeling/cell-mechanics/use-cases.md` UC-A;
  - `src/vcell_fenics/backend/assemble.py` lines 233–267 and 615–618, `multi_compartment.py`,
    `cell_force_balance.py`, `runner.py`;
  - issues #240, #249, #251.
- **vcell-core** at `8ca02eb`: `conformance/math/resolved-cell.jsonl` (E2).
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A
  free-boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13:e1005862 (2017).
  doi:10.1371/journal.pcbi.1005862.
  - *Read* here (pp. 3–8, 12), from the copy in `docs/papers/`.
  - *Use:* myosin's transport, its cut-offs and boundary condition, the singular limit, the fixed parameters,
    the iteration count (Context, decisions 2–4).
- **Keller–Segel's critical mass** (decision 4's threshold) is the review's argument, *inferred*, not checked
  against a source here. The classical result is for the parabolic–elliptic system on the plane; a bounded
  domain with no-flux conditions changes the constant.

# ADR 019 — The cell force balance for a single cell

**Date:** 2026-11-14
**Status:** Proposed. Asked for by Jim (2026-11-14): "I don't want to move so strongly into multicellular
modeling that would make modeling a single cell difficult … I don't want to lose what we had before." PR #246
(ADR 018 staging step 1) is held until this note is decided. Revised after an independent review (23
findings: none blocking, 7 major), each answered below and listed in "Review" at the end.
**Builds on:**
- ADR 018 (accepted 2026-11-14): the template T9 `cell_force_balance`, its slots (decision 3), its
  well-posedness rules (decision 4), its time stepping (decision 5), its term kinds (decision 6), its motion
  (decision 7), chemistry on the cells (decision 8) and its staging.
- ADR 017 (accepted 2026-11-13): the FEM forms of vcell-core's shared-subset laws, which T9 was written to
  carry.
- `docs/modeling/cell-mechanics/use-cases.md` (2026-10-08): three published single-cell models mapped onto the
  formalism. Its §3 asks for a *compressible single-phase active gel* family and its §4 for a boundary-law
  motion kind.
- ADR 013 (a surface's prescribed motion drives the domains it bounds; transport relative to a moving
  boundary), ADRs 014–015 (the moving partition; conservative mesh moves), formalism §1.10 (motion) and §2.2.1
  (the motion schema).
- vcell-core's `spec/semantics/motion-and-mechanics.md`: §2 (three velocities kept apart), §3 (law kinds,
  among them `active`; the mechanics template slots `viscosity`, `active_stress`, `body_force`, `drag`; the
  mechanical conditions `traction`, `slip`, `no_slip`), §4, and Addendum C (accepted 2026-11-08: material
  velocities; `FrontLaw`, reserved, for a non-material front).

**Would amend:**
- ADR 018:
  - decision 2 ("The only BC `v` accepts is … the zero-velocity wall"; "Motion, both ways");
  - decision 3 (a new `active_stress` slot; a tension optional on a free membrane; a `traction` slot);
  - decision 4 (a Dirichlet condition on a free membrane holds the cell);
  - decision 5 (no semi-implicit tension on a growing front, since a growing front carries no tension);
  - decision 6 (new term kinds; a viscous-operator choice);
  - decision 7 (a member's membrane may be a growing front);
  - decision 8 (the membrane carrier and the member assembly at a growing front);
  - its staging (one cell first; a fixed-mesh mechano-chemical test brought forward).
- ADR 017 decision 1's positivity rule, read as ADR 018 already reads it, plus: a free membrane may omit its
  tension.
- ADR 013 decision 3 (transport relative to a moving boundary, extended to a member at a growing front) and
  decision 6 (a volume with its own motion bordered by a prescribed membrane, allowed for a T9 member at a
  growing front).
- The formalism: the T9 section of §1.4.2; §1.6.4; §1.8.8 (a vector Dirichlet value); §1.10.1; §1.11 (the
  T9 BC rule and the coercivity line); §2.2.1; §2.2.4 (`surfaces:` entries; the `viscous_operator` key).
- The validator of PR #246: the wall rule, the rule that every `surfaces:` entry has `motion: unknown`, the
  tension's required-ness, and `NonUniformPressure`'s message.

*Checked* marks what was read in a source (file and line, or section, or page) or run here. *Inferred* marks
reasoning. *Background* marks a published result used without reading it here (cited in References). This
repository was read at `dd5137e` (main) and `4cce16f` (PR #246); vcell-core at `f6f45c7`.

## Context

### Nothing that ran before has been lost

*Checked:*

- ADRs 017 and 018 changed no code (`git diff --stat 1bfb91c dd5137e`: two ADRs only).
- PR #246 (`git diff --stat dd5137e 4cce16f`) changes 8 files and edits no existing test. Its check run
  passed 978 tests, 46 of them new (the run's log, not the diff). Its 26 deleted lines are rewrites of the same
  code: the variable index, the coverage count, the dumper's variable record, the schema's equation union and
  `variables` field, imports, and formalism-doc sentences that now name T9.
- T1–T5, the prescribed moving front with species (ADRs 013–016), the membrane-only unknown-motion force
  balance (`backend/unknown_motion.py`, formalism §1.10.8), and the Stokes, slip, FSI and phase-field drivers
  are unchanged. `runner.py` and `multi_compartment.py` each gain one check before routing that refuses T9 by
  name; nothing else in them changes.
- The new formalism features are fenced to T9 by the validator: a list subdomain is accepted only for a vector
  that T9 governs; `volumes:`/`surfaces:` only on a T9 equation; the boundary-condition and initial-condition
  exceptions only for a T9 velocity.

### One cell is already a valid T9 model, but not yet a moving one

- A single cell is `subdomain: cyt` (or `[cyt, nuc]` with a nucleus); no list is required (*checked*, test
  `test_a_single_member_force_balance_needs_no_list` in PR #246). Junctions, the end-pinned slide and
  `Λ_ij + T_i + T_j` arise only when a membrane meets another member's membrane or a held box face (ADR 018
  decision 7).
- On a **fixed** mesh (ADR 018 stages 2–3) a single cell needs nothing more.
- On a **moving** mesh, today's partition moves **one** closed membrane by a **prescribed** velocity and
  refuses several moving membranes (formalism §3.6.1, *checked*). So even one cell needs new work: a solved
  rather than prescribed membrane. A cell with a nucleus, or several cells that do not touch, add several closed
  loops, but no junctions. A cell held at a box face adds junctions (*inferred*, from ADR 018 decision 7).

### What the shared subset narrowed

T9 was written to carry the laws a vertex model and an FEM model share (ADR 017). For one cell, four of its
rules come from that subset rather than from the mechanics, and each excludes a published single-cell model
(*checked* against `use-cases.md` §1–§2; UC-A also in [Nickaeen 2017] pp. 3–6):

| T9 rule (ADR 018) | Its reason | What it excludes |
|---|---|---|
| `pressure` uniform per member (`NonUniformPressure`, decision 3) | a vertex cell has one pressure `P_c = K(V0 − V)` | a stress driven by a species: UC-A's contractile myosin stress `β m I`; UC-C's expansive `σ_a(ρ)` |
| a free membrane must carry a tension (decision 3) | a vertex edge's free boundary is capillary | UC-A's zero-stress edge (ZS) |
| no condition on a member's membrane (decisions 2 and 4) | the shared subset has no adhesion band or applied load | UC-A's zero-velocity edge (ZV, an adhesion band); an applied load |
| a member's membrane moves with `v` (decisions 2 and 7) | a vertex is material | UC-A's front `v_f = v_p n + u` (polymerization adds material at the edge) |

A fifth rule is in the mechanics, not the subset, but excludes UC-A as published: the viscous operator is
`2η dev ε + η_b (∇·v) I`, and UC-A's is the vector Laplacian `η Δu` ([Nickaeen 2017] p. 4, *checked*: "we
ignore … interplay between bulk and shear viscous stresses").

UC-C's operator, the full symmetric gradient `2η ε(v)`, is already expressible:
`2η ε : ε(φ) = 2η dev ε : ε(φ) + (2η/3)(∇·v)(∇·φ)`, so it is T9 with `bulk_viscosity = 2η/3`, in 3D and in 2D
with the 3D deviator (*inferred* from `ε = dev ε + (tr ε/3) I`; the review checked it numerically). The rest of
UC-C is not (see "What T9 still does not do").

`use-cases.md` §3 already asks for this family: a compressible, pressure-free active gel,
`∇·(η(∇v [+∇vᵀ])) + ∇σ_a(c) − ξ v = 0`, with a species-dependent `σ_a` and `η`, drag, and ZV/ZS boundaries,
"developed as a weak form first, then promoted to a template" (§5 step 3). T9 is that template's operator,
narrowed. This note widens it instead of adding a second template.

## Decisions

1. **T9 is the FEM compressible viscous force balance; the shared subset is a property of some T9 models,
   which vcell-core's lowering owns.**
   - **The options:**
     - **(a) One template, widened.** T9 accepts the single-cell terms below. A model that uses them is
       outside the shared subset; vcell-core decides which of its models it lowers to the subset (motion §4,
       *checked*). This repository's validator checks what makes a model well formed and solvable, and does not
       check membership of the subset.
     - **(b) Two templates.** T9 stays the shared subset, and a new template (the use cases' "active gel")
       takes the rest. Both would assemble the same operator from the same term kinds, and a model would
       change template on gaining an active stress.
     - **(c) A mode flag** on T9 (`subset: shared | full`) that the validator enforces.
   - **Why (a)** (*inferred*). The operator, the variable and the solve are the same in every case; only the
     set of terms differs. Two templates would duplicate the validator rules and split verification. A flag in
     the math description would make this repository police vcell-core's realization choice, which ADR 017
     decision 2 put in vcell-core.
   - **`NonUniformPressure` stays,** as a naming rule, not a solvability rule: `pressure` names the uniform
     per-cell law, and a field goes in `active_stress` (decision 2), so the rule refuses nothing that cannot
     be written.
   - **What the subset still governs here:** ADR 017 decision 6's contract C tests use subset models only (no
     active stress, no traction, no Dirichlet membrane, the symmetric operator, material membranes).
   - **The name** `cell_force_balance` fits a single cell (ADR 018 decision 1 chose it for that reason).

   *Recommendation: (a).*

2. **A member gains an `active_stress` slot: an isotropic stress `σ_a` that may be a field.**
   - **The term:** the stress becomes `σ = σ_visc − P I + σ_a I`, so the residual gains `+∫ σ_a ∇·φ dV` on the
     member. `σ_a > 0` is contractile: a uniform `σ_a` acts on the boundary as the traction `−σ_a n`, pulling
     inward, and the force density `∇σ_a` points up its gradient (*inferred*; confirmed by the review).
   - **The two published models in this convention:**
     - UC-A: `active_stress: "beta * m"`. Its force balance is `α Δu + β ∇m − u = 0` ([Nickaeen 2017] Eq 2,
       p. 6, *checked*) with the zero-stress condition `n·(α∇u + β m I) = 0` (p. 6, the text after Eq 2,
       *checked*), the natural condition of `∫ α∇u:∇φ + β m ∇·φ + u·φ = 0` (*inferred*).
     - UC-C: its active stress "plays the role of pressure" (`use-cases.md` UC-C, *checked*), so it is
       expansive: `active_stress: "-kappa_a * rho^2"`, in the use case's notation.
   - **What it may read:** what a T1 slot may (§1.4.4): fields, species, `geom.*`, region variables,
     `region_size`, parameters. Not `v` (ADR 018 decision 3: the solve stays linear). On a moving run,
     species and region variables wait for stage 5, as for every T9 slot (ADR 018 decision 3).
   - **Why a new slot and not a field `pressure`:**
     - **(a) A new slot; `pressure` stays uniform** (decision 1). Any field pressure `P` is written
       `active_stress: -P`, so nothing is lost, and a reader sees which term is the passive volume law and
       which is the motor.
     - **(b) Relax `pressure` to a field.** One slot, but the passive law and the motor share it.
     - **(c) Both.**

     vcell-core's mechanics templates already name the slot `active_stress` (motion §3, *checked*).
   - **Its invariant** (*inferred*, as ADR 018 decision 6's for the pressure): for a rigid motion `φ`,
     `∇·φ = 0`, so the load vanishes pointwise, and an active stress exerts no net force or torque on a cell.
     With no traction and no Dirichlet condition, testing with a constant `φ` gives `∫ ζ v dV = 0`: the drag
     transmits no net force either. A cell's centroid moves by its front's kinematics and by the interior's
     compressibility, not by a net force (*inferred*; the review's derivation). `∫ζv = 0` joins the invariant
     tests.
   - **The energy identity** (ADR 018 decision 5), case by case (*inferred*; the review confirmed the first):
     - **With the active stress,** on material membranes, no traction and no Dirichlet membrane:
       `δE[v] = −2R(v) − Δt∫γ|∇_Γ v|² − ∫σ_a ∇·v`, with `R` the Rayleigh function of the chosen operator
       (decision 3). It stays exact for the discrete forms, and the test keeps it.
     - **With a traction,** the right side gains the load's power `+∫_Γ t·v`.
     - **With a Dirichlet membrane,** `v` is not a test function, so the identity needs the condition's
       reaction power. With `g = 0` that power is zero and the identity holds as above; with `g ≠ 0` it is
       not tested.
     - **At a growing front** the identity still holds as an algebraic statement of the step, but `δE[v]` is no
       longer the rate of change of the energy, since the domain moves by `v_f`, not `v`. Its test there
       checks the algebraic identity only.
   - **Not here:** an anisotropic active stress (a tensor, such as `σ_a n⊗n` along fibres).

   *Recommendation: (a).*

3. **The viscous operator is a choice: `symmetric` (today's, the default) or `laplacian`.**
   - **The forms and their Rayleigh functions:**
     - `symmetric`: `∫ 2η dev ε(v):ε(φ) + η_b (∇·v)(∇·φ)`, with `R = ½∫(2η|dev ε|² + η_b(∇·v)² + ζ|v|²)`
       (ADR 017 decision 2);
     - `laplacian`: `∫ η ∇v:∇φ + η_b (∇·v)(∇·φ)`, with `R = ½∫(η|∇v|² + η_b(∇·v)² + ζ|v|²)`. UC-A is
       `η = α`, `η_b = 0`, `ζ = 1`.
   - **Why offer the Laplacian:** it is the published model, and using `ε(v)` instead changes its zero-stress
     condition and its phase diagram (`use-cases.md` UC-A mismatch 2, *checked*). Reproducing the paper, and
     comparing with other solvers on it, needs the paper's operator.
   - **What it costs** (*inferred*; the review confirmed both): it is not frame-indifferent. Its matrix
     annihilates translations but not rotations (`∇φ = W` for `φ = Wx`), so a rigidly rotating cell is damped,
     by `2ηω²|Ω|` in 2D. Its coercivity needs only Poincaré's inequality with the drag or a Dirichlet condition
     ([Brenner 2008], *background*), not Korn's.
   - **Where it is chosen:** one key on the equation, `viscous_operator: symmetric | laplacian`. The
     coefficients stay per member; only the operator is per equation, since no published model mixes two in
     one cell (*inferred*). A per-member choice can be added later without breaking a model.
   - **Its IR tag:** a separate kind, `VISCOUS_LAPLACIAN`, since its invariants differ from `VISCOUS_STRESS`'s
     (ADR 018's review finding 15 split `TENSION_SEMI_IMPLICIT` out for the same reason).
   - **Contract C and the vertex lowering** use `symmetric` only (decision 1).

   *Recommendation: yes.*

4. **A free membrane takes one of four conditions: a tension, zero traction, a velocity, or an applied
   traction.**
   A free membrane is a surface between a member and a non-member (ADR 018 decision 3). Between two members
   nothing changes. A box face keeps ADR 018's rule: `dirichlet` zero only, since the box does not move.
   - **(i) A tension** `γ > 0`, as ADR 018 (`NonPositiveTension` kept for a given tension).
   - **(ii) Zero traction:** the tension is omitted, and the natural condition is `σ n = 0`. This is UC-A's ZS
     edge, which "assumes that the membrane tension is small relative to the contractile and viscous stresses"
     ([Nickaeen 2017] p. 5, *checked*).
     - ADR 018 decision 3 required a tension so that a material free boundary would have a restoring force at
       the grid scale (*inferred* there). That reason concerns the motion, not the solve: each step stays
       coercive with `η > 0` and a drag or a Dirichlet condition.
     - A zero-traction membrane has no shape restoring force, whether it moves with `v` or is a growing front
       (decision 5): a front moving at a uniform normal speed forms corners, and UC-A reports concavities that
       grow by positive feedback ([Nickaeen 2017] pp. 9 ff., per the review). Whether such runs stay resolved is
       measured in stage 4a, not assumed. The remedies available are the slide (ADR 014), the remesh (ADR 016)
       and, later, a curvature term in the front law.
   - **(iii) A velocity:** `dirichlet` with a vector expression `g` on the membrane. UC-A's ZV edge is `g = 0`
     ("a very narrow band of very strong adhesions near the cell edge", [Nickaeen 2017] p. 5, *checked*).
     - `g` may read constants, `sim.t`, `geom.*` (so a localized condition is a field of `geom.x`), region
       variables and `region_size`. Not `v`, and not a field species in the first scope: a species-dependent
       `g` would couple the condition to the transport (*inferred*).
     - A condition on part of a membrane, such as a held patch, needs membrane sub-regions, which the geometry
       does not have yet (`use-cases.md` §3 item 5, *checked*).
   - **(iv) An applied traction:** a `traction` slot in `surfaces:`, a vector expression `t` (force per area on
     the cell). The residual gains `−∫_Γ t·φ dS`; a uniform `P` is the same as `t = P n`, so the sign agrees
     with ADR 018's (*inferred*; confirmed by the review).
     - It may read what `tension` may, plus `geom.*`: unlike a tension, a load is often localized, so it may be
       a field of `geom.x` and `geom.normal`.
     - It models external loads: a micropipette, an AFM tip, an optical trap. It can exert a net force, so the
       rigid-motion invariants exclude it.
     - A protrusive push of the actin network on the membrane is *not* a traction in a one-velocity model:
       membrane and network share `v`, so an internal push and its reaction cancel (*inferred*). Protrusion
       enters kinematically (decision 5), as in UC-A.
   - **How they combine on one membrane:** (i) and (iv) may combine (a tension plus a load). (iii) excludes (i),
     (ii) and (iv), since the velocity is then given. Any of them may be on a growing front (decision 5),
     except (i): see there.
   - **Rigid motion** (`UnconstrainedRigidMotion`, ADR 018 decision 4) counts a Dirichlet condition on a
     membrane as holding the cell, like a wall.
   - **The schema:** (ii) is the absence of a `tension`. (iii) is a `boundary_conditions` entry on the
     membrane, as a wall is on a box face. (iv) is a slot in `surfaces:` beside `tension`. A `surfaces:` entry
     is then any membrane with a tension or a traction, whatever its motion.

   *Recommendation: yes, all four.*

5. **A member's free membrane may be a growing front: it moves with a velocity law, while the member's material
   moves with `v`.**
   - **The physics** ([Nickaeen 2017] p. 5, *checked*): the edge moves with `v_f = v_p n + u|∂Ω`, where the
     scalar `v_p` is the protrusion speed. The edge is not material: actin is added there. In ZS, `v_p` reads
     the cell's area `A = |Ω|`; in ZV it also reads the trace of myosin, and `u|∂Ω = 0`. vcell-core calls this
     a *non-material front* (Addendum C, *checked*).
   - **The options:**
     - **(a) The existing surface `prescribed` motion,** with `trace(v)` admitted:
       `motion: {kind: prescribed, velocity: "trace(v) + (v0*a0/region_size(cyt) - k*(...)) * geom.normal"}`.
       A surface's prescribed velocity already may read `geom.normal`, `region_size` and the traces of the
       species beside it, and drives the domains it bounds (ADR 013 decisions 1–2; formalism §1.10.1,
       *checked*). This is `use-cases.md` §4's proposed `boundary_law` kind in the syntax the formalism has had
       since ADR 013.
     - **(b) A new motion kind** (`boundary_law`, or `unknown` with a `normal_growth` expression).
     - **(c) Defer** until chemistry on the moving cells (ADR 018 stage 5).
   - **The velocity law** need not contain `trace(v)`. The rate at which material crosses the front,
     `(v − v_f)·n`, is the polymerization (or depolymerization) rate; species do not cross it (below).
     Because a parameter may not read `trace(·)` (§1.8.3, *checked* by the review), a `v_p` that reads a trace
     is written inline in the velocity.
   - **What (a) changes:**
     - **Motion, both ways** (ADR 018 decision 2): a surface bounding a member moves with `v`'s trace *or* has a
       prescribed velocity. The member keeps `motion: {kind: unknown, variable: v}`: that is its material's
       velocity.
     - **ADR 013 decision 6** refuses a volume with its own motion bordered by a prescribed membrane, "because
       the two normal velocities must agree" (*checked*). At a growing front they are meant to differ, by the
       polymerization rate. The refusal is lifted for a T9 member at a free membrane with a prescribed
       velocity, and kept elsewhere.
     - **The domain** of a member follows its bounding surfaces, not `v`, at a growing front. The mesh moves by
       the harmonic extension from the skeleton, as ADR 018 decision 7 already does.
     - **Member species** use ADR 013 decision 3's assembly, transport relative to the moving boundary, with
       carrier `v` in place of 0 (*inferred*). Its flux at the front is relative to the front, so with no other
       flux the species obey UC-A's no-flux Rankine–Hugoniot condition `n·(−D∇c + (v − v_f) c) = 0` when the
       mesh moves with `v_f·n` there (`use-cases.md` UC-A mismatch 5; *inferred*). UC-A's own carrier is
       `u_eff = u(1 − m/m_max,u)`, a crowding cut-off; reproducing it needs a nonlinear drift, which is stage 5
       work (*checked*, [Nickaeen 2017] p. 4).
     - **Membrane species** on a growing front ride `v_f`, its material velocity (formalism §1.10.1:
       a surface's prescribed motion is its material velocity, tangential part included), with the mandatory
       dilution `ρ∇_Γ·v_f`. ADR 018 decision 8 said a membrane bounding a member rides `trace(v)`; that stays
       for material membranes.
     - **No tension on a growing front** in the first scope. ADR 018 decision 5's semi-implicit tension
       evaluates the tension at `x + Δt v`, but a growing front moves by `v_f`. A tension there would need its
       own scheme; UC-A's ZS front has none.
     - **The solve order:** solve `v` on the current geometry, evaluate the front velocity from it, move. This
       is the one-step lag the prescribed front already uses for species traces (ADR 013 decision 2,
       *checked*). UC-A's authors iterated the coupling to a fixed point at every step ([Nickaeen 2017] p. 6,
       *checked*, "Numerical methods"); comparing phase-diagram points needs the splitting error measured, or
       the same iteration.
   - **First scope:** a growing front on a free membrane that bounds one member and meets no junction (a
     closed loop). A front ending at a junction would need a law for the junction's motion; refused until a use
     case needs it.

   *Recommendation: (a),* in the formalism now, solved in stage 4a.

6. **One cell comes first in the staging, and the grammar lands in PR #246 before it merges.**
   - **How the formalism changes land:**
     - **(a) In PR #246,** as a second commit after acceptance: the slots, the operator key, the membrane
       conditions and the growing front are validator rules on the same files, and #246 is held anyway.
     - **(b) Merge #246 as it is,** then a follow-up PR. Relaxing a refusal is backward compatible, so nothing
       is locked in.
   - **The staging, amended** (ADR 018's, with its step numbers kept):
     1. The formalism: ADR 018's and this note's.
     2. The IR: ADR 018's five term kinds plus `ACTIVE_STRESS`, `SURFACE_TRACTION` and `VISCOUS_LAPLACIAN`, and
        Dirichlet conditions on a member's membrane; the invariants, extended by decisions 2–4 (`∫ζv = 0`, the
        Laplacian's translation-only kernel).
     3. The fixed mesh: the disk's exact solution; manufactured solutions for both operators, with an active
        stress and a traction; the energy identity, case by case. Then **3b, the first mechano-chemical test:**
        UC-A's ZV instability in a fixed circle ([Nickaeen 2017] Fig 4A, p. 12, *checked*: `μ_tot = 1.5π`,
        `α = 0.5`, myosin relocating to the edge), with myosin carried by `v`.
        - With `v = 0` on the edge and no growth, the skeleton does not move, so a fixed mesh is consistent
          (*inferred*).
        - 3b brings ADR 018 decision 8 (species on a member) forward, on a fixed mesh only. The crowding
          cut-offs are left out, so 3b measures the onset of the instability, which the cut-offs do not affect
          (*inferred*: they act only at high `m`), not the saturated state.
        - The paper's 1D threshold `β μ_tot > 1 + π² α` (p. 10, *checked*) is not posed in T9, whose domains
          are 2D and 3D. A 2D strip with zero-stress sides does not reduce to it, since `β m ≠ 0` there (the
          review's finding).
     4. **4a, one cell moving in 2D:** one closed membrane, solved rather than prescribed, with no nucleus and
        not touching the box: a material membrane with a tension, then a growing front (ZS, and ZV with
        `v_p` from the area and constants; a myosin-dependent `v_p` or `σ_a` waits for stage 5, as ADR 018
        decision 3 refuses species on a moving run until then). It reuses today's closed-loop slide and contact
        check. **4b, several closed loops** (a nucleus; cells that do not touch). **4c, junctions:** cells in
        contact or held at a box face, the end-pinned slide, the rewritten contact check, then ADR 016's
        remesh.
     5. Chemistry on the moving cells, with crowding cut-offs; UC-A's phase-diagram points.
   - **Why one cell first** (*inferred*): 4a needs none of the junction machinery and reuses ADR 014's closed
     loop, so the single-cell models run before the multicellular ones, and the multicellular work cannot hold
     them back.

   *Recommendation: (a), and the staging as amended.*

## What T9 still does not do for a single cell

After this note, these stay outside T9; each keeps or gains another home:

- **An incompressible interior** (Stokes): the drivers `stokes.py`, `fsi.py`, `multiphase.py`, and the
  membrane force balance of `unknown_motion.py` (formalism §1.10.8). T6 stays reserved for it.
- **An elastic interior** (T7, T8, reserved) and **memory** (viscoelasticity): ADR 017 decision 2 excluded
  them; nothing here adds them.
- **Bending** (Helfrich) and **a membrane force balance on an open patch** (UC-C's deformable invagination).
- **The rest of UC-C:** a viscosity that vanishes with `ρ` (T9 needs `viscosity > 0`, so a floor);
  a zero-stress far field (box faces take only `v = 0`, and a member on the box must be held,
  `MemberOnFreeBox`); a force functional `f_z` driving the motion; membrane sub-regions.
- **Two phases** (an active mixture with a fluid): the multiphase note's family.
- **Slip on a wall** (vcell-core names `slip`), and a **hard area constraint** (ADR 018 decision 5 (c)).
- **A coefficient that reads `v`** (nonlinear rheology), an **anisotropic active stress**, a **`body_force`**
  slot (vcell-core names one; no use case here yet).
- **Moving runs in 3D** (no 3D partition remesher).

## For vcell-core (to send to the vcell-core session, not changed from here)

Proposals; none of the names is settled in vcell-core.

- The FEM lowering of the `active` law kind on a resolved compartment: an isotropic active stress fills
  `active_stress` (decision 2), contractile positive.
- A Newtonian viscosity law may state its operator; the vector Laplacian lowers to `viscous_operator:
  laplacian` (decision 3) and is not in the shared subset.
- The mechanical conditions `traction` and `no_slip` on a membrane lower to decision 4's (iv) and (iii); an
  omitted surface stress lowers to (ii).
- The reserved `FrontLaw{boundary, normal_velocity}` (Addendum C) on a resolved cell's membrane lowers to
  decision 5's growing front, with the velocity `trace(v) + normal_velocity · n` when the front also carries
  the material's tangential velocity, as UC-A's does. motion §2's "bulk added at the boundary" is the
  polymerization rate `(v − v_f)·n` there.
- A model using any of these is not lowered to a vertex realization (motion §4's rule applies unchanged).

## Not decided here

- A growing front that meets a junction; a tension on a growing front; a growing front in 3D.
- A species-dependent Dirichlet velocity (decision 4 (iii)), and conditions on part of a membrane (needs
  membrane sub-regions).
- A per-member viscous operator.
- The fixed-point coupling of UC-A's scheme, if the one-step lag's splitting error matters (decision 5).
- The baseline for UC-A: whether VCell's MovingBoundary solver can run its vector force balance
  (`use-cases.md` UC-A mismatch 6; still open).
- The items in "What T9 still does not do".

## Review

An independent reviewer read the draft and its cited sources, with no view of the author's reasoning beyond the
note. It checked by derivation, and numerically where stated: the active-stress and traction signs, UC-A's
natural condition, the `bulk_viscosity = 2η/3` identity (0 in 2D, 3e-16 in 3D), the generalized energy
identity, the Laplacian's kernel and coercivity, and the Rankine–Hugoniot condition at a growing front. It found
23 issues: none blocking, 7 major, 12 minor, 4 nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | major | One cell is not one closed loop if it has a nucleus or touches the box; the partition moves one *prescribed* membrane and refuses several. | Context, "not yet a moving one"; stage 4 split into 4a (one solved loop), 4b (several loops), 4c (junctions). |
| 2 | major | ADR 013 decision 6 refuses a moving volume bordered by a prescribed membrane; decision 3's relative-to-boundary assembly is only for `motion: none`. | Decision 5: decision 6's refusal lifted for a T9 member at a growing front; member species use decision 3's assembly with carrier `v`; both in "Would amend". |
| 3 | major | Membrane species at a growing front: carrier and dilution unstated. | Decision 5: they ride `v_f` with `ρ∇_Γ·v_f`; ADR 018 decision 8 amended. |
| 4 | major | The energy identity is not exact with a traction, a nonzero Dirichlet, a growing front; the semi-implicit tension is wrong on a growing front. | Decision 2: the identity case by case; decision 5: no tension on a growing front in the first scope. |
| 5 | major | The 1D threshold cannot be posed in T9; 3b needs member species, staged at 5. | Stage 3b: the fixed-circle ZV instability (Fig 4A), with decision 8 brought forward on a fixed mesh; the 1D threshold dropped, with the reason. |
| 6 | major | "A growing front carries its own regularization" is unsupported. | Removed; decision 4 (ii): zero-traction runs, material or growing, are measured, with the remedies named. |
| 7 | major | The named uses (patch, localized loads) are not expressible without `geom.x` or sub-regions. | Decision 4: `traction` and `g` may read `geom.*`; a patch waits for membrane sub-regions. |
| 8 | minor | The diff range included ADRs 017/018; PR #246 is 8 files; the test count is from a run; the runner gains a refusal. | Context corrected. |
| 9 | minor | UC-C's active stress is expansive in this convention. | Decision 2 maps both models' signs. |
| 10 | minor | Only UC-C's operator is expressible; its vanishing viscosity, far field and `f_z` are not; slip and a hard area constraint unlisted. | "What T9 still does not do". |
| 11 | minor | Whether the front law must contain `trace(v)`; whether a Dirichlet and a growing front combine. | Decision 5, "The velocity law"; decision 4, "How they combine". |
| 12 | minor | UC-A's carrier is `u_eff`, not `v`. | Decision 5: qualified; crowding at stage 5. |
| 13 | minor | The one-step lag is ADR 013 decision 2's; the paper iterates to a fixed point. | Decision 5: cited as checked; splitting error to measure; "Not decided here". |
| 14 | minor | The Laplacian's invariants differ, so one tag with a flag repeats ADR 018 finding 15. | Decision 3: a separate tag, `VISCOUS_LAPLACIAN`. |
| 15 | minor | "Would amend" incomplete. | Completed. |
| 16 | minor | `NonUniformPressure` is a subset rule, not a solvability rule. | Decision 1: kept as a naming rule, and why it refuses nothing expressible. |
| 17 | minor | vcell-core's Addendum C reserves `FrontLaw` for non-material fronts; motion §2 bears on it. | "For vcell-core" maps the growing front to `FrontLaw`; Builds on cites §2 and Addendum C. |
| 18 | minor | Region variables and species are refused on a moving run until stage 5, so 4a cannot use myosin. | Stage 4a: mechanics only, `v_p` from area and constants. |
| 19 | minor | "Propulsion comes from the drag" is wrong: `∫ζv = 0` with no traction or Dirichlet. | Decision 2: rephrased; `∫ζv = 0` added as an invariant test. |
| 20 | nit | Page numbers and "`v_p` reads the normal". | Pages corrected (Eq 2 and the ZS condition p. 6, the threshold p. 10, Fig 4 p. 12); `v_p` is a scalar. |
| 21 | nit | The UC-C citation title is wrong; the full citations are in ACKNOWLEDGEMENTS.md. | References corrected from ACKNOWLEDGEMENTS.md. |
| 22 | nit | A parameter may not read `trace(·)`. | Decision 5: a trace-reading `v_p` is written inline. |
| 23 | nit | Define `R` for both operators. | Decision 3. |

## References

- **This repository** at `dd5137e` and `4cce16f`: ADRs 013–018; `docs/modeling/cell-mechanics/use-cases.md`
  §1–§5; `docs/modeling/active-protrusion-migration.md`; `docs/modeling/declarative-formalism.md` §1.4.2,
  §1.4.4, §1.6.4, §1.8.3, §1.8.8, §1.10, §1.11, §2.2.1, §2.2.4, §3.6.1; `formalism/validator.py` and
  `tests/test_formalism_cell_force_balance.py` (PR #246); `ACKNOWLEDGEMENTS.md`.
- **vcell-core** at `f6f45c7`: `spec/semantics/motion-and-mechanics.md` §2, §3, §4 and Addendum C.
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A
  free-boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13:e1005862 (2017).
  doi:10.1371/journal.pcbi.1005862.
  - *Read here,* from the copy in `docs/papers/` (gitignored): the Model section (pp. 3–6: Eqs 1–2, the ZV and
    ZS conditions, the boundary velocity), Numerical methods (p. 6), the 1D threshold (p. 10) and Fig 4 (p. 12).
    The concavities (pp. 9 ff.) are cited from the review.
  - *Use:* the single-cell model decisions 2–5 are measured against, and stage 3b's test.
- **[Nickaeen 2019]** Nickaeen M, Berro J, Pollard TD, Slepchenko BM. Actin assembly produces sufficient forces
  for endocytosis in yeast. *Mol. Biol. Cell* 30:2014–2024 (2019). doi:10.1091/mbc.E19-01-0059.
- **[Nickaeen 2022]** Nickaeen M, Berro J, Pollard TD, Slepchenko BM. A model of actin-driven endocytosis
  explains differences of endocytic motility in budding and fission yeast. *Mol. Biol. Cell* 33:ar16 (2022).
  doi:10.1091/mbc.E21-07-0362.
  - Both *cited second hand,* through `use-cases.md` UC-C; citations from `ACKNOWLEDGEMENTS.md`.
  - *Use:* the symmetric-gradient, expansive active gel (Context, decision 2).
- **[Brenner 2008]** Brenner SC, Scott LR. *The Mathematical Theory of Finite Element Methods*, 3rd ed.
  Springer (2008).
  - *Not read here.*
  - *Use:* Poincaré's and Korn's inequalities (decision 3).

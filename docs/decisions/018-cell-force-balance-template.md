# ADR 018 — A cell force-balance template

**Date:** 2026-11-14
**Status:** Proposed. Revised after an independent review (19 findings, none blocking, 5 major), each answered
below and listed in "Review" at the end.
**Builds on:**
- ADR 017 (accepted 2026-11-13), decision 5: the FEM forms of vcell-core's shared-subset laws enter the formalism
  as one new template, a quasi-static, overdamped force balance for a vector velocity `v` on a set of cells.
  Decisions 1–4 fix its forms, and decision 6 how they are verified. Its "Not decided here" gives this note the
  template's number, name and slot grammar, and chemistry on the moving cells.
- The formalism: §1.3.1 (a variable lives on exactly one subdomain class), §1.4.1 (the equation envelope),
  §1.4.3 (T6–T8 reserved), §1.6.4 (internal boundaries need an explicit BC), §1.8.4 (`region_size`), §1.9.5
  (initial conditions of steady equations), §1.10 (motion), §1.11 (the validator), §2.2.1 (the motion schema),
  §3.3 and §1.10.7 (what backends choose).
- ADR 004 (the tagged-term IR), ADR 013 (a surface-driven domain), ADR 014 (the moving partition), ADR 015
  (conservative mesh moves), ADR 016 (remeshing the partition).

**Would amend:**
- The formalism: §1.3.1 and §2.2.2 (a variable on a list), §1.3.5, §1.9.5 and §1.11.4 (no initial condition
  for T9's `v`), §1.4 (a new template), §1.6.4 and §1.11.4 (T9's member boundaries need no BC), §1.8.6, §1.11.3
  and §1.11.7 (resolving `v` and its trace), §1.8.8 (a zero-vector Dirichlet), §1.11.11 (one coercivity check),
  §2.2.1 and §2.2.4.
- ADR 017: decision 1's positivity rule, read as applying to surfaces that carry a tension (decision 3 here),
  and decision 5's "Lagrangian at the surfaces", refined (decision 7 here).
- Code: `formalism/schema.py`, `templates.py` and `validator.py`; `backend/discrete.py` (new term kinds and a
  quasi-static scheme); `backend/moving_partition.py` (several solved membranes, open arcs).

Nothing changes until the decisions are accepted.

*Checked* marks what was read in a source (file and line, or section) or run here. *Inferred* marks reasoning.
*Background* marks a published result used without reading it here (cited in References). This repository was
read at `1f555fc`.

## Context

ADR 017 decision 5 fixed what the template must express (*checked*, ADR 017 lines 301–337):

- a vector velocity `v` on a set of volume subdomains, the cells;
- per cell: `viscosity` `η`, `bulk_viscosity` `η_b`, `drag` `ζ`, and a `pressure` `P_c` uniform on the cell;
- per surface: a `tension` `γ`, on a cell's free membrane or on a contact;
- the residual, for every test velocity `φ`:

  `∫(2η dev ε(v):ε(φ) + η_b ∇·v ∇·φ + ζ v·φ) dV − Σ_c P_c ∮_{∂Ω_c} φ·n dS + Σ_Γ ∫_Γ γ_Γ ∇_Γ·φ dS = 0`,

  with `dev` the 3D deviator in every dimension (ADR 017 decision 2);
- the cells and their membranes move with `v`; ADR 016 remeshes them in 2D.

vcell-core's lowering fills the slots: `P_c = K(V0 − V_c)`, a free membrane's `γ` is the cell's tension
`T_c = σ_c + K_S(S_c − S0_c)`, and a contact's is `Λ_ij + T_i + T_j` (ADR 017 decisions 1 and 3).

**What the formalism has today** (*checked* in the sections and files named):

| Need | Today | Gap |
|---|---|---|
| One velocity across several cells | A variable lives on exactly one subdomain class and is identified by `(name, subdomain)` (§1.3.1; `schema.py` lines 166, 306). A subdomain's unknown motion names a variable declared on that subdomain (§2.2.1; `validator.py` lines 387–401). | A variable cannot span cells. |
| Per-subdomain coefficients in one equation | An equation has one `subdomain` and one `terms` map (§1.4.1). | No grammar for slots keyed by subdomain. |
| Terms on surfaces between volumes | Every internal boundary a variable touches needs an explicit BC (§1.6.4, line 732; §1.11.4). The weak-form language defines `ds(·)`/`dS(·)`; the backend defers them (§3.6.1). | No template assembles a force on a surface between two volumes. |
| A quasi-static solve inside the IR | `DiscreteProblem` lowers time-dependent problems only (`discrete.py` lines 17–18, 110). | No steady scheme. |
| Several moving membranes | The moving partition moves one closed membrane by a prescribed velocity and refuses several (§3.6.1). | Several membranes, solved, meeting at junctions. |
| Mechanics templates | T6–T8 are reserved: inertial Stokes with incompressibility, linear and hyper-elasticity (§1.4.3). | None fits. |

## Decisions

1. **The template is new: `cell_force_balance`, numbered T9.**
   - **Why not T6** (*inferred*). T6 is sketched as an inertial momentum balance (`ρ₀(∂_t v + v·∇v)`) with its
     stress "filled by a constitutive template", paired with T3 incompressibility (§1.4.3; *checked*). This one
     is overdamped, compressible, has a fixed constitutive law, and carries surface and per-cell terms. Folding
     both into one number would give T6 two incompatible meanings.
   - **Why not the weak-form escape hatch.** ADR 017 decision 5 answers this: a template with named slots is
     what VCell's lowering emits, and the backend defers the escape hatch's surface measures.
   - **The name** says what it balances and where; "tissue" is avoided because a single cell is a valid
     instance.

   *Recommendation:* `cell_force_balance`, T9. T6–T8 stay reserved as they are.

2. **The velocity lives on a list of volume subdomains (the members), and the equation keys its slots by
   subdomain.**
   - **The options:**
     - **(a) A variable on a list of subdomains.** `v` is declared with `subdomain: [cyt__c1, nuc__c1, cyt__c2]`,
       and one T9 equation governs it, with slots keyed by member.
     - **(b) A named group in the geometry.** The geometry declares `cells = {cyt__c1, nuc__c1, cyt__c2}`, and
       `v` lives on the group. This changes the geometry formalism, the source of truth for domains (ADR 007),
       to serve one template.
     - **(c) One velocity per cell, with value equality across contacts.** No envelope change, but value
       equality is deferred in every coupled solver (§3.6.1; *checked*), and a velocity must also be continuous
       across a nucleus's envelope inside a cell, so the equalities multiply.
   - **What (a) changes in the formalism:**
     - **Identity (§1.3.1, §2.2.2).** A variable lives on one subdomain class, *or*, for a vector governed by
       T9, on a list of volume classes. A list is accepted for no other template. A listed variable's name is
       unique in the MathDescription: no other variable of that name lives on any member. The validator keys it
       by name and maps every member to it.
     - **Resolution (§1.8.6, §1.11.3, §1.11.7).** On a member, the bare name `v` is `v`. On a surface that bounds
       a member, `trace(v)` is `v`'s trace. On a surface between two members it is not ambiguous, because `v` is
       one continuous field there; the trace-ambiguity rule exempts a listed variable.
     - **Motion (§2.2.1).** Every member has `motion: {kind: unknown, variable: v}`, and so does every surface
       that bounds a member: it moves with `v`'s trace. The validator checks both ways: a member or a bounding
       surface without it is refused, and so is `motion: unknown` naming `v` anywhere else.
     - **Envelope (§1.4.1, §2.2.4).** A T9 equation has `subdomain` equal to the variable's list, and two slot
       maps in place of `terms`: `volumes:`, keyed by member, which must name every member (each needs a
       `viscosity`), and `surfaces:`, keyed by surface.
     - **Boundaries (§1.6.4, §1.11.4).** A surface bounding a member needs no BC for `v`: its condition is the
       natural one, the tension of the `surfaces:` map (or none). The only BC `v` accepts is decision 4's
       zero-velocity wall.
     - **Initial condition (§1.3.5, §1.9.5, §1.11.4).** Those sections make an initial condition optional for an
       unknown-motion variable, since it sets the configuration at `t = 0` (*checked*, lines 1125, 1381). For a
       velocity solved quasi-statically it would set nothing, so T9 refuses one. The initial configuration is the
       geometry (§1.10.4).
   - **What a modeller writes** (vcell-core's lowering, for two touching cells in a medium `ec`):

     ```yaml
     subdomains:
       - { name: cyt__c1, kind: volume, motion: { kind: unknown, variable: v } }
       - { name: cyt__c2, kind: volume, motion: { kind: unknown, variable: v } }
       - { name: ec,      kind: volume }                                  # at rest, surface-driven
       - { name: pm__c1,  kind: surface, motion: { kind: unknown, variable: v } }
       - { name: pm__c2,  kind: surface, motion: { kind: unknown, variable: v } }
       - { name: jn__c1_c2, kind: surface, motion: { kind: unknown, variable: v } }

     variables:
       - { name: v, subdomain: [cyt__c1, cyt__c2], type: vector }

     equations:
       - template: cell_force_balance
         variable: v
         subdomain: [cyt__c1, cyt__c2]
         temporality: steady_state
         volumes:
           cyt__c1: { viscosity: eta, bulk_viscosity: eta_b, drag: zeta, pressure: "K * (V0 - region_size(cyt__c1))" }
           cyt__c2: { viscosity: eta, bulk_viscosity: eta_b, drag: zeta, pressure: "K * (V0 - region_size(cyt__c2))" }
         surfaces:
           pm__c1:    { tension: T_c1 }
           pm__c2:    { tension: T_c2 }
           jn__c1_c2: { tension: "Lambda + T_c1 + T_c2" }

     parameters:
       - { name: T_c1, expression: "sigma + K_S * (region_size(pm__c1) + region_size(jn__c1_c2) - S0)" }
       - { name: T_c2, expression: "sigma + K_S * (region_size(pm__c2) + region_size(jn__c1_c2) - S0)" }
     ```

     In 2D the lowering multiplies the 2D measures by the depth `d` (ADR 017 decision 1; *checked*); the example
     omits it. `region_size` of a moving surface or volume is re-measured after each move, as the moving
     partition does today (§1.8.4; *checked*).

   *Recommendation: (a).* It is the smallest change that expresses one continuous velocity, and it is local to
   T9.

3. **The slots, their types and what is checked.**

   | Slot | Map | Type | Default | May read | Checked |
   |---|---|---|---|---|---|
   | `viscosity` `η` | `volumes` | scalar | required | what a T1 slot may (§1.4.4), except `v` | `> 0` at the form's quadrature points, every step |
   | `bulk_viscosity` `η_b` | `volumes` | scalar | 0 | the same | `≥ 0`, likewise |
   | `drag` `ζ` | `volumes` | scalar | 0 | the same | `≥ 0`, likewise |
   | `pressure` `P` | `volumes` | scalar | 0 | constants, `sim.t`, `region_size`, region (T5) variables, and parameters that read only these | uniform on the volume, at validation |
   | `tension` `γ` | `surfaces` | scalar | see below | constants, `sim.t`, `region_size`, region variables, the surface's own species, `trace(·)` of an adjacent volume's species; not `v` | `> 0` at the form's quadrature points, every step |

   - **No slot reads `v`.** A coefficient that depends on the velocity would make the solve nonlinear, which the
     quasi-static scheme (decision 6) does not take.
   - **Pressure on each volume, assembled as a volume term** (*inferred*; confirmed by the review). For a
     uniform `P`, `P∮_{∂Ω}φ·n = ∫_Ω P ∇·φ` (the divergence theorem), so the term is `−∫_Ω P ∇·φ dV` on each
     member.
     - Where a cell has several compartments (a nucleus in a cytosol), the lowering gives each the cell's `P_c`.
       The envelope between them then receives `P_c φ·n` from both sides with opposite normals, which cancel,
       and the sum is `P_c∮_{∂Ω_c}φ·n` over the cell's outer boundary, exactly ADR 017's form.
     - A contact between cells `i` and `j` receives `(P_i − P_j) φ·n`, the pressure difference that curves it
       (ADR 017 decision 3).
     - A different `P` on a nucleus would put a pressure jump on its envelope. That is a different law, which
       this template can express but vcell-core does not lower to.
   - **Why the pressure must be uniform.** ADR 017 decision 1 requires one value per cell for `K`, `V0`, `K_S`
     and `S0`, and a field `P` would no longer equal the boundary form. The validator checks the expression's
     vocabulary, following parameters transitively: a parameter that reads `geom.*` or a field species makes it
     non-uniform. It refuses one by name (`NonUniformPressure`). Spatial T4 variables are fields (§1.4.2), so they
     are refused too.
   - **Why the tension may be a field.** ADR 017 decision 1 allows a field `σ` or `Λ` (the Marangoni force).
   - **Which surfaces carry a tension, and when one is required** (review finding 4):
     - **A surface between a member and a non-member** (a free membrane) **must have one.** It is a free boundary,
       and ADR 017 decision 1 requires its tension to be positive.
     - **A surface between two members** (a contact, or an envelope inside a cell) **may omit it.** It is then no
       surface at all to the mechanics: two parts of one continuous viscous body, carrying no force and no
       instability. ADR 017 decision 1's rule is read as applying to surfaces that carry a tension. vcell-core's
       lowering always gives a contact one.
     - **A given tension must stay positive.** A value `≤ 0` stops the run by name (`NonPositiveTension`),
       naming the surface and the time. Zero is refused too, because a free membrane at zero tension has no
       restoring force at the grid scale.
     - A surface that bounds no member is refused in `surfaces:`.
   - **The medium exerts no traction** (*inferred*, the shared subset's assumption). A non-member volume has no
     velocity in this equation. A resolved, viscous medium is outside the shared subset (ADR 017 decision 4).
   - **Fields in the first stages** (review finding 11): the moving partition refuses region variables
     (ADR 014; §3.6.1), and species on the moving cells wait for decision 8. Until those land, a pressure or
     tension that reads a region variable or a species is refused on a moving run.

   *Recommendation: yes,* as the table.

4. **Well-posedness: a member is held only by drag or by a zero-velocity wall, and the validator and the
   backend refuse rigid motion that nothing resists.**
   - **The condition** (*inferred*). The bilinear form is coercive on `H¹` when `η > 0` on every member and each
     connected component of the members' union has `ζ > 0` on a set of positive measure, or a face where
     `v = 0`.
     - In 2D, planar flow with the 3D deviator gives `|dev ε|² ≥ ⅓|ε|²` (ADR 017 decision 2), and Korn's
       inequality ([Brenner 2008], *background*) finishes it.
     - In 3D with `η_b = 0`, `dev ε` vanishes on the conformal Killing fields, a finite-dimensional space, and
       the trace-free Korn inequality ([Dain 2006], *background*) gives coercivity once the drag or the wall
       removes them. With `η_b > 0` Korn's inequality suffices.
   - **The deviator is pinned** (review finding 5): `dev ε = ε − (tr ε/3) I` in 2D and 3D. UFL's `dev` on a 2×2
     tensor is the 2D deviator (*checked* by the review in the UFL source), whose kernel in 2D includes every
     conformal map, so η_b = 0 would lose coercivity. The term builder writes the 3D form, and a structural test
     checks it.
   - **A member touching the box** (review finding 2). The box is fixed, and the partition's harmonic extension
     holds its nodes (`moving_partition.py` lines 1–24; *checked*). So:
     - every box face a member touches must carry `dirichlet` `v = 0` (a cell held at a wall). Any other value is
       refused, since the wall does not move;
     - a member touching the box with no such BC is refused (`MemberOnFreeBox`);
     - the contact stop (decision 7) does not measure a held face's gap to the box.
   - **The BC's type** (§1.8.8): a `dirichlet` on a vector variable accepts only the zero vector, written `0`.
   - **What is checked when** (review finding 7):
     - **at validation**, from the MathDescription alone: every member has a `viscosity`, and some member has a
       `drag` slot or `v` has a wall BC. Otherwise `UnconstrainedRigidMotion`. This is the validator's one
       coercivity check, which §1.11.11 gains;
     - **when the geometry is realized** (§1.11.10's cross-check, in the backend): each connected component of
       the members has a `drag` slot or touches a held face; every member on the box is held;
     - **at every step**: the drag's values are positive somewhere on each component that has no wall.

   *Recommendation: yes.*

5. **The equation is quasi-static. Each step solves `v` on the current geometry, with the tensions
   semi-implicit and the per-cell scalars lagged.**
   - **Temporality:** `steady_state` only (formalism §1.10.3 allows it for a motion variable), with no initial
     condition (decision 2).
   - **The time stepping is the backend's** (formalism §1.10.7, *checked*), but it decides what the tests
     measure, so it is proposed here. The options (ADR 017 decision 1, time stepping):
     - **(a) Explicit.** Solve `v` on the geometry at `tⁿ`, move by `Δt v`. The capillary limit `Δt ≲ ηh/γ`
       binds, and tightens at corners and triple lines like `1/ln(1/h)` (ADR 017 decisions 1 and 3).
     - **(b) Semi-implicit tensions.** Since `∇_Γ·φ = ∇_Γ x : ∇_Γ φ` on a surface (with `x` the identity; the
       review ran it to 5e-30), the tension form is evaluated at the new position `xⁿ + Δt v` on the current
       surface: `∫_{Γⁿ} γ ∇_Γ·φ + Δt ∫_{Γⁿ} γ ∇_Γ v : ∇_Γ φ` ([Dziuk 1991], [Bänsch 2001]; *background*). The added
       term is linear, symmetric and positive semidefinite for `γ > 0`. The cited works use it to remove the
       capillary limit (*background*). This repository's curve flow, `core/bgn_curve.py`, is semi-implicit in the
       same sense ([Barrett 2007]). The per-cell scalars `P_c` and `T_c` are read from the current geometry, so
       they are lagged.
     - **(c) Implicit per-cell scalars too.** Each cell gains real unknowns (`P_c`, `T_c`) tied to `v` by the
       linearized measures (`Vⁿ⁺¹ ≈ Vⁿ + Δt∮v·n`), as region variables already use real spaces (*checked*,
       `multi_compartment.py` lines 14–15).
   - **What (b) costs** (*inferred*; the review confirmed each by derivation and the second by a run):
     - It is first order in `Δt`. At an equilibrium `v = 0` the added term vanishes, so steady states are
       unchanged.
     - It damps rigid rotation slightly: for a rotation `v = Wx`, `∇_Γ v = WP` (with `P` the tangential
       projector) is non-zero, and the added term dissipates `Δt∫γ|WP|² = Δt(d−1)ω²∫γ`. Translations are
       undamped.
     - The energy identity of ADR 017 decision 6 gains the added term:
       `δE[v] = −2R(v) − Δt∫γ|∇_Γ v|²`. The tests check that form.
   - **What (b) leaves:** the lagged scalars impose a step limit from `K` and `K_S` that does not shrink with
     `h` (*inferred*, from the measures being integrals). It is measured in the disk test (ADR 017 decision 6),
     and (c) follows if it binds in practice.
   - **The step size** (review finding 16): `Δt` is the run's, as on every other path, and is fixed in the
     first stages. A step that would tangle the mesh triggers ADR 016's remesh; one that a lagged scalar makes
     unstable is caught by the energy identity's diagnostic, not adapted to. Adaptivity is later work.

   *Recommendation: (b),* with (c) as a later step on evidence.

6. **The IR gains five term kinds and a quasi-static scheme.**
   - **Term kinds** (ADR 017 decision 5, with the tension split in two; review finding 15):
     - `VISCOUS_STRESS` (with the 3D deviator, decision 4), `DRAG`, and `CELL_PRESSURE` (the volume form of
       decision 3);
     - `SURFACE_TENSION`, the load `∫γ∇_Γ·φ`;
     - `TENSION_SEMI_IMPLICIT`, decision 5's bilinear term, tagged separately because its invariants differ.

     A tension on a surface between two members is assembled on the interior facets of the members' mesh, and
     one facing a non-member on its exterior facets (*inferred*). For a continuous `φ` the tangential gradient is
     the same from both sides of a facet, so the interior integrand is single-valued (the review ran it: the
     jump is 1e-29 against 6.5 for the full gradient).
   - **Operator invariants** (*inferred*; the review ran the first two):
     - for any rigid motion `φ = c + Wx` (`W` skew), the loads of `CELL_PRESSURE` and `SURFACE_TENSION` vanish
       pointwise for any coefficients: `∇·φ = tr W = 0`, and `∇_Γ·φ = P:W = 0` since `P` is symmetric. So the
       cells feel no net force or torque from them, whatever their fields;
     - the viscous matrix annihilates rigid motions;
     - the drag matrix is symmetric and positive semidefinite;
     - `TENSION_SEMI_IMPLICIT` annihilates translations, not rotations (decision 5).
   - **A quasi-static scheme** beside `BackwardEuler`: no time-derivative term; each step assembles and solves
     the terms on the current mesh. `discrete.py` says steady-state lowering is not yet built (*checked*, lines
     17–18, 110).
   - **The element:** a continuous vector Lagrange space on the members' submesh, `P1` by default, configurable
     through the variable's `space` hint (§1.3.3).
     - `P1` and `P2` lock as `η_b/η` grows (near incompressibility); a mixed form with a pressure unknown removes
       it ([Brenner 2008], *background*, not read here). The backend warns above a ratio it will measure; the
       mixed form is later work.
     - The geometry is `P1`. With a `P2` velocity, the mesh moves by `v`'s values at the vertices, and the
       semi-implicit term uses `v` itself; the difference is a consistency error of the `P1` geometry
       (*inferred*).
   - **Results** (ADR 010): `v` is written per member compartment, as each compartment's species are today, as
     a vector field. The details belong to stage 3.

   *Recommendation: yes.*

7. **The motion: the cells and their membranes move with `v`; the medium is at rest and driven by them; the
   mesh moves by the moving partition, extended to several solved membranes.**
   - **The physics** (formalism §1.10.1): every member and every surface bounding a member moves with `v`
     (decision 2). A non-member volume is at rest; its domain moves with the membranes that bound it (§1.10.1's
     `none` row, *checked*). ADR 013 decision 1 states this for a prescribed membrane; extending it to a solved
     one is *inferred*. The box is fixed.
   - **The mesh** (*inferred*; the backend's choice, §1.10.7). This refines ADR 017 decision 5's "Lagrangian at
     the surfaces" (review finding 12):
     - **Junctions** (triple points in 2D, and points where a membrane meets a held face) move with the full `v`.
     - **Other membrane nodes** move with `v`, and in 2D may then be slid back along their arc between junctions
       to keep their spacing. Only the normal motion is geometric; the tangential part is carried by the species
       (decision 8). Today's slide works on a closed loop (a periodic spline; ADR 014 decision 4, *checked*,
       §3.6.1). An arc pinned at both junctions needs a new, end-pinned slide.
     - **Interior nodes,** of the cells and the medium, move by a harmonic extension of the skeleton's motion.
       The mesh velocity `w` inside a cell then differs from `v`. That costs nothing in the force balance, which
       has no state in the interior (the viscous law has no memory).
   - **Remeshing** is ADR 016's whole-partition remesh, from the skeleton, in 2D.
   - **The contact stop** must be rewritten, not adjusted. `_ContactCheck` (`moving_partition.py` lines 114–170;
     *checked*) measures one closed moving membrane's gaps to itself, the box and other membranes, with a
     perimeter wrap. In a tissue membranes meet at junctions by construction. The new check measures gaps
     between skeleton arcs, and treats two arcs as joined within `h` along them of a shared junction, as today's
     `same_part` does within one loop. Two arcs closing elsewhere still stop the run: a new contact is a topology
     change, refused (ADR 017 decision 3).
   - **A surface between two non-members that ends on a member** (a medium–medium boundary meeting a cell) is
     refused in the first scope: its end would move with `v` while today's partition holds every other membrane
     fixed.
   - **3D:** a fixed-mesh T9 solve is allowed (stages 2–3). A moving 3D run is refused, since there is no 3D
     partition remesher.
   - **What the moving partition gains:** several membranes, solved rather than prescribed, meeting at junctions,
     moving together.

   *Recommendation: yes.*

8. **Chemistry on the moving cells rides `v`, and comes after the mechanics.**
   - **The rule** (*inferred*, from formalism §1.10.1 and §1.10.5, with no new syntax):
     - a species on a member rides the solved `v`: the template conservation form with substrate velocity `v`,
       so it dilutes by `c∇·v`, which is non-zero because the interior is compressible;
     - a membrane species rides `v`'s trace `v_Γ`, tangential part included, with the mandatory dilution
       `ρ∇_Γ·v_Γ`;
     - a species in the medium is at rest.
   - **The assembly** (*inferred*). On the moving partition the mesh moves by `w ≠ v`, so each member species is
     transported relative to the mesh by `v − w`, in the conservative ALE form of ADR 015. Today the partition's
     compartments are at rest (carrier 0) and only the moving membrane's species ride their material
     (*checked*, §3.6.1). The member compartments generalize "at rest" to a solved carrier, and every membrane
     bounding a member carries its species as the one moving membrane does now.
   - **The coupling:** a tension that reads a species makes the step mechano-chemical. Staggered, as
     `unknown_motion.py` is (*checked*, §3.6.1): solve `v` with the species at `tⁿ`, move, then advance the
     species.
   - **Staging:** the first implementation is mechanics only, and refuses a species on a member or on a moving
     membrane by name. Chemistry follows as its own step, with ADR 015's exactness tests on the solved motion.
     How the staggered solve fits ADR 015's half-move stride, which is second order with a prescribed velocity,
     is decided then ("Not decided here").

   *Recommendation: yes.*

## For vcell-core (to send to the vcell-core session, not changed from here)

These are proposals; none of the names exists in vcell-core yet.

- The FEM lowering emits T9 in decision 2's shape: one `v` on every volume compartment of the resolved cells;
  `motion: unknown` on each and on every surface bounding one; slot maps keyed by compartment and surface.
- Every compartment of a cell gets the cell's `P_c` (decision 3), so a nucleus's envelope carries no pressure.
- A contact is one surface subdomain whose `tension` is `Λ_ij + T_i + T_j`. How the contact's three
  compartments map onto it stays open (ADR 017, "Not decided here").
- A cell on the box is held at the wall, or refused (decision 4).
- Refusal names: the lowering can refuse in advance what is static, a cell with neither friction nor a wall
  (proposed here as `UnconstrainedRigidMotion`, the refusal ADR 017's "For vcell-core" leaves unnamed).
  `NonPositiveTension` is a run-time stop that only the solver can raise.

## Staging (after acceptance)

1. The formalism: decision 2's amendments, T9 in §1.4, and the schema, registry and validator, with a test of
   each refusal.
2. The IR: the five term kinds and the quasi-static scheme, with ADR 017 decision 6's contract C tests and
   decision 6 above's invariants and the deviator test, on fixed meshes.
3. The solve on a fixed mesh: the disk's exact solution, a manufactured solution in `mms/`, the energy identity
   with the semi-implicit term; `v` in the results bundle.
4. The solve on a moving mesh in 2D: several solved membranes meeting at junctions, the end-pinned slide, the
   junction diagnostic and ADR 017 decision 3's convergence expectations, then ADR 016's remesh with the new
   contact check. Coefficients read only constants, `sim.t` and `region_size` here (decision 3).
5. Chemistry on the moving cells (decision 8), and region variables on the moving partition.

## Not decided here

- A pressure jump on a nucleus's envelope, a nuclear law of its own.
- A resolved viscous medium, and a drag relative to it; a cell sliding along a wall.
- Implicit per-cell scalars (decision 5 (c)), if the lagged ones bind; an adaptive step.
- A mixed form for near-incompressible interiors.
- How the staggered mechano-chemical step fits ADR 015's stride (decision 8).
- 3D: the remesher, quadruple points, and contract C (ADR 017).
- The contact's compartments (ADR 017).
- Exposing T9 through the CLI: the runner routes it only once stage 4 works.

## Review

An independent reviewer read the draft and its cited sources, with no view of the author's reasoning beyond the
note. It checked the mathematics by derivation and with a DOLFINx script: the residual's sign, the pressure
identity and its cancellation, the semi-implicit identity (5e-30), the rigid-motion invariants (1e-14), the
rotation damping (1e-14), the energy identity and the single-valued tangential gradient on interior facets.
It found 19 issues: none blocking, 5 major, 13 minor, 1 group of nits.

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | major | Free membranes are internal boundaries of `v`, which §1.6.4 and §1.11.4 make need an explicit BC; the example would be refused. | Decision 2, "Boundaries"; added to "Would amend". |
| 2 | major | A member touching the fixed box is unhandled: its normal velocity, the BC's value, the contact stop's box gap. | Decision 4: a member on the box must be held with `v = 0`, or it is refused; a held face is not gap-checked. |
| 3 | major | Variable identity, bare-name and trace resolution, and trace ambiguity are undefined for a list subdomain. | Decision 2, "Identity" and "Resolution"; §1.8.6, §1.11.3 and §1.11.7 added to "Would amend". |
| 4 | major | An omitted tension (allowed) conflicts with ADR 017's "every surface positive", and with an explicit 0 being refused. | Decision 3: required on free membranes, optional only between members, where there is no free surface; ADR 017's rule read as applying to surfaces with a tension, listed in "Would amend". |
| 5 | major | `dev` is not pinned; UFL's 2D `dev` loses coercivity with `η_b = 0`. | Decision 4 pins the 3D deviator; a structural test (stage 2). |
| 6 | minor | Korn's inequality does not cover `dev ε` in 3D with `η_b = 0`. | Decision 4: the trace-free Korn inequality [Dain 2006]; "positive measure". |
| 7 | minor | The coercivity check's split between validation and backend is unclear; §1.11.11 says the validator does not check coercivity. | Decision 4, "What is checked when"; §1.11.11 in "Would amend". |
| 8 | minor | "No initial condition" conflicts with §1.9.5 and §1.11.4, which make one optional for unknown motion. | Decision 2: T9 refuses one; those sections in "Would amend". |
| 9 | minor | A vector Dirichlet expression is not typed. | Decision 4: only the zero vector; §1.8.8 in "Would amend". |
| 10 | minor | Slot vocabularies admit `v` (nonlinear), do not follow parameters, and leave T4 open. | Decision 3: `v` excluded, parameters followed transitively, spatial T4 refused in `pressure`. |
| 11 | minor | Region variables and species are readable but refused on the moving partition. | Decision 3, "Fields in the first stages"; Staging step 4. |
| 12 | minor | Sliding surface nodes departs from ADR 017's "Lagrangian at the surfaces"; the slide and contact check assume closed loops. | Decision 7 says it refines ADR 017, and names the end-pinned slide and the rewritten contact check as new code. |
| 13 | minor | A non-member surface ending on a member is unspecified. | Decision 7: refused in the first scope. |
| 14 | minor | "P2 is the remedy" for locking is overstated; P2 velocity on a P1 geometry unaddressed. | Decision 6: a mixed form removes locking (later work); the mesh moves by vertex values. |
| 15 | minor | One term holds a load and a bilinear part with different invariants; "positive" should be "semidefinite". | Decision 6: `TENSION_SEMI_IMPLICIT` tagged separately; "positive semidefinite". |
| 16 | minor | Who sets `Δt`, adaptivity, and the fit with ADR 015's stride are undecided. | Decision 5, "The step size"; the stride in decision 8 and "Not decided here". |
| 17 | minor | ADR 017's decision 5 is at lines 301–337; ADR 004 says nothing about steady state; ADR 013 decision 1 covers prescribed motion only. | Corrected in Context, decision 6 and decision 7. |
| 18 | minor | "For vcell-core" states proposed names as fact; a lowering cannot raise a run-time stop. | Reworded as proposals, with the static and run-time refusals separated. |
| 19 | nits | The non-uniform-pressure refusal is unnamed; where tensions are checked; every member in `volumes:`; motion as a validator rule both ways; 3D moving runs; results output; the BGN citation. | `NonUniformPressure`; quadrature points; decision 2's "Motion" and "Envelope"; decision 7, "3D"; decision 6, "Results". The BGN citation is still unverified (References). |

## References

- **This repository** at `1f555fc`: ADRs 004, 007, 010, 013–017; `docs/modeling/declarative-formalism.md`
  §1.3.1, §1.3.3, §1.3.5, §1.4.1–§1.4.4, §1.6.4, §1.8.4, §1.8.6, §1.8.8, §1.9.5, §1.10, §1.11, §2.2.1, §2.2.4,
  §3.6.1; `formalism/schema.py`, `templates.py`, `validator.py`; `backend/discrete.py`, `moving_partition.py`,
  `multi_compartment.py`, `unknown_motion.py`; `core/bgn_curve.py`.
- **[Dziuk 1991]** Dziuk G. An algorithm for evolutionary surfaces. *Numer. Math.* 58:603–611 (1991).
  - *Not read here.*
  - *Use:* the semi-implicit tension, evaluated at the new position (decision 5).
- **[Bänsch 2001]** Bänsch E. Finite element discretization of the Navier–Stokes equations with a free
  capillary surface. *Numer. Math.* 88:203–235 (2001).
  - *Not read here.*
  - *Use:* the same, coupled to a viscous bulk, and its removal of the capillary step limit (decision 5).
- **[Barrett 2007]** Barrett JW, Garcke H, Nürnberg R. A parametric finite element method for fourth order
  geometric evolution equations. *J. Comput. Phys.* 222:441–467 (2007).
  - *Not read here.* `core/bgn_curve.py` names the method but gives no citation; this one is from memory and
    should be checked before acceptance.
  - *Use:* the existing semi-implicit curve flow cited in decision 5.
- **[Brenner 2008]** Brenner SC, Scott LR. *The Mathematical Theory of Finite Element Methods*, 3rd ed.
  Springer (2008), ch. 11.
  - *Not read here.*
  - *Use:* Korn's inequality (decision 4), and locking of low-order elements near incompressibility
    (decision 6).
- **[Dain 2006]** Dain S. Generalized Korn's inequality and conformal Killing vectors. *Calc. Var. Partial
  Differential Equations* 25:535–540 (2006).
  - *Not read here;* cited on the reviewer's suggestion, from memory.
  - *Use:* coercivity of the deviatoric viscous form in 3D with `η_b = 0` (decision 4).
- **[Alnæs 2014]** Alnæs MS, Logg A, Ølgaard KB, Rognes ME, Wells GN. Unified form language. *ACM Trans. Math.
  Softw.* 40:1–37 (2014). doi:10.1145/2566630.
  - *Use:* UFL's `dev`, read in its source by the reviewer (decision 4).

# ADR 017 — FEM forms of vcell-core's shared-subset mechanical laws

**Date:** 2026-11-13
**Status:** Proposed. Answers issue #241's four questions (decisions 1–4) and adds two that the answers
force (decisions 5–6).
**Builds on:**
- vcell-core's accepted M4a note, `spec/semantics/vertex-laws.md` (accepted 2026-11-12; D111, D112): the five
  laws, their 3D energies and units, their vertex forms, and the provisional FEM column this note answers.
- vcell-core's `motion-and-mechanics.md` §3 (the law kinds), §4 (the shared subset; "it is a reduction, not
  an identity") and §7 (M3 holds the FEM-only laws).
- vcell-core's `active-t1.md` decision 2 (D118): `JunctionViscoelasticity` is vertex-only and an FEM
  realization refuses it (`NotInSharedSubset`).
- This repository: formalism §1.3.1 (a variable lives on one subdomain class), §1.5 (the weak-form escape
  hatch), §1.8.4 (`region_size`), §3.3 (what backends choose); ADR 004 (the tagged-term IR); ADRs 014–016 (the
  moving partition and its remesh); `backend/stokes.py`, `backend/fsi.py`, `backend/unknown_motion.py`.

**Would amend:** vcell-core's `vertex-laws.md` decision 2 (its FEM column), through a vcell-core PR that the
vcell-core session owns (listed under "For vcell-core" below). Nothing in this repository changes until the
decisions are accepted.

*Checked* marks what was read in a source (file and line, or section) or run here. *Inferred* marks
reasoning. *Background* marks general knowledge not checked against a source. The sources in vcell-core were
read at its commit `3dac5b6` (the vectors at `100ce8c`); this repository at `b0132ea`.

## Context

vcell-core's M4a note settled five mechanical laws that one model carries to both a vertex realization and a
resolved (FEM) one. Each is energy-first, hosted on a compartment, with 3D coefficients (*checked*,
vertex-laws.md decision 2):

| Law (host) | 3D energy | Provisional FEM form (issue #241) |
|---|---|---|
| `AreaElasticity{K, V0}` (a cell's volume compartment) | `½K(V − V0)²` | a uniform pressure `K(V0 − V)`, a per-cell functional |
| `SurfaceElasticity{K_S, S0}` (the cell's membrane) | `½K_S(S − S0)²` | a uniform tension `K_S(S − S0)`, a per-cell functional |
| `CorticalTension{σ}` (the cell's membrane) | `σS` | a local surface tension `σ` (Laplace pressure `σκ`) |
| `InterfacialTension{Λ}` (a junction type's compartment) | `Λ · (contact area)` | a tension `Λ` on the contact surface |
| `Friction{ζ}` (a cell's volume compartment) | dissipation `½ζ∫|v|² dV` | a drag `ζv` per volume against an ambient medium |

Units (*checked*, same place): energy pN·µm; `K` pN/µm⁵, `K_S` pN/µm³, `σ` and `Λ` pN/µm, `ζ` pN·s/µm⁴. `Λ`
is a contact's tension in excess of `σ_i + σ_j`. Coefficients are expressions, and forces are taken at fixed
coefficients. A 2D realization is a cross-section of depth `d`: `V = A·d`, `S = P·d`, a contact's area `ℓ·d`
(vertex-laws.md, Jim's choices; *checked*). In a vertex realization the laws read the cell's whole membrane,
contacts included (vertex-laws.md decision 3, amended by vertex-realization.md decision 2; *checked*).

**What vcell-fenics solves today** (*checked* in the source named):

| Piece | Where | Envelope |
|---|---|---|
| A uniform surface tension `γ` on one closed membrane, weak curvature-free load `−γ∮_Γ ∇_Γ·φ ds` | `stokes.py` `solve_incompressible_stokes_surface_tension`; `fsi.py` `step_force_balance_fsi` | one cell; incompressible Taylor–Hood interior, so the enclosed volume is fixed; a Python driver, not a template |
| A volume drag (`screening`, "substrate friction") | the same drivers; `multiphase.py` (`γ_a v_a`) | a constant per solve |
| A membrane force balance `η v + σ H n = 0` on one closed membrane, with no interior | `unknown_motion.py` (formalism §1.10.8, the weak-form route) | a membrane drag, not a volume one |
| A partition of several compartments moving together, remeshed in 2D | `moving_partition.py`, `partition_remesh.py` (ADRs 014–016) | prescribed motion of one membrane only; chemistry, no mechanics |
| `region_size(<subdomain>)`, re-measured after every move on the moving partition | formalism §1.8.4; `moving_partition.py:238` | refused on ADR 013's single moving mesh (#240) |

Nothing solves more than one cell mechanically, a tension on an interior surface, a compressible viscous
interior, or a per-cell pressure. Of the five laws, `CorticalTension` and `Friction` exist for one cell, in a
driver; the other three do not.

### A spike: the FEM forms against vcell-core's vectors

Run here (*checked*; the scripts are on the unmerged branch `spike/adr-017`): each face of every complex in vcell-core's
`conformance/mechanics/terms.jsonl` (square, hexagon, sheared, pair, depth) fan-triangulated about its centroid
into a P1 DOLFINx mesh, so the mesh's boundary is exactly the complex. On it, the shape derivatives of
decision 1 (the forms `K(V−V0)∮φ·n`, `K_S(S−S0)∮∇_Γ·φ`, `σ∮∇_Γ·φ`, `Λ∫_contact ∇_Γ·φ`) were assembled and
their negatives read at the complex's vertices. **Every energy and every force matched the vectors to
2.2·10⁻¹⁶**, the depth case included. Friction's total mobility `ζ∫dA` matched `Σ ζ a_v` in all four cases
that carry it; its per-vertex values are not comparable (decision 4).

Why it is exact (*inferred*, algebra): on straight edges `∫_e ∇_Γ·φ ds = (φ(b) − φ(a))·t_e`, which is
`∂ℓ_e/∂x` paired with `φ`; `∮φ·n` on a polygon is `∂A/∂x` paired with `φ`. A P1 mesh's measures of a
polygonal cell are the polygon's own, so its energies are the vertex energies and its forms their exact
gradients. With edges subdivided by mesh nodes, the vertex force is the chain rule of the nodal forces through
the edges' linear interpolation (*inferred*, not run). The same holds for flat triangles in 3D (*inferred*).

## Decisions

1. **Area and surface elasticity are per-cell functionals of a measure, and fit as uniform scalars on the
   cell.**
   - **The forms:** with `n` the cell's outward normal and `φ` a test velocity, the first variations are
     - `AreaElasticity`: `δE[φ] = K(V − V0)∮_{∂Ω_c} φ·n dS`, i.e. a uniform pressure `P_c = K(V0 − V)`;
     - `SurfaceElasticity`: `δE[φ] = K_S(S − S0)∮_{∂Ω_c} ∇_Γ·φ dS`, i.e. a uniform tension added to `σ`;
     - `CorticalTension`: `σ∮_{∂Ω_c} ∇_Γ·φ dS`;
     - `InterfacialTension`: `Λ∫_{Γ_ij} ∇_Γ·φ dS` on the contact.
     `V`, `S` and the contact's area are the current configuration's, evaluated at the start of the solve
     (forces at fixed coefficients, and at a fixed measure, as the vertex forms are). `∂Ω_c` is the cell's
     whole boundary, contacts included, as in the vertex realization.
   - **Written in the weak, curvature-free form**, not as a Laplace pressure `σκ`. `∫σ∇_Γ·φ` equals the
     normal traction `σκn` (κ the total curvature: `κ = 2H` in 3D) only on a closed smooth surface with
     uniform `σ`. In general it also carries the conormal force at a surface's boundary (decision 3) and the
     Marangoni force `∇_Γσ` of a varying `σ`. It needs no pointwise curvature, which a piecewise-linear
     boundary lacks; `stokes.py` already uses it (*checked*, its docstring).
   - **Why per-cell functionals are acceptable** (*inferred*): they are the energies the vertex model can
     express. A global `½K(V − V0)²` is a cell whose pressure equilibrates instantly; a global `½K_S(S − S0)²`
     is a cortex whose tension is uniform, a fluid cortex with an area reservoir. A *local* surface or bulk
     elasticity (a strain energy over a reference configuration) is a different law, with memory, that the
     vertex model cannot express. It belongs to M3, FEM-only. This matches the research note's warning,
     quoted in vertex-laws.md §1, that a cortex "does not in general reduce to `(P − P₀)²`" (cited second hand).
   - **The 2D cross-section needs no new FEM coefficients** (*inferred*): with `V = A·d` and `S = P·d`, the
     2D weak form per unit depth is the 3D one with the 3D coefficients unchanged; `d` appears only inside the
     measures, which the lowering writes (`K(V0 − d·region_size(c))`).
   - **The rank-one coupling.** Each functional couples every boundary node of a cell. Lagged (measured at
     the step's start, the force explicit), the forms stay sparse, at a step-size limit set by `K` and `K_S`.
     Implicit, each cell gains one scalar unknown (its pressure or tension), with a sparse bordered system;
     scifem's real spaces provide it (*background*: scifem is a dependency here; its real-space API was not
     checked for this note). Which one is a time-stepping choice, the backend's (formalism §3.3).

   *Requires.* Formalism: the per-cell scalars are expressions of `region_size` of the cell's volume and
   membrane subdomains (§1.8.4), which exist; each cell must be its own subdomain. vcell-core's unrolling
   already names each cell's copy (`cyt__c1`; D93), so no per-region sizes (#239) are needed (*inferred*).
   `region_size` must be re-measured on a mechanically moving mesh, as the moving partition already does.
   IR: tagged terms `CELL_PRESSURE` (a per-cell scalar times `∮φ·n`) and `SURFACE_TENSION` (a scalar field
   times `∇_Γ·φ` on a named surface), decision 5.

   *Recommendation: yes,* as the forms above. The issue's column should state the weak form rather than
   "Laplace pressure `σκ`".

2. **The interior constitutive law: a compressible Newtonian viscosity, as a sixth law that is FEM-only; no
   default.**
   - **Why one is needed** (*inferred*): the vertex form has no interior. An FEM force balance with only the
     five laws, `∫ζv·φ = −δE[φ]`, puts surface forces against a volume drag; it has no `H¹` solution (the
     velocity would concentrate on the surfaces). An interior viscous stress makes it well posed. Korn's
     inequality then needs a shear viscosity `η > 0` and either `ζ > 0` or a fixed boundary (*background*).
   - **Why compressible** (*inferred*): in the vertex model a cell's volume changes freely under
     `AreaElasticity`. With an incompressible interior (today's Stokes drivers) `∮v·n = 0`, so `V` never
     changes and `AreaElasticity` is inert: its pressure is absorbed into the incompressibility multiplier. A
     compressible interior is also the published choice for a cell-scale gel against a substrate: a
     compressible viscous actin network with substrate drag (Nickaeen 2017, cited second hand through
     `docs/modeling/active-protrusion-migration.md`).
   - **The law:** `Viscosity{η, η_b}` on a cell's volume compartment, with the stress
     `2η dev ε(v) + η_b (∇·v) I`. The roles are `shear_viscosity` and `bulk_viscosity`, both pN·s/µm²
     (= Pa·s); `η > 0`, `η_b ≥ 0`. The dissipation is `∫(2η|dev ε|² + η_b(∇·v)²)`. In a 2D cross-section the
     flow is planar (`v_z = 0`, no dependence on `z`) and `dev` is the 3D deviator (*inferred*).
   - **A sixth `MechanicalTerm`, marked FEM-only.** This mirrors `JunctionViscoelasticity`, which extended
     the closed menu as a vertex-only variant (active-t1.md decision 2; *checked*). vcell-core's
     `MaterialLaw` is not the place: it states a material *velocity* (`AtRest`, `FrameVelocity`, …, `Solved`
     reserved), not a stress (D98; `biological.yaml`; *checked*).
   - **What the vertex realization does with it: drop it, with a warning, rather than refuse it.** It is a
     dissipation only: it changes rates, not energies, forces or equilibria, which are what contract C
     compares. The vertex model's dissipation, `ζ a_v`, stands in for it in the reduction (motion §4: "a
     reduction, not an identity"; *checked*). `JunctionViscoelasticity` is refused by FEM for the opposite
     reason: it carries an energy and a state. Refusing `Viscosity` in vertex would make every shared model
     unshareable, since FEM needs it.
   - **No default.** A resolved cell whose model states no interior law is refused by name
     (`MissingInteriorLaw`), as vertex-realization refuses `MissingFriction` (D113; *checked*) and as the
     cell-mechanics framework asks of unresolved choices (§6, "diagnosed rather than silently defaulted";
     *checked*).
   - **Not chosen:** an elastic interior (it needs a reference configuration and remapped history through
     remeshing, framework §3; the vertex model has neither); an incompressible interior (it disables
     `AreaElasticity`).

   *Requires.* Formalism: a vector velocity on several volume subdomains (decision 5); viscosity and drag
   slots per subdomain. IR: `VISCOUS_STRESS` and `DRAG` terms. vcell-core: the variant and its two roles, the
   vertex lowering's warning, the FEM lowering's `MissingInteriorLaw`.

   *Recommendation: yes.* Jim to choose between "drop with a warning" (recommended) and "refuse" in vertex.

3. **A triple line needs no law of its own: the force is the natural condition of the tension forms, and in
   the continuum the line sits in quasi-static balance.**
   - **The force** (*inferred*, integration by parts on an open surface): `∫_Γ γ∇_Γ·φ dS = ∫_Γ γκ n·φ dS −
     ∫_Γ ∇_Γγ·φ dS + ∮_{∂Γ} γ ν·φ dl`, with `ν` the outward conormal. Summed over the surfaces meeting at a
     line, the line terms give the force `Σ_k γ_k ν_k`, the vertex model's `Σ_e γ_e t_e`. So the weak forms of
     decision 1 apply the triple-line force with no further input, and contract C holds at triple points (the
     spike's pair case: the contact edge's endpoints are triple points; *checked*).
   - **The tensions** (*inferred*, from the definition of `Λ`): with `T_c = σ_c + K_S,c(S_c − S0_c)`, a cell's
     free surface carries `T_c` and a contact between cells `i` and `j` carries `Λ_ij + T_i + T_j`.
   - **The dynamics.** A triple line out of balance is a point force in 2D (a line force in 3D) on a viscous
     continuum, and the velocity at it is logarithmically infinite (*background*: the 2D Kelvin solution).
     The spike confirms it: an open tense segment ending inside a compressible viscous bulk with drag
     (`ζ = γ = 1`, a P1 mesh, `h` from 1/4 to 1/128). Its endpoint's velocity grew by `0.0735` per
     halving of `h`. The spike's stress was `2ηε + λ(∇·v)I` with `η = λ = 1` (in decision 2's terms,
     `λ = η_b − 2η/3`), and the Kelvin coefficient `(λ + 3η)/(4πη(λ + 2η))·ln 2` is `0.0735` there
     (*checked*). Away from the endpoint the velocity converged. Adding a point drag `μ = 1` at the endpoint did not cure it: the velocity crept
     toward `F/μ = 1` as `0.21, 0.30, 0.36, 0.42` over `h = 1/4 … 1/256`, because the bulk's resistance to a
     point vanishes only like `1/ln(1/h)` (*checked*).
   - **So the continuum limit is quasi-static balance** (*inferred*): an unbalanced line moves infinitely
     fast compared with the bulk, so it is always at Neumann's triangle `Σγ_kν_k = 0`. This is the
     resolved model's physics and differs from the vertex model, whose vertex has the finite mobility
     `ζ a_v`. It is one of the reduction's relaxation-time mismatches (motion §4; *checked*). It does not
     affect contract C, which is static.
   - **When balance exists** (*inferred*, the triangle inequality): at a free triple line between cells `i`,
     `j` and the medium, iff `−2 min(T_i, T_j) < Λ_ij < 0`. For `Λ > 0` the contact peels open; for
     `Λ < −2 min T` it spreads. Both end in a topology change, which FEM refuses (motion §4; *checked*): the
     solver stops by name when a contact's length or a cell's area collapses.
   - **What the implementation does:** the natural condition only, no constraint and no junction drag. It
     reports a diagnostic per triple point, the imbalance `|Σγ_kν_k|`, and tests convergence in `h` of the
     quantities the shared subset compares (cell areas, perimeters, contact lengths, centroids), not of
     junction velocities. If those do not converge, a regularization is a later decision with evidence.

   *Requires.* Nothing of the model; no new formalism. The backend needs the triple points of the mesh's
   interface skeleton, which the partition remesher already builds (`partition_remesh.skeleton`, ADR 016
   decision 1; *checked*).

   *Recommendation: yes.*

4. **`Friction` is a volume drag against the lab frame, and stays one.**
   - **The form:** `∫_{Ω_c} ζ v·φ dV`, the drag `ζv` per volume, against the substrate at rest. It is the
     existing `screening` term (`stokes.py`, `multiphase.py`; *checked*), which there also removes the rigid
     null modes.
   - **It matches the vertex form's measure** (*inferred*, with the spike): the vertex model shares each
     cell's area among its vertices (`a_v`), so a uniform translation's drag is `ζΣA_c`, which is `ζ∫dA` in
     FEM per unit depth (*checked* in all four complexes). Per-vertex mobilities differ: FEM's mass matrix
     shares a cell's area among its mesh nodes, interior ones included. So friction is not part of the
     generalized-force check; its check is a dynamic one (decision 6).
   - **Volume or substrate are the same in a cross-section** (*inferred*): a basal drag `ξ` over the area `A`
     equals a volume drag over `A·d` with `ξ = ζd`. They differ only in 3D, where a basal drag acts on the
     contact with a substrate. That is a different law, hosted on a surface, FEM-only until a 3D vertex model
     exists; deferred.
   - **A drag relative to a moving medium** (a resolved extracellular fluid) is outside the shared subset; the
     vertex model has no medium velocity.
   - The FEM lowering requires `ζ > 0` or a fixed boundary, as vertex-realization requires friction
     (`MissingFriction`, D113).

   *Requires.* A `DRAG` term (decision 5). Nothing in vcell-core.

   *Recommendation: yes.*

5. **The forms enter the formalism as one new template, a cell-tissue force balance, not as weak forms.**
   - **Why a template** (*inferred*): formalism §1.3.1 puts a variable on one subdomain class (*checked*),
     and the weak-form escape hatch has no `ds`/`dS` measures and no coupled multi-equation forms (§3.6.1,
     *checked*). The force balance needs a velocity continuous across the cells (the shared subset has no
     cleft: a contact is one surface, and the cells cannot slide or separate there) and tension terms on
     interior surfaces. A template with named slots also carries what VCell's lowering emits: the
     framework's "templated equations ... whose solutions this repository verifies" (§6; *checked*).
   - **The template** (quasi-static, overdamped): a vector velocity `v` on a set of volume subdomains (the
     cells), with slots per cell `viscosity`, `bulk_viscosity`, `drag`, `pressure` (a scalar expression,
     uniform on the cell), and per surface `tension` (an expression, on a cell's free membrane or a
     contact). Its residual is
     `∫(2η dev ε(v):ε(φ) + η_b ∇·v ∇·φ + ζ v·φ) − Σ_c P_c∮_{∂Ω_c}φ·n + Σ_Γ ∫_Γ γ_Γ ∇_Γ·φ = 0`.
     The cells' and the membranes' motion is `unknown`, the template's `v` (formalism §1.10). vcell-core's
     lowering fills the slots from the laws: `P_c = K(V0 − V_c)`, `γ` of a free membrane `T_c`, of a contact
     `Λ + T_i + T_j`. Whether this is T6 (Stokes, reserved, §1.4.3) with a compressible option or a new
     number is for the template's own design note.
   - **The IR** (ADR 004): new tagged terms `VISCOUS_STRESS`, `DRAG`, `CELL_PRESSURE`, `SURFACE_TENSION`. Their
     operator invariants: the tension and pressure terms of a closed cell exert no net force or torque; the
     viscous term annihilates rigid motions.
   - **The motion** is the template's: the mesh moves with `v` (Lagrangian at the surfaces), remeshed by ADR
     016 in 2D.

   *Requires.* A design note for the template (its slots, validation, envelope) before code, per the working
   agreements. It is the formalism change; this ADR fixes only what the template must express.

   *Recommendation: yes,* with that note as the next step.

6. **How the forms are verified.**
   - **Contract C, where the measures are shared** (the four conservative laws): energies and generalized
     forces on prescribed deformations, against the vertex forms. The spike above becomes a test: P1 meshes
     whose skeleton is the complex, the energies and forces at its vertices to round-off; with subdivided
     edges, through the chain rule; and in 3D on a polyhedral cell.
   - **Where the vectors come from** (a choice for Jim). vcell-core is private and declares no licence, and
     this repository is public under MIT. The options:
     - (a) a test that reads `../vcell-core/conformance/mechanics/terms.jsonl` when the sibling checkout is
       present and skips otherwise, as `cross_validation/` relies on sibling repositories, together with
       (b) a self-contained test on the same complexes, built in the test, with a small independent vertex
       evaluator (shoelace area, perimeter, their gradients) and finite differences;
     - (c) copying the five vectors into `tests/fixtures/` with their source commit, as `vcell-vertex` did
       (D112). That puts vcell-core content in a public repository.

     *Recommendation: (a) and (b).* (b) runs in CI and is independent of vcell-core's formulas. (c) waits for a
     licence decision.
   - **Manufactured and exact solutions, where the measures are not shared** (the viscous interior,
     friction, the dynamics):
     - a disk relaxing under `AreaElasticity`, whose radial velocity has a closed form
       (`v_r ∝ I₁(r/ℓ)`, `ℓ² = (η_b + 4η/3)/ζ`; *inferred*, to be derived in full in the test) and the
       area's relaxation rate with it;
     - a manufactured velocity with surface and volume residuals for the full template, in the `mms/` suite
       with its order-regression gate;
     - energy decrease at fixed coefficients: `dE/dt = −dissipation`, discretely, for a passive model;
     - the triple-line diagnostic and the `h`-convergence of decision 3.

   *Recommendation: yes.*

## For vcell-core (to send to the vcell-core session, not changed from here)

- vertex-laws.md decision 2's FEM column, replaced by decisions 1–4: the weak forms; "Laplace pressure `σκ`"
  dropped or qualified; the triple line as the forms' natural condition, in quasi-static balance;
  `Friction` as a volume drag against the lab frame, equal to a basal drag in a cross-section.
- If decision 2 is accepted: `MechanicalTerm` gains `Viscosity{η, η_b}` (FEM-only), `MechanicalRole` gains
  `shear_viscosity` and `bulk_viscosity`; the vertex lowering warns and drops it; the FEM lowering refuses a
  resolved cell without it (`MissingInteriorLaw`) and without friction or a fixed boundary.
- A model with an incompressible interior law (M3) and `AreaElasticity` could be warned about, since the law
  is inert there.
- The licence question in decision 6, if Jim wants option (c).

## Staging (after acceptance)

1. Post the accepted answers on #241, and tell the vcell-core session what to change.
2. The template's design note (decision 5), reviewed.
3. `backend/` term builders for the five laws and `Viscosity`, with decision 6's contract C tests (on fixed
   meshes, no motion).
4. The template's solve on a fixed mesh, then on a moving one, with the exact and manufactured checks.

## Not decided here

- The template's number, name and slot grammar (decision 5's note).
- Local surface or bulk elasticity, viscoelastic cytoplasm, active stress fields: M3, FEM-only.
- A substrate drag in 3D; a drag relative to a resolved medium.
- Contacts that separate or slide (a cleft); topology change.
- A regularization of triple-line dynamics, if decision 3's measurements need one.

## References

- **vcell-core** (private), read at `3dac5b6`: `spec/semantics/vertex-laws.md` (decisions 2–5, the FEM column,
  §1), `motion-and-mechanics.md` (§3, §4, §7, Addendum C), `active-t1.md` (§3, decision 2),
  `spec/DECISIONS.md` (D93, D98, D111–D118), `spec/model/biological.yaml` (`MaterialLaw`), and
  `conformance/mechanics/terms.jsonl` (at `100ce8c`). *Use:* the laws, units, conventions and vectors this
  note answers.
- **[Farhadifar 2007]** Farhadifar R, Röper J-C, Aigouy B, Eaton S, Jülicher F. The influence of cell
  mechanics, cell–cell interactions, and proliferation on epithelial packing. *Curr. Biol.* 17:2095–2104
  (2007). *Not read here*; cited by vertex-laws.md. *Use:* the area–perimeter vertex energy the laws take
  their form from.
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A free-
  boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13(11):e1005862 (2017).
  *Not read here*; cited second hand through `docs/modeling/active-protrusion-migration.md`. *Use:*
  precedent for a compressible viscous interior with a substrate drag (decision 2).
- **[Baratta 2023]** Baratta IA, Dean JP, Dokken JS, et al. DOLFINx: the next generation FEniCS problem solving
  environment. Zenodo (2023). doi:10.5281/zenodo.10447666. *Use:* the spikes (DOLFINx 0.10).
- **[Alnæs 2014]** Alnæs MS, Logg A, Ølgaard KB, Rognes ME, Wells GN. Unified form language. *ACM Trans. Math.
  Softw.* 40:1–37 (2014). doi:10.1145/2566630. *Use:* the weak forms of the spikes.
- *Background, not checked against a source:* Kelvin's 2D point-force solution of linear elasticity and
  Stokes flow (the logarithmic singularity of decision 3, whose coefficient the spike reproduced); Korn's
  inequality (decision 2); Neumann's triangle and Herring's condition for a junction of tensions
  (decision 3).

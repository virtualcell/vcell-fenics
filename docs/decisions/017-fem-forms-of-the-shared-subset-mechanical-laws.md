# ADR 017 — FEM forms of vcell-core's shared-subset mechanical laws

**Date:** 2026-11-13
**Status:** Proposed. Answers issue #241's four questions (decisions 1–4) and adds two that the answers
force (decisions 5–6). Revised after an independent review (17 findings, none blocking, 6 major), each
answered below and listed in "Review" at the end.
**Builds on:**
- vcell-core's accepted M4a note, `spec/semantics/vertex-laws.md` (accepted 2026-11-12; D111, D112): the five
  laws, their 3D energies and units, their vertex forms, and the provisional FEM column this note answers.
- vcell-core's `motion-and-mechanics.md` §3 (law kinds; "Newtonian viscosity `η`" among the constitutive
  laws), §4 (the shared subset: "a reduction, not an identity"; a law without a vertex form makes a vertex
  realization `UnsupportedForLowering`) and §7 (M3 holds the FEM-only laws).
- vcell-core's `active-t1.md` decision 2 (D118): `JunctionViscoelasticity` is vertex-only and an FEM
  realization refuses it (`NotInSharedSubset`).
- vcell-core's `vertex-realization.md` decision 2: a cell's measures are its whole volume and whole membrane.
- This repository: formalism §1.3.1 (a variable lives on one subdomain class), §1.5 (the weak-form escape
  hatch), §1.8.4 (`region_size`, `geom.mean_curvature`), §3.3 (what backends choose); ADR 004 (the tagged-term
  IR); ADRs 014–016 (the moving partition and its remesh); `backend/stokes.py`, `fsi.py`, `unknown_motion.py`.

**Would amend:** vcell-core's `vertex-laws.md` decision 2 (its FEM column) and, by decision 2 here,
`motion-and-mechanics.md` §4's rule for laws without a vertex form. Both go through vcell-core PRs that the
vcell-core session owns ("For vcell-core" below). Nothing in this repository changes until the decisions are
accepted.

*Checked* marks what was read in a source (file and line, or section) or run here. *Inferred* marks
reasoning. *Background* marks a published result used without reading it here (cited in References). The
sources in vcell-core were read at its commit `3dac5b6` (the vectors at `100ce8c`); this repository at
`b0132ea`.

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
| `Friction{ζ}` (a cell's volume compartment) | Rayleigh function `½ζ∫|v|² dV` | a drag `ζv` per volume against an ambient medium |

Units (*checked*, same place): energy pN·µm; `K` pN/µm⁵, `K_S` pN/µm³, `σ` and `Λ` pN/µm, `ζ` pN·s/µm⁴. `Λ`
is a contact's tension in excess of `σ_i + σ_j`. Coefficients are expressions; a force is the gradient at
fixed coefficients, i.e. ignoring a coefficient's own dependence on positions. A 2D realization is a
cross-section of depth `d`: `V = A·d`, `S = P·d`, a contact's area `ℓ·d` (vertex-laws.md, Jim's choices;
*checked*). A cell's `V` is its whole volume, whichever volume compartment hosts the law, and its `S` its whole
membrane, patches and contacts included (vertex-realization.md decision 2; *checked*).

**What vcell-fenics solves today** (*checked* in the source named):

| Piece | Where | Envelope |
|---|---|---|
| A surface tension `γ` (a constant or a field) on one closed membrane, as the weak load `−∮_Γ γ∇_Γ·φ ds` | `stokes.py` `solve_incompressible_stokes_surface_tension` (lines 245–256); `fsi.py` `step_force_balance_fsi` | one cell; incompressible Taylor–Hood interior, so the enclosed volume is fixed; a Python driver, not a template |
| A volume drag (`screening`, "substrate friction") | the same drivers; `multiphase.py` (`γ_a v_a`) | a constant per solve |
| A membrane force balance `η v + σ H n = 0` on one closed membrane, with no interior | `unknown_motion.py` (formalism §1.10.8, the weak-form route) | a membrane drag, not a volume one |
| A partition of several compartments moving together, remeshed in 2D | `moving_partition.py`, `partition_remesh.py` (ADRs 014–016) | prescribed motion of one membrane only; chemistry, no mechanics |
| `region_size(<subdomain>)`, re-measured after every move on the moving partition | formalism §1.8.4; `moving_partition.py:238` | refused on ADR 013's single moving mesh (#240) |
| One scalar unknown per region (a real space) | `multi_compartment.py` (line 15, region variables) | used for T5 `region_ode` |

Nothing solves more than one cell mechanically, a tension on an interior surface, a compressible viscous
interior, or a per-cell pressure. Of the five laws, `CorticalTension` and `Friction` exist for one cell, in a
driver; the other three do not.

### A spike: the FEM forms against vcell-core's vectors

Run here (*checked*; the scripts are on the unmerged branch `spike/adr-017`). Every face of each complex in
vcell-core's `conformance/mechanics/terms.jsonl` (square, hexagon, sheared, pair, depth) was fan-triangulated
about its centroid into a P1 DOLFINx mesh, so the mesh's boundary is exactly the complex. On it, the forms of
decision 1 (`K(V−V0)∮φ·n`, `K_S(S−S0)∮∇_Γ·φ`, `σ∮∇_Γ·φ`, `Λ∫_contact ∇_Γ·φ`) were assembled and their
negatives read at the complex's vertices. **Every energy and force in the vectors matched to 2.2·10⁻¹⁶**, the
depth case included (the pair case carries only cortical and interfacial values). Friction's total mobility
`ζ∫dA` matched `Σ ζ a_v` in all four cases that carry it. Its per-vertex values are not comparable
(decision 4).

Why it is exact (*inferred*, algebra): on a straight edge, `∫_e ∇_Γ·φ ds = (φ(b) − φ(a))·t_e`, which is
`∂ℓ_e/∂x` paired with `φ`; on a polygon, `∮φ·n` is `∂A/∂x` paired with `φ`. A P1 mesh measures a polygonal cell
exactly, so its energies are the vertex energies and its forms are their exact gradients. Given that, the
check is close to tautological. Its value is in pinning conventions: depth, signs, a contact edge counted in
both cells. With edges subdivided by mesh nodes, the vertex force is the chain rule of the nodal forces through
the edges' linear interpolation (*inferred*, not run).

## Decisions

1. **Area and surface elasticity are per-cell functionals of a measure, and fit as uniform scalars on the
   cell.**
   - **The forms.** With `n` the cell's outward normal and `φ` a test velocity, the first variations are:
     - `AreaElasticity`: `δE[φ] = K(V − V0)∮_{∂Ω_c} φ·n dS`, a uniform pressure `P_c = K(V0 − V)`;
     - `SurfaceElasticity`: `δE[φ] = K_S(S − S0)∮_{∂Ω_c} ∇_Γ·φ dS`, a uniform tension added to `σ`;
     - `CorticalTension`: `δE[φ] = σ∮_{∂Ω_c} ∇_Γ·φ dS`;
     - `InterfacialTension`: `δE[φ] = Λ∫_{Γ_ij} ∇_Γ·φ dS` on the contact.

     `∂Ω_c` is the cell's whole boundary, contacts included. In a quasi-static step `V`, `S` and the contact's
     area are those of the geometry the step is solved on, which is a time-stepping choice (below). These are
     the exact gradients of the energies, as the vertex forces are.
   - **The measures** (review finding 6): `V_c` is the sum of `region_size` over all of the cell's volume
     compartments (a nucleus included). `S_c` is the sum over its membrane, its patches and its contacts.
     vcell-core's lowering writes these sums, as it writes `Size_` expressions for the vertex realization
     (D113).
   - **Written in the weak, curvature-free form, not as a Laplace pressure `σκ`.**
     - For a closed smooth surface with uniform `σ`, `∫σ∇_Γ·φ` equals the normal traction
       `σ(d−1)H n`, with `H` the signed mean curvature and `∇_Γ·n = (d−1)H` (formalism §1.8.4).
     - In general it also carries the conormal force at a surface's boundary (decision 3) and the Marangoni
       force of a varying `σ`.
     - It needs no pointwise curvature, which a piecewise-linear boundary lacks. `stokes.py` already uses this
       form (*checked*).
   - **Coefficients that read species** (review finding 5):
     - `σ` and `Λ` may be fields on their surface. "Fixed coefficients" then means `σ` held at material
       points, and the form is `∫σ∇_Γ·φ`. A species-dependent `σ` thereby gives the Marangoni force the
       resolved model should have; the vertex model reads the per-cell value (motion §4's "active tension"
       row; *checked*).
     - `K`, `V0`, `K_S` and `S0` multiply a per-cell measure, so each must have one value per cell. Recommended:
       the FEM lowering accepts parameters, `t`, measures and well-mixed (region) variables there, and refuses
       a resolved field by name until a per-cell mean is asked for.
   - **Why per-cell functionals are acceptable** (*inferred*). They are the energies the vertex model can
     express:
     - a global `½K(V − V0)²` is a cell whose pressure equilibrates instantly;
     - a global `½K_S(S − S0)²` is a cortex whose tension is uniform: a fluid cortex with an area reservoir.

     A *local* surface or bulk elasticity (a strain energy over a reference configuration) is a different law,
     with memory, which the vertex model cannot express. It is M3's, FEM-only. This matches the research note's
     warning, quoted in vertex-laws.md §1 (cited second hand), that a cortex "does not in general reduce to
     `(P − P₀)²`".
   - **The 2D cross-section needs no new FEM coefficients** (*inferred*). With `V = A·d` and `S = P·d`, the 2D
     weak form per unit depth is the 3D one with the 3D coefficients unchanged. `d` appears only inside the
     lowered measures (`K(V0 − d·A)`).
   - **Positive tension is required** (review finding 1; *inferred*, capillary stability):
     - **The requirement:** a resolved surface with tension `γ ≤ 0` is unstable at all wavelengths, fastest at
       the grid scale; a straight vertex edge cannot wrinkle. The FEM realization therefore requires every
       surface's current tension to be positive at every step: `T_c = σ_c + K_S,c(S_c − S0_c)` on a free
       membrane and `Λ_ij + T_i + T_j` on a contact.
     - **Otherwise:** the solver stops by name (`NonPositiveTension`). Going further needs bending, which is
       M3's.
     - **What it rules out:** vertex models run in this regime (an effective tension below zero, vertex-laws.md
       decision 2's `τ_c` and decision 5's `p_eff`; *checked*). It is a mismatch of the reduction.
   - **Time stepping** (review finding 3; *inferred*; the scheme is the backend's, formalism §3.3):
     - **The capillary limit:** moving the geometry explicitly under a surface tension imposes a step limit of
       order `ηh/γ`. It shrinks with `h` and is usually the binding one, more so than the limits from `K` and
       `K_S`.
     - **At corners and triple points:** the mobility there grows like `ln(1/h)` (decision 3), which tightens
       the limit further.
     - **The remedy:** a semi-implicit treatment, with the surface's new position inside the tension form
       ([Dziuk 1991], [Bänsch 2001]; *background*).
     - **The per-cell scalars:** they can be lagged, which keeps the system sparse. Or each cell gains one
       scalar unknown in a real space, as region variables already do here (*checked*).
     - The template's note (decision 5) chooses among these.

   *Requires.*
   - Formalism: the per-cell scalars are expressions of `region_size`, which exists. Each cell must be its
     own subdomain. vcell-core's unrolling already names each cell's copy (`cyt__c1`; D93), so per-region
     sizes (#239) are not needed (*inferred*).
   - `region_size` re-measured on a mechanically moving mesh, as the moving partition already does.
   - IR: tagged terms `CELL_PRESSURE` and `SURFACE_TENSION` (decision 5).

   *Recommendation: yes,* as the forms above. The issue's column should state the weak form.

2. **The interior constitutive law: a compressible Newtonian viscosity, an M3 law; no default. A vertex
   realization drops it with a warning.**
   - **Why one is needed** (*inferred*). The vertex form has no interior. An FEM force balance with only the
     five laws, `∫ζv·φ = −δE[φ]` for all `φ ∈ H¹`, makes `ζv` equal a surface functional, which is not in
     `L²`: there is no `H¹` solution. An interior viscous stress makes the problem well posed. Coercivity needs
     a shear viscosity `η > 0` and either `ζ > 0` or a fixed boundary, by Korn's inequality ([Brenner 2008],
     *background*). This holds with `η_b = 0` in planar flow, since the 3D deviator gives
     `|dev ε|² ≥ ⅓|ε|²` (*inferred*, confirmed by the review).
   - **Why compressible** (*inferred*). In the vertex model a cell's volume changes freely under
     `AreaElasticity`. An incompressible interior (today's Stokes drivers) keeps `∮v·n = 0`, so `V` never
     changes and `AreaElasticity` is inert: its pressure is absorbed into the incompressibility multiplier. A
     compressible viscous interior with a substrate drag also has published precedent: the motile-cell actin
     network of [Nickaeen 2017], cited second hand through `docs/modeling/active-protrusion-migration.md`.
     That model's viscous term is `ηΔU` and its boundary is not material, so it supports only "compressible,
     with a drag", not this stress.
   - **The law:** `Viscosity{η, η_b}` on a cell's volume compartment.
     - The stress is `2η dev ε(v) + η_b (∇·v) I`, with the roles `shear_viscosity` and `bulk_viscosity`, both
       in pN·s/µm² (= Pa·s). Admissible values: `η > 0`, `η_b ≥ 0`.
     - Its Rayleigh function is `½∫(2η|dev ε|² + η_b(∇·v)²)`.
     - In a 2D cross-section the flow is planar (`v_z = 0`, no dependence on `z`), and `dev` is the 3D
       deviator (*inferred*).
   - **Where it lives: M3's constitutive laws, not M4a's menu** (review finding 2):
     - motion §3 already lists "Newtonian viscosity `η`" as a constitutive law, and vertex-laws.md decision 1
       keeps the FEM-only laws M3's (*checked*). `MechanicalTerm` already has six variants, with
       `JunctionViscoelasticity` (D118; `biological.yaml`; *checked*).
     - vcell-core's `MaterialLaw` is not the place either: it states a material *velocity* (D98; *checked*).
   - **What a vertex realization does with it.** motion §4 says a law without a vertex form makes a vertex
     realization `UnsupportedForLowering` (*checked*). Applied here, a model that FEM can run (it needs
     `Viscosity`) could never be realized as vertex, so the shared subset would be empty. The options:
     - **(a) Drop it with a warning,** amending motion §4's rule for laws that are dissipation only. They change
       rates, not energies or generalized forces, which are what contract C compares. The vertex model's own
       dissipation, `ζ a_v`, stands in for it in the reduction. `JunctionViscoelasticity` is refused by FEM for
       the opposite reason: it carries an energy and a state.
     - **(b) Carry the interior law on the `resolved` realization** (the application layer), as the vertex
       realization carries `depth` and thresholds. The model stays shareable and §4's rule stands. Against it:
       Addendum B question 1 puts constitutive properties in the model (*checked*), and a cytoplasm's viscosity
       is a measurable property, not a discretization parameter.
     - **(c) Refuse,** as §4 says. Shared models are then impossible.

     *Recommendation: (a).* Jim to choose.
   - **No default.** A resolved cell whose model states no interior law is refused by name
     (`MissingInteriorLaw`). This follows vertex-realization's `MissingFriction` (D113; *checked*) and the
     cell-mechanics framework's rule that unresolved choices be "diagnosed rather than silently defaulted"
     (§6; *checked*).
   - **Not chosen:**
     - an elastic interior: it needs a reference configuration and remapped history through remeshing
       (framework §3), and the vertex model has neither;
     - an incompressible interior: it disables `AreaElasticity`.

   *Requires.*
   - Formalism: a vector velocity on several volume subdomains (decision 5), and viscosity and drag slots per
     subdomain.
   - IR: `VISCOUS_STRESS` and `DRAG` terms.
   - vcell-core: the M3 law and its two roles, the vertex lowering's warning, and the FEM lowering's
     `MissingInteriorLaw`.

   *Recommendation: yes,* with (a) above.

3. **A triple line needs no law of its own. The force is the natural condition of the tension forms, and in
   the continuum the line sits in quasi-static balance.**
   - **The force** (*inferred*, integration by parts on an open surface, confirmed by the review):
     `∫_Γ γ∇_Γ·φ dS = ∫_Γ γ(d−1)H n·φ dS − ∫_Γ ∇_Γγ·φ dS + ∮_{∂Γ} γ ν·φ dl`, with `ν` the outward conormal.
     - Summed over the surfaces meeting at a line, the line terms of `δE` are `Σ_k γ_k ν_k·φ`. The force on the
       line is therefore `−Σ_k γ_k ν_k = Σ_k γ_k t_k`, with `t_k` the unit tangent pointing from the line into
       surface `k`. That is the vertex model's `Σ_e γ_e t_e`, with `t_e` pointing from the vertex along edge `e`.
     - So the forms of decision 1 apply the triple-line force with no further input, and contract C holds at
       triple points. In the spike's pair case the contact edge's endpoints are triple points (*checked*).
   - **The tensions** (*inferred*, from the definition of `Λ`): a cell's free surface carries `T_c`, and a
     contact between cells `i` and `j` carries `Λ_ij + T_i + T_j`.
   - **Corners are the same thing** (review finding 4). A corner of a polygonal free membrane is an
     unbalanced point force `2T sin(θ/2)` (`θ` the turning angle), like an unbalanced triple point.
   - **The dynamics.** An unbalanced triple point or corner is a point force in 2D (a line force in 3D) on a
     viscous continuum, and the velocity at it is logarithmically infinite (the 2D Kelvin solution,
     [Thomson 1848], *background*).
     - **Spike** (*checked*, confirmed by the review's rerun): an open tense segment ending inside a
       compressible viscous bulk with drag (`ζ = γ = 1`, P1, `h` from 1/4 to 1/128).
       - The endpoint's velocity grew by `0.0735` per halving of `h`. The spike's stress was
         `2ηε + λ(∇·v)I` with `η = λ = 1` (in decision 2's terms `λ = η_b − 2η/3`), and the Kelvin coefficient
         `(λ + 3η)/(4πη(λ + 2η))·ln 2` is `0.0735` there.
       - Away from the endpoint the velocity converged.
       - A point drag `μ = 1` at the endpoint did not cure it: the velocity crept toward `F/μ = 1` as
         `0.21, 0.30, 0.36, 0.42` over `h = 1/4 … 1/256`, because the bulk's compliance at a point grows like
         `ln(1/h)`.
   - **So the continuum limit is quasi-static balance** (*inferred*).
     - An unbalanced line moves infinitely fast compared with the bulk, so in the limit it is always at
       Neumann's triangle `Σγ_k t_k = 0`: Herring's condition ([Herring 1951], *background*). A free cell's
       corners round at once.
     - This is the resolved model's physics and differs from the vertex model's in two ways:
       - a vertex has the finite mobility `ζ a_v`;
       - its edges stay straight, while FEM contacts curve under pressure differences.

       So FEM and vertex **equilibria differ** as well as their transients. Both are mismatches of the
       reduction (motion §4's "relaxation-time mismatch"; *checked*). Neither affects contract C, which
       compares energies and generalized forces on prescribed deformations.
   - **When balance exists** (*inferred*, the triangle inequality, with `T_i, T_j > 0` from decision 1):
     - at a free triple line (cells `i` and `j` and the medium), iff `−2 min(T_i, T_j) < Λ_ij < 0`;
     - at a junction of three cells `i`, `j`, `k`, iff each contact's tension is below the sum of the other two,
       i.e. `Λ_ij < Λ_jk + Λ_ki + 2T_k` and its two permutations, which positive `Λ`s usually satisfy.

     Outside these conditions a contact peels open or spreads. With `K_S > 0` the tensions change with `S`, so
     the system may rebalance; if it does not, a contact's length or a cell's area collapses. That is a
     topology change, which FEM refuses (motion §4; *checked*), and the solver stops by name.
   - **What the implementation does:**
     - the natural condition only, with no constraint and no junction drag;
     - a diagnostic per triple point and corner: the imbalance `|Σγ_k t_k|`;
     - convergence tests that expect what the spike predicts (*inferred*):
       - equilibria, the energy minimizers on the mesh, converge algebraically in `h`;
       - transients in contact lengths and areas converge only like `1/ln(1/h)`, so they are measured, not
         gated on an order;
       - junction velocities do not converge.

     If equilibria do not converge, a regularization is a later decision, taken with evidence.

   *Requires.* Nothing of the model and no new formalism. The backend needs the triple points of the mesh's
   interface skeleton. The partition remesher's skeleton shares junction vertices
   (`partition_remesh.py` lines 1–5, 67; *checked*), so the triple points are derivable from it.

   *Recommendation: yes.*

4. **`Friction` is a volume drag against the lab frame, and stays one.**
   - **The form:** `∫_{Ω_c} ζ v·φ dV`, the drag `ζv` per volume, against the substrate at rest. It is the
     existing `screening` term (`stokes.py`, `multiphase.py`; *checked*), which there also removes the rigid
     null modes.
   - **It matches the vertex form's measure** (*inferred*, and the spike). The vertex model shares each cell's
     area among its vertices (`a_v`), so a uniform translation's drag is `ζΣA_c`, which is `ζ∫dA` in FEM per
     unit depth (*checked* in all four complexes).
     - Per-vertex mobilities differ: FEM's mass matrix shares a cell's area among its mesh nodes, interior ones
       included.
     - So friction is not part of the generalized-force check; its check is dynamic (decision 6).
   - **Volume or substrate are the same in a cross-section** (*inferred*). A basal drag `ξ` over the area `A`
     equals a volume drag over `A·d` with `ξ = ζd`. They differ only in 3D, where a basal drag acts on the
     contact with a substrate. That is a different law, hosted on a surface, FEM-only until a 3D vertex model
     exists; deferred.
   - **A drag relative to a moving medium** (a resolved extracellular fluid) is outside the shared subset; the
     vertex model has no medium velocity.
   - The FEM lowering requires `ζ > 0` or a fixed boundary, as vertex-realization requires friction
     (`MissingFriction`, D113).

   *Requires.* A `DRAG` term (decision 5). In vcell-core, only the doc wording "against an ambient medium".

   *Recommendation: yes.*

5. **The forms enter the formalism as one new template, a cell-tissue force balance.**
   - **Why a template** (*inferred*):
     - Formalism §1.3.1 puts a variable on one subdomain class (*checked*). The force balance needs one
       velocity, continuous across all the cells: the shared subset has no cleft, so a contact is one surface,
       and the cells can neither slide nor separate there.
     - The weak-form language defines `ds(·)`, `dS(·)` and coupled forms (§1.5.2–§1.5.3), but the backend
       defers them (§3.6.1; *checked*).
     - Above all, a template with named slots is what VCell's lowering emits: the cell-mechanics framework's
       "templated equations ... whose solutions this repository verifies" (§6; *checked*).
   - **The template** is quasi-static and overdamped:
     - **Variable:** a vector velocity `v` on a set of volume subdomains, the cells.
     - **Slots per cell:** `viscosity`, `bulk_viscosity`, `drag`, and `pressure` (a scalar expression, uniform
       on the cell).
     - **Slot per surface:** `tension` (an expression), on a cell's free membrane or a contact.
     - **Residual:**
       `∫(2η dev ε(v):ε(φ) + η_b ∇·v ∇·φ + ζ v·φ) − Σ_c P_c∮_{∂Ω_c}φ·n + Σ_Γ ∫_Γ γ_Γ ∇_Γ·φ = 0`.
       Its sign agrees with `δE` (confirmed by the review).
     - **Motion:** the cells' and the membranes' motion is `unknown`, the template's `v` (formalism §1.10).
     - **How vcell-core's lowering fills it:** `P_c = K(V0 − V_c)`; a free membrane's `γ` is `T_c`; a contact's
       is `Λ + T_i + T_j`.
     - Whether this is T6 (Stokes, reserved, §1.4.3) with a compressible option, or a new number, is for the
       template's own note.
   - **An alternative considered** (review finding 8): one cell class with many regions, the velocity on that
     class, and per-region sizes (#239). It fits §1.3.1, but it needs per-region coefficients as well as sizes,
     whereas vcell-core's unrolling already gives each cell its own subdomains.
   - **The IR** (ADR 004) gains tagged terms `VISCOUS_STRESS`, `DRAG`, `CELL_PRESSURE` and `SURFACE_TENSION`.
     Their operator invariants:
     - the tension and pressure terms of a closed cell exert no net force or torque;
     - the viscous term annihilates rigid motions.
   - **The motion** is the template's: the mesh moves with `v` (Lagrangian at the surfaces) and is remeshed by
     ADR 016 in 2D. There is no 3D remesher yet.

   *Requires.* A design note for the template (its slots, validation and envelope) before code, per the
   working agreements. That note also owns chemistry on the moving cells ("Not decided here"). This ADR fixes
   only what the template must express.

   *Recommendation: yes,* with that note as the next step.

6. **How the forms are verified.**
   - **Contract C, where the measures are shared** (the four conservative laws). The check is energies and
     generalized forces on prescribed deformations, against the vertex forms, in 2D: the spike above as a test,
     and with subdivided edges through the chain rule. 3D waits for a 3D vertex convention (vertex-laws.md §4
     leaves 3D vertex models undecided).
   - **Where the vectors come from** (a choice for Jim). vcell-core is private and declares no licence; this
     repository is public under MIT. The options:
     - (a) a test that reads `../vcell-core/conformance/mechanics/terms.jsonl` when the sibling checkout is
       present and skips otherwise, as `cross_validation/` relies on sibling repositories;
     - (b) a self-contained test on the same complexes, built in the test, with a small independent vertex
       evaluator (shoelace area, perimeter, their gradients) and finite differences;
     - (c) copying the five vectors into `tests/fixtures/` with their source commit, as `vcell-vertex` did
       (D112). That puts vcell-core content in a public repository.

     *Recommendation: (a) and (b).* (b) runs in CI and is independent of vcell-core's formulas. (c) waits for a
     licence decision.
   - **Exact and manufactured solutions, where the measures are not shared** (the viscous interior, friction,
     the dynamics):
     - a disk relaxing under `AreaElasticity`: its radial velocity has a closed form, `v_r ∝ I₁(r/ℓ)` with
       `ℓ² = (η_b + 4η/3)/ζ` (*inferred*, confirmed by the review), and so does the area's relaxation rate;
     - a manufactured velocity with surface and volume residuals for the full template, in the `mms/` suite and
       its order-regression gate (on smooth closed surfaces, with no corners or triple points);
     - **energy and dissipation.** For a passive model, at fixed coefficients, the solved `v` satisfies
       `δE[v] = −2R(v)`, with `R` the Rayleigh function (friction plus viscosity). This identity is exact for
       the discrete forms at every step. The energy's decrease from one step to the next holds only to `O(Δt)`
       with explicit motion;
     - the triple-point diagnostic and decision 3's convergence expectations.
   - **Periodic boxes:** the vertex realization's (vertex-realization decision 5) are out of this template's
     first scope. A periodic FEM mesh is later work.

   *Recommendation: yes.*

## For vcell-core (to send to the vcell-core session, not changed from here)

- vertex-laws.md decision 2's FEM column, replaced by decisions 1–4. That means the weak forms; "Laplace
  pressure `σκ`" dropped or qualified; the positivity of tensions; the triple line as the forms' natural
  condition, in quasi-static balance; and `Friction` as a volume drag against the lab frame, equal to a basal
  drag in a cross-section.
- motion-and-mechanics.md §4: its FEM column ("a surface stress `σ` (Laplace pressure `σH`)") in the same
  terms. If decision 2(a) is accepted, its rule for laws without a vertex form gains the dissipation-only
  exception.
- `biological.yaml`: the `MechanicalTerm` and `Friction` docs ("against an ambient medium", "provisional FEM
  form"), generated in three languages.
- If decision 2 is accepted:
  - the M3 law `Viscosity{η, η_b}` with roles `shear_viscosity` and `bulk_viscosity`;
  - the vertex lowering's warning;
  - the FEM lowering's refusals: `MissingInteriorLaw`, a cell with neither friction nor a fixed boundary, and a
    resolved field in `K`, `V0`, `K_S` or `S0`.
- `MaterialVelocity.Solved` (reserved, `biological.yaml`) un-reserved for a resolved cell.
- A warning for a model with an incompressible interior law (M3) and `AreaElasticity`, since the law is inert
  there.
- The licence question of decision 6, if Jim wants option (c).

## Staging (after acceptance)

1. Post the accepted answers on #241, and tell the vcell-core session what to change.
2. The template's design note (decision 5), reviewed.
3. `backend/` term builders for the five laws and `Viscosity`, with decision 6's contract C tests on fixed
   meshes.
4. The template's solve on a fixed mesh, then on a moving one, with the exact and manufactured checks.

## Not decided here

- The template's number, name and slot grammar (decision 5's note).
- **Chemistry on the moving cells.** Species ride the solved, compressible, tangentially flowing `v`: bulk
  species dilute by `c∇·v`, membrane species by the mandatory `ρ∇_Γ·v_Γ`. The moving partition must carry
  several solved membranes, not one prescribed one. This is for the template's note.
- How a contact's three compartments (the patch of `i`, the patch of `j`, the junction's own compartment) map
  onto one FEM surface.
- A per-cell mean of a resolved field in `K`, `V0`, `K_S` or `S0`.
- Local surface or bulk elasticity, viscoelastic cytoplasm, active stress fields, bending: M3, FEM-only.
- 3D: a contract C convention, quadruple points, a remesher.
- Periodic boxes.
- A substrate drag in 3D; a drag relative to a resolved medium.
- Contacts that separate or slide (a cleft); topology change.
- A regularization of triple-point dynamics, if decision 3's measurements need one.

## Review

An independent reviewer read the first draft and its sources, and reran the spikes. It confirmed the first
variations and their signs, the integration-by-parts identity, the Kelvin coefficient, the stress and its
dissipation, the `I₁` solution, the units, the no-`H¹` argument, and the spikes' results. Its 17 findings, and
how each was answered:

| # | Severity | Finding | Answer |
|---|---|---|---|
| 1 | major | Negative tension makes FEM ill posed; no positivity condition | Decision 1: tensions must stay positive, `NonPositiveTension`; a mismatch of the reduction |
| 2 | major | "Drop with a warning" contradicts motion §4's rule; the menu already has six variants; the realization-level option is missing | Decision 2: an M3 law, not a menu variant; options (a)–(c) with §4's amendment named |
| 3 | major | The time-step limit comes from the tensions, not `K` and `K_S` | Decision 1, "Time stepping": the capillary limit, the corner mobility, the semi-implicit remedy |
| 4 | major | Corners behave like triple points; equilibria differ; convergence is logarithmic | Decision 3: corners, differing equilibria, convergence expectations; "equilibria" dropped from the contract C sentence |
| 5 | major | Species-dependent coefficients, the variation convention, chemistry and `Solved` | Decision 1, "Coefficients"; "Not decided here"; "For vcell-core" |
| 6 | major | A cell's measures with several compartments | Decision 1, "The measures": sums written by vcell-core's lowering |
| 7 | minor | The three-cell balance condition; "both end" too strong | Decision 3: added and softened |
| 8 | minor | The formalism defines `ds`/`dS`; the backend defers them; one class with many regions not weighed | Decision 5: corrected; alternative weighed |
| 9 | minor | "At a fixed measure, as the vertex forms are" is wrong | Decision 1: reworded |
| 10 | minor | The sign of the line force, `t_e` undefined | Decision 3: `−Σγν = Σγt`, `t` defined |
| 11 | minor | Rayleigh-function convention; which discrete identity | Context table and decision 6 |
| 12 | minor | Three *checked* claims overstated | Decision 3 (skeleton: derivable), context (tension may be a field), real spaces *checked* |
| 13 | minor | "For vcell-core" incomplete | motion §4's column, the `biological.yaml` docs, `Solved` added |
| 14 | minor | 3D and periodic boxes thin | Decision 6 and "Not decided here" |
| 15 | nit | "Total curvature" | Decision 1: the formalism's `(d−1)H` |
| 16 | nit | Background results without citations | References |
| 17 | nit | Nickaeen precedent stronger than it reads | Decision 2: narrowed |

## References

- **vcell-core** (private), read at `3dac5b6`. *Use:* the laws, units, conventions and vectors this note
  answers.
  - `spec/semantics/vertex-laws.md`: decisions 1–5, the FEM column, §1, §4.
  - `motion-and-mechanics.md`: §3, §4, §7, Addendum B question 1, Addendum C.
  - `active-t1.md`: §3, decision 2.
  - `vertex-realization.md`: decisions 2 and 5.
  - `spec/DECISIONS.md`: D93, D98, D111–D118.
  - `spec/model/biological.yaml`: `MaterialVelocity`, `MechanicalTerm`.
  - `conformance/mechanics/terms.jsonl`, at `100ce8c`.
- **[Farhadifar 2007]** Farhadifar R, Röper J-C, Aigouy B, Eaton S, Jülicher F. The influence of cell
  mechanics, cell–cell interactions, and proliferation on epithelial packing. *Curr. Biol.* 17:2095–2104
  (2007).
  - *Not read here*; cited by vertex-laws.md.
  - *Use:* the area–perimeter vertex energy the laws take their form from.
- **[Nickaeen 2017]** Nickaeen M, Novak IL, Pulford S, Rumack A, Brandon J, Slepchenko BM, Mogilner A. A free-
  boundary model of a motile cell explains turning behavior. *PLoS Comput. Biol.* 13(11):e1005862 (2017).
  - *Not read here*; cited second hand through `docs/modeling/active-protrusion-migration.md`.
  - *Use:* precedent for a compressible viscous interior with a substrate drag (decision 2).
- **[Thomson 1848]** Thomson W (Lord Kelvin). Note on the integration of the equations of equilibrium of an
  elastic solid. *Cambridge and Dublin Math. J.* 3:87–89 (1848).
  - *Not read here.*
  - *Use:* the point-force solution, whose 2D form gives decision 3's logarithmic singularity. The spike
    reproduced its coefficient.
- **[Brenner 2008]** Brenner SC, Scott LR. *The Mathematical Theory of Finite Element Methods*, 3rd ed.
  Springer (2008), ch. 11.
  - *Not read here.*
  - *Use:* Korn's inequality, for the coercivity of decision 2.
- **[Herring 1951]** Herring C. Surface tension as a motivation for sintering. In: Kingston WE (ed.), *The
  Physics of Powder Metallurgy*, McGraw-Hill (1951), pp. 143–179.
  - *Not read here.*
  - *Use:* the force balance of tensions at a triple junction (Neumann's triangle), decision 3.
- **[Dziuk 1991]** Dziuk G. An algorithm for evolutionary surfaces. *Numer. Math.* 58:603–611 (1991).
  - *Not read here.*
  - *Use:* the semi-implicit treatment of surface tension (decision 1, time stepping).
- **[Bänsch 2001]** Bänsch E. Finite element discretization of the Navier–Stokes equations with a free
  capillary surface. *Numer. Math.* 88:203–235 (2001).
  - *Not read here.*
  - *Use:* the same, coupled to a viscous bulk.
- **[Baratta 2023]** Baratta IA, Dean JP, Dokken JS, et al. DOLFINx: the next generation FEniCS problem solving
  environment. Zenodo (2023). doi:10.5281/zenodo.10447666.
  - *Use:* the spikes (DOLFINx 0.10).
- **[Alnæs 2014]** Alnæs MS, Logg A, Ølgaard KB, Rognes ME, Wells GN. Unified form language. *ACM Trans. Math.
  Softw.* 40:1–37 (2014). doi:10.1145/2566630.
  - *Use:* the spikes' weak forms.

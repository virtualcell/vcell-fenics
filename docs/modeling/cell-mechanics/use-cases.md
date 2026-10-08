# Worked use cases: three published models against the framework

**Status:** analysis, 2026-10-08 (PR #213). Three models from `docs/papers/` (gitignored) are
restated in the [modeling framework](modeling-framework.md)'s vocabulary and mapped onto the
[declarative formalism](../declarative-formalism.md) and the backend as they are today, to find
what is missing and where the framework's assumptions do not fit before the plan is hardened. The
extractions were made from the main texts only; the supplements (2017 S1 Appendix, the 2019/2022
supplemental methods and COMSOL file) are not in the repository, and the gaps they leave are named.

| Use case | Paper | What it exercises |
|---|---|---|
| **UC-A** Motile cell | Nickaeen, Novak, Pulford, Rumack, Brandon, Slepchenko, Mogilner, *PLoS Comput Biol* 2017 | a free boundary whose velocity is *derived* from bulk mechanics, a compressible single-phase active gel, substrate drag, emergent symmetry breaking |
| **UC-B** Prescribed moving domain | Novak & Slepchenko, *J Comput Phys* 2014 (the MovingBoundary solver's algorithm) | VCell's own moving-boundary semantics: lab-frame species, Rankine–Hugoniot front condition, exact conservation; the comparison baseline |
| **UC-C** Endocytic actin patch | Nickaeen et al., *MBoC* 2019 and 2022 | nanometre-scale mechanics, density-dependent rheology, ~15-species kinetics in a domain whose membrane invagination moves, surface species on ring sub-regions, a force functional driving the motion. The authors solved it axisymmetric with a rigid invagination; both were tractability approximations, not the model (see UC-C) |

The short version: **every one of the three needs something the formalism cannot say or the backend
cannot run today, and two of the three (A and C) need the same missing thing** — a compressible,
single-phase, pressure-free active gel with its velocity solved on a moving *volume* and a boundary
motion law derived from it. The framework's §4 "active viscous mixture" family (two phases, mixture
incompressibility, a pressure) is not what either paper does. One reading rule throughout: a paper's
*numerical* choices (an axisymmetric reduction, a rigid boundary, a fitted resistance curve) are
recorded as approximations with the model they stand in for, so the framework targets the model and
the papers' numbers become acceptance tests of the approximated case. Details follow.

## 1. The three models in the framework's vocabulary

The framework (§2) asks each model to declare the boundary velocity `v_b`, the material velocity of
each phase `v_a`, the carrier of each species `v_c`, the mesh velocity `w`, and the measures.

### UC-A — Nickaeen–Novak–Mogilner 2017

- **Domain.** 2D, the cell interior `Ω(t)` in the lab frame; the edge `∂Ω(t)` is a closed curve; no
  substrate domain, no membrane field, no exterior species.
- **Unknowns.** Actin-network velocity `u` (bulk, vector); myosin areal density `m` (bulk, scalar).
  Nothing else: no pressure, no actin density, no membrane variable.
- **Balances.** Force balance, quasi-static and **compressible with no pressure**:
  `α Δu + β ∇m − u = 0` (vector Laplacian, isotropic active stress `β m I`, uniform substrate drag).
  Myosin: `∂t m = ∇·(d_eff ∇m − u_eff m)` with crowding cut-offs `u_eff = u (1 − m/m_max,u)⁺`,
  `d_eff = 1 − m/m_max,d`; no turnover, total myosin conserved.
- **Velocities.** `v_a = u` (the network). The myosin carrier is `v_c = u_eff`, a *lab-frame* velocity
  (the paper is Eulerian). The boundary velocity is **derived**: `v_b = v_p n + u|∂Ω`, with the
  protrusion speed `v_p = v₀ a₀/a − k(a − a₀(a₀/a)ⁿ)` (zero-stress variant, n = 2) or
  `v_p = v₀ a₀ /(a (1 + m|∂Ω)) − k(a − a₀)` (zero-velocity variant), where `a(t) = |Ω(t)|` is the
  current area. So `v_b` depends on a **global functional** of the geometry, on the **boundary trace
  of a bulk field** (ZV), on the **normal**, and (ZS) on the **tangential** network velocity.
- **Boundary conditions.** ZV: `u = 0` on `∂Ω`; ZS: zero traction `n·(α∇u + β m I) = 0`. Myosin:
  zero flux relative to the moving boundary, `n·(−d_eff∇m + (u_eff − v_b) m) = 0`.
- **Measures.** Per current area; `∇·u ≠ 0`, so myosin is compressed by the flow wherever it
  converges — the compressible-carrier dilution the framework's §2 calls out.
- **Mesh velocity.** Free: ZV needs `w·n = v_p` and any tangential choice; ZS needs `w·n = (v_p + u·n)`.
  The paper's front tracker moved marker points with the full `v_b`.
- **What membrane tension is.** Not a force. It is folded into the kinematic area-restoring term of
  `v_p`; the steady area deviates from `a₀`. A hard area constraint would be a different model.

### UC-B — Novak–Slepchenko 2014

- **Domain.** 2D (tested), `Ω(t)` with a piecewise-linear front whose position is **known at all
  times**; the exterior carries no unknowns.
- **Unknowns.** One scalar bulk concentration `u` (several species are independent).
- **Balance.** `∂t u = ∇·(D∇u − v u) + R`, with `v` a **lab-frame species velocity**, independent of
  the front velocity `v_b`; `D`, `v` time-independent in the paper; `R` present in the formulation but
  never tested.
- **Front condition.** Rankine–Hugoniot for an impermeable membrane: `(−D∇u + (v − v_b) u)·n = 0`.
- **Velocities.** `v_c = v` (default `0`: the cytoplasm is at rest and the front *sweeps* it);
  `v_b` prescribed; `w` does not exist (fixed grid, cut-cell Voronoi control volumes). Carrying the
  cytoplasm with the cell is modelled by setting `v = v_b`, not by changing frame.
- **Measures.** Per current area; conservation is exact by construction (mass-based time term,
  mass-preserving natural-neighbour remap when a grid node enters or leaves `Ω`).
- **Not in the paper.** Membrane species, species-dependent front velocity, topology change, 3D, any
  reaction test.

### UC-C — Nickaeen et al. 2019 / 2022

- **Domain.** A cylinder of cytoplasm (sub-micron) under a flat plasma membrane, with a membrane
  invagination (2019: spherocylinder of radius 30 nm; 2022: a head–neck "flask" with neck radius 3–10
  nm) protruding into it. **Two approximations the authors made for tractability, not as modeling
  statements:** the problem was solved **axisymmetric in (r, z)**, and the invagination was treated
  as a **rigid body** of fixed shape that translates along the axis against a prescribed turgor
  resistance (membrane mechanics is listed by the authors as future work). The *model* is a 3D
  cytoplasmic gel pushing a deformable membrane invagination against turgor pressure, membrane
  tension and the coat's elasticity; the papers' numbers are what the approximations produced. Under
  the approximations the flat membrane and the far field stay put, so `Γ` grows and `Ω` shrinks.
- **Unknowns.** Network velocity `v` (bulk, vector, compressible); ~8 bulk species of the Berro 2010
  patch kinetics (new, aged and cofilin-bound filament subunits, active and capped barbed ends,
  pointed ends, bound and active Arp2/3) with the polymerized density `ρ` as their sum; 3 membrane
  species on 20-nm NPF **ring sub-regions** of `Γ` (ODEs, no surface transport); two scalars, the
  axial force on the invagination `f_z(t)` and its velocity `u(t)`.
- **Balances.** Force balance `∇·(η(ρ, L)(∇v + ∇vᵀ)) − ∇σ_a(ρ) = 0`, **single-phase, compressible,
  no pressure, no cytosolic drag** (the active stress "plays the role of pressure"); species
  `∂t[X] = −∇·([X] v) + R_X` in conservative form (pure advection, no diffusion, except a local
  diffusion added near the rings so a non-zero influx is well posed); ring ODEs with bulk traces in
  their rates.
- **Constitutive.** `σ_a = κ_a ρ²`; `η = κ_v ρ (1/N + ρ δ² L)` with `N`, `L` ratios of species — **η
  vanishes where ρ does** (far field; everywhere at t = 0 bar a seed). Rate constants carry crowding
  and load factors in `(1 − ρ/ρ_max)`.
- **Boundary conditions.** No-slip `v = u e_z` on the invagination, `v = 0` on the flat membrane
  (a velocity jump at the junction), zero stress far away, axis conditions on r = 0; zero relative
  species flux (automatic under no-slip); the Arp2/3 influx as a flux condition on the rings.
- **Motion law (as approximated).** `u(t) = μ ⟨f_z(t) − f_c(t)⟩₊` with `f_z = ∫_S e_z·σ·n ds` over
  the invagination and `f_c(t)` a prescribed (logistic-in-time) turgor resistance; a rigid translation
  standing in for a shape equation. **The modeling intent** is a membrane force balance on the
  invagination — the gel's traction `σ·n` against turgor pressure, membrane tension/bending and the
  clathrin coat — with the shape solved; the 2022 paper's time-varying resistance is a hand-fitted
  surrogate for the neck narrowing that such a balance would produce.
- **Velocities.** `v_a = v_c = v` for every bulk species (carried by the network); `v_b = u e_z` on
  the invagination, `0` elsewhere; `w` free in the interior. The rings are *geometric* regions that
  translate rigidly, so `∇_Γ·v_Γ = 0` on them and no surface dilution arises — but only because they
  are rigid; a ring treated as a material patch on a stretching neck would need the `ρ ∇_Γ·v_Γ`
  term.

## 2. Mapping to the formalism and the backend

Legend for the "today" columns: ✓ exists · ◐ partly / with a workaround · ✗ missing. "CLI" means
reachable through `vcell_fenics.cli`; a Python driver is not.

### UC-A

| Ingredient | Formalism construct | Backend today | CLI |
|---|---|---|---|
| Compressible force balance `α Δu + β ∇m − u = 0` on a bulk | `weak_form` on a `volume` subdomain governing a vector `u` (§1.5); no template (T6 is Stokes, with a pressure) | ◐ `assemble_weak_form` lowers one linear weak form, but **constant parameters only**, no Dirichlet BC support, no coupling to a template equation in the same step; `slip.solve_overdamped_slip` solves the screened vector Laplacian `−ν∇²v + γv = f` (the right operator) but with a Nitsche normal-slip BC, not ZV/ZS, and as a stand-alone Python call | ✗ |
| Active stress as a forcing `β ∇m` (ZS: enters the natural BC) | the weak form `β m div(u_test)` with `m` another variable | ◐ allowed syntactically; the lowering splits with `lhs/rhs` so `m` must be a lagged `Function` — the staggered scheme `unknown_motion.py` uses on a membrane, not available on a bulk | ✗ |
| ZV: `u = 0` on `∂Ω`; ZS: zero traction | Dirichlet on a weak-form variable (§1.5.6); natural BC in the form | ✗ `weakform.py` defers BCs on weak-form variables; natural BC fine | ✗ |
| Myosin transport with lab-frame carrier `u_eff`, crowding cut-offs, Rankine–Hugoniot no-flux | T1 with the lab-frame `advection` slot (`advection: "u * max(0, 1 − m/m_max_u)"`), `diffusion: "1 − m/m_max_d"`; the slot's integration by parts gives zero flux relative to the mesh (`v − w`), i.e. the paper's BC when `w = v_b` on the boundary | ◐ the operator is right (`assemble.py`, ADR 009 addendum) but the slot expression references `m` and `u`: **nonlinear**, so backward Euler refuses it and only the method of lines runs it; `advection` is single-mesh bulk only (fine here) | ◐ the CLI moving path is backward-Euler only (`runner._run_moving`) |
| Boundary velocity `v_b = v_p n + u|∂Ω` with `a(t) = |Ω|`, `m|∂Ω` | **no construct.** `motion.kind: prescribed` wants a velocity *field* on the volume; `kind: unknown` wants a motion *variable on the same subdomain* governed by an equation. A law given only on the boundary, from the normal, a bulk trace and a global integral, fits neither | ✗ `_MeshMotion` extends a boundary displacement harmonically — the right mechanism — but it is fed by evaluating the prescribed expression on the boundary nodes; `geom.normal` is scoped to surfaces; `region_size(Ω)` is **refused on a moving mesh** | ✗ |
| Area functional `a(t)` | `region_size(<subdomain>)` (§1.8.4) | ✗ on moving meshes ("its sizes change in time", a follow-up) | ✗ |
| Tangential boundary velocity (ZS) | the formalism allows tangential components on a moving surface; for a volume, the mesh's tangential motion is a backend choice (§1.10.7) | ◐ harmonic extension accepts any boundary velocity; BGN redistribution exists only for the membrane-only unknown-motion path | — |
| Remeshing (concave rear, large travel) | solver-side (§1.10.7) | ✓ 2D bulk remesh-and-continue (Netgen, conservative remap); **BCs across a remesh unsupported** — the ZV Dirichlet on `u` would have to be re-applied after each rebuild | ◐ |
| Symmetry breaking from a small initial gradient | IC expression | ✓ | ✓ |

**Impedance mismatches (UC-A).**

1. *The motion law lives on the boundary, the formalism's motion lives on the subdomain.* VCell's
   `MembraneSubDomain Velocity` has the same shape as this paper's `v_b` (a front law), and the bridge
   maps it to a *prescribed volume velocity* `v = v_b` on the inside compartment — acceptable for the
   analytic fields VCell authors, wrong in general (a law in `n`, `a(t)` and `m|∂Ω` has no volume
   extension to evaluate). The framework needs a third motion kind: **a boundary velocity law on a
   labelled boundary, extended into the volume by the backend** (§6 below).
2. *The force balance is not Stokes.* No pressure, no incompressibility, no inf-sup; the viscous
   operator is the vector Laplacian, not the symmetric gradient (using `ε(u)` changes the ZS traction
   condition and the phase diagram). None of `stokes.py`, `stokes_hdiv.py`, `multiphase.py`,
   `fsi.py` applies; they all carry a pressure. The closest existing operator is the screened vector
   Laplacian in `slip.py`. The framework's family table (§4) has "compressible active-gel force
   balance" only under the migration row and marks it planned; it should be a family of its own
   (shared with UC-C).
3. *Two couplings in one step.* `u` depends on `m` (forcing), `m` is advected by `u`, and `v_b` reads
   both. The backend has that staggering only for a membrane with one weak-form equation and one T2
   (`unknown_motion.py`), not for a bulk. The paper used backward Euler with a segregated fixed-point
   iteration (up to 35 iterations, tolerance 1e-10) and validated against a monolithic COMSOL solve.
4. *Nonlinear transport on a moving mesh.* The crowding cut-offs make the T1 slots nonlinear; the
   CLI's moving path is backward-Euler only, and the method-of-lines moving stepper has no output
   hooks. Either the moving path grows a MOL route or the cut-offs are lagged (then the hard cap
   `m = m_max,u`, which the ZV steady states reach, needs care).
5. *Conservation is the authors' quality criterion.* Total myosin conserved to machine precision; the
   backend's conservative time term plus the relative no-flux condition gives that on the single-mesh
   path, and the remap keeps it across remeshes (2D supermesh). Fine — provided the mesh velocity on
   the boundary is exactly the `v_b` used in the flux condition, which the boundary-law construct
   must guarantee.
6. *Which solver is the baseline?* The migration note says `../vcell-mbsolver` runs this model. The
   paper's code is `RotatingCell.zip` (a VCell-based implementation of the 2014 algorithm generalized
   to free boundaries); whether the production MovingBoundary solver can solve the *vector* force
   balance is not established here — VCell's MathDescription has no vector PDE. Confirm before
   planning the "apples-to-apples" cross-validation.

**Acceptance tests UC-A provides.** (i) The 1D zero-velocity linear-stability threshold
`β μ_tot > 1 + π² α` (fixed domain, `λ(q) = q²(β μ_tot/(1 + αq²) − 1)`): an exact, discriminating,
cheap test of the coupled operator before any boundary moves. (ii) The fixed-circle 2D instability
(myosin relocates to the rim by t ≈ 10 at `μ_tot = 1.5π`, `α = 0.5`). (iii) Interior points of the
phase diagram, not its borders: ZV `(v₀, μ_tot, α) = (2.5, 2π, 0.5)` translates at speed ≈ 0.18 with
a rear myosin band at `m ≈ m_max,u = 15`; ZS `(2.5, 0.75π, 0.5)` rotates on a radius ≈ 1; ZS
`(2.5, 0.125π)` returns to a circle. (iv) Aspect ratio and speed versus `α` (Fig. 3). Dimensional
parameter values are not in the main text (the S1 Appendix is missing); the dimensionless groups
`α, β = 5, μ_tot, v₀, k = 1.5, m_max,u = 15, m_max,d = 125, a₀ = π` are complete.

### UC-B

| Ingredient | Formalism construct | Backend today | CLI |
|---|---|---|---|
| Prescribed front velocity, species at rest in the lab frame (swept) | `motion: prescribed` on the inside compartment + the T1 `advection` slot (lab-frame carrier, default 0) | ✓ `tests/test_backend_lab_frame_advection.py`; `cross_validation/mb_swept*.py` through the SimulationTask path | ✓ |
| Species carried with the cell (`v = v_b`) | `motion: prescribed` with no advection slot (co-moving) | ✓ `mb_translation`, `mb_expansion` | ✓ |
| Rankine–Hugoniot front condition | implied by the `advection` slot's integration by parts (zero total flux relative to the mesh) | ✓ | ✓ |
| Exact conservation under domain change | solver-side; the conservative time term | ✓ (1e-12 across remeshes) | ✓ |
| Reaction `R` | T1 `source` | ✓ (untested in the paper) | ✓ |
| Species-dependent front velocity | the bridge's explicit one-step lag | ✓ for VCell models | ✓ |
| Membrane species on the front, exterior species, topology change, 3D | — | membrane species on a moving front ✗ (refused); 3D ✓ beyond the paper; topology change ✗ by design | — |

**Impedance mismatches (UC-B).** Few, and already handled: the lab-frame-versus-mesh semantics is
the ADR 009 addendum and the SWEPT cross-validation. Two findings matter for the framework:

1. *"No species velocity" is not "no dilution".* With `v = 0` and an expanding front the paper's
   scheme carves mass from neighbours for each new interior node: total mass is constant and the mean
   concentration falls as `1/|Ω(t)|`. An ALE implementation with the `advection` slot and the
   conservative time term reproduces this; one with a mesh-following species and an explicit
   `u ∇·w` term reproduces the *carried* case instead. The framework's §2 statement that "declaring a
   boundary impermeable differs from setting only the diffusive flux to zero" is exactly this and is
   right; the use case gives it a closed-form test that does not yet exist (next item).
2. *Order and norm.* The paper is first order in time and between 1 and 2 in space (first-order
   locally at front cells from the remap); agreement with the P1 ALE scheme should be judged in L2,
   with the front position error separated from the field error (the cross-validation README already
   does this).

**Acceptance tests UC-B provides.** (i) The travelling exponential `u₀ e^{−k(x − v_b t)}`,
`k = v_b/D`, for *any* rigidly translating shape with the species at rest — exists as a test for the
wall case; the circle version (paper T2/T3) is a one-line addition. (ii) The carried translating
disk with the exact Neumann eigenmode `e^{−λ²Dt} J₁(λr′) cos θ′`, `λ = j′₁,₁ = 1.8412` (paper T4;
check the angular factor, the printed form is ambiguous): an exact moving-domain test the suite lacks
(`mb_translation` compares homogenisation, not an eigenmode). (iii) New: a **swept expansion** —
radial `v_b`, `v = 0` — with the exact mean `ū(t) = u₀ |Ω(0)|/|Ω(t)|` and a transient boundary layer;
it discriminates swept from carried as cleanly as the dilution case does for carried.

### UC-C

| Ingredient | Formalism construct | Backend today | CLI |
|---|---|---|---|
| Axisymmetric (r, z) reduction — the authors' cost-saving approximation | **none** (Cartesian 2D/3D only); the faithful formulation is 3D, which the formalism and the 3D body-fitted realization support | ◐ 3D is available; the cost is resolution (a 0.3 µm domain with 20-nm rings and a 3–10 nm neck needs local refinement the remesher does not yet do). An `r`-weighted measure with `ε_θθ = v_r/r` would be an optional backend optimisation to reproduce the papers' runs cheaply, not a modeling requirement | ◐ |
| Compressible force balance with `η(ρ, L)`, `σ_a = κ ρ²`, no pressure | `weak_form` on a bulk governing `v`, coefficients from other variables | ◐ as UC-A, plus **degenerate viscosity** where `ρ → 0` (a floor is presumably in the COMSOL file; must be chosen and shown insensitive) and the lhs/rhs split cannot take `η(ρ)` as a lagged coefficient without a `Function` parameter | ✗ |
| ~8 bulk species, conservative advection by `v`, Berro kinetics with `(1 − ρ/ρ_max)` factors | T1 with the lab-frame `advection` slot (`v` the carrier) and nonlinear `source`s; `ρ` as a derived expression (a parameter in the species) | ◐ the MOL single-mesh path runs stiff nonlinear kinetics on a fixed mesh; **on the translating domain it needs the moving MOL stepper**, which has no CLI/output route; pure advection needs stabilisation (unspecified in the paper) | ✗ |
| Local diffusion `D(x)` near the rings so an influx is well posed | T1 `diffusion` as an expression in `geom.x` | ✓ | ✓ |
| Ring sub-regions of the membrane carrying ODEs with bulk traces | T4 `lumped_ode` on a `surface` subdomain with `trace(·)` in `rate`; but the rings are **parts of one membrane** — the geometry formalism names a surface class per membrane *pair*, not sub-patches | ◐ surface species with bulk traces exist (`coupled.py`, `multi_compartment.py`); a membrane partitioned into labelled patches does not | ✗ |
| Arp2/3 influx into the bulk from the rings | Neumann on the bulk species referencing the ring variable (the §1.6.6 composable pattern) | ✓ on a fixed geometry (`coupled.py`) | ✗ |
| No-slip `v = u e_z` on the invagination, `v = 0` on the flat membrane, zero stress far away | Dirichlet on a weak-form variable, per labelled boundary | ✗ (as UC-A); the velocity jump at the junction is a lid-driven corner — refine there | ✗ |
| Force functional `f_z = ∫_S e_z·σ·n ds` | T5 `region_ode`'s boundary term `∫_∂R j ds` is the region variable's own flux BCs (`neumann` / `interface_flux` expressions), so the traction would have to be written as a flux expression in `grad(v)`, `geom.normal` and the stress law on one labelled boundary | ◐ region variables are hosted only by the multi-compartment solver on a **fixed** geometry; a flux expression in the gradient of a vector variable is not an existing integrand | ✗ |
| Rigid translation `u(t) = μ⟨f_z − f_c(t)⟩₊ e_z` of **part** of the boundary — the authors' approximation | `motion: prescribed` with an expression in a region variable and `sim.t`, piecewise on the boundary | ◐ the prescribed velocity is evaluated on boundary nodes and extended harmonically, so a piecewise `if(on invagination, u e_z, 0)` expression would move the right nodes; nothing names "this labelled boundary moves rigidly, that one does not" | ✗ |
| The modeled law: a **membrane force balance** on the invagination (gel traction vs turgor, tension/bending, coat) with the shape solved | `motion: unknown` on a `surface` subdomain with a weak-form force balance referencing the bulk traction and `geom.mean_curvature` / `geom.normal` (§1.10.8), coupled to the bulk's motion | ◐ the membrane-mechanics machinery exists for a *closed* membrane around an **incompressible** interior (`unknown_motion.py`, `fsi.step_force_balance_fsi`: tension, projected curvature, BGN redistribution); here the membrane is an open patch pinned to a flat wall, loaded by a *compressible* gel from one side and turgor from the other, with a coat stiffness — none of which is built | ✗ |
| Prescribed inputs | `WASp0(t)` bell curve, rate constants, `D(x)` support, `Ω` size, viscosity floor, mesh and time step | **not in the main texts** (supplements, `Figure2.mph`); Berro 2010 is a VCell model, so the kinetics exist as VCML | — |

**Impedance mismatches (UC-C).**

1. *Separate the model from the papers' approximations.* Axisymmetry and the rigid invagination
   reduced the authors' computational cost; they are not biology. The faithful formulation is 3D
   with a deformable invagination under a membrane force balance. The framework's stance ("the
   existing solver is a foundation, not a constraint on how biology must be described") says to
   model the former and treat the latter as (a) acceptance tests for the papers' numbers and (b)
   optional cost reductions. Consequences: 3D at nanometre scale needs **local refinement** (the
   remesher rebuilds the whole region at one `h`); an axisymmetric measure is a worthwhile backend
   optimisation, not a formalism construct; and the membrane mechanics the framework's §3 lists
   (traction balance, tension, bending) *is* on this use case's path, with a twist — an open membrane
   patch pinned to a wall, a compressible gel on one side, turgor and a coat on the other.
2. *The mechanics has no solvent and no pressure* — the same gap as UC-A, with the added degeneracy
   that `η` and `σ_a` vanish with `ρ`. The framework's §3 remark "mixture incompressibility is not
   automatically incompressibility of each phase" is beside the point here: there is no mixture.
3. *Two tiers of motion law.* The papers' tier is a rigid translation of part of the boundary driven
   by a force functional against a prescribed resistance — reproducible with a piecewise prescribed
   velocity plus a region variable, and the right first target because it isolates gel mechanics
   from membrane mechanics and has published numbers. The model's tier is a solved membrane shape;
   for it the existing closed-membrane force-balance machinery is the nearest piece but assumes an
   incompressible interior and a closed curve. Both tiers need a functional of the bulk stress
   (`f_z`, or the traction field) available to the motion.
4. *Scales and units.* Lengths 3–300 nm, forces 10²–3·10³ pN, densities to 18 mM with `n_A` factors
   inside the coefficients; two typos in the papers (`μ` 0.4 vs 0.04, "µm/s/pN" vs nm/(s·pN)) that a
   dimensional check catches. The framework asks for units and dimensionless groups up front; this
   model is where that discipline earns its keep.
5. *Biochemistry is VCell's.* The Berro kinetics are a VCell model; the mechanics is not expressible
   in VCell's MathDescription (no vector unknowns). This is the clearest test of the ownership split
   the workspace proposes: VCell would supply kinetics and geometry, and the mechanics would have to
   be either new VCell math constructs or a backend-side contract. See §4.

**Acceptance tests UC-C provides.** (i) Static spherocylinder (2019 Fig. 4–6): peak axial force
2538 pN with `κ_v = 3.93 n_A⁻¹ Pa·s/µM`, `κ_a/κ_v = 0.94 n_A⁻¹ s⁻¹ mM⁻¹`, 6500 subunits in the patch
cylinder, density ≈ 14 mM near the rings, inward normal force at the base rising to ≈ 320 pN. (ii) The
fixed-threshold elongation (2019 Fig. 5): onset at t ≈ −5 s, peak 2.4 nm/s, 70 → 84 nm. (iii) The
2022 head–neck run (Fig. 2): peak drive ≈ 1100 pN, 179 nm in 6.2 s, max 38 nm/s. These all require
the full kinetics; a *reduced* first test is a prescribed `ρ(x)` field → force balance → `f_z`, which
checks the mechanics, the force integral and the corner singularity without the biochemistry.

## 3. What the use cases say about the framework

Where the [modeling framework](modeling-framework.md) is confirmed, and where it must change.

**Confirmed.**

- §2's four-velocity vocabulary (`v_b`, `v_a`, `v_c`, `w`) is exactly what separates the three models:
  UC-B is `v_c ≠ v_b` by default; UC-A is `v_c = u_eff ≠ v_b`; UC-C is `v_c = v_a = v` with `v_b`
  piecewise rigid. The relative-flux statement `(c(v_c − v_b) + j)·n` is the Rankine–Hugoniot
  condition all three use.
- §2's warning that compressible carriers dilute by their own divergence is load-bearing in A and C
  (both have `∇·v ≠ 0` as the mechanism).
- §3's "a constitutive law is not always an algebraic stress substitution" is not exercised in the
  bulk: none of the three has memory, a reference configuration or elasticity in the gel. The
  poroelastic row of §4 and the Biot benchmark of §5 are not justified by these use cases; they stay
  proposed. §3's *interface* laws (traction balance, tension, bending) **are** exercised by UC-C's
  modeling intent — the deformable invagination the papers approximated away — and by nothing else.
- §5's ladder rows "different mesh and carrier velocities" and "static diffusion and prescribed
  expansion" are what UC-B tests; UC-B adds two closed forms to them.

**Must change.**

1. **A missing problem family.** Add to §4: *compressible single-phase active gel, no pressure*
   (`∇·(η(∇v [+ ∇vᵀ])) + ∇σ_a(c) − ξ v = 0`, quasi-static, with a species-dependent `σ_a` and
   optionally `η`), coupled to transport by `v`. It covers UC-A (constant `η`, drag `ξ`, vector
   Laplacian, `σ_a ∝ m`) and UC-C (`η(ρ, L)`, no drag, symmetric gradient, `σ_a ∝ ρ²`). Its numerics
   are a vector Helmholtz / elliptic solve with no saddle point — simpler than Stokes, and absent from
   the backend. The "active viscous mixture" row (two phases, mixture pressure) stays for the
   multiphase note's use case but is not what these papers do.
2. **Motion laws are a dimension of the model, not a detail.** §2's table has one row for `v_b`. The
   use cases show three distinct *kinds* of boundary law: a prescribed field (B), a law **derived on
   the boundary** from the normal, bulk traces, the boundary velocity of a bulk phase and a global
   functional (A), and a **rigid motion of part of the boundary driven by a functional** (C). The
   formalism has constructs for the first (`prescribed`) and for a solved velocity field on the moving
   subdomain itself (`unknown`); it has none for the other two. This is the single most important
   gap for the migration goal.
3. **Global functionals.** `a(t) = |Ω(t)|` (A) and `f_z = ∫_S e_z·σ·n ds` (C) are scalar functionals
   of the solution that feed the motion. `region_size` exists but is refused on moving meshes; T5's
   boundary-integral slot exists but only on fixed geometries and for scalar fluxes. The framework
   should name "functionals of the state that drive motion" as a first-class need.
4. **Dimensional reductions are approximations to record, not domain kinds to add.** §1's "state
   whether 2D is a planar model, a cross-section, or a thickness-averaged approximation" should also
   say *what the reduction costs and why it was made*: UC-A's thin-film 2D is part of the model
   (the lamellipodium is flat); UC-C's axisymmetry is a cost reduction of a 3D model. The formalism
   should keep describing the 3D model; an axisymmetric measure is a backend optimisation to add
   when reproducing the papers' runs cheaply matters.
5. **Sub-regions of a membrane** (UC-C's rings) need a geometry construct: labelled patches of one
   surface class, each a subdomain for surface equations.
6. **The verification ladder** should gain, in this order: the UC-A 1D instability threshold (exact,
   fixed domain, tests the coupled operator); UC-B's Bessel-mode carried disk and the swept expansion
   (exact, moving domain); a manufactured solution for the screened vector Laplacian with a
   species-dependent forcing; UC-C's prescribed-density force integral.

## 4. What the use cases say about the formalism and the ownership split

- **Weak-form escape hatch.** Both mechanics models are one linear vector weak form per step given the
  lagged species, which is what §1.5 allows — but the backend's lowering accepts only constant
  parameters, no Dirichlet BCs, and no coupling to template equations on a bulk. Closing those three
  in `weakform.py` (or a dedicated compressible-gel driver that `assemble()` dispatches to, the way
  `unknown_motion.py` does for membranes) is the enabling step for A and C.
- **A boundary-law motion kind.** Proposed shape, for discussion: on a *labelled boundary* of a
  volume subdomain, `motion: { kind: boundary_law, normal_velocity: <expr>, velocity: <expr> }`, with
  the expressions allowed to reference `geom.normal`, traces of the subdomain's variables, region
  variables and `region_size(·)`; the backend extends the boundary velocity into the volume (harmonic
  extension, as today) and uses the same velocity in every relative-flux condition. UC-A's ZV is
  `normal_velocity: "v0*a0/(a*(1+m)) - k*(a - a0)"` with `a` a region variable equal to
  `region_size(cell)`; ZS adds `velocity: u`. UC-C's invagination is `velocity: "[0, u]"` with `u` a
  region variable, on one labelled boundary, and `none` on the flat membrane. VCell's
  `MembraneSubDomain Velocity` is the degenerate case (an analytic field, no traces) and would map to
  the same kind instead of to a prescribed volume velocity.
- **`region_size` on moving meshes** and **region variables outside the multi-compartment solver** are
  prerequisites of the above; both are recorded follow-ups already.
- **Ownership.** The workspace expects the biological model → mathematical description transformation
  to live in VCell. UC-B is VCell math already. UC-A and UC-C are *not expressible* in VCell's
  MathDescription (no vector unknowns, no mechanics, membrane velocity only as an expression).
  Either VCell's math grows mechanics constructs (vector variables, force-balance templates, the
  boundary-law motion kind — a large change to the Java platform), or the contract is: VCell supplies
  geometry, kinetics and membrane velocity laws; the mechanics families are declared in this
  formalism and solved here. The second is the realistic near-term split, and it should be stated in
  the workspace README rather than left implicit.
- **mbsolver as baseline.** For UC-B the baseline is exact. For UC-A, confirm what the production
  MovingBoundary solver can run (the 2017 code is separate). For UC-C the authors used COMSOL's ALE
  solver; the only baseline is the paper's numbers and the `.mph` file.

## 5. Recommended order

1. UC-B's two exact tests (Bessel-mode carried disk; swept expansion) — cheap, closes a gap in the
   suite, no new constructs.
2. UC-A's 1D instability threshold on a fixed interval — the first test of the compressible-gel
   operator coupled to transport; needs the weak-form lowering to accept a lagged `Function` parameter
   and a Dirichlet BC, nothing else.
3. The compressible single-phase gel driver on a bulk (vector Laplacian / symmetric gradient,
   species-dependent `σ_a` and `η` with a floor, drag, ZV/ZS boundary conditions), staggered with T1
   transport by the lab-frame `advection` slot under the method of lines; verified by a manufactured
   solution and the fixed-circle instability.
4. The boundary-law motion kind with `region_size` on moving meshes — then UC-A's phase-diagram
   points through remeshing, and the mbsolver question settled.
5. Membrane sub-regions and local 3D refinement (or, as a shortcut to the papers' numbers, an
   axisymmetric measure) — then UC-C's prescribed-density force test with the rigid invagination,
   then the full patch with the Berro kinetics imported from VCML, against the papers' numbers.
6. The deformable invagination: a membrane force balance on an open patch (gel traction, turgor,
   tension/bending, coat stiffness) with the shape solved — the framework's §3 interface laws, built
   on the closed-membrane force-balance machinery but for a compressible interior.

The poroelastic family and the two-phase mixture are not on this path; none of the three published
models needs them. Membrane mechanics is — at step 6, as the thing UC-C's authors approximated away.

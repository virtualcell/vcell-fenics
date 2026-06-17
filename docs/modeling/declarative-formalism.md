# A declarative formalism for cell-biology PDE/ODE systems

**Status:** complete first draft as of 2026-05-22. All three parts are drafted: **Part 1 (Mathematical formalism)** covers §1.1–§1.11; **Part 2 (Data model)** covers §2.1–§2.7; **Part 3 (Solver contract)** covers §3.1–§3.6. The document is now a discussion artifact for review and revision rather than an outline with TBD sections.

This document describes a declarative data model for capturing a well-posed mathematical problem — partial and ordinary differential equations on labelled geometric domains — *without* encoding how to solve it. It is the formalism that `vcell-fenics` will use to drive its DOLFINx backend, and it is intended to remain importable from VCell `MathDescription` artifacts while not inheriting VCell's historical quirks (Cartesian box faces, Neumann-only internal interfaces, single-velocity-per-subdomain restrictions).

The driving design decisions, recorded in the project memory, are:

1. **Boundaries are labelled codim-1 sub-boundaries** of the geometry; external and internal boundaries are treated uniformly, and any boundary condition kind (Dirichlet, Neumann, Robin, interface-coupling) may be applied to any boundary.
2. **Equation form is operator-template first, weak-form (UFL) as escape hatch.** Templates cover the common cases with named term slots; an escape hatch handles unusual physics.
3. **Each equation declares its temporality explicitly** (`time_dependent` or `steady_state`). Temporality is not inferred from the presence of a time derivative — mixed systems (algebraic constraints alongside time-evolving variables) need to be expressible cleanly.
4. **One MathDescription per coupled system.** All variables, equations, and coupling that share unknowns or boundaries live in a single well-posed math object.
5. **Moving geometry is first-class.** A subdomain's motion is either prescribed or solved-for by an equation in the same MathDescription. This is non-negotiable: mechanics-driven cell migration is the project's central use case.
6. **Motion is a property of the subdomain**, not of any equation on it. The substrate velocity comes from `subdomain.motion`; species-relative drift comes from a per-equation `relative_advection` slot.

Excluded from this v1 by design: VCell-style FastSystem (solver-side reduction), Events (discrete state transitions), region variables (compartment-aggregate piecewise constants), stochastic constructs, and pre-built constitutive templates for adhesion/slippage. Each is expressible-in-principle within the formalism or recoverable via change of variables; none ship in v1.

---

## Part 1 — Mathematical formalism

### 1.1 Goals and non-goals

#### 1.1.1 Purpose

This formalism captures a well-posed mathematical problem in cell-biology modelling — deterministic partial- and ordinary-differential equations on labelled geometric domains, with possibly moving subdomains — as data, independent of how the problem is solved. Two consequences follow from that single design choice:

- **One math description, many solvers.** The same MathDescription can be run by a finite-element backend (the `vcell-fenics` DOLFINx implementation), by a finite-volume backend, or by VCell's existing moving-boundary solver, with no edits to the math description. Solver choice is a separate decision made in a separate object (§3).
- **One math description, many geometries.** The math description references geometric entities by name; the concrete mesh, region-to-class assignment, and labelled boundaries live in a separate Geometry object. A model written for a 2D test disk runs against a 3D real-cell mesh by swapping the Geometry, not by rewriting the math.

The goal is **a stable, declarative description language that survives changes in both solver technology and concrete geometry**, while still being precise enough that any compliant backend produces the same well-posed problem when handed the same MathDescription + Geometry pair.

#### 1.1.2 In scope

The formalism covers:

- **Bulk reaction-advection-diffusion** on volume subdomains, in 1D / 2D / 3D, with scalar or vector or tensor species, with prescribed or solved-for substrate motion (§1.4 T1, §1.10).
- **Surface PDEs with stretch-dilution** on codim-1 subdomains, including the canonical case of moving cell membranes carrying receptor / activator densities (§1.4 T2).
- **Algebraic constraints** alongside time-evolving equations, producing differential-algebraic systems naturally — incompressibility constraints, conserved-total constraints, instantaneous force balances (§1.4 T3, §1.9).
- **Lumped ODE dynamics** at zero-dimensional points or in non-spatial models (§1.4 T4).
- **Mechanics-driven moving geometries** — subdomain motion may be prescribed (a known velocity or displacement field) or itself unknown (solved by an equation in the same MathDescription, which is the path to mechanics-driven cell migration; §1.10).
- **Boundary conditions** on labelled codim-1 sub-boundaries: Dirichlet, Neumann, Robin, and interface BCs (value-equality and flux-balance), uniformly applied to external and internal boundaries (§1.6).
- **Coupled multi-physics within a single MathDescription** — reaction-diffusion + mechanics + electrical signalling can coexist in one well-posed math object, coupled through shared variables, traces, and matched BC / source expressions (§1.6.5, §1.8).
- **A weak-form escape hatch** for equations no operator template covers — custom constitutive laws, higher-order operators, mixed-FE-pair problems (§1.5).
- **VCell-importability** — the formalism is intended to round-trip with a useful subset of VCell `MathDescription` artifacts. The full subset and the mapping rules are deferred to §2.4.

#### 1.1.3 Non-goals

The formalism does **not** cover and is **not intended** to cover:

- **Solver settings.** Time-stepping scheme (BE, CN, BDF2, RK), mesh resolution, FE polynomial order, linear-solver and preconditioner choice, nonlinear-iteration tolerances, DAE-index reduction strategy. These live in a separate solver-configuration object (§3); they are not part of the math problem.
- **Stochastic dynamics.** Particle-based simulators (Smoldyn, MCell), stochastic PDEs, Langevin systems, random fields. These require their own formalism with different primitives; this document is the deterministic PDE/ODE formalism only.
- **Agent-based and cellular-automaton models.** Individual-based simulators where rules act on discrete entities rather than continuous fields.
- **Reactions as first-class entities.** VCell-style explicit `Reaction` objects with kinetic-law metadata. In this formalism, reactions are source terms — algebraic combinations of variables and parameters in the `source` slot of an equation (§1.4.2 T1, T2). Users with reaction-network models can still capture them; the formalism just does not impose a separate Reaction abstraction.
- **Code generation.** This is a *description* of a problem, not a procedure for solving one. Backends translate the description into executable code, but the formalism itself does not prescribe how.
- **Be all things to all PDE problems.** The targeted audience is cell-biology modellers: cytoplasmic reaction-diffusion, membrane signalling, cell mechanics, cell migration. PDE problem classes outside that scope (geophysical fluid dynamics, computational electromagnetics, solid mechanics of structural assemblies) may be partially expressible but are not the design target.

#### 1.1.4 Audience and assumed background

The intended reader is a cell-biology modeller comfortable with PDE / ODE language at the level used in papers like Contri–Massing–Rangamani 2025 (`docs/research/2026-05-21-fenicsx-ecosystem.md`) — they know what a Laplace operator is, what a reaction-diffusion equation looks like, and what a moving membrane means in math. They do not need to be FE practitioners. This document deliberately leaves FE-specific concerns (function spaces, weak-form assembly patterns, integration-by-parts mechanics) at the boundary — they appear in §1.3.3 as hints with sensible defaults, and in §1.5 as the escape hatch for users who *do* know UFL.

For the FE-savvy reader, the formalism is best understood as a domain-specific subset of UFL plus a coupling vocabulary (`trace`, `subdomain.motion`, labelled boundaries) and explicit per-equation temporality / well-posedness machinery.

For a modeller coming from VCell, the formalism is a direct generalisation of `MathDescription`: subdomain classes mapped to mesh regions (same as VCell), variables scoped to subdomains (same), per-equation operator templates filling named slots (same), with three intentional widenings — labelled boundaries replacing per-face Cartesian slots, geometry-as-unknown for mechanics-driven motion, and a UFL escape hatch for non-templated physics.

#### 1.1.5 v1 status and roadmap

This document is the v1 design of the formalism. The headline deferrals are:

- **Mechanics templates T5–T7** (Stokes / Navier-Stokes, linear elasticity, hyperelasticity).
- **Constitutive templates for adhesion / slippage** at moving membrane-substrate interfaces.
- **Three known topological limitations** of the class-based subdomain abstraction (squashed thin layers, per-cell connected components, same-class-on-both-sides interface ambiguity).
- **FastSystem / Events / region variables / stochastic constructs** from the VCell heritage.
- **Per-region term overrides (Tier 2)** for structural variation within a subdomain class.
- **General-algebraic interface BCs** for couplings that the value-equality + flux-balance kinds cannot express.
- **Lift / extension operators** (surface → bulk) — explicitly *not planned*; cases that need such lifts express them structurally via auxiliary variables and BCs.

The full consolidated list — every item explicitly deferred or rejected, organised by category with source-section cross-references — is **Appendix B (v2 Roadmap)**. The driving principle, applied throughout: each item ships when a model that genuinely cannot be expressed without it is concretely needed — not on speculation.

### 1.2 Geometry vocabulary

#### 1.2.1 What the geometry is, and what it is not

A MathDescription does not embed a geometry. It **references** one by name. The geometry — mesh, named regions, region-to-subdomain-class assignments, boundary labels — lives in a separate object that is loaded alongside the MathDescription at solve time. This separation lets the same math description run against different geometries (a 2D test disk vs. a real cell mesh, a simple disk-in-disk vs. a multi-cell tissue patch) without modification, and lets different math descriptions share a geometry.

The MathDescription is responsible for declaring the geometric *vocabulary* it uses — what subdomain classes and labelled boundaries it references. The Geometry is responsible for providing concrete regions and labels matching those names. Schema validation cross-checks the two: every name the MathDescription references must resolve in the Geometry, with matching kind.

#### 1.2.2 Subdomain classes, not regions

A **subdomain** in this formalism is a *class* — a named topological entity, not a contiguous mesh region. Multiple physical regions in a geometry may belong to the same subdomain class. The canonical example: a tissue patch geometry with two cells, each cell's interior tagged `cytoplasm` and each cell's surface tagged `membrane`. There are two `cytoplasm` regions in the mesh and two `membrane` regions, but the MathDescription has *one* `cytoplasm` subdomain class and *one* `membrane` subdomain class. Equations and BCs are written once per class and apply uniformly to all regions of that class.

This is the VCell / SBML-Spatial convention (the user authored SBML Spatial; VCell `MathDescription.SubDomain` is the same idea). Its compactness comes from "say once, apply to all instances"; its limitation is that region-level distinction is not visible to the math description. Region-specific variation in coefficients is addressed by region-keyed parameter maps in §1.2.5 / §1.4. Region-specific variation in equation *structure* requires either Tier-2 overrides (deferred to v2) or splitting into distinct subdomain classes.

A MathDescription declares its subdomain classes in a top-level `subdomains:` block. Each entry has:

| Field | Meaning |
|---|---|
| `name` | A unique identifier within the MathDescription. Must resolve to a same-named class in the referenced Geometry. |
| `kind` | One of `volume`, `surface`, `curve`, `point`. Determines topological dimension and which operator templates may live on the subdomain. |
| `motion` | Optional; declares how the subdomain moves. See §1.10. Default `{ kind: none }`. |

The subdomain block is *required* (not optional shorthand): `subdomain.motion` is a property of the subdomain and has nowhere else to live.

#### 1.2.3 Subdomain kinds

| Kind | Topological dimension | Examples | Templates available |
|---|---|---|---|
| `volume` | $d$ (ambient) | cytoplasm, extracellular space, nucleus interior, ECM | T1 (bulk RAD), T3 (constraint), T4 (lumped ODE on a point-like volume), T5–T7 (mechanics) |
| `surface` | $d - 1$ | plasma membrane, nuclear membrane, ECM interface | T2 (surface PDE with dilution), T3, T4 |
| `curve` | $d - 2$ | filaments, actin bundles, cytokinetic furrow | (v2; not in v1 templates) |
| `point` | 0 | single localised pole, spindle pole body | T4 (ODE), T3 (constraint) |

"Ambient dimension" $d$ is the dimension of the embedding geometry (2 for a 2D model, 3 for a 3D model). Subdomain kinds are forms of dimensionality relative to that — a `surface` is always codim-1 in the ambient space.

#### 1.2.4 Labelled boundaries

Boundaries are first-class named entities in the geometry. Every boundary is a codim-1 sub-manifold of *some* subdomain. The MathDescription does not declare boundaries as top-level entries (the geometry owns the boundary list); they appear only via references in BC entries (§1.6) and in the natural-BC default behaviour (§1.6.4).

A boundary's incidence — which subdomain class(es) it bounds — is geometry-side metadata. From it the schema derives:

- **External boundaries**: incident to exactly one subdomain. Default no-flux applies if no BC declared.
- **Internal boundaries**: incident to two subdomains (one on each side). No default — every variable that lives on either side and touches the boundary requires an explicit BC.

If a labelled codim-1 entity in the geometry is *itself* a subdomain class (the canonical cell-membrane case — `membrane` is both an internal boundary between `cytoplasm` and `extracellular` *and* a surface subdomain that carries its own PDEs), it appears in both roles: as a name in the `subdomains:` block, and as a name available for BC references. Bulk-surface coupling via this same-name double role is the §1.6.5 composable pattern.

#### 1.2.5 Region-keyed parameters (Tier 1 region variation)

Region-specific coefficient values within a class are expressed via `region_map` parameters. A parameter declared this way provides a value per region of the relevant subdomain class; expressions that reference it resolve to the region-appropriate value at evaluation time.

```yaml
parameters:
  - name: D
    kind: region_map
    subdomain: cytoplasm
    values:
      cytoplasm_left_cell:  0.10
      cytoplasm_right_cell: 0.25
```

Every region the geometry assigns to the parameter's subdomain class must have a value in `values`. Missing regions are validation errors (no implicit defaults — silent missing values are how subtle physical inconsistencies leak in).

Class-level (region-independent) parameters are declared as plain scalars, as in earlier examples:

```yaml
parameters:
  - { name: k_on, value: 0.10 }
```

A class-level parameter resolves to the same value in every region of its subdomain class — the v1 common case.

**Tier 2 (per-region term overrides) is deferred.** When term *structure*, not just coefficient values, needs to vary by region, v1 requires splitting into distinct subdomain classes (Tier 3) or shaping the equation so the variation lives in a coefficient (Tier 1). Tier 2 ships if it earns its keep through a concrete model that cannot be expressed in Tier 1.

#### 1.2.6 Known topological limitations

These three patterns from the user's experience with VCell and SBML Spatial are recognised as poorly supported by class-based subdomain abstractions and intentionally **not** addressed in v1. They are recorded here so a future v2 has them in scope. The latter two share a root cause — the class abstraction loses region-instance information when a single class is realised by multiple regions in the geometry.

**(1) Squashed thin layers — multi-role mesh entities.** When two cells are adjacent and the extracellular space between them is too thin to resolve at the mesh scale, the geometric reduction maps three physical entities onto a single codim-1 surface in the mesh: the left cell's membrane, the right cell's membrane, and the squashed-to-zero-thickness extracellular volume between them. The formalism currently assumes each region in the geometry belongs to exactly one subdomain class. The squashed-layer case requires *multi-role* region assignment — one mesh entity bearing multiple subdomain-class labels simultaneously, with thickness-derived weights so that each class's measure over that entity is physically correct (each membrane contributes its own surface integral; the squashed extracellular contributes a volume integral scaled by its collapsed thickness). This is a geometry-layer extension; v1 has no mechanism for it.

**(2) Per-cell connected components within a subdomain class.** When a subdomain class (e.g. `membrane`) is realised as many topologically disconnected regions — one membrane per cell in a tissue patch — and a species on the class is supposed to diffuse *within each cell's membrane* but **not** *between cells*, the class-level equation as written would couple all components into one connected field. The checkerboard case makes this concrete: a single cell's membrane spans four boundary patches shared with its four neighbours; the species on that cell should diffuse across those four patches, but not onto a neighbour's four patches. Tier 3 (one subdomain class per cell) handles this but scales poorly with cell count. Cleaner mechanisms — per-region variable instancing where one class declaration generates one variable instance per region with independent value fields, or a topology-aware diffusion operator that respects connected components — are deferred to v2.

**(3) Same-class-on-both-sides ambiguity for membrane expressions.** When a membrane subdomain is adjacent to two regions of the *same* subdomain class — both sides are the same cell type's cytoplasm, both compartments are extracellular space, etc. — variables on either side become ambiguous in expressions written on the membrane. A Na/Ca exchanger sitting in a membrane between two `cytoplasm` regions needs `Na` and `Ca` on each side separately to compute the exchange flux, but `trace(Na)` cannot distinguish them — both sides resolve to the same subdomain class. The v2 syntax `trace(u, from=<subdomain>)` anticipated in §1.8.7 does *not* solve this: the subdomain is the same on both sides; the *region* or *orientation* is what differs.

The recommended v2 path is **intrinsic disambiguation by region index** — each region in the geometry carries a unique integer ID, and an internal interface between two regions inherits a deterministic ordering from those IDs. Syntax sketch: `trace(Na, side=a)` and `trace(Na, side=b)` (or `inside`/`outside`, naming-to-be-finalised) where one label refers to the lower-ID adjacent region and the other to the higher-ID. This aligns with how the FE level already distinguishes the two sides of an internal facet (UFL's `("+")` / `("-")` convention is precisely index-based), needs no geometric metadata beyond stable region IDs, and handles symmetric cases — Na/Ca on two identical cytoplasm regions — with no model-side naming overhead.

The trade-off is that index-based labels carry no physical meaning: if the geometry is regenerated with a different region ordering, the model's "side a" and "side b" silently swap. The geometry is therefore expected to expose a readable region-ID-to-label mapping (e.g. `region_id 17 → cyto_left_cell`) for results interpretation and post-processing, even though the math description does not reference those labels.

When physical names *must* appear in the math (because the cell biology depends on knowing which region is which — e.g. a polarised tissue where left and right cells genuinely differ in role even if their compartment classes match), the opt-in fallback is **explicit region naming**: `trace(Na, region=<region_name>)`, requiring the geometry to declare names for the regions in question. This trades some compactness for unambiguous physical meaning.

v1 ships neither mechanism. Cases that need either fall back to Tier 3 (declare distinct subdomain classes for each region — `cytoplasm_left` and `cytoplasm_right` — even when their physics is identical), trading some duplication for unambiguous addressability.

These three limitations are flagged here rather than buried in v2 roadmap notes because they are the most likely surprises to bite a modeller coming from a mature tool like VCell that has accumulated workarounds for all three.

#### 1.2.7 Worked sketch — geometry block of the §1.6.6 example

The MathDescription side of the §1.6.6 ligand-receptor model declares its geometric vocabulary:

```yaml
math_description:
  geometry: cell_with_extracellular   # external geometry object

  subdomains:
    - { name: extracellular, kind: volume }
    - name: membrane
      kind: surface
      motion: { kind: prescribed, velocity: "0" }
    # No cytoplasm in this example — ligand only diffuses in the extracellular bulk.

  # BCs reference the boundary names `outer` and `membrane`; both must
  # exist as labelled boundaries in the geometry object.
```

The `cell_with_extracellular` geometry is expected to provide:

- A region or regions tagged `extracellular` (kind: volume).
- A region or regions tagged `membrane` (kind: surface). The same name appears both as a subdomain (the surface carries its own PDEs) and as a labelled boundary (the bulk's BC at the membrane references it).
- A labelled boundary `outer` (codim-1 surface of the extracellular volume), incident to one subdomain (`extracellular`).

The math description is independent of which mesh is used, so long as the geometry honours these names and kinds.

### 1.3 Variables

#### 1.3.1 What a variable is

A variable is a named unknown function defined on exactly one subdomain class. Its envelope:

| Field | Meaning |
|---|---|
| `name` | Unique identifier within the MathDescription. |
| `subdomain` | The name of the subdomain class this variable lives on. |
| `type` | One of `scalar`, `vector`, `symmetric_tensor`. Defines the value type at each point. Default `scalar`. |
| `space` | Optional FE function-space hint. Default `lagrange_p1` (continuous H¹ piecewise-linear). Other values listed in §1.3.3. |

A MathDescription may contain many variables; multiple variables on the same subdomain class is the normal case (one membrane carries multiple species; cytoplasm carries ligand plus an intracellular signal plus a regulator). The pairing of `(name, subdomain)` is what uniquely identifies a variable — the same name `c` on `cytoplasm` and on `extracellular` would be two distinct variables.

A variable's *value field* exists on every region of its subdomain class; the user does not declare per-region values or per-region existence. If the subdomain has motion (§1.10), the value field is defined on the deforming manifold; the time derivative in any equation referring to this variable is taken in the appropriate frame, with the convention fixed by the equation's operator template (§1.4) or the user's UFL form (§1.5).

#### 1.3.2 Variable types

Three value types in v1:

- **`scalar`** — a single real number per point. The default. Covers concentrations, densities, voltages, pressures, scalar order parameters.
- **`vector`** — a tuple in $\mathbb{R}^d$ where $d$ is the ambient dimension. Covers velocities, displacements, fluxes (when treated as primary unknowns), gradients of scalar fields when those are first-class unknowns.
- **`symmetric_tensor`** — a symmetric $d \times d$ tensor per point. Covers stress, strain, diffusivity-as-an-unknown, and similar quantities. Less common in v1 but included so mechanics templates have somewhere natural to live.

Non-symmetric tensor variables are **not supported in v1.** Cases that need them (e.g. velocity-gradient as a primary unknown) must decompose into a symmetric part and a skew part — each declared as its own variable with its own equation — or wait for the `general_tensor` type, which is on the v2 roadmap when a concrete use case demands it. The weak-form escape hatch (§1.5) does not change this: a weak-form equation governs a *declared* variable, and the declared types are `scalar | vector | symmetric_tensor` only.

#### 1.3.3 Function-space hints

The `space` field provides an FE-method hint to the backend. v1 default is `lagrange_p1`:

| Value | Meaning |
|---|---|
| `lagrange_p1` | Continuous piecewise-linear ($H^1$ on the subdomain). The workhorse for cell-biology reaction-diffusion. **Default.** |
| `lagrange_p2` | Continuous piecewise-quadratic. Higher accuracy where worth the cost. |
| `lagrange_pk` for `k ∈ {3,4,...}` | Higher-order Lagrange. |
| `discontinuous_galerkin_pk` | Element-wise polynomial of degree $k$, discontinuous across element boundaries. For advection-dominated problems and certain flux-conservative schemes. |
| `taylor_hood` | Reserved for mechanics-template vector velocities paired with $P_1$ pressure (Stokes / Navier–Stokes); not in v1. |

The `space` field is a *hint*, not a contract. A backend may choose a different but compatible discretisation if it justifies the change. Backends are not required to support every space value; an unsupported hint is a solver-side error, not a MathDescription error.

The reason for default-then-override: cell-biology modellers should not have to think about FE function spaces for the common case. A scalar concentration variable on a bulk subdomain is overwhelmingly likely to want $P_1$ Lagrange; making them spell that out adds friction without adding signal.

#### 1.3.4 Motion variables

A motion variable is just a regular vector variable that happens to be referenced from a subdomain's `motion` slot. There is no separate "motion variable" entity kind.

```yaml
subdomains:
  - name: membrane
    kind: surface
    motion: { kind: unknown, variable: membrane_velocity }

variables:
  - { name: membrane_velocity, subdomain: membrane, type: vector }

equations:
  - template: <constitutive_template_for_motion>   # e.g. force balance, viscous slip
    variable: membrane_velocity
    subdomain: membrane
    temporality: ...
    terms: { ... }
```

The motion variable lives on the same subdomain whose motion it represents. The equation governing it can reference any other variable visible from that subdomain — including bulk variables via `trace(·)` (e.g. cortical actin stress imported from the cytoplasm), parameters, etc. The composability of motion-as-unknown comes from this plain-variable treatment.

#### 1.3.5 Variables on moving subdomains

When a variable's subdomain has `motion.kind` of `prescribed` or `unknown`, the value field is defined on the deforming manifold. The semantic conventions:

- **Spatial coordinates and geometry quantities** (`geom.x`, `geom.normal`, `geom.mean_curvature`, etc.) evaluate against the current (deformed) configuration at every time step.
- **Time derivative `∂_t u`** in an operator template's `∂_t` slot is the partial-time derivative at fixed lab-frame coordinate (Eulerian convention). The template writes the equation in conservation form $\partial_t u + \nabla \cdot (u \mathbf{v}_\Omega) = \ldots$ where $\mathbf{v}_\Omega$ is the substrate velocity from `subdomain.motion`; expanding the flux divergence yields the compression / dilution term $u \nabla \cdot \mathbf{v}_\Omega$ automatically. Full discussion in §1.10.5.
- **Initial conditions** are evaluated on the initial (t=0) configuration. For subdomains with `motion.kind = unknown`, the initial configuration is determined by the motion variable's initial condition (a displacement field at t=0).

These conventions are imposed *by* the operator templates; the user writing a template's `source`, `diffusion`, or BC expression does not need to think about which frame they are working in. The escape-hatch weak form (§1.5) requires the user to be explicit about time-derivative conventions — that is a v1.5 problem.

### 1.4 Equations — operator templates

#### 1.4.1 Common equation structure

Every equation in a MathDescription has the same envelope:

| Field | Meaning |
|---|---|
| `template` | Which operator template this equation instantiates (e.g. `bulk_radv_diff`). |
| `variable` | The unknown this equation governs (named; must exist in the MathDescription's variable list). |
| `subdomain` | The subdomain on which the equation holds (named; must exist in the geometry). |
| `temporality` | `time_dependent` or `steady_state` — declared explicitly per §1.9. |
| `terms` | A map from the template's term-slot names to expressions. Optional slots default to zero. |
| `initial_condition` | Required iff `temporality = time_dependent`. |

The template fixes the *shape* of the equation (which terms exist and how they combine); `terms` provides the *content* (expressions for each slot, with optional slots omitted or zero). One MathDescription may contain many equations; the same subdomain may carry many variables, each with its own equation.

A weak-form escape hatch (§1.5) replaces `template` + `terms` with a UFL expression. The rest of the envelope is identical.

#### 1.4.2 Templates for v1

##### T1 — Scalar reaction-advection-diffusion (bulk)

On a bulk subdomain Ω of dimension d ∈ {1, 2, 3}, in conservative form:

$$\partial_t u \;+\; \nabla \cdot \bigl(u \, \mathbf{v}_\Omega\bigr) \;+\; \nabla \cdot \bigl(u \, \mathbf{w}\bigr) \;=\; \nabla \cdot (D \, \nabla u) \;+\; s$$

For `temporality = steady_state`, the `∂_t u` term is omitted.

| Slot | Type | Required | Meaning |
|---|---|---|---|
| `diffusion` | scalar or symmetric tensor on Ω | no | D (omit ⇒ 0, pure reaction or advection) |
| `relative_advection` | vector field on Ω | no | **w**, the species' drift relative to the substrate. Default 0. |
| `source` | scalar expression | no | s. May depend on u, `geom.x`, `sim.t`, parameters, traces of variables on other subdomains (§1.8). |

The substrate velocity **v_Ω** is not a slot — it comes from `subdomain.motion` (§1.10). When the bulk is static (no motion), the `∇ · (u v_Ω)` term vanishes; when it is moving, the compression contribution `u ∇ · v_Ω` is included automatically and cannot be forgotten. The Eulerian-vs-Lagrangian distinction is therefore a modelling choice expressed entirely through `subdomain.motion`:

- **Eulerian fluid setup** ⇒ `bulk.motion = none`, fluid velocity goes into each species' `relative_advection`.
- **Lagrangian setup** ⇒ `bulk.motion = v_fluid`, `relative_advection` defaults to zero.

At least one of `diffusion` or `source` must be present for the equation to be non-trivial.

Covers: existing `BulkPDE`, every cytoplasmic reaction-diffusion model in the cell-biology literature, drift-diffusion of charged species, electromigration.

##### T2 — Surface PDE with stretch-dilution (codim-1)

On a surface subdomain Γ of codimension 1 (a 1-curve in 2D, a 2-surface in 3D), in conservative form:

$$\partial_t \rho \;+\; \nabla_\Gamma \cdot \bigl(\rho \, \mathbf{v}_\Gamma\bigr) \;+\; \nabla_\Gamma \cdot \bigl(\rho \, \mathbf{w}\bigr) \;=\; \nabla_\Gamma \cdot (D \, \nabla_\Gamma \rho) \;+\; s_\Gamma$$

Expanding the first divergence reveals the **mandatory dilution term** and the substrate-advection term separately:

$$\partial_t \rho \;+\; \mathbf{v}_\Gamma \cdot \nabla_\Gamma \rho \;+\; \rho \, \nabla_\Gamma \cdot \mathbf{v}_\Gamma \;+\; \nabla_\Gamma \cdot (\rho \mathbf{w}) \;=\; \nabla_\Gamma \cdot (D \, \nabla_\Gamma \rho) \;+\; s_\Gamma$$

The third term, $\rho \, \nabla_\Gamma \cdot \mathbf{v}_\Gamma$, is the canonical bug in moving-membrane surface-density code (see `docs/modeling/approaches.md`). Forgetting it produces models that visibly violate mass conservation under uniform stretch. The template eliminates this failure mode: the substrate velocity is not the user's responsibility per-equation — it comes from `subdomain.motion`, and the divergence is computed automatically.

| Slot | Type | Required | Meaning |
|---|---|---|---|
| `diffusion` | scalar or tensor on Γ | no | D (surface Laplace-Beltrami coefficient) |
| `relative_advection` | tangential vector field on Γ | no | **w**, species drift relative to the membrane material (active transport, motor-driven flow). Default 0. |
| `source` | scalar expression | no | s_Γ |

The substrate velocity **v_Γ** is not a slot — it comes from `subdomain.motion`. When Γ is itself a moving subdomain (the typical case for a cell membrane), v_Γ is either Γ's prescribed motion expression or the unknown motion field solved by an equation elsewhere in the MathDescription (§1.10).

Covers: existing `SurfacePDE`, the cos(kθ) eigenmode test, the dilution-discriminator test, every membrane-bound receptor / activator / signalling model.

##### T3 — Algebraic / divergence constraint

On any subdomain of any dimension:

$$f(u_1, u_2, \ldots, \mathbf{x}, t) \;=\; 0$$

A pure constraint — no time derivative. Most common instance: incompressibility $\nabla \cdot \mathbf{v} = 0$ for a velocity unknown.

| Slot | Type | Required | Meaning |
|---|---|---|---|
| `constraint` | scalar expression | yes | The expression to be zeroed |

T3 is always `temporality = steady_state` (per §1.9: the constraint holds instantaneously). When written alongside time-dependent equations on the same or overlapping subdomains, it produces a DAE-like coupled system — the canonical mixed-temporality case from §1.9.

##### T4 — Scalar ODE (lumped / zero-spatial-dim)

On a 0-dimensional subdomain (a single geometric point, or the entire domain in a non-spatial lumped model):

$$\frac{du}{dt} \;=\; r(u, t)$$

For `temporality = steady_state`, this collapses to the algebraic $r(u) = 0$.

| Slot | Type | Required | Meaning |
|---|---|---|---|
| `rate` | scalar expression | yes | r — the right-hand side |

Covers: non-spatial signalling-network models, lumped-compartment kinetics, any case where a variable is constant in space within its subdomain.

#### 1.4.3 Sketched for v2+

The following templates are designed-for but not implemented in v1. The schema (§2) reserves space for them; the solver contract (§3) treats them as "the backend may decline."

- **T5 — Stokes / Navier-Stokes momentum balance** (vector unknown, on bulk Ω): $\rho_0 (\partial_t \mathbf{v} + (\mathbf{v} \cdot \nabla) \mathbf{v}) = \nabla \cdot \mathbf{\sigma} + \mathbf{f}$, with $\mathbf{\sigma}$ filled by a constitutive template. Pairs with T3 incompressibility.
- **T6 — Linear elasticity** (vector displacement, on bulk Ω): $\nabla \cdot \mathbf{\sigma}(\mathbf{u}) + \mathbf{f} = \rho_0 \, \partial_{tt} \mathbf{u}$, with linearized strain. Steady-state form drops the inertia term.
- **T7 — Hyperelasticity** (same shape as T6 with a non-linear $\mathbf{\sigma}(F)$ constitutive slot).

These exist on the roadmap because mechanics-driven cell migration is the project's central goal; they are deferred only because their FEniCSx implementations are larger than the first concrete prototype warrants. A v2 effort will also add a library of **constitutive templates for adhesion / slippage** (Stokes drag, Coulomb friction, viscous slippage between membrane and substrate) so users do not write force balances from scratch each time a membrane has non-trivial coupling to its substrate.

#### 1.4.4 Term-slot semantics

- Every slot expression is evaluated in the subdomain's coordinate frame.
- Slot expressions may reference:
  - The equation's own variable (e.g. `u` in T1).
  - Other variables in the same MathDescription, via traces / restrictions when they live on a different subdomain (mechanism in §1.8).
  - The time variable `t`.
  - Spatial coordinates `geom.x`, plus geometry quantities (the outward unit normal `geom.normal`, tangent basis, mean curvature `geom.mean_curvature`, etc.) — provided as built-in functions; full list in §2.
  - Named parameters (constants declared in the MathDescription).
- Optional slots default to zero. There is no magic inference of non-zero values from elsewhere: what is not written is not there.
- The schema enforces type compatibility (a vector-typed slot cannot hold a scalar expression).

#### 1.4.5 Worked example — 2D single-cell receptor density

The project's first concrete goal in `CLAUDE.md`: a 2D circular cell with surface PDEs for receptor density on a prescribed-motion membrane. Two species — an active and an inactive form — coupled by mass-action kinetics, both diffusing on the membrane, both experiencing the same prescribed radial expansion via `membrane.motion`.

The data form below is illustrative (the schema in §2 may render as Python dataclasses, JSON, or YAML — that decision is not yet made):

```yaml
math_description:
  geometry: disk_radius_1            # external reference, see §1.2

  subdomains:
    - name: membrane
      kind: surface
      motion:
        kind: prescribed
        velocity: "r_dot * geom.x / geom.radius"    # uniform radial expansion in R^2 (geom.radius = |geom.x|, §1.8.4)

  variables:
    - { name: rho_active,   type: scalar, subdomain: membrane }
    - { name: rho_inactive, type: scalar, subdomain: membrane }

  equations:
    - template: surface_pde_with_dilution
      variable: rho_active
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.1
        source: "k_on * rho_inactive - k_off * rho_active"
      initial_condition: "1.0 + 0.5 * cos(2 * geom.azimuth)"

    - template: surface_pde_with_dilution
      variable: rho_inactive
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.1
        source: "-k_on * rho_inactive + k_off * rho_active"
      initial_condition: "1.0"

  parameters:
    - { name: k_on,   value: 0.10 }
    - { name: k_off,  value: 0.05 }
    - { name: r_dot,  value: 1.00 }
```

Notes on what this example demonstrates:

- **Two variables share one subdomain.** Coupling between them is expressed in the `source` slots via mass-action expressions — no special template needed for "reactions"; reactions are just source terms.
- **`surface_velocity` is absent** from both equations. The membrane's motion is declared once on the subdomain; both species inherit it. The dilution term $\rho \, \nabla_\Gamma \cdot \mathbf{v}_\Gamma$ is computed automatically from `membrane.motion` and is impossible to forget.
- **`relative_advection` is absent**, which means both species move purely with the membrane material (no active transport). Adding motor-driven retrograde flow on `rho_active` would mean a single `relative_advection: ...` slot on that equation; nothing else changes.
- **No solver settings appear** — no time step, no mesh resolution, no FE order, no integrator. Those belong to a separate solver-configuration object (§3.2).

### 1.5 Equations — weak-form escape hatch

#### 1.5.1 When to use it

The operator templates of §1.4 cover the equation forms common in cell-biology PDE/ODE modelling: scalar reaction-advection-diffusion in a bulk (T1), surface PDEs with stretch-dilution (T2), algebraic constraints (T3), lumped ODEs (T4). They do not cover everything. The **weak-form escape hatch** is the route for equations that no template fits:

- Custom constitutive laws (non-Fickian flux, anisotropic stress responses, history-dependent material).
- Higher-order spatial operators (Cahn–Hilliard's biharmonic, gradient-flow models with $\nabla^4$).
- Mixed-space problems requiring inf-sup-stable FE pairs (Stokes / Navier–Stokes with Taylor–Hood, mixed elasticity).
- Mechanics force balances in v1 (since mechanics templates T5–T7 ship in v2; see §1.10.3, §1.10.8).

The trade-off is straightforward: the user gives up the template's guardrails (auto-compression term, narrow calculus rule, schema-level term-type validation) in exchange for full UFL expressiveness. The validator can no longer check the equation's differential structure; the user owns well-posedness.

Use it deliberately. For everything that fits a template, the template path is shorter and safer.

#### 1.5.2 Form envelope

A weak-form equation uses the same envelope as a template equation (§1.4.1), with two slot-level changes:

| Field | Meaning |
|---|---|
| `template` | The literal string `weak_form` — signals the escape hatch. |
| `variable` | The unknown this equation governs (named; must exist in the MathDescription's variable list). |
| `subdomain` | The subdomain on which the equation holds. The form's default integration measure is determined by this subdomain's kind (see §1.5.3). |
| `temporality` | `time_dependent` or `steady_state` (§1.9). |
| `form` | A UFL-style expression describing the residual. The equation is interpreted as `form = 0` for all admissible test functions. Detailed semantics in §1.5.3–§1.5.5. |
| `initial_condition` | Required iff `temporality = time_dependent`. |

One weak-form equation governs one variable. Mixed-space problems (Stokes' coupled velocity and pressure, mixed elasticity) are expressed as multiple coupled weak-form equations whose forms share variables — same pattern operator templates already use (T1 momentum + T3 incompressibility for Stokes).

#### 1.5.3 What the form contains

The form is a residual expression: the equation is `form = 0` for all admissible test functions. The expression is built from:

- **The governed variable**, written by its declared name (e.g. `v_membrane` for the §1.10.8 motion variable).
- **The variable's test function**, written as `<variable>_test` (e.g. `v_membrane_test`). The user does not declare the test function; it is implicit, has the same function-space as the governed variable, and is the function the equation is satisfied against.
- **Other variables**, including via `trace(·)` when they live on a different subdomain (§1.8.2). Calculus operators (`grad`, `div`, `lapl`, `grad_surf`, `div_surf`, `lapl_beltrami`) may be applied to **any** variable — the narrow rule of §1.8.5 does *not* apply to the weak-form escape hatch. The user is writing UFL and is responsible for the resulting equation's well-posedness.
- **Parameters, time `sim.t`, position `geom.x`, geometry quantities** (`geom.normal`, `geom.mean_curvature`, etc.) per §1.8.
- **Standard functions** (`sin`, `cos`, `exp`, `if`, …) per §1.8.5.
- **Integration measures**:

| Measure | Meaning | Default availability |
|---|---|---|
| `dx` | Volume / interior measure on the equation's own subdomain (for a `volume` subdomain). | Default for bulk equations. |
| `dx_Gamma` | Surface measure on the equation's own subdomain (for a `surface` subdomain). | Default for surface equations. |
| `dl` | Line measure (for a `curve` subdomain). | Default for curve equations (v2+). |
| `dp` | Point measure (for a `point` subdomain). | Default for point equations. |
| `ds(<boundary>)` | Surface measure on a labelled boundary of the equation's subdomain. | Used for boundary integrals (natural BC terms). |
| `dS(<boundary>)` | Internal-facet measure on a labelled internal boundary. | Used for jump-flux terms in DG-flavoured forms (v2+). |
| `dl_Gamma(<boundary>)` | Line measure on the boundary of a surface subdomain. | For surface PDEs with edges. |

#### 1.5.4 Time-derivative convention

For `temporality: time_dependent` equations, the time derivative is written symbolically as **`partial_t(u)`**, where `u` is any variable in scope. This is the **Eulerian** partial-time derivative at fixed lab-frame coordinate, consistent with the operator-template convention from §1.10.5.

The backend handles time stepping. The math description does not pin the time-integration scheme (BE, CN, BDF2, …); that is a solver-side choice (§3). Users who need fine-grained control over time integration use the solver-configuration object, not the math description.

**On moving subdomains**, `partial_t(u)` is the Eulerian time derivative — the template machinery's auto-compression behaviour does **not** apply to weak-form equations. If a weak-form equation's variable lives on a moving subdomain and the user wants the conservation-form $\partial_t u + \nabla \cdot (u \mathbf{v}_\Omega)$ behaviour, they must write that flux divergence explicitly in the form. Similarly, material-derivative-style equations (where $D_t u = \partial_t u + \mathbf{v}_\Omega \cdot \nabla u$ is what's intended) require the user to add the substrate-advection term themselves. The escape hatch buys flexibility at the price of explicitness.

Discrete-time forms (with reserved `u_prev` / `dt` symbols) are intentionally **not** part of v1. They would encode discretization choices in the math description, which conflicts with the formalism's solver-agnostic stance. v2 may add a discrete-time mode as an opt-in for users who need fine time-integration control.

#### 1.5.5 Test functions

Each weak-form equation has exactly one test function, implicitly named `<variable>_test`. It is in the same function space as the governed variable and is the function the residual is satisfied against (the equation is `form = 0` for all admissible `<variable>_test`).

Test functions for *other* variables appearing in the form (via `trace(·)` or directly) are **not** in scope. If the user wants those, the right pattern is to write a separate weak-form equation governing each variable. Coupled multi-variable forms (Stokes' joint velocity-pressure form with mixed test functions) become multiple weak-form equations sharing variables — same coupling pattern as the rest of the formalism.

#### 1.5.6 Interaction with §1.6 boundary conditions

A weak-form equation's variable may carry §1.6 boundary conditions. The rule is split:

- **Dirichlet BCs** declared in §1.6 are imposed as **strong** constraints (DOF elimination, exactly as for template equations). The form is solved subject to the Dirichlet condition; the user does not need to encode the Dirichlet value in the form.
- **Neumann, Robin, value-equality, and flux-balance interface BCs** declared in §1.6 are **not** automatically added to the form. The user must encode their boundary terms in the form they write. This matches FEniCSx idiom: natural BCs become boundary integrals in the variational form via integration by parts; the form's author writes them where they belong.

The schema rejects (or at minimum warns about) a non-Dirichlet §1.6 BC on a weak-form-governed variable, to avoid silent double-counting (the user encoding the Neumann term in the form *and* declaring a §1.6 Neumann) or silent omission (the user expecting §1.6 to handle a Neumann BC the form does not encode). The clean rule: for weak-form equations, all natural BCs live in the form; only Dirichlet is in §1.6.

#### 1.5.7 Validation

The validator checks that:

- Every name referenced in the form resolves — variables, parameters, geometric helpers, measures.
- The form returns a scalar value (it's a residual, which integrates to zero).
- The variable's `<variable>_test` reference is present in the form (a form without any test-function reference is almost certainly wrong).
- The form does not reference test functions of variables other than the governed one.
- If `temporality = time_dependent`, an initial condition is provided.
- If the variable has §1.6 BCs, they are all Dirichlet (per §1.5.6).

The validator **cannot** check:

- Well-posedness (does the form have a unique solution?).
- Differential structure (does the form make physical sense?).
- Sign conventions (is the user's $\nabla \cdot (D \nabla u)$ in the right direction?).
- Consistency between coupled weak-form equations (do they reconcile at shared interfaces?).

These are the user's responsibility — the price of the escape hatch.

#### 1.5.8 Worked example — viscous force balance for membrane motion

This is the same viscous-force-balance + receptor-density model that appears in §1.10.8 (presented there with focus on unknown motion) and §2.7 (presented there in complete YAML / JSON / Python carrier round-trip form). Here the focus is the **weak-form escape hatch**: how to write the force-balance equation as a UFL form when no template fits. A closed 2D membrane whose velocity is solved by a quasi-static viscous force balance. The membrane is in mechanical equilibrium at every instant; surface tension and a prescribed active traction drive motion, and viscous drag from the surrounding cytosol resists it.

The strong form of the force balance is

$$\eta \, \mathbf{v}_\Gamma \;+\; \sigma_T \, H(\mathbf{x}) \, \mathbf{n}(\mathbf{x}) \;-\; \mathbf{f}_{\text{active}}(\mathbf{x}, t) \;=\; \mathbf{0} \quad \text{on } \Gamma$$

where $\eta$ is viscous drag, $\sigma_T$ surface tension, $H$ mean curvature, $\mathbf{n}$ outward normal, and $\mathbf{f}_{\text{active}}$ a user-supplied driving traction (e.g. a polarised contractile force). Multiplying by a test function $\mathbf{v}_{\Gamma,\,\text{test}}$ and integrating over the membrane:

$$\int_\Gamma \Bigl[ \eta \, \mathbf{v}_\Gamma \cdot \mathbf{v}_{\Gamma,\,\text{test}} \;+\; \sigma_T \, H \, \mathbf{n} \cdot \mathbf{v}_{\Gamma,\,\text{test}} \;-\; \mathbf{f}_{\text{active}} \cdot \mathbf{v}_{\Gamma,\,\text{test}} \Bigr] \, \mathrm{d}\Gamma \;=\; 0.$$

This is a steady-state (quasi-static) weak form on the membrane subdomain. In the formalism's syntax:

```yaml
math_description:
  geometry: cell_2d

  subdomains:
    - name: membrane
      kind: surface
      motion: { kind: unknown, variable: v_membrane }

  variables:
    - { name: v_membrane, subdomain: membrane, type: vector }
    - { name: rho,        subdomain: membrane, type: scalar }

  equations:
    # Motion equation: weak-form viscous force balance.
    - template: weak_form
      variable: v_membrane
      subdomain: membrane
      temporality: steady_state
      form: |
        ( eta * inner(v_membrane, v_membrane_test)
          + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)
          - inner(f_active, v_membrane_test)
        ) * dx_Gamma
      initial_condition: "0"      # zero default (memory decision 11c)

    # Receptor density: standard T2 surface PDE — unchanged from the §1.4.5
    # / §1.6.6 examples. The T2 template picks up dilution from
    # membrane.motion automatically.
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.05
        source: "-k_off * rho"
      initial_condition: "1.0 + 0.3 * cos(2 * geom.azimuth)"

  parameters:
    - { name: eta,     value: 1.0  }
    - { name: sigma_T, value: 0.10 }
    - { name: k_off,   value: 0.02 }
    - { name: f0,      value: 0.3  }              # active-traction amplitude
    - name: f_active                              # polarised active traction (vector)
      type: vector
      subdomain: membrane                         # uses geom.azimuth — scope required (§2.2.3)
      expression: "[f0 * cos(geom.azimuth), 0]"
```

What this example demonstrates:

- **The weak-form equation is a residual that integrates to zero.** The expression following `form:` is the entire residual; the convention is that the assembled equation reads "form = 0 for all `v_membrane_test`."
- **Test function is implicit.** `v_membrane_test` is the test function for `v_membrane`, in the same function space. The user did not declare it.
- **`f_active` is an expression-valued parameter (§2.2.3), not a state variable.** It carries a vector-valued expression body (`[f0 * cos(geom.azimuth), 0]`) and is referenced from the form as a bare name. Because the body uses `geom.azimuth`, the parameter declares `subdomain: membrane` (§1.11.10). At assembly time the parameter resolves to its expression's value at the current point — the same mechanism that lets `L_reservoir = "1.0 + 0.5 * sin(omega * sim.t)"` carry time-varying boundary data.
- **No automatic compression term.** The membrane has unknown motion, but because this is a weak-form equation, the auto-dilution that T2 would apply does **not** apply here. The force-balance equation has no time derivative anyway, so there is nothing to reconcile — but if the user had wanted a transient force balance with `partial_t(v_membrane)` on a moving substrate, they would have had to write the appropriate Eulerian / material-derivative terms themselves.
- **Steady-state weak form is fine.** Mechanics at low Reynolds is quasi-static; the membrane velocity at each instant is determined by the instantaneous force balance, not by inertia. `temporality: steady_state` makes this explicit — the equation has no time derivative and is solved as an algebraic problem at each time step.
- **Two equations coexisting cleanly.** The motion equation (weak-form, steady) and the receptor equation (T2 template, time-dependent) share the membrane subdomain. The T2 equation reads `membrane.motion.variable = v_membrane` and uses that variable's solved value at each time step to compute its own dilution term. Composability across template and weak-form paths is direct.
- **Solver-side concerns absent.** No time-stepping scheme, no FE order, no remeshing trigger. The MathDescription specifies the math; the backend picks the rest.

### 1.6 Boundary conditions

#### 1.6.1 What a BC is

Every boundary condition has the same envelope:

| Field | Meaning |
|---|---|
| `variable` | The variable the BC constrains (named; must exist in the MathDescription's variable list). |
| `boundary` | A labelled codim-1 sub-boundary of the geometry (§1.2). |
| `kind` | One of the BC kinds catalogued in §1.6.2. |
| `expression` | The right-hand-side expression for the BC. Type depends on `kind`. Interpretation depends on `kind`. |

Interface-kind BCs (§1.6.5) additionally carry a second `variable` (the variable on the other side of the interface). All other kinds reference one variable on one subdomain.

There is no external/internal distinction in the schema. A boundary is just a labelled codim-1 entity in the geometry; whether it bounds one subdomain (external) or two (internal interface) is a property of the geometry, not the BC. This eliminates the VCell historical split between Cartesian box-face slots and `JumpCondition` machinery: the same primitive expresses both.

A single boundary may carry many BCs — one per variable that needs constraining there. The schema does not require *every* variable to have a BC on *every* boundary it touches; defaults apply per §1.6.4.

#### 1.6.2 BC kinds for v1

##### Dirichlet

$$u \;=\; g(\mathbf{x}, t)$$

Fixes the value of the variable on the boundary. `expression` is a scalar expression.

##### Neumann

$$D \, \nabla u \cdot \mathbf{n} \;=\; h(\mathbf{x}, t)$$

Fixes $D \, \nabla u \cdot \mathbf{n}$ on the boundary, where $\mathbf{n}$ is the outward unit normal of the variable's home subdomain. **Sign convention:** this is the natural BC of the diffusive weak form, so it enters as $\frac{\mathrm d}{\mathrm dt}\!\int_\Omega u = \int_\Gamma h$ — i.e. **positive $h$ is an influx** (a source adding to the subdomain), negative $h$ a sink. A *consumption* flux (binding, capture) is therefore written with a negative sign, e.g. `expression: "-(k_on * trace(L) * rho_f - k_off * rho_b)"`. (An earlier draft of this section described positive $h$ as "flux flowing out"; that contradicted the equation $D\nabla u\cdot\mathbf n = h$ and is corrected here.) `expression` is a scalar expression. For variables governed by a template with non-isotropic diffusion, $D \, \nabla u \cdot \mathbf{n}$ generalises to $(D \nabla u) \cdot \mathbf{n}$ in the natural way.

##### Robin

$$\alpha \, u \;+\; \beta \, D \, \nabla u \cdot \mathbf{n} \;=\; h(\mathbf{x}, t)$$

Linear combination of value and flux. Covers permeability-type conditions (membrane permeability with a fixed external reference, semi-permeable wall, etc.) without a separate template. Robin is the one BC kind that carries three coefficients rather than a single `expression`; in the schema (§2.2.6) they appear as separate named fields `alpha`, `beta`, and `expression` (where `expression` is the right-hand side $h$).

##### Interface — value-equality

At an internal boundary between two subdomains, with the variable's "left side" $u_L$ and the partner variable's "right side" $u_R$:

$$u_L \;=\; k(\mathbf{x}, t) \cdot u_R$$

Covers continuity ($k = 1$, the same physical quantity expressed as variables on either side — e.g. voltage continuous across a passive membrane) and partition equilibrium ($k \ne 1$, e.g. Nernst-style partitioning across a barrier). `expression` carries the partition coefficient $k$ (scalar; defaults to 1 for pure continuity).

The BC names both variables: `variable` (the one on the left), `partner_variable` (the one on the right). Both must be defined on subdomains incident to `boundary`. The formalism does not privilege either side — `(u_L, u_R)` and `(u_R, u_L)` express the same condition.

##### Interface — flux-balance

At an internal boundary between two **bulk** subdomains, the outward normal flux of the home variable equals a user-supplied constitutive expression:

$$D \, \nabla u_L \cdot \mathbf{n} \;=\; f(u_L, u_R, \text{partner traces}, \mathbf{x}, t, \text{parameters})$$

The equal-and-opposite flux into the partner subdomain is enforced automatically by mass conservation — the partner side's Neumann condition is not separately written. The constitutive expression $f$ may reference the home variable, the partner variable (named via `partner_variable`), traces of any other variable on either side's subdomain, geometric helpers, time, and parameters. This is the generalisation of VCell's `JumpCondition` without the Neumann-only restriction — $f$ can express any constitutive relation (linear permeability $P(u_L - u_R)$, saturating transport $V_{max} u_L / (K + u_L)$, voltage-gated channel kinetics, …).

`partner_variable` must be supplied. The BC's sign convention is fixed by which subdomain `variable` belongs to: positive $f$ means flow *out of* the home subdomain *into* the partner subdomain.

**Scope: bulk-bulk only.** This BC kind is for transport between two bulk compartments where mass *crosses* the interface but does not *accumulate* on it. Use it for channel kinetics, semi-permeable wall transport, paracellular flux between cells, and similar bulk-to-bulk crossings.

**For bulk-surface coupling where mass accumulates on the surface** (binding reactions, receptor capture, membrane-bound complex formation), use the **composable Neumann + source pattern** documented in §1.6.5 — not this BC kind. The composable pattern writes a regular `kind: neumann` BC on the bulk variable and a matching `source:` term on the surface variable's equation, with the user enforcing mass-balance by matching the expressions with appropriate signs. Worked end-to-end in §1.6.6.

A v2 "general algebraic" interface kind — any expression in traces and fluxes from either side $= 0$ — is anticipated as an escape hatch but deferred. Value-equality and flux-balance plus the §1.6.5 composable pattern together cover every interface coupling in the project's foreseeable use cases.

#### 1.6.3 Sign convention

The outward unit normal $\mathbf{n}$ in Neumann, Robin, and flux-balance kinds is the outward normal of the **variable's home subdomain**. Positive flux means flow *out* of that subdomain across the boundary. On an internal interface, $\mathbf{n}$ for the left-side variable and $\mathbf{n}$ for the right-side variable therefore point in opposite directions; the formalism keeps track of this automatically when the interface BC is assembled.

This is the standard PDE textbook convention and matches the natural BC of the weak form: $\int_\Gamma (D \nabla u \cdot \mathbf{n}) \, v \, \mathrm{d}S$ appears with $\mathbf{n}$ outward.

#### 1.6.4 Defaults and well-posedness

**External boundaries.** When a variable has no BC declared on an external sub-boundary it touches, the implicit default is **zero-Neumann (no-flux)**: $D \nabla u \cdot \mathbf{n} = 0$. This is the natural BC of the diffusive weak form and produces conservative closed-domain models by default — the right behaviour for typical cell-shaped geometries with insulating outer boundaries.

**Internal boundaries.** No default. Every internal sub-boundary that a variable touches must carry an explicit BC — Dirichlet, Neumann, Robin, value-equality interface, or flux-balance interface. The schema validates this; a missing internal BC is an error.

**Compatibility checks.** Validation rules (deferred in detail to §1.11) include:

- A variable cannot have two BCs of conflicting kind on the same boundary (e.g. Dirichlet and Neumann simultaneously).
- An interface BC's `variable` and `partner_variable` must live on subdomains that are actually incident to `boundary` from opposite sides.
- A Dirichlet BC on a variable with no boundary in the subdomain it constrains is a definition error (cannot fix a value on a non-existent boundary).
- All expressions must type-check against the variable's function space.

#### 1.6.5 Bulk-surface coupling — the composable pattern

When a boundary is *itself* a subdomain that carries its own PDE — the canonical cell-membrane case, where Γ_mem is both the interface between cytoplasm and extracellular space *and* a surface subdomain carrying receptor density variables — the coupling between bulk and surface is expressed entirely through trace operators in expressions. Note that this is **not** the same as `interface_flux_balance` (§1.6.2): flux-balance is for bulk-bulk interfaces where mass *crosses* the boundary without accumulating; the composable pattern below is for bulk-surface interfaces where mass *accumulates* on the surface (binding, capture, complex formation).

**The trace operator.** `trace(u)` is the value of a higher-dimensional variable $u$ restricted to a lower-dimensional boundary or interface within its domain. For a bulk variable $L$ defined throughout the cytoplasm Ω_cyto, `trace(L)` evaluated on the membrane Γ_mem is the value of $L$ at the membrane — formally, the limit of $L$ as you approach the membrane from inside Ω_cyto. Surface variables like $\rho_f$ that already live on Γ_mem are written directly; only higher-dimensional variables need an explicit `trace(·)` when used in a lower-dimensional expression. The trace is mathematically well-defined for the Sobolev spaces our variables live in (H¹ bulk functions have H^{1/2} traces on the boundary); FEniCSx handles trace assembly automatically. The full vocabulary of cross-dimensional reference operators is catalogued in §1.8.

**The composable pattern, then, is:**

- **Membrane equation source terms** reference bulk variables via `trace(·)`: e.g. `k_on * trace(L) * rho_f - k_off * rho_b`.
- **Bulk BCs at the membrane** reference surface variables directly (no `trace` needed; the surface variable already lives on the membrane): e.g. a Neumann BC for L with expression `-(k_on * trace(L) * rho_f - k_off * rho_b)` — negated because binding *consumes* L and positive Neumann $h$ is an influx (§1.6.2).
- **Mass conservation** — what is consumed from the bulk equals what is produced on the surface — is the user's responsibility, expressed by writing matched expressions in both places with the appropriate signs. The schema does *not* auto-balance.

There is no dedicated "reaction" BC entity in v1; the composable form covers all cases. A future v2 may add a sugar template that desugars to the same two expressions a careful user would have written by hand, once enough use cases accumulate to justify standardising it.

#### 1.6.6 Worked example — ligand-receptor binding with bulk diffusion

Extending the §1.4.5 receptor example: now the extracellular ligand L diffuses in the bulk Ω_ext, binds reversibly to the membrane-bound free receptor $\rho_f$ to form the bound complex $\rho_b$, and the outer boundary of the extracellular space is held at a fixed bulk concentration (Dirichlet) — a stirred reservoir.

```yaml
math_description:
  geometry: cell_with_extracellular   # external reference; provides Ω_ext, Γ_mem, ∂Ω_outer

  subdomains:
    - { name: extracellular, kind: volume,  motion: { kind: none } }
    - name: membrane
      kind: surface
      motion:
        kind: prescribed
        velocity: "0"                   # static membrane in this example

  variables:
    - { name: L,     type: scalar, subdomain: extracellular }
    - { name: rho_f, type: scalar, subdomain: membrane }
    - { name: rho_b, type: scalar, subdomain: membrane }

  equations:
    - template: bulk_radv_diff
      variable: L
      subdomain: extracellular
      temporality: time_dependent
      terms:
        diffusion: 0.5
      initial_condition: "1.0"

    - template: surface_pde_with_dilution
      variable: rho_f
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.05
        source: "-(k_on * trace(L) * rho_f - k_off * rho_b)"
      initial_condition: "0.5"

    - template: surface_pde_with_dilution
      variable: rho_b
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.05
        source: "  k_on * trace(L) * rho_f - k_off * rho_b"
      initial_condition: "0.0"

  boundary_conditions:
    # Reservoir at the outer boundary of the extracellular space.
    - variable: L
      boundary: outer
      kind: dirichlet
      expression: "L_reservoir"

    # Coupling at the membrane: ligand is *consumed* from the extracellular bulk
    # at the net binding rate, so the Neumann flux is the negated rate (positive h
    # is an influx; §1.6.2 sign convention). Matched to the rho_b source above with
    # the opposite sign — user-enforced conservation (ligand lost = receptor bound).
    - variable: L
      boundary: membrane
      kind: neumann
      expression: "-(k_on * trace(L) * rho_f - k_off * rho_b)"

    # rho_f and rho_b live on a closed membrane (no edge), so they need
    # no BCs of their own — the surface PDE on a closed manifold is
    # well-posed without them.

  parameters:
    - { name: k_on,         value: 0.10 }
    - { name: k_off,        value: 0.02 }
    - { name: L_reservoir,  value: 1.00 }
```

What this example demonstrates:

- **External BC on a labelled boundary.** `outer` is just another labelled codim-1 entity in the geometry; the Dirichlet BC references it by name, with no special "external" syntax.
- **No BC needed on the closed membrane for the surface variables.** $\rho_f$ and $\rho_b$ live on a closed manifold and the surface PDE template is well-posed without edge BCs. The schema does not require one.
- **Composable bulk-surface coupling.** The Neumann BC for L at the membrane and the source terms for $\rho_f$ and $\rho_b$ all reference the same constitutive expression $k_{on}\, \mathrm{trace}(L)\, \rho_f - k_{off}\, \rho_b$. The user writes it three times with correct signs; the formalism does not auto-balance. If the signs are wrong, mass is not conserved — there is no schema-level check for that.
- **`trace(L)` versus `rho_f`.** L is a bulk variable; on the membrane its value is the boundary trace, written `trace(L)`. $\rho_f$ already lives on the membrane and is referenced directly.
- **Zero-Neumann default applies nowhere here**, because every external boundary touched by every variable has an explicit BC — but if `extracellular` had a second outer boundary that we did not declare, it would default to no-flux.
- **Time-varying reservoir is a one-line swap.** `L_reservoir` is a constant here, but to model a pulse-stimulation experiment it can be replaced by an expression-valued parameter (§2.2.3) — e.g. `{ name: L_reservoir, type: scalar, expression: "1.0 + 0.5 * sin(omega * sim.t)" }` (with `omega` added as another parameter). The Dirichlet BC declaration does not change; the value it delivers becomes time-varying automatically because the parameter resolves to its expression at every evaluation.

If the membrane were itself moving (replace `motion.velocity: "0"` with a real expression), the dilution term in both surface PDEs picks up automatically from `membrane.motion`; the BC structure does not change.



### 1.7 Initial conditions

#### 1.7.1 When required

An initial condition is required for a variable iff its governing equation declares `temporality: time_dependent`. Variables governed by `steady_state` equations have no IC and the schema rejects one if provided — **with one exception: an unknown-motion variable** (§1.10.3) governed by a `steady_state` (quasi-static) equation may carry an IC, because that IC sets the $t = 0$ configuration of the moving subdomain (§1.7.7, §1.10.4), not a time-evolution starting value. The IC is optional there and defaults to zero. Variables that are not governed by any equation (a corner case that should not occur in a well-formed MathDescription, caught by §1.11) likewise have no IC.

Each variable's IC appears as the `initial_condition` field on its governing equation. This placement — IC on the equation, not on the variable's declaration — reflects that the IC is part of the well-posed time-evolution problem (variable + equation + IC + BCs) and is meaningless without the equation context.

#### 1.7.2 Form, scope, and evaluation context

An initial condition is a typed expression evaluated at $t = 0$ on the variable's subdomain. Its type must match the variable's `type` (scalar IC for scalar variable, vector IC for vector variable, symmetric-tensor IC for tensor variable).

The evaluation context is the **reference configuration**: the geometry's initial mesh, with no motion applied. For subdomains with `motion.kind` of `prescribed` or `unknown` (§1.10), the reference configuration is what the geometry provides; the motion field's effect on positions is applied for $t > 0$, not at $t = 0$. Geometric helpers (`geom.normal`, `geom.mean_curvature`, etc.) in IC expressions therefore evaluate against the reference configuration.

Spatial coordinates `geom.x` in an IC expression refer to the reference configuration's coordinate. There is no IC equivalent of "current configuration coordinates" — at $t = 0$ the two coincide.

#### 1.7.3 Vocabulary restrictions

IC expressions use the §1.8 vocabulary with two restrictions:

- **No references to other state variables.** An IC expression for variable $u$ may not reference any other variable in the MathDescription (whether via `trace(·)`, by bare name, or otherwise). This avoids ordering ambiguity — what does "$u(\mathbf{x}, 0) = 0.5 \cdot v(\mathbf{x}, 0)$" mean if $v$ is itself defined by an IC that references $u$? Worse, it avoids cycles. v1 sidesteps both concerns by forbidding the references; if a concrete use case demands coupled ICs, v2 may relax with topological-sort resolution.

- **No bare reference to time `sim.t`.** ICs are evaluated at $t = 0$ by definition; a bare `sim.t` in an IC expression has no useful meaning beyond a constant substitution. The schema rejects `sim.t` in IC expressions to catch the misconception cleanly (a user writing `initial_condition: "exp(-sim.t)"` likely meant a *forcing* expression, not an IC, and should be told). This rule applies to the IC expression directly; if the IC references a parameter (§2.2.3) whose body expression contains `sim.t`, the parameter is evaluated at $t = 0$ in the usual way — referencing such a parameter from an IC is permitted and produces the parameter's value at $t = 0$.

What IC expressions **may** reference: the spatial coordinate `geom.x` and its accessors (`geom.x[0]`, `geom.azimuth`, `geom.radius`, …); named parameters, including region-keyed parameter maps (§1.2.5); geometry quantities (`geom.normal`, `geom.mean_curvature`, principal curvatures, tangent basis); standard functions (`sin`, `cos`, `exp`, `if`, `step`, etc.).

#### 1.7.4 Type matching

The IC's value type must match the variable's `type`:

| Variable type | IC must produce |
|---|---|
| `scalar` | scalar |
| `vector` | vector in $\mathbb{R}^d$ |
| `symmetric_tensor` | symmetric $d \times d$ tensor |

No implicit broadcasting. A scalar where a vector is expected is an error; the user must write the broadcast explicitly (e.g. `0.0 * geom.normal` for the zero vector along normal, or `[0.0, 0.0]` for an explicit 2D vector).

#### 1.7.5 Compatibility with Dirichlet boundary conditions

At every point on a labelled boundary where a Dirichlet BC is declared for variable $u$, the IC value must equal the Dirichlet BC value at $t = 0$:

$$u_{\text{IC}}(\mathbf{x}) \;=\; g(\mathbf{x}, 0) \quad \text{for all } \mathbf{x} \in \Gamma_{\text{Dirichlet}}$$

Incompatibility produces a discontinuity at $t = 0^+$ as the solution snaps from the IC to the BC at the boundary, typically manifesting as a spurious boundary layer or oscillation depending on the discretization. The validator (§1.11) checks compatibility for syntactically simple cases (constant IC vs. constant Dirichlet, both evaluable at validation time); complex expressions are the user's responsibility, with the validator emitting a warning if it cannot prove compatibility but cannot prove incompatibility either.

There is no compatibility requirement between ICs and Neumann or Robin BCs — these are flux conditions and impose nothing on the IC value.

#### 1.7.6 Region-keyed initial conditions

ICs are class-level, just like equations: one IC expression for variable $u$ on subdomain class `cytoplasm` applies to every region of that class. Region-specific initial fields are expressed by using region-keyed parameter maps (§1.2.5) in the IC expression. The resolution happens at evaluation time, per region, transparently:

```yaml
parameters:
  - name: c0
    kind: region_map
    subdomain: cytoplasm
    values:
      cytoplasm_left_cell:  1.0
      cytoplasm_right_cell: 0.3

equations:
  - template: bulk_radv_diff
    variable: c
    subdomain: cytoplasm
    temporality: time_dependent
    terms:
      diffusion: 0.1
    initial_condition: c0           # resolves per region from the map
```

The IC expression `c0` references the region-keyed parameter; each region of `cytoplasm` receives its own initial value from the map. No per-region IC syntax is needed in §1.7.

#### 1.7.7 Motion-variable initial conditions

The IC for an unknown-motion variable (§1.10.3) follows the standard rules of this section, with the additional default established in §1.10.4: the IC defaults to zero, with the interpretation "initial configuration equals the reference (geometry) configuration." A displacement-typed motion variable's zero IC means "geometry starts where the mesh is"; a velocity-typed motion variable's zero IC means "starts at rest." Users may override the default when a non-zero initial deformation or initial motion is wanted.

The IC for a motion variable is, like every other IC, evaluated on the reference configuration — there is no other configuration available at $t = 0$.

### 1.8 Coupling between subdomains and the expression vocabulary

This section formalises the closed vocabulary that every right-hand-side expression in a MathDescription draws on — term-slot fillers in §1.4, BC expressions in §1.6, initial conditions in §1.7, motion-velocity expressions in §1.10, and constitutive expressions for unknown-motion subdomains. The escape-hatch weak forms in §1.5 use a strict superset of this vocabulary (full UFL); operator-template slots use the subset described here.

#### 1.8.1 What an expression is

An **expression** is a typed formula that evaluates to a scalar, a vector in $\mathbb{R}^d$, or a $d \times d$ tensor at a point in space and time. The spatial point is implicit from the expression's *evaluation context*: the subdomain on which the equation lives (for term slots), the labelled boundary the BC constrains (for BC expressions), and so on. The time point is the current solver time, or $t = 0$ for initial conditions.

Expressions are built from:

1. References to variables and named parameters (§1.8.6),
2. Time and spatial coordinates (§1.8.3),
3. Geometric helpers (§1.8.4),
4. Standard mathematical functions (§1.8.5),
5. Calculus operators on variables, subject to the rule in §1.8.5,
6. The cross-dimensional `trace(·)` operator (§1.8.2),
7. Arithmetic combinators (`+`, `-`, `*`, `/`, `**`) and tensor algebra (`·` for inner product, `:` for double-contraction, `⊗` for outer product).

The concrete syntactic carrier — parsable string, Python AST, SymPy expression, etc. — is a data-model decision deferred to §2. This section specifies *what may appear*, not *how it is written*.

#### 1.8.2 Cross-dimensional reference: the trace operator

**Definition.** For a variable $u$ defined on a higher-dimensional subdomain $\Sigma_{\text{high}}$ and a lower-dimensional subdomain $\Sigma_{\text{low}}$ that is a sub-manifold of $\Sigma_{\text{high}}$ or of its boundary, the **trace** of $u$ on $\Sigma_{\text{low}}$ is the restriction $u|_{\Sigma_{\text{low}}}$ — formally the boundary trace operator from the Sobolev space of $u$ on $\Sigma_{\text{high}}$ to its image on $\Sigma_{\text{low}}$ (the standard $H^1 \to H^{1/2}$ result for second-order PDEs on Lipschitz domains).

**Syntax.** Written `trace(u)` in expressions evaluated on $\Sigma_{\text{low}}$ when $u$ lives on $\Sigma_{\text{high}}$.

**When required.** Whenever an expression evaluated on $\Sigma_{\text{low}}$ references a variable defined on a strictly higher-dimensional $\Sigma_{\text{high}}$. Variables defined on the *same* subdomain as the expression are referenced directly. Variables defined on a *strictly lower-dimensional* subdomain than the expression's evaluation domain are intentionally not referenceable inside an expression (§1.8.7): the surface → bulk direction is mathematically non-unique, and the cases that need it are better expressed as a named bulk variable with its own equation tied to the surface variable via a boundary condition.

**Side specifier (deferred to v2).** When the higher-dim subdomain $\Sigma_{\text{high}}$ contributes exactly one connected region on one side of the lower-dim evaluation context, `trace(u)` is unambiguous. Two cases require a side specifier and are deferred (see §1.2.6): (a) a single variable defined on both sides of an internal interface where the two sides are *different* subdomain classes — resolved by `trace(u, from=<subdomain>)`; (b) an interface between two regions of the *same* subdomain class — resolved by intrinsic index-based disambiguation `trace(u, side=a)` / `trace(u, side=b)` (recommended path, mirrors UFL `+`/`-` semantics) or, when physical names matter, by explicit region naming `trace(u, region=<region_name>)`. v1 rejects ambiguous traces at validation time and the modeller falls back to Tier 3 (separate subdomain classes) for unambiguous addressing.

**Implementation.** Trace evaluation is a backend concern. In FEniCSx 0.10, traces of bulk variables on internal facets are realised through native mixed-dimensional assembly (see `docs/research/2026-05-21-fenicsx-ecosystem.md`). The user-facing formalism does not commit to a particular evaluation strategy.

#### 1.8.3 Time, space, and parameters

Built-in quantities are **namespaced** under `geom.*` (geometry) and `sim.*` (simulation); the bare
identifier namespace belongs to the user (ADR 006). So time is `sim.t` and the position is `geom.x`
— a modeller is free to name a variable or parameter `x`, `t`, `r`, `phi`, … without collision.

| Symbol | Meaning | Type |
|---|---|---|
| `sim.t` | The time. Equals current solver time, $t = 0$ in IC expressions (where it is disallowed). | scalar |
| `geom.x` | Spatial coordinate at the evaluation point, in the embedding-space dimension. | vector in $\mathbb{R}^d$ |
| `<param_name>` | A named parameter declared in the MathDescription. May resolve to a constant, to the value of an expression in `sim.t` / `geom.*` / other parameters, or to a region-keyed value (§2.2.3). | scalar, vector, or symmetric tensor — per the parameter's declared type |

Component access on `geom.x` is by index: `geom.x[0]`, `geom.x[1]`, `geom.x[2]`. The curvilinear
accessors derived from `geom.x` are listed under geometric quantities (§1.8.4).

**Parameter resolution at evaluation time.** A bare-name reference to a parameter is replaced by the parameter's value at the current evaluation point. For **constant** parameters the value is the declared scalar. For **expression-valued** parameters (§2.2.3) the value is the parameter's body expression evaluated against the surrounding context — same `x`, same `t`, same subdomain. For **region-keyed** parameters the value is the entry corresponding to the current region. From the call-site's perspective the parameter is just a typed value at a point; the declaration form determines how that value is computed.

Expression-valued parameters may carry an optional `subdomain:` scope (§2.2.3) and must declare one if their body references a subdomain-relative geometry quantity (anything under `geom.*` except `geom.x`). A reference from outside the scoping subdomain is a validation error (§1.11.10).

#### 1.8.4 Geometric quantities (`geom.*`)

Namespaced geometry quantities (ADR 006), addressed as `geom.<member>`. Available in expressions evaluated on subdomains for which the relevant notion is defined:

| Quantity | Meaning | Defined where |
|---|---|---|
| `geom.normal` | Outward unit normal. On a boundary or codim-1 subdomain, the outward direction relative to the home subdomain. | codim-1 entities |
| `geom.mean_curvature` | Mean curvature. | codim-1 entities embedded in higher-dim space |
| `geom.curvature1`, `geom.curvature2` | Principal curvatures. | codim-1 surfaces in 3D |
| `geom.tangent` | Tangent unit vector. | 1-curves in 2D, or codim-2 edges in 3D |
| `geom.radius`, `geom.azimuth` | Polar accessors. Sugar for `sqrt(dot(geom.x, geom.x))` and `atan2(geom.x[1], geom.x[0])`. | any subdomain |

`geom.x` (position) is available everywhere; the quantities above are subdomain-relative. For subdomains with `motion.kind` of `prescribed` or `unknown` (§1.10.6), these are evaluated against the current (deformed) configuration at every time step. For `unknown` motion the $t = 0$ configuration comes from the motion variable's initial condition (§1.7, §1.10); for `prescribed` motion the $t = 0$ configuration is the reference configuration (no displacement has yet been applied). At any $t > 0$, `geom.normal`, `geom.mean_curvature`, the principal curvatures, and the tangent basis reflect the deformed shape — a moving membrane's outward normal is the *current* outward normal, not the reference one.

#### 1.8.5 Standard functions and calculus operators

**Standard mathematical functions.** The usual elementary, transcendental, and piecewise primitives: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `pow`, `if(cond, a, b)` for conditional evaluation, `step(x)` for Heaviside, `sign(x)`. These have no usage restrictions — they appear anywhere an expression appears.

**Calculus operators on variables:**

| Operator | Meaning | Input type | Output type |
|---|---|---|---|
| `grad(u)` | Gradient of scalar field $u$. | scalar variable | vector |
| `div(v)` | Divergence of vector field $v$. | vector variable | scalar |
| `lapl(u)` | Laplacian of scalar field $u$ (full-space). | scalar variable | scalar |
| `grad_surf(u)` | Surface (tangential) gradient on a codim-1 subdomain. | scalar variable on the surface | tangent vector |
| `div_surf(v)` | Surface divergence on a codim-1 subdomain. | tangent vector variable | scalar |
| `lapl_beltrami(u)` | Laplace–Beltrami operator on a codim-1 subdomain. | scalar variable on the surface | scalar |

**Usage rule — the narrow rule.** In operator-template slots (T1–T7), a calculus operator may be applied to *any variable except the one the slot's equation governs*. In the weak-form escape hatch (§1.5), there is no restriction — the user is writing a UFL form and is responsible for the resulting equation's well-posedness.

**Usage rule — smoothness requirement.** Second-order operators (`lapl`, `lapl_beltrami`) require their argument variable to live in a function space that admits a meaningful strong second derivative. v1's rule: the argument's `space` must be `lagrange_p2` or higher; applying `lapl(u)` to a `lagrange_p1` variable is a validation error (§1.11.9), because the strong Laplacian of a piecewise-linear function is element-wise zero and singular on facets — the user almost certainly did not mean that. First-order operators (`grad`, `div`, `grad_surf`, `div_surf`) have no smoothness restriction beyond the default `lagrange_p1`. In the weak-form escape hatch (§1.5), users who need second-order behaviour on a P1 variable should apply integration-by-parts in the form they write — that is exactly the kind of FE-method maneuver the escape hatch is for, and it sidesteps the strong-second-derivative issue.

The rule exists because operator templates make assumptions about the differential order and integration-by-parts pattern of the assembled weak form. Allowing arbitrary calculus on the equation's own variable can silently violate those assumptions: a `lapl(u)` in T1's `source` slot for $u$ embeds a second-order operator on $u$ where the template expects a coefficient; a `grad(u)` in the same slot creates a first-order advective term outside the template's integration-by-parts machinery. UFL would compile these into *some* form; the result is unlikely to be what the user meant.

Calculus on *other* variables is safe because it produces a coefficient-shaped value (scalar, vector, or tensor) that the template uses positionally — chemotaxis source `-grad(phi) * u`, voltage-gradient drift in a relative-advection slot, or a custom flux on an internal interface BC referencing both sides' gradients. The PDE structure for the slot's governing variable is preserved.

The validator (§1.11) enforces this by walking the expression AST: if any `grad`/`div`/`lapl`/`grad_surf`/`div_surf`/`lapl_beltrami` is applied to (or transitively reduces to) the slot's governing variable, that's an error.

#### 1.8.6 Variable, parameter, and bare-name resolution

Built-in quantities are **namespaced** (`geom.*`, `sim.*`; §1.8.3, ADR 006), so they are never bare — a bare name belongs to the user. A **bare name** (no `trace(·)`, no calculus operator) in an expression resolves in this order:

1. **Local variable** — a variable defined on the same subdomain as the expression's evaluation context. Resolves to the function value at the current point.
2. **Named parameter** — a top-level parameter declared in the MathDescription. Resolves to the parameter's value at the current point: a constant if declared so, the body expression evaluated in context if expression-valued, or the per-region value if region-keyed (§2.2.3, §1.8.3).

(A qualified name `geom.<member>` / `sim.<member>` resolves to the corresponding built-in quantity; a bare operator/function name like `sin` or `grad` is valid only in call position, `name(...)`.)

A name that matches none of the above is an error (caught by §1.11). A variable defined on a *different* subdomain than the expression's evaluation context may not be referenced by bare name — higher-dimensional variables must go through `trace(·)`; lower-dimensional variables cannot be referenced inside an expression at all (§1.8.7), and the coupling must be expressed structurally via a boundary condition.

Name shadowing — a parameter and a local variable with the same name — is an error at MathDescription construction time, not a precedence resolution.

#### 1.8.7 Reserved for v2+

- **`trace(u, from=<subdomain>)` side specifier.** Disambiguator for a single variable defined on both sides of an internal interface.
- **DG flux operators.** `jump([u])`, `avg({u})` for discontinuous-Galerkin formulations. Deferred until the variable schema admits DG function spaces.
- **General-algebraic interface BC expression vocabulary.** The general-algebraic interface BC kind (anticipated for v2, §1.6.2) will need operators for trace fluxes (`flux_trace(u)` or similar) on both sides of an interface; those will be specified when that BC kind is.

There is intentionally **no** lift / extension operator (sometimes called `extend(·)`) in the formalism, even as a reservation. The membrane–bulk case that most uses suggest — a surface reaction needing the bulk concentration at the membrane — is exactly the `trace(·)` direction (§1.8.2): the bulk basis functions evaluate uniquely at the boundary, no smoothing or auxiliary problem needed. The reverse direction (surface → bulk lift) is mathematically non-unique and shows up only in a few specialised settings — ALE mesh-motion is the most common — where it is more honest to express the extension as a *named bulk variable with its own equation and a boundary condition tying it to the surface variable* than to hide a substantive computational choice behind a one-symbol operator.

#### 1.8.8 Type rules summary

Every slot has a declared type; expressions in that slot must produce a matching type.

| Slot category | Expected type |
|---|---|
| Scalar source, scalar diffusion $D$, Dirichlet `expression`, Neumann `expression`, IC, partition coefficient $k$, constraint, rate, source on T2/T3/T4 | scalar |
| Vector advection, vector `relative_advection`, vector `motion.velocity` | vector in $\mathbb{R}^d$ |
| Tensor diffusion $D$ | symmetric $d \times d$ tensor |
| Robin coefficient fields `alpha`, `beta`, `expression` ($\alpha$, $\beta$, $h$) | three scalars |
| Interface value-equality partition coefficient $k$ | scalar |

Type mismatches are validation errors. A scalar where a vector is expected is **not** implicitly broadcast; the user must write the broadcast explicitly (e.g., `c * geom.normal` to turn a scalar `c` into a vector along the outward normal). The **one exception is the numeric literal `0`**, which is the zero of whichever type a slot expects — `velocity: "0"` and a vector variable's `initial_condition: "0"` are both accepted as the zero vector, so a zero default need not be spelled `[0, 0]`. Any *other* scalar (a nonzero literal, a named scalar, an expression) in a vector or tensor slot is still a no-broadcast error.

### 1.9 Temporality, mixed systems, and DAE structure

#### 1.9.1 Per-equation temporality declaration

Every equation in a MathDescription declares a `temporality` field. The value is one of:

| Value | Meaning |
|---|---|
| `time_dependent` | The equation describes how the variable evolves: it produces a time derivative of the governed variable, which the solver integrates. |
| `steady_state` | The equation describes an algebraic constraint that must hold at every instant: it produces no time derivative of the governed variable, and the solver determines the variable's value instantaneously. |

`temporality` is required. There is no default and no inference from the equation's structure — the user states their intent and the validator (§1.9.5) checks that the equation's form is consistent with the declaration. The rationale (recorded in memory decision 3) is that inferring temporality from "is ∂_t present?" forecloses the mixed-systems case below, where the user genuinely wants to say "this equation has no time derivative but is part of the time-coupled system."

#### 1.9.2 Time-dependent equations

A `time_dependent` equation contributes a rule for how the governed variable evolves. Operationally:

- **Template equations (T1, T2, T4):** the template's `∂_t` slot is generated and assembled. The user does not write `∂_t` explicitly; it is part of the template's mathematical form, activated by the `time_dependent` flag.
- **Weak-form equations:** the user writes `partial_t(u)` for the governed variable $u$ inside the form (§1.5.4). The expression must appear at least once.

In both cases the variable acquires a time evolution, requires an **initial condition** (§1.7), and is part of the solver's time integration loop.

#### 1.9.3 Steady-state equations

A `steady_state` equation contributes an algebraic constraint. Operationally:

- **Template equations:** for templates with an optional `∂_t` slot (T1, T2, T4), the slot is omitted. T3 (algebraic constraint) is always `steady_state`; declaring it otherwise is an error.
- **Weak-form equations:** the form must not contain `partial_t(u)` for the governed variable $u$.

The governed variable does not have a time evolution rule from this equation — its value is determined at every instant by satisfying the constraint. **No initial condition is required, and providing one is an error** (§1.7.1).

A `steady_state` equation may still be re-solved at every time step. Whether it is solved once (in an all-steady model) or repeatedly (in a mixed model) is a model-level property, not a per-equation one — see §1.9.6.

#### 1.9.4 Mixed systems and DAE structure

A MathDescription is **mixed-temporality** when it contains both `time_dependent` and `steady_state` equations. The combined system is a **differential-algebraic equation (DAE)**: time derivatives appear for some variables, algebraic constraints for others, all coupled through shared variables in their expressions.

Canonical examples:

- **Stokes flow.** Time-dependent momentum equation `∂_t v = ∇·σ + f` paired with steady-state continuity `∇·v = 0`. The continuity equation determines pressure (via Lagrange multiplier structure) at every instant; momentum evolves $v$ in time.
- **Quasi-static cell mechanics with diffusing species** (the §1.10.8 worked example). Steady-state force balance for the membrane velocity (mechanical equilibrium at every instant); time-dependent surface PDE for receptor density (evolves on the moving membrane).
- **Reaction-diffusion with a conserved-total constraint.** Time-dependent species equations plus a steady-state integral constraint enforcing $\int_\Omega u \, \mathrm{d}\Omega = M_0$.

The DAE structure is the natural way to express these. Without a per-equation temporality declaration the modeller would have to either pin everything as time-dependent (and somehow encode "this variable has no time derivative") or rewrite the math to eliminate constraints (which loses physical clarity). Per-equation temporality is the cleaner abstraction.

**DAE index.** Mathematically, DAEs have an *index* that measures how far they are from a pure ODE — index-1 systems are tractable with standard implicit solvers; index-2 and higher require careful treatment. The math description does not pin the index. The validator does not compute it. This is a solver-side concern: the backend may choose how to handle the DAE (full DAE solver, index reduction, partitioned approach), and that choice is independent of the math description.

#### 1.9.5 Validation rules

The validator checks the following at MathDescription construction time:

**Temporality ↔ time-derivative consistency:**

- An equation declared `time_dependent` must contain $\partial_t$ of its governed variable. For T1/T2/T4 templates, this means the template's `∂_t` slot is present (always true when `temporality: time_dependent`). For weak-form equations, the form must contain `partial_t(u)` where $u$ is the governed variable.
- An equation declared `steady_state` must not contain $\partial_t$ of its governed variable. T3 may only be `steady_state`. Weak-form equations must not contain `partial_t(u)` for the governed $u$.
- References to *other* variables' time derivatives (e.g. a steady-state weak form that references `partial_t(v)` for some other variable $v$ as a coefficient) are unrestricted. The temporality flag controls the governed variable's evolution; other variables' time derivatives are just coefficient values.

**IC consistency (cross-references §1.7):**

- A `time_dependent` equation must have an `initial_condition` field for its governed variable.
- A `steady_state` equation must not have an `initial_condition` field — **except** when its governed variable is an unknown-motion variable (§1.10.3), whose IC sets the $t = 0$ configuration of the moving subdomain (§1.7.1, §1.7.7). There the IC is optional, not forbidden.

These are the strict-matching rules; they catch a class of silent errors where a `time_dependent` equation forgets its time derivative (and silently becomes an algebraic constraint inside the time loop) or a `steady_state` equation accidentally introduces $\partial_t$ (and silently becomes an evolution equation without an IC).

#### 1.9.6 Whole-model classifications

The model itself acquires a temporality from its equations:

| Model has... | Model is... | Solver behaviour |
|---|---|---|
| All equations `steady_state` | A **steady-state model** | Solved once as a nonlinear algebraic system. No time stepping. No ICs required anywhere. |
| At least one equation `time_dependent` | A **time-dependent (possibly DAE) model** | Solved over a time interval $[0, T]$. Time-stepping required. ICs required for every time-dependent variable. Steady-state equations are re-solved at every time step alongside the integration. |

The model's temporality is implied, not declared. The user does not write a top-level `temporality: ...` field; it is determined by aggregating the per-equation declarations.

#### 1.9.7 Solver-side concerns (briefly)

The following are explicitly *not* part of the MathDescription, even though they materially affect a mixed-temporality simulation:

- **Time-stepping scheme.** BE, CN, BDF2, IMEX splits, Runge-Kutta variants — backend's choice. The math description's `partial_t(u)` is continuous-time and discretization-agnostic.
- **DAE index reduction.** If the system is high-index, the backend may rewrite it (Pantelides algorithm and friends). The math description states the well-posed problem; rewriting is a solver-side optimisation.
- **Predictor-corrector iteration for stiff couplings.** Monolithic vs partitioned solve of the coupled system at each step — backend's choice.
- **Tolerance and convergence criteria.** Always solver-side.

The math description says *what* the well-posed time-coupled DAE is; the backend says *how* it is integrated.

#### 1.9.8 Worked example reference

The §1.10.8 example (viscous force balance for membrane motion with a receptor density on the moving membrane) is the canonical mixed-temporality model in this document:

- The motion equation is `temporality: steady_state` — mechanical equilibrium at every instant.
- The receptor density equation is `temporality: time_dependent` — diffusion plus reaction plus auto-dilution from the membrane's motion.
- The two are coupled through `subdomain.motion.variable = v_membrane`: the receptor's surface-PDE template reads the solved motion at each time step to compute its dilution term.

A pure steady-state model would have all equations declared `steady_state` and no ICs anywhere; a pure time-dependent model (no constraints) would have all equations `time_dependent` and ICs for every variable. Most cell-biology models with mechanics fall between these — quasi-static mechanics balances paired with time-evolving species — which is exactly what mixed-temporality systems are for.

### 1.10 Moving subdomains

#### 1.10.1 The three motion kinds

Every subdomain carries a `motion` field whose `kind` is one of:

| Kind | Meaning |
|---|---|
| `none` | Subdomain is static. Default when `motion` is omitted. Substrate velocity is zero everywhere; compression / dilution terms in operator templates vanish; geometric quantities (`geom.normal`, `geom.mean_curvature`, etc.) evaluate against the initial (and only) configuration. |
| `prescribed` | Subdomain moves according to an expression the user supplies. The expression may be a velocity field or a displacement field (§1.10.2); the formalism converts internally. The substrate velocity feeds compression / dilution terms automatically. |
| `unknown` | Subdomain moves according to a motion variable solved by an equation in the same MathDescription. The motion variable is a regular vector variable (§1.3.4); the equation governing it is any equation that produces a matching vector field on the right subdomain. This is the mechanics-driven-migration path; it is the central capability v1 builds toward even though v1's template library does not yet include mechanics templates (T5–T7). |

The default is `none`. When `kind: prescribed` or `kind: unknown`, the substrate velocity at every point on the subdomain is the value of the relevant motion field at that point. All variables on the subdomain experience the same substrate velocity (memory decision 6); equation templates that have a substrate-velocity-dependent term (compression in T1, dilution in T2) handle it without per-equation user input.

#### 1.10.2 Prescribed motion: velocity and displacement forms

Either form may be used; not both at once:

| Form | Schema | Meaning |
|---|---|---|
| Velocity | `motion: { kind: prescribed, velocity: <vector_expr> }` | The substrate velocity field $\mathbf{v}_\Omega(\mathbf{x}, t)$. The current configuration is obtained by integrating $\dot{\mathbf{x}} = \mathbf{v}_\Omega$ from the reference (initial) configuration. |
| Displacement | `motion: { kind: prescribed, displacement: <vector_expr> }` | The displacement field $\mathbf{d}(\mathbf{X}, t)$ at reference point $\mathbf{X}$. The current position is $\mathbf{X} + \mathbf{d}$. The substrate velocity at a given current point is $\partial \mathbf{d}/\partial t$, computed by the backend. |

The two forms are mathematically equivalent: a displacement field uniquely determines a velocity field (by time-differentiation along the reference point), and a velocity field uniquely determines a displacement field (by time-integration along the material point, given the reference configuration). The form a modeller picks should match the natural specification of the motion they intend:

- **Velocity form is natural for**: radial expansion (`r_dot * geom.x / geom.radius`), fluid-driven motion, prescribed steady flow.
- **Displacement form is natural for**: rigid translations (`[v_x, v_y, v_z] * sim.t`), oscillations (`A * sin(omega * sim.t) * e_1`), prescribed wall deformations.

The backend's job is to convert as needed for its assembly; the user does not need to do this conversion.

**Type constraints.** For a `volume` subdomain, the velocity or displacement is a vector in the ambient dimension $\mathbb{R}^d$. For a `surface` subdomain, the velocity may include both tangential and normal components (memory decision 6a — tangential material flow is a real physical phenomenon, not an edge case). For a `point` subdomain, the velocity is a vector specifying how the point moves through space.

#### 1.10.3 Unknown motion: motion variable and governing equation

When `motion.kind: unknown`, the user provides the *name* of a motion variable; that variable is declared in the `variables:` block like any other vector variable, on the same subdomain whose motion it represents. An equation governing this motion variable must exist elsewhere in the MathDescription.

```yaml
subdomains:
  - name: membrane
    kind: surface
    motion: { kind: unknown, variable: v_membrane }

variables:
  - { name: v_membrane, subdomain: membrane, type: vector }

equations:
  - template: <any template producing a vector field on `membrane`>
    variable: v_membrane
    subdomain: membrane
    temporality: <as appropriate>
    # ...slots / form / etc.
```

**Equation choice is open.** Any equation that produces a matching vector field on the motion variable's subdomain is acceptable: a mechanics template (T5 Stokes, T6/T7 elasticity — v2+); a T1-style steady Laplace equation for ALE harmonic mesh-velocity extension; a custom force balance via the weak-form escape hatch (§1.5). v1 validates only that *some* equation governs the variable, on the right subdomain, producing the right type.

**Practical v1 caveat.** Because v1 does not ship mechanics templates, unknown-motion equations in v1 will typically use the weak-form escape hatch (§1.5). The schema and validator support unknown motion fully; the limitation is template-library coverage, not formalism design.

**Temporality of the motion equation.** May be `time_dependent` (the motion variable evolves under, e.g., an inertia-bearing momentum equation) or `steady_state` (the motion is solved as a quasi-static balance at each time step — typical for low-Reynolds cell-mechanics where inertia is negligible). The choice belongs to the user and is independent of the temporality of other equations in the same MathDescription.

#### 1.10.4 Reference configuration and initial conditions

The **reference configuration** is the initial mesh as loaded from the geometry. The MathDescription does not provide a mechanism to specify a different reference configuration; if a model needs a non-mesh reference, the geometry is the right place to express that, not the math.

**Initial conditions for unknown-motion variables** default to zero. The interpretation:

- For a displacement-typed motion variable: $\mathbf{d}(\mathbf{X}, 0) = \mathbf{0}$ means "the initial configuration equals the reference configuration." This is the overwhelmingly common case — the cell starts where the mesh says it is.
- For a velocity-typed motion variable: $\mathbf{v}(\mathbf{x}, 0) = \mathbf{0}$ means "starts at rest." Also the common case.

Users may override the IC to specify a non-zero initial displacement (a deformed starting configuration) or initial velocity (an in-progress motion). The schema accepts this; the validator enforces type-compatibility (vector IC for vector variable, etc.).

**ICs for other variables on the moving subdomain** are evaluated on the initial (t=0) configuration. For unknown-motion subdomains, that is the configuration produced by the motion variable's IC, which by default is the reference configuration. Composability is direct: writing `initial_condition: "1.0 + 0.5 * cos(2 * geom.azimuth)"` for a receptor density refers to angular coordinate $\theta$ on the *initial* membrane — exactly what the modeller intends.

#### 1.10.5 Material vs. Eulerian conventions in operator templates

Operator templates write equations in **Eulerian conservation form**:

$$\partial_t u \;+\; \nabla \cdot (u \, \mathbf{v}_\Omega) \;=\; \nabla \cdot (D \, \nabla u) \;+\; s$$

with $\partial_t u$ the partial-time derivative at fixed lab-frame coordinate and $\mathbf{v}_\Omega$ the substrate velocity from `subdomain.motion`. Expanding the flux divergence,

$$\nabla \cdot (u \, \mathbf{v}_\Omega) \;=\; \mathbf{v}_\Omega \cdot \nabla u \;+\; u \, \nabla \cdot \mathbf{v}_\Omega,$$

gives two pieces: an advection-by-substrate-motion term $\mathbf{v}_\Omega \cdot \nabla u$, and a compression / dilution term $u \, \nabla \cdot \mathbf{v}_\Omega$. The template computes both from `subdomain.motion` and folds them into assembly. The user never writes them; the user never even needs to choose a frame.

This convention applies uniformly across T1 (bulk, $u \, \nabla \cdot \mathbf{v}_\Omega$), T2 (surface, $\rho \, \nabla_\Gamma \cdot \mathbf{v}_\Gamma$), and any future template with a $\partial_t$ slot.

**Connection to the material derivative.** If $\mathbf{v}_\Omega$ is the actual material velocity of the substrate, the conservation form is exactly equivalent to the Lagrangian form $D_t u + u \, \nabla \cdot \mathbf{v}_\Omega = \ldots$ where $D_t = \partial_t + \mathbf{v}_\Omega \cdot \nabla$ is the material derivative. The Eulerian form is preferred in the formalism because it matches the standard FE assembly pattern on a moving mesh (with ALE mapping handled by the backend) and because $\partial_t u$ is unambiguous at the user-expression level — it is the time derivative the FE solver computes.

**The weak-form escape hatch (§1.5) is different.** When a user writes a UFL form directly, they pick the convention — they may write a Lagrangian D_t form, an Eulerian conservation form, or an ALE form with explicit mesh velocity. The user is responsible for consistency with the geometry's motion and with any other equation that interacts with the same variables.

#### 1.10.6 Moving labelled boundaries

A labelled boundary of a moving subdomain moves *with* the subdomain. Specifically:

- The boundary's incidence — which subdomain class(es) it bounds — is invariant under motion. A `membrane` that bounds `cytoplasm` and `extracellular` continues to bound them at every $t$, no matter how it deforms.
- The geometric position of the boundary at time $t$ is the image of its initial position under the subdomain's motion map.
- BC expressions evaluated on the boundary use the boundary's current (deformed) position. Geometric helpers (`geom.normal`, `geom.mean_curvature`, etc.) on a moving boundary reflect the current configuration.

The implication for BCs is mostly transparent: a Dirichlet BC `u = f(x, t)` on a moving boundary evaluates `f` at the current position $x$ on the deformed boundary. A Neumann BC's flux is the flux through the current boundary surface; the outward normal $\mathbf{n}$ is the current outward normal, not the reference one.

#### 1.10.7 What is solver-side, not in the math description

The following are explicitly *not* part of the MathDescription, even though they materially affect a moving-domain simulation:

- **ALE mesh-motion algorithm.** Whether the bulk mesh follows the membrane motion exactly, follows a harmonic-extension velocity field, follows a fictitious elastic-extension, or uses a different recipe entirely — backend choice. Different backends may make different choices for the same MathDescription.
- **Mesh remeshing for large deformations.** When the deformation is large enough that mesh quality degrades, some backends remesh adaptively. Triggering conditions, remeshing algorithms, and field-transfer schemes are solver-side concerns.
- **Time-stepping for the motion variable.** Whether the motion variable's update is implicit or explicit, whether it is solved monolithically with field variables or in a partitioned manner — all backend choices.
- **Numerical stabilisation for advection-dominated regimes.** SUPG, GLS, entropy-viscosity stabilisations are stabilisation choices; they do not change the math problem.

The math description states the well-posed problem; the backend chooses how to solve it.

#### 1.10.8 Worked sketch — mechanics-driven membrane motion with a surface species

The first model that pushes beyond the §1.4.5 / §1.6.6 prescribed-motion examples: a closed membrane whose motion is solved by a simple viscous force balance, with a receptor density on the membrane that experiences the resulting motion via the standard T2 dilution. Here the focus is **unknown motion** — the mechanics-driven path to a non-prescribed substrate velocity. The same model returns in §1.5.8 with the weak-form escape hatch in the foreground, and in §2.7 as the complete YAML / JSON / Python end-to-end example.

```yaml
math_description:
  geometry: cell_2d

  subdomains:
    - name: membrane
      kind: surface
      motion:
        kind: unknown
        variable: v_membrane

  variables:
    - { name: v_membrane, subdomain: membrane, type: vector }   # motion field, vector in R^2
    - { name: rho,        subdomain: membrane, type: scalar }   # receptor density

  equations:
    # Motion equation: simple viscous force balance with prescribed active traction.
    # In v1, this uses the weak-form escape hatch (§1.5) because mechanics templates
    # (T5-T7) ship in v2. The form below is illustrative; full UFL is left to §1.5.
    #
    #   eta * v_membrane = -sigma_T * geom.mean_curvature * geom.normal + f_active
    #
    # where eta is drag, sigma_T is surface tension, H is mean curvature,
    # n is outward normal, and f_active is a vector-valued expression
    # parameter (§2.2.3) declared below.
    - template: weak_form               # §1.5
      variable: v_membrane
      subdomain: membrane
      temporality: steady_state          # quasi-static at each time step
      # The form is the residual; the equation is "form = 0" for all
      # admissible test functions (§1.5). Inline YAML comments inside a `|`
      # block are part of the string, so the "= 0" note lives outside the form.
      form: |
        ( eta * inner(v_membrane, v_membrane_test)
          + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)
          - inner(f_active, v_membrane_test) ) * dx_Gamma
      initial_condition: "0"              # zero default (memory decision 11c)

    # Receptor density: standard T2 surface PDE. Dilution from v_membrane is automatic.
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.05
        source: "-k_off * rho"
      initial_condition: "1.0 + 0.3 * cos(2 * geom.azimuth)"

  parameters:
    - { name: eta,     value: 1.0  }     # viscous drag coefficient
    - { name: sigma_T, value: 0.10 }     # surface tension
    - { name: k_off,   value: 0.02 }     # receptor decay rate
    - { name: f0,      value: 0.3  }     # active-traction amplitude
    - name: f_active                     # polarised active traction (vector)
      type: vector
      subdomain: membrane                # uses geom.azimuth — scope required (§2.2.3)
      expression: "[f0 * cos(geom.azimuth), 0]"
```

What this sketch demonstrates:

- **`motion: { kind: unknown, variable: v_membrane }`** wires the membrane's substrate velocity to a vector unknown solved on the same subdomain.
- **The motion equation uses the §1.5 weak-form escape hatch** — because v1 has no surface-mechanics template, the user writes a UFL form directly. v2 will replace this with a T5/T6/T7-style mechanics template.
- **The receptor density equation is unchanged from §1.4.5** in shape — only `motion.kind` changed from `prescribed` to `unknown`. The T2 template picks up the dilution automatically from `membrane.motion`, whether prescribed or solved. The user does not edit the receptor equation when switching motion modes.
- **`initial_condition: 0` for `v_membrane`** uses the zero default (memory decision 11c). The mesh starts at rest at t = 0; the force balance immediately produces a non-zero velocity in response to the initial curvature and traction.
- **Geometric helpers `geom.mean_curvature`, `geom.normal`** evaluate against the current deformed membrane configuration (§1.10.6). At t = 0 that is the geometry's initial configuration.

This sketch is intentionally minimal — a single membrane, one mechanics balance, one surface species. Real cell-migration models add bulk hydrodynamics, multiple surface species, intracellular signalling, and adhesion-with-slippage to substrates. Each of those is expressible in the formalism (the deferred constitutive templates from §1.4.3 and the mechanism for unknown motion + governing equation) once the matching templates and adhesion vocabulary ship.

### 1.11 Well-posedness checks

#### 1.11.1 Purpose and stance

A MathDescription that passes validation is *structurally* well-posed: every name resolves, every required field is provided, every typed slot receives a value of the right type, and the temporal / boundary structure is internally consistent. The validator does **not** prove the resulting math problem has a unique solution — that is a deeper property of the equations themselves, depending on coefficient signs, coercivity, inf-sup conditions, and many other things no static check can verify on arbitrary user expressions. The validator's job is to catch the class of errors that produce silently wrong models: missing time derivatives, mismatched types, dangling references, conflicting BCs. The user remains responsible for the math.

The rules are stated below in categories, with forward-references back to the section where each rule was originally introduced. They are checks the schema runs at MathDescription construction time; they do not require a solver, a geometry beyond the name-level interface, or any numerical computation.

#### 1.11.2 Errors versus warnings

Validation outputs distinguish two severities:

- **Errors** prevent MathDescription construction. The model is malformed; no solve can run.
- **Warnings** allow construction but flag a suspicious or possibly-wrong pattern that the validator cannot prove is incorrect.

Most rules below are errors. The few warning cases are noted explicitly. Examples of warnings: IC ↔ Dirichlet compatibility with complex expressions the validator cannot evaluate symbolically; non-trivial expression patterns that suggest user intent might be different from what the syntax says.

#### 1.11.3 Reference resolution

**Errors.** Every name appearing in a MathDescription must resolve:

- Variable names referenced in equations, BCs, ICs, or motion-variable slots must be declared in the `variables:` block.
- Parameter names referenced in expressions must be declared in the `parameters:` block.
- Subdomain class names referenced anywhere must be declared in the `subdomains:` block (§1.2).
- Labelled boundary names referenced in BCs must exist in the referenced geometry (§1.11.10).
- Reserved names — the namespace roots `geom` / `sim` and the integration measures — must not be taken by variable, parameter, or subdomain names. Operator and function names (`grad`, `sin`, …) are reserved only in call position and never collide with the bare value namespace (ADR 006). Shadowing a reserved name is an error at construction (§1.8.6, §2.4.1).
- A bare name in an expression must resolve to a local variable (defined on the same subdomain as the expression's evaluation context), a parameter, or a reserved name. References to variables on a *different* subdomain require `trace(·)`; lower-dimensional variables cannot be referenced at all (§1.8.6, §1.8.7).
- **Parameter expression cycles are an error.** Parameter expressions may reference other parameters (§2.2.3); the validator topologically sorts the parameter graph and rejects any cycle. The error message names the cycle's members.

#### 1.11.4 Coverage rules

**Errors.**

- Every variable must be governed by exactly one equation. A variable without an equation is undetermined; a variable with two or more equations is overdetermined.
- Every equation declared `temporality: time_dependent` must have an `initial_condition` (§1.7.1, §1.9.5).
- Every equation declared `temporality: steady_state` must **not** have an `initial_condition` (§1.7.1, §1.9.5) — except when its governed variable is an unknown-motion variable, whose IC sets the $t = 0$ configuration (§1.7.7); there the IC is optional, not forbidden.
- Every variable referenced in any equation, BC, or IC must be declared.
- Every internal boundary touched by a variable must have at least one explicit BC for that variable (no zero-Neumann default on internal boundaries; §1.6.4).
- For region-keyed parameter maps (`kind: region_map`), every region of the named subdomain class must have a value in the map (§1.2.5).

External boundaries have a zero-Neumann default, so missing BCs there are *not* errors.

#### 1.11.5 Type rules

**Errors** (cross-references §1.4.4, §1.8.8, §1.7.4):

- A slot expression's value type must match the slot's declared type. Scalar slots require scalar values; vector slots require vectors in $\mathbb{R}^d$; tensor slots require symmetric $d \times d$ tensors.
- No implicit broadcast. A scalar where a vector is expected must be made explicit (e.g., `c * geom.normal` to broadcast scalar $c$ along the normal).
- Calculus operators' argument and result types must be honoured (`grad(u)` for scalar $u$ returns a vector; `div(v)` for vector $v$ returns a scalar; etc., per §1.8.5).
- Variable initial conditions must match the variable's declared type (§1.7.4).
- The Robin coefficient fields `alpha`, `beta`, `expression` ($\alpha$, $\beta$, $h$ in §1.6.2) must each be scalar.
- Interface BC `partner_variable` must be defined on a subdomain incident to the BC's boundary from the opposite side (§1.6.2).

#### 1.11.6 Temporality consistency

**Errors** (cross-references §1.9.5):

- An equation declared `time_dependent` must contain $\partial_t$ of its governed variable. For T1/T2/T4 templates this is automatic when the flag is set. For weak-form equations, the form must contain `partial_t(u)` where $u$ is the governed variable.
- An equation declared `steady_state` must **not** contain $\partial_t$ of its governed variable.
- T3 (algebraic constraint) may only be `steady_state`.
- References to $\partial_t$ of *other* variables (variables not governed by this equation) in an expression are unrestricted — those are coefficient values, not evolution rules for this equation.

#### 1.11.7 Boundary-condition consistency

**Errors** (cross-references §1.6.4, §1.5.6):

- No two BCs for the same (variable, boundary) pair may declare conflicting kinds (e.g. Dirichlet and Neumann on the same variable and boundary).
- Interface BCs must have a `partner_variable` defined on a subdomain incident to the BC's boundary from the opposite side.
- Dirichlet BCs must be declared on a labelled boundary that exists; the boundary must be incident to a subdomain on which the variable lives.
- For weak-form equations (template = `weak_form`), §1.6 BCs on the governed variable must all be Dirichlet. Non-Dirichlet §1.6 BCs on a weak-form-governed variable are an error (§1.5.6); the user must encode natural BCs in the form itself.
- **`interface_flux_balance` requires both sides to be `volume` (bulk) subdomains.** The kind enforces mass conservation across an interface where mass *crosses* but does not *accumulate*. Bulk-surface couplings (where mass accumulates on the surface, e.g. ligand binding to membrane receptors) are not expressible as flux-balance and are an error here; use the composable Neumann + source pattern in §1.6.5 instead. The validator rejects flux-balance entries whose `variable` or `partner_variable` lives on a non-volume subdomain, with an error message pointing the user at §1.6.5.
- Trace ambiguity: when an expression's `trace(u)` would have multiple resolutions (e.g. higher-dim $u$ contributes regions on both sides of the lower-dim evaluation context), the reference is ambiguous and rejected. In v1 the modeller's recourse is Tier 3 (separate subdomain classes); v2 will add side-or-region specifiers (§1.2.6, §1.8.2).

#### 1.11.8 Initial-condition consistency

**Errors** (cross-references §1.7):

- ICs must not reference other state variables.
- ICs must not reference time `t`.
- IC type must match the variable's declared type.

**Warnings**:

- At points on a Dirichlet boundary where both an IC and a Dirichlet BC apply, the IC value should equal the Dirichlet BC value at $t = 0$. The validator checks this for syntactically simple cases (constant expressions, polynomial expressions whose boundary trace is computable) and emits a warning when it cannot prove agreement but cannot prove disagreement either (§1.7.5).

#### 1.11.9 Operator usage rules

**Narrow rule — errors** (cross-references §1.8.5):

- In operator-template slots (T1–T7), calculus operators (`grad`, `div`, `lapl`, `grad_surf`, `div_surf`, `lapl_beltrami`) may not be applied to the equation's own governed variable. Applications to *other* variables are unrestricted.
- The validator walks the slot expression's AST: if any calculus operator's argument resolves (directly or via composition) to the slot's governed variable, that is an error. The rule does not apply to weak-form equations (§1.5.3); there, the user has full UFL expressiveness and owns well-posedness.

**Smoothness requirement — errors** (cross-references §1.8.5):

- Second-order calculus operators (`lapl`, `lapl_beltrami`) require their argument to live in a function space that admits a meaningful strong second derivative. v1 enforces this by requiring the argument variable's `space` to be `lagrange_p2` or higher. Applying `lapl(u)` to a variable with `space: lagrange_p1` (the default) is an error, because the strong Laplacian of a piecewise-linear function is element-wise zero and singular on facets — the user almost certainly did not intend that.
- The validator walks the expression AST and checks the `space` of every variable that appears under a second-order operator. The error message names the offending operator, the offending variable, and the suggested fix (raise the variable's `space` to `lagrange_p2`, or use the weak-form escape hatch with explicit integration-by-parts).
- First-order calculus operators (`grad`, `div`, `grad_surf`, `div_surf`) have no v1 smoothness restriction beyond the default `lagrange_p1`.
- This rule does not apply to weak-form equations (§1.5.3); the escape-hatch user owns the smoothness consequences of their UFL forms.

#### 1.11.10 Geometry interface compatibility

**Errors.** Cross-checks between the MathDescription and its referenced Geometry (§1.2.1):

- Every subdomain class name declared in the MathDescription must have a same-named class in the Geometry, with the same `kind`.
- Every labelled boundary name referenced in a BC must exist in the Geometry.
- Every internal boundary referenced by an interface BC must in fact bound two subdomains in the Geometry, matching the BC's `variable.subdomain` and `partner_variable.subdomain`.
- Region-keyed parameter maps (`kind: region_map`) must cover every region the Geometry assigns to the parameter's subdomain class — missing regions are errors, not silent zero defaults.
- **Expression-valued parameter scoping (§2.2.3).** A parameter whose body expression references any subdomain-relative geometry quantity (`geom.normal`, `geom.mean_curvature`, `geom.curvature1`, `geom.tangent`, `geom.azimuth`, `geom.radius`, …) must declare a `subdomain:` scope. Any *use* of a scoped parameter must be from an expression whose evaluation context is on (or a sub-entity of) the parameter's scope subdomain. A use from an incompatible context is an error pointing both to the parameter declaration and the offending use site.

These checks require the Geometry to be available at MathDescription validation time. If the Geometry is loaded lazily (typical at solve time), some of these checks are deferred until both are present. The validator may still run all *intra*-MathDescription checks without the Geometry.

#### 1.11.11 What the validator cannot check

The following are explicitly outside the validator's scope. They are the user's responsibility, and they are the deepest reason validation cannot prove a model is well-posed:

- **Coercivity, ellipticity, inf-sup conditions.** Whether a given bilinear form admits a unique solution depends on the sign and structure of its terms (positive-definite diffusion, properly-paired mixed spaces, etc.). The validator does not check these.
- **Sign conventions.** Whether the user wrote $\nabla \cdot (D \nabla u)$ with the correct sign for a diffusion equation (or accidentally $-\nabla \cdot (D \nabla u)$) is invisible to a static walk of the AST.
- **Physical consistency.** Whether $k_{on}$ and $k_{off}$ in a binding reaction have the right dimensions, whether $D$ has units of length²/time, whether the chosen $\sigma_T$ produces a meaningful tension — none of these can be checked without unit annotations the formalism does not currently require. Unit-aware validation is a possible v2 enhancement.
- **Conservation across coupled bulk-surface expressions.** The §1.6.5 composable pattern relies on the user writing matched expressions in three places (bulk BC, surface source for the bound form, surface source for the free form, with appropriate signs). The validator does not enforce that they are consistent; silent mass leakage is a class of error the user is responsible for catching, e.g. through dedicated mass-balance test cases.
- **Well-posedness of the user's weak forms.** The escape hatch trades guardrails for expressiveness. A form whose bilinear part is singular, whose test-function structure does not match the trial-function space, or whose boundary integrals are missing terms required for integration-by-parts consistency will silently produce a malformed problem. The validator catches name-resolution and type-level errors only.

These limits define the validator's stance: catch the structural mistakes a static check *can* catch, and trust the modeller for the rest.

---

## Part 2 — Data model

Part 1 specified what is in a MathDescription. Part 2 specifies *how* a MathDescription is represented — its concrete syntactic carrier, its schema, the expression-language parser, and the rules by which references resolve. The carrier is the bridge from the abstract formalism to something a modeller can write and a backend can load.

### 2.1 Schema overview

#### 2.1.1 Three carriers, one model

A MathDescription ships in three forms that describe the same underlying data model:

| Carrier | Purpose | Audience |
|---|---|---|
| **YAML** | Human-edited form; the primary authoring interface. | Modellers writing models by hand. Matches every worked example in Part 1. |
| **JSON** | Canonical machine interchange and storage. | Tools, CI, archives, and any consumer that wants a parser without a YAML dependency. |
| **Python dataclasses** | In-memory representation. | The `vcell-fenics` runtime; `pyvcell` integration; programmatic model construction. |

A single parser / serializer round-trips among the three: YAML ↔ JSON ↔ dataclasses, with the dataclass form as the canonical AST. No information is lost going through any of the three forms.

YAML and JSON differ only syntactically. YAML's nested-block format is what the documentation has shown; JSON is the same shape with `{}` / `[]` punctuation and quoted keys. A YAML model and the equivalent JSON model parse to identical dataclass instances.

#### 2.1.2 The top-level envelope

A MathDescription has the following top-level fields:

```yaml
math_description:
  geometry: <string>            # name of the external Geometry object — required
  subdomains: [...]             # list of Subdomain entries (§2.2.1) — required, non-empty
  variables: [...]              # list of Variable entries (§2.2.2) — required, non-empty
  equations: [...]              # list of Equation entries (§2.2.4 / §2.2.5) — required, non-empty
  parameters: [...]             # list of Parameter entries (§2.2.3) — optional, defaults to []
  boundary_conditions: [...]    # list of BoundaryCondition entries (§2.2.6) — optional, defaults to []
```

`geometry`, `subdomains`, `variables`, and `equations` are required (a model with no equations cannot be solved). `parameters` and `boundary_conditions` are optional and default to empty lists when omitted — a model with no named constants or no boundary conditions is a normal case (closed-membrane surface PDEs need no BCs at all, §1.6.6). The order within any list is not semantically significant; references are by name.

There is no top-level `temporality` or `motion` declaration — both are derived from per-equation and per-subdomain fields. There is no top-level `solver` block either; solver configuration is a separate object (Part 3).

### 2.2 Schema by entity

#### 2.2.1 Subdomain

```yaml
subdomains:
  - name: <string>              # unique within the MathDescription
    kind: volume | surface | curve | point
    motion:                     # optional; defaults to { kind: none }
      kind: none | prescribed | unknown
      # additional fields per `kind`, below
```

`motion` field shapes:

- **`kind: none`** — no further fields. Subdomain is static.
- **`kind: prescribed`** — exactly one of `velocity` or `displacement`:
  ```yaml
  motion: { kind: prescribed, velocity: "<vector_expr>" }
  motion: { kind: prescribed, displacement: "<vector_expr>" }
  ```
- **`kind: unknown`** — references a motion variable:
  ```yaml
  motion: { kind: unknown, variable: <variable_name> }
  ```
  The referenced variable must be declared in `variables:`, must be `type: vector`, must have `subdomain: <this subdomain>`, and must be governed by some equation (§1.10.3, §1.11.4).

The `kind` discriminator in `motion` selects which fields are valid. Cross-mixing (e.g. `kind: none` with a `velocity` field, or `kind: prescribed` with both `velocity` and `displacement`) is rejected at construction time.

#### 2.2.2 Variable

```yaml
variables:
  - name: <string>              # unique within the MathDescription
    subdomain: <subdomain_name>
    type: scalar | vector | symmetric_tensor    # default: scalar
    space: <function_space_hint>                # optional; default: lagrange_p1
```

The `space` hint values are catalogued in §1.3.3. Backends are not required to honour every hint; an unsupported hint is a solver-side error.

The pair `(name, subdomain)` uniquely identifies a variable. The same variable name on different subdomains denotes two distinct variables (per §1.3.1).

#### 2.2.3 Parameter

A parameter is a named input to the math problem that does not evolve in time as a state variable. v1 supports three forms.

**(a) Constant scalar — the common case:**

```yaml
parameters:
  - { name: <string>, value: <number> }
```

Shorthand for `{ name, kind: scalar, value }`. Most rate constants, diffusion coefficients, and reservoir values are constants.

**(b) Expression — scalar, vector, or symmetric tensor:**

```yaml
parameters:
  - name: <string>
    type: scalar | vector | symmetric_tensor   # default: scalar
    subdomain: <subdomain_name>                 # optional; see scoping below
    expression: "<expression>"
```

The `expression` body is an expression in the §1.8 vocabulary: `sim.t`, the `geom.*` quantities, standard functions, and references to other parameters. It is **not** a function declaration (no formal arguments) — it is the parameter's *value*, which happens to depend on the evaluation point. At each use, the parameter resolves to its expression evaluated in the surrounding context (current `geom.x`, current `sim.t`, current subdomain).

This covers time-varying boundary values (`L_reservoir = "1.0 + 0.5 * sin(omega * sim.t)"`), prescribed forcing fields (`f_active = "[f0 * cos(geom.azimuth), 0]"`), and any data field that is a known function of space-time but not a state variable.

**Scoping rules for expression parameters:**

- The `subdomain:` field is optional. When provided, the parameter may only be referenced from expressions whose evaluation context is on (or a sub-entity of) that subdomain — geometric helpers like `geom.mean_curvature` and tangent-frame quantities are only meaningful inside the scope they were written for.
- **`subdomain:` is required** if the expression body uses any subdomain-relative geometry quantity (`geom.normal`, `geom.mean_curvature`, `geom.curvature1`, `geom.tangent`, `geom.azimuth`, `geom.radius`, etc.). Without a scope, the validator cannot tell where the quantity is meaningful.
- Parameters with no geometric-helper references are unscoped by default and may be used from any expression context.

**Parameter-to-parameter references** are permitted (e.g. `omega = 2 * pi * freq`). The validator topologically sorts the parameter graph at construction time and rejects any cycle (§1.11.3).

**(c) Region-keyed — Tier 1 region variation (§1.2.5):**

```yaml
parameters:
  - name: <string>
    kind: region_map
    subdomain: <subdomain_name>
    values:
      <region_name>: <number>
      <region_name>: <number>
      # ... one entry per region of the subdomain class
```

Region-keyed parameters resolve at expression-evaluation time using the geometry's region-to-class assignment. Every region of the named subdomain class must have an entry in `values` (validator-checked, §1.11.4). v1 supports per-region *constants* only; per-region expressions are deferred to v2 (memory: the v1-vs-v2 scope was explicitly chosen to keep region-keyed parameters simple).

#### 2.2.4 Equation — template form

```yaml
equations:
  - template: <template_name>           # e.g. bulk_radv_diff, surface_pde_with_dilution, ...
    variable: <variable_name>
    subdomain: <subdomain_name>
    temporality: time_dependent | steady_state
    terms:
      <slot_name>: "<expression>"
      # one entry per template slot the user wants to fill;
      # omitted slots default to zero per §1.4.4
    initial_condition: "<expression>"   # required iff temporality = time_dependent
```

Template names for v1: `bulk_radv_diff` (T1), `surface_pde_with_dilution` (T2), `algebraic_constraint` (T3), `lumped_ode` (T4). Template slot names per template are catalogued in §1.4.2.

The validator checks that every slot in `terms` is a valid slot name for the template and that the expression types match the slot's declared types (§1.11.5).

#### 2.2.5 Equation — weak-form

```yaml
equations:
  - template: weak_form
    variable: <variable_name>
    subdomain: <subdomain_name>
    temporality: time_dependent | steady_state
    form: |
      <UFL-style residual expression — see §1.5 and §2.3.5>
    initial_condition: "<expression>"   # required iff temporality = time_dependent
```

The `form` field carries the full residual; the equation is `form = 0` for all admissible test functions, per §1.5. The form references `<variable>_test` for the test function and uses the measure vocabulary catalogued in §1.5.3.

#### 2.2.6 Boundary condition

```yaml
boundary_conditions:
  - variable: <variable_name>
    boundary: <boundary_name>            # labelled boundary in the Geometry
    kind: dirichlet | neumann | robin | interface_value_equality | interface_flux_balance
    # additional fields per `kind`:

  # Dirichlet / Neumann: scalar expression
  - { variable: u, boundary: outer, kind: dirichlet, expression: "<expr>" }
  - { variable: u, boundary: outer, kind: neumann,   expression: "<expr>" }

  # Robin: triple
  - variable: u
    boundary: outer
    kind: robin
    alpha: "<scalar_expr>"
    beta: "<scalar_expr>"
    expression: "<scalar_expr>"          # the h in alpha*u + beta*D*grad(u)·n = h

  # Interface value-equality
  - variable: u_left
    partner_variable: u_right
    boundary: membrane
    kind: interface_value_equality
    expression: "<scalar_expr>"          # the partition coefficient k; default 1

  # Interface flux-balance
  - variable: u_left
    partner_variable: u_right
    boundary: membrane
    kind: interface_flux_balance
    expression: "<scalar_expr>"          # the constitutive flux expression
```

Interface kinds require `partner_variable` per §1.6.2. The validator checks all the BC consistency rules from §1.11.7 (no conflicting kinds on the same `(variable, boundary)`; partner subdomain incidence is correct; Dirichlet-only restriction on weak-form-governed variables; etc.).

### 2.3 Expression language

#### 2.3.1 Surface syntax

Expression strings in YAML use math-like infix notation. The syntactic primitives:

- **Numeric literals.** `0`, `1.0`, `0.5`, `1.0e-3`, scientific notation per usual.
- **Bare names** resolve per §1.8.6: local variable → parameter. Examples: `rho`, `k_on`. (Built-ins are qualified — `geom.x`, `sim.t` — not bare.)
- **Indexed access** on the spatial coordinate: `geom.x[0]`, `geom.x[1]`, `geom.x[2]`.
- **Binary arithmetic**: `+`, `-`, `*`, `/`, `**` (power). Standard precedence.
- **Unary minus**: `-expr`.
- **Function calls**: `f(arg1, arg2, ...)` for every function in the vocabulary — `sin`, `cos`, `exp`, `sqrt`, `trace`, `grad`, `inner`, `if`, `partial_t`, etc.
- **Vector and tensor literals**: `[a, b]`, `[a, b, c]` for vectors; tensor literals are nested lists, e.g. `[[a, b], [c, d]]`.
- **Parentheses** for grouping, as expected.

There are no statements, no assignments, no control flow outside the `if(cond, a, b)` form. Expressions are pure.

Examples seen in Part 1's worked models:

```
"1.0"
"1.0 + 0.5 * cos(2 * geom.azimuth)"
"-k_off * rho_active"
"k_on * trace(L) * rho_f - k_off * rho_b"
"r_dot * geom.x / geom.radius"
"k_on * rho_inactive - k_off * rho_active"
```

#### 2.3.2 Parsing

The string is parsed at MathDescription construction time. The parser is hand-written (a small recursive-descent for math-like infix; no third-party dependency); errors point to the offending position in the string.

Parsing happens in two stages with distinct inputs. The **parser** sees only the string and produces a *syntactic* AST: every bare identifier becomes a single `Name` node, every `f(...)` becomes a `FunctionCall`, and no node carries a type. Distinguishing a local variable from a parameter from a reserved name (`t`, `x`) from a measure (`dx`, `ds`, …) requires the surrounding MathDescription — which variables and parameters exist, and on which subdomain the expression lives — that the parser does not have. The **validator's resolution pass** (§1.11, §2.5), which does have that context, walks the syntactic AST, resolves each `Name` into one of the typed reference kinds below, and assigns every node a type. The fully-resolved, typed AST is the canonical internal form the backend consumes.

#### 2.3.3 AST node kinds

After the validator's resolution pass, the AST has the following node kinds. The parser itself produces only the syntactic subset noted below; `Name` is the unresolved form the resolution pass rewrites into `VariableRef` / `ParameterRef` / `ReservedRef` / `MeasureRef`.

| Node | Produced by | Carries | Notes |
|---|---|---|---|
| `Number` | parser | numeric value | Scalar numeric literal. |
| `Name` | parser | name | Unresolved bare identifier. The resolution pass rewrites it into one of the four `*Ref` kinds; it does not survive into the canonical AST. |
| `VariableRef` | resolution | name | A `Name` that resolved to a local variable's value field. |
| `ParameterRef` | resolution | name | A `Name` that resolved to a parameter: constant value for `kind: scalar`; the parameter's body expression (already a parsed AST) evaluated against the current context for expression-valued parameters; per-region value for `kind: region_map` (§2.2.3). |
| `BuiltinRef` | resolution | which (`geom.x`, `sim.t`, …) | A qualified `Name` that resolved to a built-in quantity — time, the spatial coordinate, or a `geom.*` geometry quantity (ADR 006). |
| `MeasureRef` | resolution | which (`dx`, `dx_Gamma`, `ds`, …) | A `Name` (or `FunctionCall` for the parametrised `ds(<boundary>)` forms) that resolved to a measure. Only valid in weak-form `form:` expressions. |
| `IndexAccess` | parser | object, index | `geom.x[0]` parses to `IndexAccess(Name("geom.x"), Number(0))`; resolution rewrites the base to the built-in `geom.x`. |
| `FunctionCall` | parser | callee, args | Both built-in functions (sin, cos, …) and special operators (`trace`, `grad`, geometric helpers, `partial_t`, `inner`). The callee is a raw name; resolution dispatches on it (and rewrites measure callees to `MeasureRef`). |
| `BinaryOp` | parser | op (`+`, `-`, `*`, `/`, `**`), left, right | Standard arithmetic. |
| `UnaryOp` | parser | op (`+`, `-`), operand | |
| `VectorLiteral`, `TensorLiteral` | parser | components / rows | A bracket whose every element is itself a bracket is a `TensorLiteral`; otherwise a `VectorLiteral`. |

After resolution, each node carries a type (`scalar`, `vector`, `symmetric_tensor`, or an error type for un-resolved cases). The validator (§1.11, §2.5) walks the AST type-checking each node.

#### 2.3.4 Operator and function vocabulary

The vocabulary is the union of:

- **Standard mathematical functions** from §1.8.5 (`sin`, `cos`, `tan`, inverse trig, `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `pow`, `if`, `step`, `sign`).
- **Geometry quantities** from §1.8.4, namespaced under `geom.*` and used as *values*, not calls (`geom.normal`, `geom.mean_curvature`, `geom.curvature1`, `geom.curvature2`, `geom.tangent`, `geom.radius`, `geom.azimuth`).
- **Calculus operators** from §1.8.5, subject to the narrow rule of §1.8.5 in template slots and unrestricted in weak-form (`grad`, `div`, `lapl`, `grad_surf`, `div_surf`, `lapl_beltrami`).
- **Cross-dimensional reference** from §1.8.2 (`trace`).
- **Tensor algebra** for weak-form expressions: `inner(a, b)` for the inner product (works on any matching-rank pair — scalar*scalar, vector·vector, tensor:tensor — returns a scalar), `outer(a, b)` for outer product, `cross(a, b)` in 3D. There is intentionally no separate `dot(·, ·)` — on vectors it would coincide with `inner` and the duplication invites confusion; tensor-contraction beyond the matching-rank inner case is deferred to the v2 vocabulary if a use case warrants it.
- **Time derivative**: `partial_t(u)`, valid only inside weak-form `form:` expressions and only for time-dependent equations.
- **Measures**: `dx`, `dx_Gamma`, `dl`, `dp`, `ds(<boundary>)`, `dS(<boundary>)`, `dl_Gamma(<boundary>)`. Valid only inside weak-form `form:` expressions; multiply expressions by a measure to integrate them.

#### 2.3.5 Weak-form expression specifics

A weak-form `form:` expression has additional rules:

- Must reference `<variable>_test` for the governed variable's test function at least once.
- Must end up with measure multiplications producing an integrated scalar — the residual.
- For `temporality: time_dependent`, must contain `partial_t(<variable>)` at least once.
- For `temporality: steady_state`, must not contain `partial_t(<variable>)`.

Weak-form expressions may span multiple lines via YAML's `|` (literal block) syntax. Whitespace and line breaks are not semantic except as token separators.

### 2.4 Naming and reference resolution

#### 2.4.1 Naming conventions

Names within a MathDescription:

- **Identifiers** match `[A-Za-z_][A-Za-z0-9_]*` — letters, digits, underscores, leading non-digit. Case-sensitive. A *qualified* built-in name adds dotted members (`geom.x`, `sim.t`); the dot is not legal in a user identifier.
- **Snake_case** is conventional but not enforced; `rho_active`, `k_on`, `cytoplasm_left_cell` are typical.
- **Reserved names** (cannot be used as variable, parameter, or subdomain names): only the two **namespace roots** `geom` / `sim`, and the integration measures (`dx`, `ds`, …). Per ADR 006 the bare value namespace belongs to the user, so the short physical names that used to collide — `t`, `x`, `r`, `theta`, `phi`, `n`, `H` — are now **free for the modeller**: `parameter: theta` for an angle offset, `variable: phi` for a phase field, and so on are all legal. Operator and function names (`sin`, `trace`, `grad`, …) are reserved only in *call* position; they never occupy a value name, so a variable may even be named `grad` (and `grad(...)` still calls the operator). Built-in quantities are reached through their namespace — `geom.azimuth`, `geom.radius`, `geom.normal`, `sim.t` — never as bare names.
- **Boundary and region names** are also identifiers; their assignment is the Geometry's responsibility.

#### 2.4.2 Scope rules

Names resolve in the order specified in §1.8.6:

1. Local variable (on the expression's subdomain).
2. Named parameter.
3. Reserved name.

A name that matches more than one of these at construction time is an error (shadowing — §1.11.3). A name that matches none is an error.

Variable names on *different* subdomains are not conflicts — `(c, cytoplasm)` and `(c, extracellular)` are two distinct variables with the same simple name. References must use `trace(·)` to cross subdomain boundaries (§1.8.2); within an expression, the bare name `c` resolves to whichever variable lives on the expression's evaluation subdomain.

### 2.5 Validation

The validation rules of §1.11 are implemented by a single validation pass over the MathDescription dataclass tree, run during construction (after YAML/JSON parsing, before the MathDescription is exposed to downstream code). The pass:

1. Resolves every name (§1.11.3) — every reference must point to a declared entity.
2. Topologically sorts parameter expressions (§1.11.3) — parameter-to-parameter references must be acyclic.
3. Type-checks every expression AST (§1.11.5) — node types must match slot expectations.
4. Checks coverage (§1.11.4) — every variable governed, every internal boundary BC'd, etc.
5. Checks temporality consistency (§1.11.6) — strict matching of `temporality` and `∂_t` presence.
6. Checks BC consistency (§1.11.7) — no conflicts, interface partner-subdomain validity, weak-form Dirichlet-only restriction.
7. Checks IC consistency (§1.11.8) — type match, no inter-variable references, Dirichlet-compatibility warnings.
8. Applies the operator usage rules (§1.11.9) — template slot expressions must not contain calculus on the governed variable (narrow rule), and second-order operators (`lapl`, `lapl_beltrami`) require argument variables with `space: lagrange_p2` or higher (smoothness rule).
9. Checks parameter scoping (§1.11.10) — expression parameters with geometric helpers carry a `subdomain:` scope; their uses must be from compatible contexts.

Errors prevent construction and are reported with the offending field's location (line / column for YAML, JSON pointer for JSON, attribute path for dataclasses). Warnings are emitted but allow construction.

Geometry-compatibility checks (§1.11.10) run when the Geometry object is loaded — typically at solve time. The MathDescription itself can be validated without a Geometry, catching intra-model errors early.

### 2.6 Mapping to / from VCell `MathDescription`

This section specifies which VCell `MathDescription` constructs map to which formalism constructs, which don't map and why. **No converter ships in v1**; this is documentation of the translation rules to inform the eventual `pyvcell` integration when concrete VCell models need to be imported.

#### 2.6.1 Constructs that map directly

| VCell construct (cbit.vcell.math) | This formalism |
|---|---|
| `CompartmentSubDomain` | `Subdomain` with `kind: volume` |
| `MembraneSubDomain` | `Subdomain` with `kind: surface` |
| `FilamentSubDomain` (planned) | `Subdomain` with `kind: curve` |
| `PointSubDomain` | `Subdomain` with `kind: point` |
| `VolVariable` | `Variable` with appropriate `subdomain` and `type: scalar` |
| `MemVariable` | `Variable` with surface `subdomain` and `type: scalar` |
| `PdeEquation` with `bSteady = false` | `Equation` template `bulk_radv_diff` or `surface_pde_with_dilution`, `temporality: time_dependent` |
| `PdeEquation` with `bSteady = true` | Same template, `temporality: steady_state` |
| `OdeEquation` | `Equation` template `lumped_ode`, `temporality: time_dependent` |
| Constant parameters | `Parameter` with plain `value` |
| Initial expression on a PDE/ODE | `initial_condition` field on the equation |
| `BoundaryConditionType` (DIRICHLET/NEUMANN/PERIODIC/ROBIN) on a compartment face | `BoundaryCondition` with corresponding `kind` |

#### 2.6.2 Constructs that need transformation

| VCell construct | Transformation needed |
|---|---|
| `JumpCondition` | Translates to one or two `boundary_condition` entries with `kind: interface_flux_balance`. The Neumann-only restriction in VCell is relaxed here; flux-balance interface BCs may have any constitutive expression. |
| Per-face Cartesian BCs (Xp/Xm/Yp/Ym/Zp/Zm) on a CompartmentSubDomain | Each face becomes a separate labelled-boundary BC. The Geometry must expose those faces as named boundaries (`x_minus`, `x_plus`, etc.). |
| `MembraneSubDomain.velocityX`, `velocityY` | A `subdomain.motion` with `kind: prescribed` and `velocity: "[velocityX, velocityY]"`. Note the formalism uses a single vector expression, not per-component scalars. |
| VCell `Expression` strings (infix math) | Parse using §2.3.2; the syntax is largely compatible. VCell's parser supports a few constructs ours does not (e.g. integer division semantics, certain function names) — translation may need minor rewrites. |
| `ReservedSymbols` (Faraday, gas constant, …) | Become reserved parameter names in the MathDescription, with values from a shared constants module. |

#### 2.6.3 Constructs that don't map in v1

| VCell construct | Reason it does not map |
|---|---|
| `FastSystem`, `FastInvariant`, `FastRate` | Solver-side QSSA reduction; not part of this formalism (memory decision: "out of scope"). |
| `Event` | Discrete state transitions are deferred to v2 (no template). |
| `VolumeRegionVariable`, `MembraneRegionVariable` | Piecewise-constant region variables are deferred to v2. |
| `ParticleMolecularType`, `StochVolVariable` | Stochastic dynamics are out of scope for this formalism entirely. |
| `PostProcessingBlock` | Observables / derived outputs are out of scope for the formalism — they belong with the solver-configuration / output-spec object. |

Models using any of these constructs cannot be imported losslessly in v1.

#### 2.6.4 Round-trip considerations

For models that fall entirely within §2.6.1 (direct mappings), bidirectional round-trip is feasible. For models requiring §2.6.2 transformations, round-trip is *lossy in one direction*: the formalism's richer constructs (labelled non-Cartesian boundaries, flux-balance interface BCs with arbitrary expressions, unknown motion) cannot in general be expressed back into VCell `MathDescription` without losing information. The natural direction is **VCell → formalism** (importing legacy models); **formalism → VCell** is offered only for the subset that fits in VCell's quirks (recorded as v2 if a use case earns it).

SBML Spatial compatibility (the standard interchange format for spatial cell biology, which the user authored) is **not** part of v1. The schemas diverge non-trivially — SBML Spatial does not have a weak-form escape hatch, lacks first-class unknown motion, and has its own per-face BC machinery. A v2 SBML Spatial converter would be valuable but is a substantial project on its own.

### 2.7 End-to-end worked example

The mechanics-driven membrane motion model from §1.10.8 (also discussed in §1.5.8 as a weak-form example), now shown end-to-end in all three carriers — YAML, JSON, and Python dataclass — to demonstrate that they describe the same data and round-trip without loss. The focus here is the **data model and its three syntactic forms**; the model itself is exactly the same.

```yaml
math_description:
  geometry: cell_2d                       # external Geometry name

  subdomains:
    - name: membrane
      kind: surface
      motion:
        kind: unknown
        variable: v_membrane

  variables:
    - { name: v_membrane, subdomain: membrane, type: vector }
    - { name: rho,        subdomain: membrane, type: scalar }

  parameters:
    - { name: eta,     value: 1.0   }
    - { name: sigma_T, value: 0.10  }
    - { name: k_off,   value: 0.02  }
    - { name: f0,      value: 0.3   }              # active-traction amplitude
    - name: f_active                                # polarised active traction (vector)
      type: vector
      subdomain: membrane                           # uses geom.azimuth — scope required (§2.2.3)
      expression: "[f0 * cos(geom.azimuth), 0]"

  equations:
    # Motion equation — weak-form viscous force balance, quasi-static.
    - template: weak_form
      variable: v_membrane
      subdomain: membrane
      temporality: steady_state
      form: |
        ( eta * inner(v_membrane, v_membrane_test)
          + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)
          - inner(f_active, v_membrane_test)
        ) * dx_Gamma
      initial_condition: "0"

    # Receptor density — standard T2 surface PDE with auto-dilution.
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: "0.05"
        source: "-k_off * rho"
      initial_condition: "1.0 + 0.3 * cos(2 * geom.azimuth)"

  boundary_conditions: []
```

The Geometry object `cell_2d` is expected to provide:

- A region or regions tagged `membrane` (surface kind).
- No labelled boundaries are referenced in this model, so `boundary_conditions` is empty — but the geometry may still have labelled boundaries that the model could use later.

What the equivalent JSON looks like (same data, different syntax):

```json
{
  "math_description": {
    "geometry": "cell_2d",
    "subdomains": [
      {
        "name": "membrane", "kind": "surface",
        "motion": { "kind": "unknown", "variable": "v_membrane" }
      }
    ],
    "variables": [
      { "name": "v_membrane", "subdomain": "membrane", "type": "vector" },
      { "name": "rho",        "subdomain": "membrane", "type": "scalar" }
    ],
    "parameters": [
      { "name": "eta",     "value": 1.0  },
      { "name": "sigma_T", "value": 0.10 },
      { "name": "k_off",   "value": 0.02 },
      { "name": "f0",      "value": 0.3  },
      {
        "name": "f_active",
        "type": "vector",
        "subdomain": "membrane",
        "expression": "[f0 * cos(geom.azimuth), 0]"
      }
    ],
    "equations": [
      {
        "template": "weak_form",
        "variable": "v_membrane",
        "subdomain": "membrane",
        "temporality": "steady_state",
        "form": "( eta * inner(v_membrane, v_membrane_test) + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test) - inner(f_active, v_membrane_test) ) * dx_Gamma",
        "initial_condition": "0"
      },
      {
        "template": "surface_pde_with_dilution",
        "variable": "rho",
        "subdomain": "membrane",
        "temporality": "time_dependent",
        "terms": { "diffusion": "0.05", "source": "-k_off * rho" },
        "initial_condition": "1.0 + 0.3 * cos(2 * geom.azimuth)"
      }
    ],
    "boundary_conditions": []
  }
}
```

And what programmatic dataclass construction looks like (sketched, exact API in `vcell-fenics`):

```python
from vcell_fenics.formalism import (
    MathDescription, Subdomain, Variable, Parameter,
    TemplateEquation, WeakFormEquation, Motion,
)

md = MathDescription(
    geometry="cell_2d",
    subdomains=[
        Subdomain(name="membrane", kind="surface",
                  motion=Motion(kind="unknown", variable="v_membrane")),
    ],
    variables=[
        Variable(name="v_membrane", subdomain="membrane", type="vector"),
        Variable(name="rho",        subdomain="membrane", type="scalar"),
    ],
    parameters=[
        Parameter(name="eta",     value=1.0),
        Parameter(name="sigma_T", value=0.10),
        Parameter(name="k_off",   value=0.02),
        Parameter(name="f0",      value=0.3),
        Parameter(
            name="f_active",
            type="vector",
            subdomain="membrane",
            expression="[f0 * cos(geom.azimuth), 0]",
        ),
    ],
    equations=[
        WeakFormEquation(
            variable="v_membrane",
            subdomain="membrane",
            temporality="steady_state",
            form=(
                "( eta * inner(v_membrane, v_membrane_test)"
                " + sigma_T * geom.mean_curvature * inner(geom.normal, v_membrane_test)"
                " - inner(f_active, v_membrane_test)"
                ") * dx_Gamma"
            ),
            initial_condition="0",
        ),
        TemplateEquation(
            template="surface_pde_with_dilution",
            variable="rho",
            subdomain="membrane",
            temporality="time_dependent",
            terms={"diffusion": "0.05", "source": "-k_off * rho"},
            initial_condition="1.0 + 0.3 * cos(2 * geom.azimuth)",
        ),
    ],
)
```

All three forms describe the same MathDescription and round-trip among each other without loss. The validator runs once after construction, producing the same errors / warnings regardless of which carrier was used.

---

## Part 3 — Solver contract

Part 1 specified what is in a MathDescription. Part 2 specified how it is represented. Part 3 specifies what a backend must do with it — the contract that lets the formalism claim "the same MathDescription + Geometry + SolverConfiguration produces equivalent results regardless of which compliant backend runs it."

### 3.1 Purpose

A **backend** is software that consumes a MathDescription, a Geometry, and a SolverConfiguration, and produces approximate numerical solutions of the well-posed math problem the MathDescription describes. `vcell-fenics`'s DOLFINx implementation is one backend; a finite-volume implementation would be another; VCell's existing moving-boundary solver could be wrapped as a third.

The **contract** in Part 3 is what makes the formalism's "one math description, many solvers" promise (§1.1.1) operationally meaningful. It specifies:

- **Required operations** — what every compliant backend must implement (§3.2).
- **Discretion zone** — what backends are free to choose, with the understanding that those choices affect numerical accuracy and performance but not the well-posed math problem being solved (§3.3).
- **The SolverConfiguration object** — how a user expresses preferences within the discretion zone, in a way that is portable across backends (§3.4).
- **Conformance** — what makes a backend compliant; a small reference suite of canonical models that any backend must reproduce within tolerance (§3.5).
- **The vcell-fenics DOLFINx backend's v1 status** — which formalism subset it implements, what it punts (§3.6).

### 3.2 Backend interface — required operations

Every compliant backend implements the following operations. The names below describe semantics; the concrete API in `vcell-fenics` may differ in spelling but not in capability.

#### 3.2.1 Loading

- **Accept a `MathDescription`** in any of the three carriers from §2.1 (YAML, JSON, or in-memory dataclass). Run the validation pass (§2.5) and reject any MathDescription that fails.
- **Accept a `Geometry`** referenced by name from the MathDescription. The geometry provides at minimum: the mesh, the region-to-subdomain-class assignment, the labelled-boundary identifiers and their incidence to subdomain classes. The precise Geometry API is not specified here; it is a separate design (and a separate document) at the Geometry layer.
- **Accept a `SolverConfiguration`** specifying discretisation choices (§3.4). The SolverConfiguration may declare backend-specific options that compliant backends are free to ignore if they do not understand them — but only after the backend has confirmed it has its own sensible default for the relevant choice.
- **Resolve names through the loader.** All three artifacts reference each other by name (§3.4 "Name resolution"); the backend uses whichever loader the runtime provides — filesystem search, in-memory registry, package-bundled artifacts, or a combination — to turn the names into concrete objects. The loader API is not specified by the formalism.
- **Cross-validate** the three artifacts: the SolverConfiguration's `geometry` name matches the MathDescription's `geometry` name (otherwise the binding is incoherent); subdomain class names in the MathDescription resolve to regions in the Geometry; labelled-boundary names referenced in BCs resolve to entities in the Geometry; region-keyed parameters cover every region (§1.11.10). Any mismatch is a hard error before any solving begins.

#### 3.2.2 Assembly

- **Operator templates (T1–T4)** — assemble the equation each template specifies, in Eulerian conservation form (§1.10.5), filling slot expressions provided by the user. Compression / dilution terms from `subdomain.motion` are computed automatically and added to the assembled form. The narrow rule on calculus operators (§1.8.5) is enforced at validation time; the backend trusts the validator's result.
- **Weak-form equations** — assemble the user-supplied form directly, treating it as a UFL-equivalent residual. The backend chooses any FE-method primitives needed to support the form (mixed function spaces, internal facet integrals, etc.) according to the SolverConfiguration's space hints.
- **Boundary conditions** — assemble Dirichlet BCs as strong constraints (DOF elimination or equivalent); Neumann / Robin / interface BCs as boundary integrals added to the relevant equations' assembled forms; for weak-form-governed variables, only Dirichlet BCs from §1.6 are honoured (the rest must live in the form, §1.5.6).
- **Initial conditions** — for `time_dependent` equations, evaluate the IC expression on the reference (t=0) configuration and use it as the starting state for time stepping.

#### 3.2.3 Solving

- **Steady-state models** (all equations declared `steady_state`, §1.9.6) — solve as a nonlinear algebraic system using the backend's chosen nonlinear solver.
- **Time-dependent models** (any equation declared `time_dependent`) — advance over a time interval $[0, T]$ using the backend's chosen time-stepping scheme. Steady-state equations in a mixed model are re-solved at every time step alongside the time-dependent ones (DAE structure, §1.9.4).
- **Moving subdomains** — for prescribed motion, update the subdomain's configuration each time step using the user-supplied velocity or displacement expression. For unknown motion, the motion variable's equation participates in the per-step solve; the backend chooses an ALE algorithm to propagate the resulting motion through the bulk mesh (§1.10.7).

#### 3.2.4 Results exposure

- **At each output time** the backend must expose, for every variable in the MathDescription, its current value field on its subdomain. The form of this exposure (in-memory arrays, on-disk file in XDMF / VTKHDF / similar, callbacks) is backend-discretion, but a compliant backend must be able to provide values at user-requested output times within the simulation interval.
- **The geometric configuration** at each output time (for subdomains with non-`none` motion) must be available alongside the field values. Without it, the values cannot be interpreted spatially.

### 3.3 What backends choose

The following are explicitly *not* part of the MathDescription. They are the backend's discretion (sometimes guided by the SolverConfiguration). Choices in this zone affect numerical accuracy, stability, and performance — but not the well-posed math problem being solved.

| Choice | Examples |
|---|---|
| **Finite-element order** | $P_1$ Lagrange, $P_2$ Lagrange, $P_k$ for higher $k$, Taylor–Hood pair for Stokes-type problems, DG variants. The MathDescription's `space` hint (§1.3.3) is a suggestion; the backend may honour or override it. |
| **Mesh resolution and refinement** | Element size; adaptive refinement triggers; remeshing for large deformations. |
| **Time-stepping scheme** | Backward Euler, Crank–Nicolson, BDF$k$, implicit Runge-Kutta, IMEX splits. |
| **Linear solver** | Direct (LU, Cholesky) or iterative (GMRES, CG, MINRES, BiCGSTAB). |
| **Preconditioner** | Multigrid (algebraic, geometric), block-diagonal, AMG, ILU. |
| **Nonlinear solver** | Newton, Picard, damped Newton, line search, trust region. |
| **Stabilisation** | SUPG, GLS, entropy viscosity for advection-dominated regimes; pressure stabilisation for Stokes; ghost penalty for cut-FEM. |
| **Parallelisation** | MPI domain decomposition, threading, GPU offload. |
| **ALE algorithm** | Whether the bulk mesh follows the surface motion exactly, follows a harmonic-extension velocity field, follows an elastic-extension velocity field, or uses a different recipe. |
| **Remeshing strategy** | When to remesh during large deformations; what algorithm; how to transfer fields. |
| **Convergence tolerances** | Linear solver relative tolerance, nonlinear solver convergence threshold, time-stepping error control. |

The discretion is asymmetric: the backend may choose anything in this zone, but the choice must not change *what* equation is being solved — only *how* approximately. A backend that, say, drops a slot expression to "simplify" the equation is not compliant.

### 3.4 The SolverConfiguration object

Discretion-zone choices that a user wants to pin (for reproducibility, for parameter sweeps, for cross-backend comparisons) go in a separate **SolverConfiguration** artifact, with its own YAML/JSON schema.

```yaml
solver_configuration:
  math_description: my_model              # name; resolved by the loader (see below)
  geometry: cell_2d                       # name; must match the MathDescription's `geometry:` field

  fe_order:
    default: 1                            # P1 Lagrange unless overridden per-variable
    overrides:
      v_membrane: 2                       # P2 for the velocity field

  time_stepping:
    scheme: backward_euler                # or crank_nicolson, bdf2, ...
    dt: 0.01
    t_final: 1.0
    output_times: [0.0, 0.5, 1.0]             # or `every: 0.1`

  linear_solver:
    type: gmres
    preconditioner: amg
    rel_tol: 1.0e-8
    abs_tol: 1.0e-12

  nonlinear_solver:
    type: newton
    max_iterations: 25
    convergence_tol: 1.0e-9

  ale:
    bulk_velocity_extension: harmonic     # or material, elastic, none
    remesh_when_quality_below: 0.3        # backend may ignore if it does not support remeshing

  stabilisation:
    advection: none                       # or supg, gls, ...

  parallelisation:
    mpi_processes: 4                      # backend may honour or override
```

Most fields have backend-side defaults. A SolverConfiguration may be entirely empty (`solver_configuration: { math_description: m, geometry: g }`) and the backend will use sensible defaults for everything else.

Backend-specific options are permitted under a `backend_specific:` key per-backend; compliant backends ignore options they do not understand, after confirming they have their own default for the relevant choice. This lets a user pin DOLFINx-specific behaviour (e.g. PETSc options) without breaking other backends that do not interpret PETSc strings.

A run is `(MathDescription, Geometry, SolverConfiguration)` — three files / objects. The MathDescription stays the same across, e.g., a time-step convergence study; only the SolverConfiguration's `time_stepping.dt` varies.

**Name resolution.** All three artifacts reference each other by **name**, not by file path. The MathDescription declares the geometric vocabulary it requires (`math_description.geometry: <name>`); the SolverConfiguration names the MathDescription and Geometry it wants to run (`solver_configuration.math_description: <name>`, `solver_configuration.geometry: <name>`). The SolverConfiguration's `geometry` name must match the MathDescription's declared geometry name; the validator checks this at load time.

How names resolve to concrete objects is a **loader concern**, not a formalism concern. A loader is provided by the runtime (e.g. the `vcell-fenics` backend supplies its own) and is responsible for taking a name and returning the concrete artifact. Common loader strategies — none mandated:

- **Filesystem search path.** The loader walks a configured set of directories looking for `<name>.yaml`, `<name>.json`, or similar.
- **Programmatic registration.** Code constructs a dataclass instance and registers it under a name with the loader; subsequent lookups by that name resolve to the registered instance. The natural pattern for `pyvcell` integration and for unit tests.
- **Package-bundled artifacts.** The runtime ships reference Geometries / MathDescriptions under canonical names (e.g. `cell_2d`, `disk_radius_1`); user code can reference them without supplying files.
- **Hybrid.** A loader may chain strategies, e.g. check the in-memory registry first, then a search path.

This separation keeps the math model independent of filesystem layout (the same `math_description: my_model` works whether the file lives in `./models/`, in a package, or was built programmatically), and lets `pyvcell` consume MathDescriptions as library objects without needing a file system at all.

### 3.5 Conformance

#### 3.5.1 What conformance means

A backend is **compliant** if, for every (MathDescription, Geometry, SolverConfiguration) triple it claims to support, it produces a numerical approximation of the well-posed math problem the MathDescription describes — within the tolerance specified in the SolverConfiguration. Two compliant backends, given the same input triple, should produce results that agree up to discretisation error and solver tolerance.

Compliance is *scoped*. A backend may claim conformance for a subset of the formalism — say, "this backend implements T1 and T2 with prescribed motion, no weak-form, no Robin BCs." Models that fall outside the claimed subset are not the backend's responsibility; the backend should reject them with a clear error rather than silently produce a wrong result.

#### 3.5.2 The conformance reference suite

The formalism ships a small reference suite of canonical models. Any backend that claims conformance for the relevant subset of the formalism must reproduce these results within stated tolerance. The suite formalises what "implements the formalism correctly" means.

**v1 reference models** (corresponding to existing `vcell-fenics` tests — `tests/test_*`):

| Reference model | Formalism subset exercised | Expected behaviour |
|---|---|---|
| **Bulk diffusion eigenmode decay** | T1 (bulk RAD), `temporality: time_dependent`, no motion, zero-Neumann external BC | A Bessel eigenmode $J_0(\lambda r)$ initial condition decays as $\exp(-D \lambda^2 t)$ at the analytical rate, within $\le 2\%$ relative error at $t = 0.5$ on a moderately refined mesh. |
| **Closed-surface mass conservation** | T2 (surface PDE), `temporality: time_dependent`, no motion | Total mass $\int_\Gamma \rho \, \mathrm{d}\Gamma$ is conserved to $\le 10^{-10}$ relative error over $\ge 50$ time steps on a closed manifold (no boundary). |
| **Closed-surface eigenmode decay** | T2 (surface PDE), `temporality: time_dependent`, no motion | A $\cos(k\theta)$ initial condition on a circle of radius $r$ decays as $\exp(-D k^2 / r^2 \cdot t)$ within $\le 2\%$ relative error. |
| **Moving-membrane dilution — positive control** | T2 (surface PDE), `temporality: time_dependent`, prescribed radial motion | A uniform $\rho_0$ initial condition under prescribed radial expansion conserves mass to $\le 2\%$ relative error over the full motion. |

The reference models are deliberately small (single-variable, simple geometries, analytical reference solutions). The conformance suite is the *floor*; backends may layer arbitrarily rich additional tests above it.

The vcell-fenics test suite additionally carries a **negative-control discriminator** for the moving-membrane case: the same model with the dilution term explicitly suppressed in the backend code produces $M(T) = 2 M(0)$ when the membrane doubles in length, confirming that the positive-control test would fail if the auto-dilution machinery were ever removed or bypassed. This is a backend-implementation check, not a formalism conformance test — T2 has no schema-level switch to disable dilution, by design (§1.4.2, §1.10.5), so the suppressed-dilution variant is not expressible in the formalism and not required of any compliant backend.

When templates T5–T7 ship (v2+), the reference suite extends to cover them: a simple Stokes flow (analytical Poiseuille), a linear-elastic deformation under known load, and a mechanics-driven membrane-motion case with an analytical-or-converged-solution reference.

#### 3.5.3 Determinism and reproducibility

Backends are *encouraged* to produce deterministic results across runs with identical inputs (same MathDescription, Geometry, SolverConfiguration, hardware, software stack). Determinism is not required — some legitimate backend strategies (randomised algebraic preconditioners, asynchronous parallelism) sacrifice strict reproducibility — but a backend that is non-deterministic must say so in its documentation.

Across compliant backends, results agree *up to discretisation error*. The math problem is the same; the numerical approximation is not. Cross-backend agreement is convergence under refinement, not bit-identical output.

### 3.6 The vcell-fenics DOLFINx backend

This backend is the canonical implementation of the formalism and the reference backend for v1 conformance.

#### 3.6.1 v1 status

**Implemented** (formalism-driven, end-to-end):

- The full Part 2 layer: schema dataclasses, YAML/JSON loader + dumper, the expression parser → typed AST (`src/vcell_fenics/formalism/`), and the validation pass (`formalism/validator.py`) — every check of §1.11 except the geometry cross-check and BC-expression contents, which run at the backend boundary.
- The formalism → DOLFINx backend (`src/vcell_fenics/backend/`, ADR 004): an expression→UFL compiler, a `DiscreteProblem` IR with backward-Euler lowering via a residual `ufl.lhs`/`ufl.rhs` split, a geometry adapter + name loader + §1.11.10 cross-check, and the `assemble` / `run` driver with a `SolverConfiguration`.
- T1 (bulk RAD) and T2 (surface PDE with dilution), `temporality: time_dependent`, with: scalar variables and **coupled multi-species systems** (one solve over a vector space); the `diffusion`, `source` (a source linear in the unknowns, including cross-variable coupling), and **`relative_advection`** slots; prescribed-**velocity** motion with **automatic dilution** `ρ ∇_Γ·v_Γ` and a per-step mesh advance guarded by a mesh-quality check; external **Dirichlet / Neumann / Robin** BCs on a labelled boundary (with the zero-Neumann no-flux default where none is declared); constant and expression parameters; spatially-varying ICs. Backward Euler, $P_1$ Lagrange (configurable), direct LU.
- **`relative_advection` — the Eulerian volume term** (`backend/assemble.py`, `TermKind.ADVECTION`): the species' drift `w_rel·∇c` *relative to the substrate/mesh* (`grad` on a submesh is the surface gradient, so the same term serves bulk and surface). This is what makes a volume species **Eulerian** rather than Lagrangian/co-moving: the documented Eulerian-fluid setup (`motion: none` + the fluid velocity in `relative_advection`) advects a profile at exactly the prescribed velocity; and on a *moving* mesh it supplies the `(u − w)·∇c` correction the Eulerian (no-material-points) volume needs — verified by the discriminator that a lab-frame field stays put while a rotating mesh sweeps through it (with `relative_advection = −v_mesh`), where the co-moving treatment instead carries the field around. The dilution term `c ∇·v_mesh` remains the co-moving (`u = w`) special case; the two terms together express the general moving-domain volume transport.
- **ALE remeshing** (`src/vcell_fenics/core/`, `backend/ale.py`): conservative surface and bulk field remaps, a gmsh region remesher, a `rebuild_on_mesh` teardown+reassemble of the build-once IR, harmonic-extension bulk mesh-motion, and a `step_with_remeshing` / `run_with_remeshing` driver that turns the mesh-quality guard into remesh-and-continue for a moving membrane or bulk region (`docs/modeling/ale-remesh-driver.md`).
- **Multi-compartment geometry** (`approaches/multicompartment/`, `make_cell_extracellular_geometry`): a concentric two-compartment cell (cytosol + extracellular, meeting at the membrane) with an *internal* interface boundary incident to both compartments, the substrate for interface BCs and bulk↔surface coupling. Plus `create_extracellular_annulus` — the §1.6.6 substrate (extracellular bulk + membrane inner boundary + outer reservoir).
- **Mixed-dimensional bulk↔surface coupling through `assemble()`** (`backend/coupled.py`, dispatched by a `CoupledGeometry`): a multi-subdomain MathDescription — one `volume` bulk + one `surface` on its boundary, coupled by `trace(·)` in the surface sources and a Neumann BC on the bulk variable referencing the surface variables (the §1.6.6 composable pattern) — is read *structurally* and assembled as a two-mesh block system, returning a `CoupledProblem`. Any diffusion/rate/source expressions and any number of surface species work, driven by the model. Mechanics: the residual splits into a local part (per-mesh mass+diffusion) and a coupling part (the cross-subdomain terms on the bulk's interface facets via DOLFINx 0.10 `entity_maps`), assembled as two block matrices via `ufl.extract_blocks` and summed; bilinear coupling is linearised semi-implicitly by lagging the bulk variable. **Moving membrane (conservative ALE):** when the surface has a prescribed velocity, both fields co-move with the deforming domain, so each gains a dilution term — the surface `ρ ∇_Γ·v_Γ` and the bulk `L ∇·v_mesh` — and the consumption flux at the co-moving membrane is the ordinary diffusive Neumann (no relative-flux correction). Verified: §1.6.6 through `assemble()` reproduces receptor conservation, binding equilibrium, the reservoir Dirichlet, ρ_f-non-negativity, and the local-vs-coupling source split; under membrane expansion the receptor total *and* (in a closed cell) the total ligand are conserved — the mass-balance gate on the moving-boundary flux. Scope: one bulk + one surface-on-its-boundary; >2 subdomains and interface (bulk-bulk) BCs reuse the same machinery but are not yet wired.
- Visualization helpers — PyVista in-process, XDMF for ParaView (`src/vcell_fenics/viz.py`).

The three v1 conformance models run **through the formalism** (`tests/test_backend_*.py`): bulk diffusion, the surface cos(kθ) eigenmode decay, and the dilution mass-balance with its negative control — plus the §1.4.5 two-species receptor model end-to-end. The bespoke single-physics prototypes that preceded the backend have been removed.

- Convergence-rate verification (`tests/test_backend_convergence.py`): h-refinement against analytical diffusion eigenmodes confirms the P1 operator is **second-order in L2** on both the bulk path (cos(πx)cos(πy) on an exactly-meshed unit square) and the surface path (cos(kθ) on the circle membrane, where the O(h²) polygonal-geometry error matches the FE rate); dt-refinement confirms backward Euler is **first-order in time** by self-convergence on a fixed mesh, cross-checked against the closed-form decay. This is the order-of-accuracy axis the single-resolution reference tests cannot cover.

**Weak-form escape hatch** (`backend/weakform.py`, `assemble_weak_form`): one `weak_form` equation governing a scalar or vector variable on a subdomain, with the residual `form` compiled to UFL — the variable, its implicit `<variable>_test`, the tensor-algebra (`inner`/`dot`/`outer`/`cross`) and first-order calculus (`grad`/`div`/`lapl` + `_surf`/`_beltrami`) operators, vector literals `[·,·]`, parameters, `geom.x`, and the subdomain measure (`dx`/`dx_Gamma`). `partial_t(u)` lowers to the backward-Euler difference for a time-dependent form; steady-state forms solve directly; the residual is split with `ufl.lhs`/`rhs`. This is the **membrane-mechanics** route in v1 (T5–T7 are v2): verified by reproducing the T2 surface diffusion as a weak form, a scalar membrane force balance `α u − σ Δ_Γ u = f` (analytic), and a vector viscous balance `η v = f_active` (analytic). The `geom.normal`/`geom.mean_curvature` geometric helpers compile, but resolve only where the curvature projection is bound — the unknown-motion mechanics solve (below); a bare `weak_form` that references them raises `CompileError`. *Deferred:* labelled-boundary measures `ds(·)`/`dS(·)`, BCs on a weak-form variable, and coupled multi-equation weak forms.

**Unknown (mechanics-driven) motion** (`backend/unknown_motion.py`, `assemble_unknown_motion`): the §1.10.8 model — a `motion: { kind: unknown, variable: v }` membrane whose substrate velocity is *solved* from a weak-form force balance (not prescribed), with a T2 receptor that dilutes with the solved motion. The discretisation is **staggered**: each step solves the force balance for `v` on the current membrane, moves it by `dt·v`, then advances the receptor (mass + diffusion + dilution `ρ ∇_Γ·v`) on the deformed membrane. It reuses the existing machinery — the velocity is solved into a `Function` handed to a `DiscreteProblem` as its `motion_velocity`, so the dilution term and the per-step `_MeshMotion` read the freshly-solved field. This is the path to genuine cell migration (the membrane moves under force, not by fiat). Verified against a known-answer force `η v = f₀ x/r`: the solved velocity is `|v| = f₀/η` (radial), the membrane expands at that speed, and the receptor's ∫_Γ ρ ds is conserved across the solved expansion.

**Curvature forces** (`geom.normal`/`geom.mean_curvature`): a force balance with surface tension, `η v + σ H n = 0`, drives mean-curvature flow. Because a discrete membrane's curvature is vertex-concentrated (not pointwise), `geom.normal` and `geom.mean_curvature` are resolved from a **projected mean-curvature vector** κ = H·n — the weak surface Laplacian of position, `∫κ·φ ds = ∫∇_Γ X : ∇_Γ φ ds` — re-solved on the deformed membrane each step (`_CurvatureProjection`). Then `geom.mean_curvature` = |κ|, `geom.normal` = κ/|κ|, and `σ geom.mean_curvature inner(geom.normal, test)` evaluates to the correct weak force `inner(κ, test)`. The receptor is now optional, so a pure-mechanics (motion-only) membrane is supported. Verified: a unit circle's solved velocity is `σ/(η r)` radially **inward**, and the membrane shrinks as `r² = r₀² − 2σt/η` (mean-curvature flow). A stability note: without tangential redistribution (BGN-style) the mesh degenerates at large `dt`, so the verification uses a small step. *Deferred:* vector *expression* parameters, multiple receptors, tangential mesh redistribution, and a moving membrane coupled to a bulk.

**Punted in v1**:

- Operator templates T3 (algebraic constraint), T4 (lumped ODE), T5–T7 (mechanics).
- The two **interface** BC kinds (value-equality, flux-balance) — bulk-bulk coupling on an *internal* interface — and bulk↔surface coupling for >2 subdomains or a moving surface. The bulk↔surface composable pattern (§1.6.6) now runs through `assemble()` (`backend/coupled.py`); these remaining cases reuse the same cross-mesh block machinery but are not yet wired (the single-mesh `assemble()` branch still rejects interface BCs with `NotImplementedError`). External Dirichlet/Neumann/Robin are implemented. Time-dependent BC expressions are also deferred (the v1 backend has no `t` handle).
- Prescribed-*displacement* motion. (**Unknown / mechanics-driven** motion is now implemented for the §1.10.8 class — see the unknown-motion bullet above; remeshing / conservative field transfer is also implemented — see the ALE bullet.)
- Region-keyed parameter maps; advection (`relative_advection`) slots; non-linear sources.
- The full §3.4 SolverConfiguration (linear/nonlinear solver, ALE, stabilisation knobs) and a YAML carrier for it; intermediate output-time snapshots.
- A formal conformance-subset declaration.

#### 3.6.2 What v1 conformance means for this backend

The vcell-fenics backend claims conformance for a strict subset: T1 and T2 (including coupled multi-species), prescribed-velocity or no motion, the diffusion/source slots, external Dirichlet/Neumann/Robin (and zero-Neumann default) BCs, no weak-form. Models within that subset run correctly and pass the conformance reference suite's relevant entries. Models outside that subset are not supported in v1; the assembler rejects them with a clear `NotImplementedError` at build time rather than silently producing wrong results.

This honest scoping is deliberate. The formalism is broader than what v1 implements; the v1 backend is one slice. The roadmap to broader coverage is the same as the formalism's v2 roadmap (Appendix B) — driven by concrete use cases, not by speculative feature addition.

---

## Appendix A — Glossary

Alphabetical. Each entry links to the section that defines or first uses the term in full.

- **Auto-dilution** — The $\rho \, \nabla_\Gamma \cdot \mathbf{v}_\Gamma$ term that operator templates with a $\partial_t$ slot generate automatically when their subdomain has non-zero motion. The canonical correctness check for moving-membrane surface PDEs; not user-settable (§1.4.2 T2, §1.10.5).
- **Bare name** — A reference in an expression with no `trace(·)`, no calculus operator, and no subscripts. Resolves to a local variable, parameter, or reserved name in that order (§1.8.6).
- **Boundary (labelled)** — A codim-1 entity in the geometry with a name. Carries BCs in the MathDescription; can also be a subdomain in its own right (membrane double-role, §1.2.4).
- **Composable pattern** — The §1.6.5 bulk-surface coupling idiom: a regular Neumann BC on the bulk variable plus a matching `source:` term on a surface variable. Used for accumulation (binding, capture). Contrast with `interface_flux_balance` (conservation, no accumulation).
- **Conformance** — A backend claim that, for some declared subset of the formalism, it reproduces the reference suite's models within tolerance (§3.5).
- **Current configuration** — The deformed mesh position at the current time step for moving subdomains. Geometric helpers (`geom.normal`, `geom.mean_curvature`, …) evaluate against it. Contrast with reference configuration (§1.7.2, §1.10.6).
- **DAE** — Differential-algebraic equation. The combined system when a MathDescription has both `time_dependent` and `steady_state` equations (§1.9.4).
- **Equation envelope** — The fixed five-field envelope every equation has: `template`, `variable`, `subdomain`, `temporality`, `terms` (or `form` for weak-form), plus `initial_condition` when time-dependent (§1.4.1, §1.5.2).
- **Expression-valued parameter** — A parameter whose value is an expression in `sim.t`, `geom.*`, and other parameters rather than a constant. Scalar, vector, or symmetric-tensor (§2.2.3 (b)).
- **Form (weak)** — The UFL-style residual expression in a weak-form equation. Equation is interpreted as `form = 0` for all admissible test functions (§1.5.3).
- **Geometry** — The external object that provides the mesh, region-to-class assignment, and labelled-boundary identifiers. Referenced by name from the MathDescription (§1.2.1).
- **Interface BC** — A boundary condition on an internal boundary (two-sided). Two kinds in v1: value-equality and flux-balance. Both require `partner_variable` (§1.6.2). Flux-balance is bulk-bulk only; bulk-surface accumulation uses the composable pattern.
- **Loader** — Runtime-provided machinery that turns a name into a concrete MathDescription / Geometry / SolverConfiguration object. Not specified by the formalism; common strategies are filesystem search, registry, package-bundled artifacts (§3.4).
- **MathDescription** — The top-level declarative artifact this whole document defines. A self-contained mathematical problem in data form, independent of solver and (largely) of geometry (§2.1.2).
- **Motion** — A subdomain field declaring how the subdomain moves: `none`, `prescribed`, or `unknown` (§1.10). Property of the subdomain, not of any equation.
- **Motion variable** — A vector-typed variable referenced from a subdomain's `motion.variable` slot when motion is `unknown`. Governed by some equation in the same MathDescription (§1.3.4, §1.10.3).
- **Narrow rule** — The §1.8.5 / §1.11.9 restriction that calculus operators cannot be applied to the slot's own governed variable in operator-template equations. Does not apply to weak-form equations.
- **Operator template** — A named equation shape (T1–T7) that fills a fixed differential form with user-supplied slot expressions. Contrast with weak-form escape hatch (§1.4, §1.5).
- **Parameter** — A named, non-state input. Three forms: constant scalar, expression, region-keyed (§2.2.3).
- **Reference configuration** — The geometry's initial mesh, with no motion applied. ICs evaluate against it; geometric helpers in IC expressions evaluate against it (§1.7.2, §1.10.4).
- **Region** — A concrete mesh entity (a connected piece of the mesh tagged with a subdomain class name). One subdomain class may correspond to many regions (§1.2.2).
- **Region map** — A `kind: region_map` parameter providing per-region constant values within a subdomain class (§1.2.5, §2.2.3 (c)). Tier 1 region variation.
- **Slot** — A named field in an operator template (e.g. `diffusion`, `source`, `relative_advection`). Each slot has a declared type; optional slots default to zero (§1.4.4).
- **Smoothness rule** — The §1.8.5 / §1.11.9 requirement that arguments of second-order operators (`lapl`, `lapl_beltrami`) have `space: lagrange_p2` or higher.
- **SolverConfiguration** — A separate artifact that pins discretisation choices (FE order, time-stepping scheme, linear solver, tolerances, …). Same MathDescription can run with different SolverConfigurations (§3.4).
- **Subdomain (class)** — A named topological entity in the MathDescription (e.g. `cytoplasm`, `membrane`). One class may be realised by multiple regions in the geometry. Carries `kind`, `motion`, and a set of equations / BCs / variables (§1.2.2).
- **Substrate velocity** — The motion velocity of a subdomain at a point: `subdomain.motion.velocity` for prescribed motion, or the resolved value of the motion variable for unknown motion. Feeds compression / dilution terms automatically (§1.10).
- **Temporality** — A per-equation field declaring `time_dependent` or `steady_state`. Determines whether the equation produces a time derivative of the governed variable. Required, not inferred (§1.9.1).
- **Tier 1 / Tier 2 / Tier 3** — Mechanisms for region-specific behaviour within a subdomain class. Tier 1 = region-keyed parameter maps (v1). Tier 2 = per-region term overrides (v2). Tier 3 = distinct subdomain classes for genuinely different physics (always available) (§1.2.5).
- **Trace operator** — `trace(u)` is the restriction of a higher-dimensional variable to a lower-dimensional evaluation context. Used to reference bulk variables from surface equations (§1.6.5, §1.8.2).
- **Variable** — A named unknown function on exactly one subdomain class. Types: scalar, vector, symmetric_tensor. Governed by exactly one equation (§1.3, §1.11.4).
- **Weak-form escape hatch** — `template: weak_form`. A UFL-residual equation for cases no operator template covers; user gives up template guardrails in exchange for full UFL expressiveness (§1.5).

---

## Appendix B — v2 Roadmap

A consolidated record of everything explicitly deferred to v2 (or beyond), pulled together from the various sections that introduce each item. Cross-referenced for context. The driving principle (§1.1.5): each item ships when a model that genuinely cannot be expressed without it is concretely needed — not on speculation.

### B.1 Operator templates

| Item | Source | Notes |
|---|---|---|
| **T5 — Stokes / Navier–Stokes momentum balance** | §1.4.3 | Vector unknown on bulk; pairs with T3 incompressibility constraint. |
| **T6 — Linear elasticity** | §1.4.3 | Vector displacement on bulk; steady-state form drops inertia. |
| **T7 — Hyperelasticity** | §1.4.3 | Non-linear $\sigma(F)$ constitutive slot. |
| **Adhesion / slippage constitutive templates** | §1.4.3 | Stokes drag, Coulomb friction, viscous slippage between membrane and substrate — so users do not write force balances from scratch. |
| **Reaction-BC sugar template** | §1.6.5 | A sugar over the composable Neumann + source pattern that desugars to the matching expressions a careful user would write by hand. Ships once enough cases accumulate to justify standardising. |

### B.2 Variable types and function spaces

| Item | Source | Notes |
|---|---|---|
| **`general_tensor` variable type** | §1.3.2 | For non-symmetric tensors (velocity gradient as a primary unknown, etc.). v1 workaround: decompose into symmetric + skew. |
| **`taylor_hood` function space** | §1.3.3 | Reserved for mechanics-template vector velocities paired with $P_1$ pressure. Lands with T5. |
| **`discontinuous_galerkin_pk` for $k \ge 2$** | §1.3.3 | Listed as a space-hint value but no operator template uses it in v1. |

### B.3 Expression language

| Item | Source | Notes |
|---|---|---|
| **Side specifiers for `trace`** | §1.2.6, §1.8.2, §1.8.7 | `trace(u, from=<subdomain>)` for different-class interfaces; `trace(u, side=a/b)` for same-class interfaces (intrinsic by region index); `trace(u, region=<name>)` opt-in when physical names matter. |
| **DG flux operators** | §1.8.7 | `jump([u])`, `avg({u})` for discontinuous-Galerkin formulations. Deferred until DG spaces are used by a template. |
| **Tensor contraction beyond `inner`** | §2.3.4 | The matching-rank inner case is sufficient for v1; explicit contraction operators land if a use case warrants. |
| **`flux_trace(u)` and friends** | §1.8.7 | Operators needed by the general-algebraic interface BC; specified alongside that BC kind. |

### B.4 Parameters

| Item | Source | Notes |
|---|---|---|
| **Per-region expressions for region-keyed parameters** | §2.2.3 (c) | v1 region maps carry per-region constants only; expressions land if a model needs per-cell time-varying inputs etc. |

### B.5 Boundary conditions

| Item | Source | Notes |
|---|---|---|
| **General-algebraic interface BC** | §1.6.2, §1.8.7 | Any expression in traces and fluxes from either side $= 0$. Escape hatch for couplings value-equality and flux-balance cannot express. |
| **Surface-surface flux-balance** | §1.6.2 (implied) | v1 flux-balance is bulk-bulk only; surface-surface conservation across a curve interface is unaddressed. |

### B.6 Initial conditions

| Item | Source | Notes |
|---|---|---|
| **Inter-variable IC references with topological-sort resolution** | §1.7.3 | v1 forbids ICs referencing other state variables to avoid ordering ambiguity. v2 may relax with explicit ordering. |

### B.7 Weak-form escape hatch

| Item | Source | Notes |
|---|---|---|
| **Discrete-time forms** | §1.5.4 | Reserved `u_prev` and `dt` symbols for users who need explicit time-integration control. v1 is continuous-time only. |

### B.8 Topological limitations (geometry-side)

| Item | Source | Notes |
|---|---|---|
| **Squashed thin layers — multi-role region assignment** | §1.2.6 (1) | One mesh entity bearing multiple subdomain-class labels with thickness-derived weights. Geometry-layer extension; not yet a formalism-level concern. |
| **Per-cell connected components within a class** | §1.2.6 (2) | Per-region variable instancing, or a topology-aware diffusion operator. v1 workaround is Tier 3. |
| **Same-class-on-both-sides disambiguation** | §1.2.6 (3), §1.8.2 | Addressed by the side-specifier work in B.3. |

### B.9 Validator

| Item | Source | Notes |
|---|---|---|
| **Unit-aware validation** | §1.11.11 | Catch dimensional errors in expressions (D in length²/time, etc.). Requires unit annotations the formalism does not currently require. |
| **Bulk-surface conservation check** | §1.6.5, §1.11.11 | Currently user-enforced — sign-matched expressions in three places. A static check would need a mass-balance solver; out of scope for the schema-level validator. |

### B.10 VCell / SBML-Spatial compatibility

| Item | Source | Notes |
|---|---|---|
| **VCell → formalism converter (executable)** | §2.6 | v1 ships docs only. Executable converter develops alongside concrete pyvcell integration use cases. |
| **formalism → VCell converter** | §2.6.4 | Lossy in one direction; offered only for the subset that fits in VCell's quirks. |
| **SBML Spatial converter** | §2.6.4 | Schemas diverge non-trivially (no weak-form, no first-class unknown motion, different per-face BC machinery). Substantial standalone project. |
| **VCell `Event`** | §2.6.3 | Discrete state transitions; needs its own design pass. |
| **`VolumeRegionVariable` / `MembraneRegionVariable`** | §2.6.3 | Piecewise-constant region variables (compartment-aggregate quantities). |
| **`FastSystem` / `FastInvariant` / `FastRate`** | §2.6.3 (memory decision) | Solver-side QSSA reduction; user's read is "could be done via change of variables, not worth it." |
| **Particle / `StochVolVariable` / stochastic constructs** | §2.6.3 | Out of scope for this formalism entirely — different primitives. |
| **`PostProcessingBlock`** | §2.6.3 | Observables / derived outputs belong with the solver-configuration / output-spec object. |

### B.11 SolverConfiguration

| Item | Source | Notes |
|---|---|---|
| **Adaptive time-stepping** | §3.4 (implied) | Schema currently has a fixed `dt`; adaptive control would add tolerance / error-estimate fields. |
| **Output specification** | §3.2.4 | A formal output-spec sub-schema (which variables, at which times, in which format) rather than the current backend-discretion exposure. |

### B.12 Explicitly NOT planned

These were considered and deliberately rejected, not deferred:

- **Lift / extension operators** (surface → bulk). Mathematically non-unique; cases that need them express the extension as a named bulk variable with its own equation and a boundary condition (§1.8.7).
- **Stochastic constructs in this formalism.** Stochastic dynamics get their own formalism with different primitives; this document is the deterministic PDE/ODE formalism only (§1.1.3).
- **Code generation.** The formalism is a *description* of a problem, not a procedure for solving one (§1.1.3).

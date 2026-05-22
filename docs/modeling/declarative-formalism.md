# A declarative formalism for cell-biology PDE/ODE systems

**Status:** work in progress. The document is being built section-by-section through discussion. As of 2026-05-21, §1.2 (geometry vocabulary), §1.3 (variables), §1.4 (equation templates), §1.5 (weak-form escape hatch), §1.6 (boundary conditions), §1.7 (initial conditions), §1.8 (coupling and expression vocabulary), §1.9 (temporality and DAE structure), and §1.10 (moving subdomains) have been drafted in detail; the surrounding sections (§1.1, §1.11, and Parts 2 and 3) are sketched as headings only.

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

*To be written.* Will state explicitly what is in scope (well-posed deterministic PDE/ODE systems on labelled geometric domains with possibly moving subdomains) and what is out of scope (solver settings — time-stepper, mesh resolution, tolerances, preconditioner; stochastic dynamics; agent-based simulation).

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

#### 1.2.7 Known topological limitations

These three patterns from the user's experience with VCell and SBML Spatial are recognised as poorly supported by class-based subdomain abstractions and intentionally **not** addressed in v1. They are recorded here so a future v2 has them in scope. The latter two share a root cause — the class abstraction loses region-instance information when a single class is realised by multiple regions in the geometry.

**(1) Squashed thin layers — multi-role mesh entities.** When two cells are adjacent and the extracellular space between them is too thin to resolve at the mesh scale, the geometric reduction maps three physical entities onto a single codim-1 surface in the mesh: the left cell's membrane, the right cell's membrane, and the squashed-to-zero-thickness extracellular volume between them. The formalism currently assumes each region in the geometry belongs to exactly one subdomain class. The squashed-layer case requires *multi-role* region assignment — one mesh entity bearing multiple subdomain-class labels simultaneously, with thickness-derived weights so that each class's measure over that entity is physically correct (each membrane contributes its own surface integral; the squashed extracellular contributes a volume integral scaled by its collapsed thickness). This is a geometry-layer extension; v1 has no mechanism for it.

**(2) Per-cell connected components within a subdomain class.** When a subdomain class (e.g. `membrane`) is realised as many topologically disconnected regions — one membrane per cell in a tissue patch — and a species on the class is supposed to diffuse *within each cell's membrane* but **not** *between cells*, the class-level equation as written would couple all components into one connected field. The checkerboard case makes this concrete: a single cell's membrane spans four boundary patches shared with its four neighbours; the species on that cell should diffuse across those four patches, but not onto a neighbour's four patches. Tier 3 (one subdomain class per cell) handles this but scales poorly with cell count. Cleaner mechanisms — per-region variable instancing where one class declaration generates one variable instance per region with independent value fields, or a topology-aware diffusion operator that respects connected components — are deferred to v2.

**(3) Same-class-on-both-sides ambiguity for membrane expressions.** When a membrane subdomain is adjacent to two regions of the *same* subdomain class — both sides are the same cell type's cytoplasm, both compartments are extracellular space, etc. — variables on either side become ambiguous in expressions written on the membrane. A Na/Ca exchanger sitting in a membrane between two `cytoplasm` regions needs `Na` and `Ca` on each side separately to compute the exchange flux, but `trace(Na)` cannot distinguish them — both sides resolve to the same subdomain class. The v2 syntax `trace(u, from=<subdomain>)` anticipated in §1.8.7 does *not* solve this: the subdomain is the same on both sides; the *region* or *orientation* is what differs.

The recommended v2 path is **intrinsic disambiguation by region index** — each region in the geometry carries a unique integer ID, and an internal interface between two regions inherits a deterministic ordering from those IDs. Syntax sketch: `trace(Na, side=a)` and `trace(Na, side=b)` (or `inside`/`outside`, naming-to-be-finalised) where one label refers to the lower-ID adjacent region and the other to the higher-ID. This aligns with how the FE level already distinguishes the two sides of an internal facet (UFL's `("+")` / `("-")` convention is precisely index-based), needs no geometric metadata beyond stable region IDs, and handles symmetric cases — Na/Ca on two identical cytoplasm regions — with no model-side naming overhead.

The trade-off is that index-based labels carry no physical meaning: if the geometry is regenerated with a different region ordering, the model's "side a" and "side b" silently swap. The geometry is therefore expected to expose a readable region-ID-to-label mapping (e.g. `region_id 17 → cyto_left_cell`) for results interpretation and post-processing, even though the math description does not reference those labels.

When physical names *must* appear in the math (because the cell biology depends on knowing which region is which — e.g. a polarised tissue where left and right cells genuinely differ in role even if their compartment classes match), the opt-in fallback is **explicit region naming**: `trace(Na, region=<region_name>)`, requiring the geometry to declare names for the regions in question. This trades some compactness for unambiguous physical meaning.

v1 ships neither mechanism. Cases that need either fall back to Tier 3 (declare distinct subdomain classes for each region — `cytoplasm_left` and `cytoplasm_right` — even when their physics is identical), trading some duplication for unambiguous addressability.

These three limitations are flagged here rather than buried in v2 roadmap notes because they are the most likely surprises to bite a modeller coming from a mature tool like VCell that has accumulated workarounds for all three.

#### 1.2.6 Worked sketch — geometry block of the §1.6.6 example

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

Tensor variables that are *not* symmetric (rare in cell-biology but possible for e.g. gradient of velocity as an unknown) require the weak-form escape hatch in v1; v2 may add a `general_tensor` type.

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

- **Spatial coordinates and geometric helpers** (`x`, `n(x)`, `H(x)`, etc.) evaluate against the current (deformed) configuration at every time step.
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
| `source` | scalar expression | no | s. May depend on u, x, t, parameters, traces of variables on other subdomains (§1.8). |

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
  - Spatial coordinates `x`, plus geometric helpers (the outward unit normal `n(x)`, tangent basis, mean curvature `H(x)`, etc.) — provided as built-in functions; full list in §2.
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
        velocity: "r_dot * (x / |x|)"   # uniform radial expansion in R^2

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
      initial_condition: "1.0 + 0.5 * cos(2 * theta(x))"

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
- **Parameters, time `t`, spatial coordinates `x`, geometric helpers** (`n(x)`, `H(x)`, etc.) per §1.8.
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

Expanded from the §1.10.8 sketch: a closed 2D membrane whose velocity is solved by a quasi-static viscous force balance. The membrane is in mechanical equilibrium at every instant; surface tension and a prescribed active traction drive motion, and viscous drag from the surrounding cytosol resists it.

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
          + sigma_T * H(x) * inner(n(x), v_membrane_test)
          - inner(f_active(x, t), v_membrane_test)
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
      initial_condition: "1.0 + 0.3 * cos(2 * theta(x))"

  parameters:
    - { name: eta,     value: 1.0  }
    - { name: sigma_T, value: 0.10 }
    - { name: k_off,   value: 0.02 }
```

What this example demonstrates:

- **The weak-form equation is a residual that integrates to zero.** The expression following `form:` is the entire residual; the convention is that the assembled equation reads "form = 0 for all `v_membrane_test`."
- **Test function is implicit.** `v_membrane_test` is the test function for `v_membrane`, in the same function space. The user did not declare it.
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

Fixes the *outward* normal flux on the boundary. $\mathbf{n}$ is the outward unit normal of the variable's home subdomain; positive $h$ means flux flowing out of the subdomain across the boundary. `expression` is a scalar expression. For variables governed by a template with non-isotropic diffusion, $D \, \nabla u \cdot \mathbf{n}$ generalises to $(D \nabla u) \cdot \mathbf{n}$ in the natural way.

##### Robin

$$\alpha \, u \;+\; \beta \, D \, \nabla u \cdot \mathbf{n} \;=\; h(\mathbf{x}, t)$$

Linear combination of value and flux. Covers permeability-type conditions (membrane permeability with a fixed external reference, semi-permeable wall, etc.) without a separate template. `expression` here is a tuple `(α, β, h)` of three scalar expressions.

##### Interface — value-equality

At an internal boundary between two subdomains, with the variable's "left side" $u_L$ and the partner variable's "right side" $u_R$:

$$u_L \;=\; k(\mathbf{x}, t) \cdot u_R$$

Covers continuity ($k = 1$, the same physical quantity expressed as variables on either side — e.g. voltage continuous across a passive membrane) and partition equilibrium ($k \ne 1$, e.g. Nernst-style partitioning across a barrier). `expression` carries the partition coefficient $k$ (scalar; defaults to 1 for pure continuity).

The BC names both variables: `variable` (the one on the left), `partner_variable` (the one on the right). Both must be defined on subdomains incident to `boundary`. The formalism does not privilege either side — `(u_L, u_R)` and `(u_R, u_L)` express the same condition.

##### Interface — flux-balance

The jump in outward normal flux across an internal boundary, evaluated from the perspective of `variable`'s subdomain, equals a constitutive expression:

$$\bigl[D \, \nabla u \cdot \mathbf{n}\bigr] \;=\; f(\text{traces from either side}, \mathbf{x}, t, \text{parameters})$$

The expression may reference the variable itself, the partner variable, traces of any other variable defined on either side's subdomain, geometric helpers, time, and parameters. This is the generalisation of VCell's `JumpCondition` without the Neumann-only restriction — the flux expression can express any constitutive relation (linear permeability $P(u_L - u_R)$, saturating transport $V_{max} u_L / (K + u_L)$, voltage-gated channel kinetics, …).

`partner_variable` must be supplied. As with value-equality, the BC is symmetric across the two sides; the sign convention is fixed by which subdomain `variable` belongs to.

A v2 "general algebraic" interface kind — any expression in traces and fluxes from either side $= 0$ — is anticipated as an escape hatch but deferred. Value-equality and flux-balance together cover every interface coupling in the project's foreseeable use cases.

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

When a boundary is *itself* a subdomain that carries its own PDE — the canonical cell-membrane case, where Γ_mem is both the interface between cytoplasm and extracellular space *and* a surface subdomain carrying receptor density variables — the coupling between bulk and surface is expressed entirely through trace operators in expressions.

**The trace operator.** `trace(u)` is the value of a higher-dimensional variable $u$ restricted to a lower-dimensional boundary or interface within its domain. For a bulk variable $L$ defined throughout the cytoplasm Ω_cyto, `trace(L)` evaluated on the membrane Γ_mem is the value of $L$ at the membrane — formally, the limit of $L$ as you approach the membrane from inside Ω_cyto. Surface variables like $\rho_f$ that already live on Γ_mem are written directly; only higher-dimensional variables need an explicit `trace(·)` when used in a lower-dimensional expression. The trace is mathematically well-defined for the Sobolev spaces our variables live in (H¹ bulk functions have H^{1/2} traces on the boundary); FEniCSx handles trace assembly automatically. The full vocabulary of cross-dimensional reference operators is catalogued in §1.8.

**The composable pattern, then, is:**

- **Membrane equation source terms** reference bulk variables via `trace(·)`: e.g. `k_on * trace(L) * rho_f - k_off * rho_b`.
- **Bulk BCs at the membrane** reference surface variables directly (no `trace` needed; the surface variable already lives on the membrane): e.g. a Neumann BC for L with expression `k_on * trace(L) * rho_f - k_off * rho_b`.
- **Mass conservation** — what is consumed from the bulk equals what is produced on the surface — is the user's responsibility, expressed by writing matched expressions in both places with the appropriate signs. The schema does *not* auto-balance.

There is no dedicated "reaction" BC entity in v1; the composable form covers all cases. A future v2 may add a sugar template that desugars to the same two expressions a careful user would have written by hand, once enough use cases accumulate to justify standardising it.

#### 1.6.6 Worked example — ligand-receptor binding with bulk diffusion

Extending the §1.4.5 receptor example: now the extracellular ligand L diffuses in the bulk Ω_ext, binds reversibly to the membrane-bound free receptor $\rho_f$ to form the bound complex $\rho_b$, and the outer boundary of the extracellular space is held at a fixed bulk concentration (Dirichlet) — a stirred reservoir.

```yaml
math_description:
  geometry: cell_with_extracellular   # external reference; provides Ω_ext, Γ_mem, ∂Ω_outer

  subdomains:
    - { name: extracellular, kind: bulk,    motion: { kind: none } }
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

    # Coupling at the membrane: ligand flux out of the extracellular bulk
    # equals the net binding rate produced on the membrane. Matched
    # expression to the rho_b source above; user-enforced conservation.
    - variable: L
      boundary: membrane
      kind: neumann
      expression: "k_on * trace(L) * rho_f - k_off * rho_b"

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

If the membrane were itself moving (replace `motion.velocity: "0"` with a real expression), the dilution term in both surface PDEs picks up automatically from `membrane.motion`; the BC structure does not change.



### 1.7 Initial conditions

#### 1.7.1 When required

An initial condition is required for a variable iff its governing equation declares `temporality: time_dependent`. Variables governed by `steady_state` equations have no IC and the schema rejects one if provided. Variables that are not governed by any equation (a corner case that should not occur in a well-formed MathDescription, caught by §1.11) likewise have no IC.

Each variable's IC appears as the `initial_condition` field on its governing equation. This placement — IC on the equation, not on the variable's declaration — reflects that the IC is part of the well-posed time-evolution problem (variable + equation + IC + BCs) and is meaningless without the equation context.

#### 1.7.2 Form, scope, and evaluation context

An initial condition is a typed expression evaluated at $t = 0$ on the variable's subdomain. Its type must match the variable's `type` (scalar IC for scalar variable, vector IC for vector variable, symmetric-tensor IC for tensor variable).

The evaluation context is the **reference configuration**: the geometry's initial mesh, with no motion applied. For subdomains with `motion.kind` of `prescribed` or `unknown` (§1.10), the reference configuration is what the geometry provides; the motion field's effect on positions is applied for $t > 0$, not at $t = 0$. Geometric helpers (`n(x)`, `H(x)`, etc.) in IC expressions therefore evaluate against the reference configuration.

Spatial coordinates `x` in an IC expression refer to the reference configuration's coordinate. There is no IC equivalent of "current configuration coordinates" — at $t = 0$ the two coincide.

#### 1.7.3 Vocabulary restrictions

IC expressions use the §1.8 vocabulary with two restrictions:

- **No references to other state variables.** An IC expression for variable $u$ may not reference any other variable in the MathDescription (whether via `trace(·)`, by bare name, or otherwise). This avoids ordering ambiguity — what does "$u(\mathbf{x}, 0) = 0.5 \cdot v(\mathbf{x}, 0)$" mean if $v$ is itself defined by an IC that references $u$? Worse, it avoids cycles. v1 sidesteps both concerns by forbidding the references; if a concrete use case demands coupled ICs, v2 may relax with topological-sort resolution.

- **No reference to time `t`.** ICs are evaluated at $t = 0$ by definition; referencing $t$ in an IC expression has no useful meaning beyond a constant substitution. The schema rejects `t` in IC expressions to catch the misconception cleanly (a user writing `initial_condition: "exp(-t)"` likely meant a *forcing* expression, not an IC, and should be told).

What IC expressions **may** reference: the spatial coordinate `x` and its accessors (`x[0]`, `theta(x)`, `r(x)`, …); named parameters, including region-keyed parameter maps (§1.2.5); geometric helpers (`n(x)`, `H(x)`, principal curvatures, tangent basis); standard functions (`sin`, `cos`, `exp`, `if`, `step`, etc.).

#### 1.7.4 Type matching

The IC's value type must match the variable's `type`:

| Variable type | IC must produce |
|---|---|
| `scalar` | scalar |
| `vector` | vector in $\mathbb{R}^d$ |
| `symmetric_tensor` | symmetric $d \times d$ tensor |

No implicit broadcasting. A scalar where a vector is expected is an error; the user must write the broadcast explicitly (e.g. `0.0 * n(x)` for the zero vector along normal, or `[0.0, 0.0]` for an explicit 2D vector).

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

**Side specifier (deferred to v2).** When the higher-dim subdomain $\Sigma_{\text{high}}$ contributes exactly one connected region on one side of the lower-dim evaluation context, `trace(u)` is unambiguous. Two cases require a side specifier and are deferred (see §1.2.7): (a) a single variable defined on both sides of an internal interface where the two sides are *different* subdomain classes — resolved by `trace(u, from=<subdomain>)`; (b) an interface between two regions of the *same* subdomain class — resolved by intrinsic index-based disambiguation `trace(u, side=a)` / `trace(u, side=b)` (recommended path, mirrors UFL `+`/`-` semantics) or, when physical names matter, by explicit region naming `trace(u, region=<region_name>)`. v1 rejects ambiguous traces at validation time and the modeller falls back to Tier 3 (separate subdomain classes) for unambiguous addressing.

**Implementation.** Trace evaluation is a backend concern. In FEniCSx 0.10, traces of bulk variables on internal facets are realised through native mixed-dimensional assembly (see `docs/research/2026-05-21-fenicsx-ecosystem.md`). The user-facing formalism does not commit to a particular evaluation strategy.

#### 1.8.3 Time, space, and parameters

| Symbol | Meaning | Type |
|---|---|---|
| `t` | The time variable. Reserved name; equals current solver time, $t = 0$ in IC expressions. | scalar |
| `x` | Spatial coordinate at the evaluation point, in the embedding-space dimension. | vector in $\mathbb{R}^d$ |
| `<param_name>` | A named scalar parameter declared in the MathDescription. | scalar (declared dtype) |

Component access on `x` is by index: `x[0]`, `x[1]`, `x[2]`. Named coordinate accessors derived from `x` are listed under geometric helpers (§1.8.4).

#### 1.8.4 Geometric helpers

Available in expressions evaluated on subdomains for which the relevant notion is defined:

| Helper | Meaning | Defined where |
|---|---|---|
| `n(x)` | Outward unit normal. On a boundary or codim-1 subdomain, the outward direction relative to the home subdomain. | codim-1 entities |
| `H(x)` | Mean curvature. | codim-1 entities embedded in higher-dim space |
| `kappa1(x)`, `kappa2(x)` | Principal curvatures. | codim-1 surfaces in 3D |
| `tangent(x)` | Tangent unit vector. | 1-curves in 2D, or codim-2 edges in 3D |
| `theta(x)`, `phi(x)`, `r(x)` | Polar / spherical accessors. Sugar for `atan2(x[1], x[0])`, etc. | any subdomain |

For subdomains with `motion.kind = unknown`, geometric helpers are evaluated against the current (solver-computed) configuration at every time step including $t = 0$, where the configuration comes from the motion variable's initial condition (§1.7, §1.10).

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

The rule exists because operator templates make assumptions about the differential order and integration-by-parts pattern of the assembled weak form. Allowing arbitrary calculus on the equation's own variable can silently violate those assumptions: a `lapl(u)` in T1's `source` slot for $u$ embeds a second-order operator on $u$ where the template expects a coefficient; a `grad(u)` in the same slot creates a first-order advective term outside the template's integration-by-parts machinery. UFL would compile these into *some* form; the result is unlikely to be what the user meant.

Calculus on *other* variables is safe because it produces a coefficient-shaped value (scalar, vector, or tensor) that the template uses positionally — chemotaxis source `-grad(phi) * u`, voltage-gradient drift in a relative-advection slot, or a custom flux on an internal interface BC referencing both sides' gradients. The PDE structure for the slot's governing variable is preserved.

The validator (§1.11) enforces this by walking the expression AST: if any `grad`/`div`/`lapl`/`grad_surf`/`div_surf`/`lapl_beltrami` is applied to (or transitively reduces to) the slot's governing variable, that's an error.

#### 1.8.6 Variable, parameter, and bare-name resolution

A **bare name** (no `trace(·)`, no calculus operator) in an expression resolves in this order:

1. **Local variable** — a variable defined on the same subdomain as the expression's evaluation context. Resolves to the function value at the current point.
2. **Named parameter** — a top-level parameter declared in the MathDescription. Resolves to its constant value.
3. **Reserved name** — `t` (time), `x` (space), or a name from the geometric-helper or standard-function tables.

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
| Robin coefficient tuple $(\alpha, \beta, h)$ | tuple of three scalars |
| Interface value-equality partition coefficient $k$ | scalar |

Type mismatches are validation errors. A scalar where a vector is expected is **not** implicitly broadcast; the user must write the broadcast explicitly (e.g., `c * n(x)` to turn a scalar `c` into a vector along the outward normal).

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
- A `steady_state` equation must not have an `initial_condition` field.

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
| `none` | Subdomain is static. Default when `motion` is omitted. Substrate velocity is zero everywhere; compression / dilution terms in operator templates vanish; geometric helpers (`n`, `H`, etc.) evaluate against the initial (and only) configuration. |
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

- **Velocity form is natural for**: radial expansion (`r_dot * x / |x|`), fluid-driven motion, prescribed steady flow.
- **Displacement form is natural for**: rigid translations (`[v_x, v_y, v_z] * t`), oscillations (`A * sin(omega * t) * e_1`), prescribed wall deformations.

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

**ICs for other variables on the moving subdomain** are evaluated on the initial (t=0) configuration. For unknown-motion subdomains, that is the configuration produced by the motion variable's IC, which by default is the reference configuration. Composability is direct: writing `initial_condition: "1.0 + 0.5 * cos(2 * theta(x))"` for a receptor density refers to angular coordinate $\theta$ on the *initial* membrane — exactly what the modeller intends.

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
- BC expressions evaluated on the boundary use the boundary's current (deformed) position. Geometric helpers (`n(x)`, `H(x)`, etc.) on a moving boundary reflect the current configuration.

The implication for BCs is mostly transparent: a Dirichlet BC `u = f(x, t)` on a moving boundary evaluates `f` at the current position $x$ on the deformed boundary. A Neumann BC's flux is the flux through the current boundary surface; the outward normal $\mathbf{n}$ is the current outward normal, not the reference one.

#### 1.10.7 What is solver-side, not in the math description

The following are explicitly *not* part of the MathDescription, even though they materially affect a moving-domain simulation:

- **ALE mesh-motion algorithm.** Whether the bulk mesh follows the membrane motion exactly, follows a harmonic-extension velocity field, follows a fictitious elastic-extension, or uses a different recipe entirely — backend choice. Different backends may make different choices for the same MathDescription.
- **Mesh remeshing for large deformations.** When the deformation is large enough that mesh quality degrades, some backends remesh adaptively. Triggering conditions, remeshing algorithms, and field-transfer schemes are solver-side concerns.
- **Time-stepping for the motion variable.** Whether the motion variable's update is implicit or explicit, whether it is solved monolithically with field variables or in a partitioned manner — all backend choices.
- **Numerical stabilisation for advection-dominated regimes.** SUPG, GLS, entropy-viscosity stabilisations are stabilisation choices; they do not change the math problem.

The math description states the well-posed problem; the backend chooses how to solve it.

#### 1.10.8 Worked sketch — mechanics-driven membrane motion with a surface species

The first model that pushes beyond the §1.4.5 / §1.6.6 prescribed-motion examples: a closed membrane whose motion is solved by a simple viscous force balance, with a receptor density on the membrane that experiences the resulting motion via the standard T2 dilution.

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
    #   eta * v_membrane = -sigma_T * H(x) * n(x) + f_active(x, t)
    #
    # where eta is drag, sigma_T is surface tension, H is mean curvature,
    # n is outward normal, f_active is a user-supplied driving traction.
    - template: weak_form               # §1.5
      variable: v_membrane
      subdomain: membrane
      temporality: steady_state          # quasi-static at each time step
      form: |
        ( eta * inner(v_membrane, v_membrane_test)
          + sigma_T * H(x) * inner(n(x), v_membrane_test)
          - inner(f_active(x, t), v_membrane_test) ) * dx_Gamma  # = 0
      initial_condition: "0"              # zero default (memory decision 11c)

    # Receptor density: standard T2 surface PDE. Dilution from v_membrane is automatic.
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: 0.05
        source: "-k_off * rho"
      initial_condition: "1.0 + 0.3 * cos(2 * theta(x))"

  parameters:
    - { name: eta,     value: 1.0  }     # viscous drag coefficient
    - { name: sigma_T, value: 0.10 }     # surface tension
    - { name: k_off,   value: 0.02 }     # receptor decay rate
```

What this sketch demonstrates:

- **`motion: { kind: unknown, variable: v_membrane }`** wires the membrane's substrate velocity to a vector unknown solved on the same subdomain.
- **The motion equation uses the §1.5 weak-form escape hatch** — because v1 has no surface-mechanics template, the user writes a UFL form directly. v2 will replace this with a T5/T6/T7-style mechanics template.
- **The receptor density equation is unchanged from §1.4.5** in shape — only `motion.kind` changed from `prescribed` to `unknown`. The T2 template picks up the dilution automatically from `membrane.motion`, whether prescribed or solved. The user does not edit the receptor equation when switching motion modes.
- **`initial_condition: 0` for `v_membrane`** uses the zero default (memory decision 11c). The mesh starts at rest at t = 0; the force balance immediately produces a non-zero velocity in response to the initial curvature and traction.
- **Geometric helpers `H(x)`, `n(x)`** evaluate against the current deformed membrane configuration (§1.10.6). At t = 0 that is the geometry's initial configuration.

This sketch is intentionally minimal — a single membrane, one mechanics balance, one surface species. Real cell-migration models add bulk hydrodynamics, multiple surface species, intracellular signalling, and adhesion-with-slippage to substrates. Each of those is expressible in the formalism (the deferred constitutive templates from §1.4.3 and the mechanism for unknown motion + governing equation) once the matching templates and adhesion vocabulary ship.

### 1.11 Well-posedness checks

*To be written.* Static rules a MathDescription must satisfy: every variable referenced has an equation in every subdomain where it lives; every time-dependent variable has an IC; every BC references a labelled boundary that exists; types of slot expressions match the template's declared types; etc.

---

## Part 2 — Data model

*To be written.* Schema (Python dataclasses, with YAML/JSON round-trip), naming and reference resolution, validation rules, and the mapping to/from VCell `MathDescription` (which subset round-trips, which parts deliberately do not and why).

---

## Part 3 — Solver contract

*To be written.* What a backend must implement to claim "I solve any model in this class": assemble forms from templates, assemble user-supplied UFL, apply BCs, advance time or solve steady. What the backend is free to choose (FE order, mesh, time-stepper, linear solver, preconditioner — none of which appear in the math description). The vcell-fenics DOLFINx backend: which templates it handles in v1 and what it punts.

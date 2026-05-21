# A declarative formalism for cell-biology PDE/ODE systems

**Status:** work in progress. The document is being built section-by-section through discussion. As of 2026-05-21, §1.4 (equation templates), §1.6 (boundary conditions), and §1.8 (coupling and expression vocabulary) have been drafted in detail; the surrounding sections are sketched as headings only.

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

*To be written.* Subdomains by topological dimension (volume Ω_k, surface Γ_k, curve Λ_k, point P_k). Labelled codim-1 sub-boundaries (∂Ω_k = ⋃ Γ_kj). Reference-by-name to a geometry object that lives outside the MathDescription.

### 1.3 Variables

*To be written.* Function spaces (scalar, vector, tensor; continuous, discontinuous), scope (defined on which subdomain — multiple variables on the same subdomain is the normal case), domain-of-definition semantics on moving subdomains.

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

*To be written.* Schema for "this equation is a UFL form; here are the test/trial functions, integration measures, and boundary terms." For unusual physics that no operator template covers (Cahn-Hilliard, custom constitutive laws, mixed FE pairs).

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

*To be written.* Required iff the model has any `time_dependent` equation. Specified per variable, must be well-defined on the variable's subdomain at t = 0.

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

**Side specifier (deferred to v2).** When a single variable is defined on a single subdomain — the v1 norm — `trace(u)` is unambiguous: there is exactly one side. If v2 admits a variable defined on both sides of an internal interface, the syntax `trace(u, from=<subdomain>)` is reserved for disambiguation; the schema will reject ambiguous traces at validation time until then.

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

*To be written.* Per-equation `time_dependent` / `steady_state` declaration; what it means for a model to contain both; validation rules ("you said steady but have ∂_t" — error).

### 1.10 Moving subdomains

*To be written.* `subdomain.motion ∈ {none, prescribed: <expr>, unknown: <motion_variable>}`. The unknown case requires an equation governing the motion variable. Variables on a moving subdomain are defined on the deforming manifold; the operator templates handle the time-derivative convention. ALE / mesh-motion algorithm choices are solver-side.

### 1.11 Well-posedness checks

*To be written.* Static rules a MathDescription must satisfy: every variable referenced has an equation in every subdomain where it lives; every time-dependent variable has an IC; every BC references a labelled boundary that exists; types of slot expressions match the template's declared types; etc.

---

## Part 2 — Data model

*To be written.* Schema (Python dataclasses, with YAML/JSON round-trip), naming and reference resolution, validation rules, and the mapping to/from VCell `MathDescription` (which subset round-trips, which parts deliberately do not and why).

---

## Part 3 — Solver contract

*To be written.* What a backend must implement to claim "I solve any model in this class": assemble forms from templates, assemble user-supplied UFL, apply BCs, advance time or solve steady. What the backend is free to choose (FE order, mesh, time-stepper, linear solver, preconditioner — none of which appear in the math description). The vcell-fenics DOLFINx backend: which templates it handles in v1 and what it punts.

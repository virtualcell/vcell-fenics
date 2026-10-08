# ADR 004 — A DiscreteProblem IR for backend translation

**Date:** 2026-05-24
**Status:** Accepted (not yet implemented)

## Context

Part 2 of the declarative formalism — the MathDescription data model, expression parser, and validator — is complete: a model can be loaded, parsed, and validated, but not yet executed. Part 3 (`docs/modeling/declarative-formalism.md` §3) is *backend translation*: turning a validated `MathDescription` (plus a `Geometry` and a `SolverConfiguration`) into a running DOLFINx solve for the v1 subset (§3.6: scalar variables, T1/T2 templates, prescribed or no motion, no weak-form escape hatch).

The existing bespoke prototypes (`approaches/static/bulk_pde.py`, `approaches/submesh/surface_pde.py`) encode the FEniCSx target assumptions *imperatively* — backward Euler, natural-Neumann as the default BC, P1 Lagrange, "∇ on a codim-1 submesh is the tangential gradient ∇_Γ", direct LU — buried inside lines like `a = u*w*dx + dt*D*grad(u)·grad(w)*dx`. Two questions arise when generalising this to formalism-driven assembly:

1. Should the backend build DOLFINx/UFL objects directly, or pass through an explicit intermediate representation that makes the FEniCSx target assumptions first-class rather than buried in assembly code?
2. Is correct *translation* (math → discrete operator) independently verifiable, or do we only have accurate-solution tests as holistic, tolerance-bound evidence that the whole pipeline is approximately right?

Note the layering: UFL is *already* a symbolic IR sitting between the math and the assembled PETSc matrices. So the real question is not "objects vs. an IR" at the expression level — it is whether to add a representation at the *equation / problem* level.

## Decision

Introduce a **`DiscreteProblem` intermediate representation** between expression compilation and DOLFINx object construction. The translation pipeline is:

```
expression AST → UFL (expression compiler) → DiscreteProblem IR → lowering → dolfinx / PETSc
```

The concrete decisions:

1. **The IR is FEniCSx-backend-internal — not the cross-backend seam.** The backend-neutral boundary remains the `MathDescription` itself (§3.1). The `DiscreteProblem` IR is specific to a finite-element backend; a finite-volume backend would not consume it.

2. **The expression compiler emits UFL directly.** UFL is the expression-level IR; we do not introduce a second one. Name resolution into concrete UFL objects (a variable's `Function`, a parameter's `Constant`, `x` → `SpatialCoordinate`, geometric helpers, calculus operators → `ufl.grad`/`div`/tangential variants) happens inline via a compile context built from the geometry and function spaces at solve time. The resolved/typed AST described in §2.3.3 is realised *transiently* during compilation, not materialised.

3. **The IR uses a semi-discrete residual model.** A `DiscreteProblem` records the spatial weak-form residual as a small, fixed enum of **tagged terms** — `TIME_DERIVATIVE`, `DIFFUSION`, `ADVECTION`, `DILUTION`, `SOURCE` — each carrying a real UFL integrand, plus a **separate time-scheme object** (backward Euler for v1). Lowering applies the scheme to the term list to compose the bilinear/linear forms (`a`, `L`) and then constructs the DOLFINx problem.

   *Rejected alternative:* a "discrete-forms + flags" record that holds the already-composed `a`/`L` UFL forms alongside boolean metadata (`has_dilution`, `scheme="backward_euler"`). Rejected because the flags are assertions sitting *next to* an opaque form and can silently drift from what the form actually contains — they do not provide independent verification, which is the entire motivation for the IR.

4. **Lifecycle: built once + mutable handles + an explicit per-step protocol.** The structure (spaces, term integrands, BCs) is built once. Time-varying inputs are held as named mutable handles — the time `Constant`, previous-step `Function`s, motion-derived coefficients — and updated each step through a documented protocol (set time, set previous solution, apply motion, solve). Mesh motion mutates node coordinates between solves; motion-derived coefficients such as the dilution coefficient ∇_Γ·v_Γ are recomputed from the velocity expression on the moved mesh.

5. **The IR unit is one coupled solve over a (possibly mixed) function space, not one per equation.** Coupling between equations is detected *structurally* — a `source` expression's AST references another governed variable on the same subdomain — never semantically; there is no "reaction" concept (§1.1.3 makes reactions-as-entities a non-goal; a reaction is just an algebraic `source:` string). Because v1 source terms are linear in the unknowns, a monolithic mixed-space backward-Euler step is exact and a single linear solve, preferred over operator splitting. The choice between monolithic and split assembly is a discretisation decision (§3.3 discretion zone); it does not change the math being solved.

6. **Verification proceeds in four layers; accurate-solution tests are not relied on alone.**
   - *IR-structural* — assert the tagged terms, function space, and time scheme directly on the `DiscreteProblem`, with no solve (e.g. T2-with-motion has a `DILUTION` term; static bulk diffusion has exactly `{TIME_DERIVATIVE, DIFFUSION}` and no Dirichlet object).
   - *Compiler ↔ UFL equivalence* — the assembled matrix from a compiled expression equals a hand-written UFL form's.
   - *Operator invariants* — exact-ish properties the assembled operator must have, independent of any solution: `K·1 ≈ 0` for a pure-Neumann Laplacian, `A·1 = M·1` for the backward-Euler operator, symmetry of self-adjoint operators, mass-matrix row-sums equal to ∫φᵢ.
   - *Analytical / manufactured-solution integration* — the existing conformance suite (cos(kθ) decay, the dilution discriminator).

   The layers narrow from "translated correctly" to "converges correctly," and isolate a fault to a stage rather than only signalling that the whole pipeline is off.

## Consequences

**Positive:**

- The FEniCSx target assumptions become an explicit, reviewable, testable artifact instead of being buried in imperative assembly.
- Translation correctness is verifiable independently of numerical accuracy; a bug is isolated to a pipeline stage.
- The time scheme is swappable (e.g. Crank–Nicolson, BDF2) by changing the scheme object, not the term list.
- The bespoke `BulkPDE` / `SurfacePDE` weak-form logic becomes term-builders feeding the IR, so the standalone classes can be deleted rather than retained as parallel code.

**Negative:**

- One more layer to maintain, with a risk of drifting toward a redundant mirror of UFL/DOLFINx. Mitigated by keeping the IR to translation *decisions* (tagged terms + scheme + lifecycle) with UFL forms as leaves, and by using a closed term enum rather than an open term-algebra.
- The term enum is fixed for v1's templates (T1/T2). New templates (T3/T4, the mechanics templates T5–T7) will extend it — a deliberate, incremental growth driven by real assemblers, not a general framework specified up front. *(2026-10-08: T5 became `region_ode`; the proposed mechanics templates are now numbered T6–T8, formalism §1.4.3. The enum did grow by `ADVECTION` and by boundary terms, as this predicted.)*

## Notes

- **Implementation status:** decided, not yet implemented. No Part 3 code exists as of this date; Part 2 (schema, loader/dumper, parser, validator) is complete.
- **Scope:** this ADR records the *design* of the translation layer. The increment sequence / rollout plan is intentionally not recorded here, as sequencing is expected to churn during implementation.
- **Cross-references:** `docs/modeling/declarative-formalism.md` §3 (solver contract), §3.3 (discretion zone — where the coupling-assembly choice lives), §3.6 (vcell-fenics backend scope), §1.1.3 (reactions are not first-class), §2.3.3 (the resolved/typed AST this realises transiently).

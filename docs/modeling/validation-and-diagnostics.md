# Model validation and runtime diagnostics — a strategy

> **Documentation review (2026-10-08):** See the [documentation map](../README.md) and
> [discrepancy register](../reviews/2026-10-08-documentation-discrepancies.md) for distinctions
> between historical plans and current implementation. New modeling discussion lives in the
> [cell-mechanics workspace](../modeling/cell-mechanics/README.md).

**Status (2026-10-08):** strategy note, partly built. The build-time layers exist
(`formalism/validator.py`, including the weak-form dilution and inf-sup warnings of §6 rows 1
and 6), backward Euler refuses nonlinear sources by name (`NonlinearTermError`, row 7), and the
runtime-failure translation of §5 is `backend/diagnostics.py` (`SolveError`: the non-finite
residual localised to a term at the initial condition, `TS` and linear-step failures named) —
but only on the single-mesh backward-Euler and method-of-lines paths; the multi-compartment and
interface-coupled solvers still surface raw PETSc errors. Units inference (§4) is not built.
The table in §3 and the *done* notes in §5–§7 record the state. Captures the strategy and a
growing registry of known failure modes; the build order in §7 is the implementation plan.

## 1. The problem, and why it is harder for us than for VCell

A modeler should learn that a model is wrong **as early as possible**, with a message that
points at *their model* — not at a PETSc reason code three layers down. Virtual Cell does
this well, and the reason is structural: VCell models are built from **templated reactions
and equations**, and the platform is constructed so that a model that *parses* is, by and
large, a model that is **well-posed**. The space of expressible models is deliberately
narrow enough that consistency can be guaranteed by construction.

We have made a different bet. The declarative formalism keeps the templated core
(`docs/modeling/declarative-formalism.md` §1.4, templates T1/T2/…) **but also** opens a
**weak-form escape hatch** (§1.5): arbitrary UFL residual terms, so a user can express
physics no template covers (mechanics, active stress, custom couplings). On top of that the
backend spans moving meshes (ALE), saddle points (Stokes), mixed-dimensional coupling, and
multiple time integrators. That expressiveness is the whole point of the project — but it
means **we cannot guarantee well-posedness a priori the way VCell can.** An arbitrary weak
form can be inconsistent, unstable, or ill-posed in ways no symbolic check will catch.

So the strategy cannot be "make every model provably correct before it runs." It has to be:

> **Validation strength degrades gracefully as expressiveness grows.** A fully templated
> model should get close to VCell's a-priori guarantee. An escape-hatch model gets weaker
> static guarantees and leans harder on **runtime diagnostics** that translate a solver
> failure back into a model-level explanation.

The two halves — strong static checks for the templated subset, good runtime translation
for everything — are complementary, not alternatives. But there is a stronger move available
than treating the escape hatch as a permanent user-facing surface we can only mitigate — see
§1.1.

## 1.1 The architectural resolution — templates are the user surface, weak forms are the compiler

The framing above quietly assumes the modeler *authors* weak-form terms directly, and that we
must therefore live with un-guaranteeable models. A better architecture **dissolves the
tension instead of mitigating it**: treat the weak-form layer as an **intermediate
representation — a compilation target — not an authoring surface.** The user never (or only in
an explicit expert mode) writes raw weak forms; they work with **well-posed templates**, and
those templates *expand* into weak-form terms underneath.

This recovers VCell's guarantee where it matters — at the surface the modeler touches:

- **A problem-generation layer is the front door.** Concretely, a VCell import (the
  `../pyvcell` integration target) emits models built only from **well-posed templates** —
  VCell already guarantees that on its side. Templates, not weak forms, are the unit of
  authoring. A fully templated model is well-posed by construction, exactly as in VCell.
- **New physics enters as a new template, authored once by an expert**, and *that* template
  is what compiles to weak-form terms in this formalism. The expressiveness of the escape
  hatch is still used — but as the *implementation* of a template, paid for once at authoring
  time, not re-incurred by every modeler. The template carries a well-posedness contract; the
  weak-form expansion is an implementation detail behind it.
- **Maturation pipeline — "solidify" proven templates back into the formal layer.** A template
  begins as an expert-authored weak-form expansion (flexible, fast to prototype). After it has
  earned trust through use, it is *solidified*: promoted to **first-class formal support** — its
  own tagged term kinds (ADR 004), dedicated validation rules, dimensional signatures, dedicated
  diagnostics — so it no longer rides the generic escape hatch. The path is **weak-form
  expansion → reusable template → first-class formal construct**, in increasing order of
  guarantee and decreasing order of flexibility. Physics matures *toward* the formal layer over
  time, rather than each model re-deriving it.

What this does to the strategy:

- The **a-priori guarantee is recovered at the user layer** — because the user layer is
  templates, not arbitrary weak forms. The strong static checks (§4) apply to the templated
  surface, which is now *the* surface.
- The escape hatch's weaker guarantees and the **runtime diagnostics (§5) become primarily a
  template-author's tool** — used while developing and hardening a new template, and as a
  backstop — rather than something an end modeler routinely hits. (This reframes the layer just
  built: it serves the author hardening physics, plus the rare expert-mode user.)
- **Raw weak-form authoring still exists**, but as a clearly-marked, unsupported expert mode —
  the place new templates are prototyped — never the default surface, and never carrying a
  well-posedness promise.

The rest of this note still holds; this section changes *who* each layer serves and *where*
the guarantee lives, not the layers themselves. §2–§6 describe the machinery; §1.1 says the
templated surface is the one that must be excellent, and the weak-form layer is the compiler
behind it.

## 2. The organizing idea — the *assumed solution class*

The single most useful lens: **every discretization choice carries implicit assumptions
about the solution** — the function space it lives in, its regularity, its conservation
structure, the null spaces of the operator, the compatibility of the boundary data. A large
fraction of real failures are not "bugs" but the **model violating an assumption the chosen
discretization makes about its own solution.**

The user's example is exactly this class: *boundary conditions not consistent with the
assumed solution class.* A pure no-penetration BC `v·n = 0` on a rotationally symmetric
disk leaves **rigid rotation as a null mode** of the viscous operator; a forcing aligned
with it has no solution, and the symptom is a divergence that *grows under mesh refinement*
— the classic signature of an inconsistent (rather than merely inaccurate) problem. Nothing
about the BC is malformed; it is inconsistent with the *solution class* the operator admits.

Framing validation as **"does the model respect the assumptions of its discretization?"**
unifies most of the registry in §6 and tells us where checks belong:

- assumptions that are **structural and symbolic** (a saddle point needs inf-sup-stable
  elements; a moving surface density needs the dilution term) → **build-time** checks;
- assumptions that depend on **data and geometry** (is this forcing in the operator's null
  space? does this prescribed motion conserve volume on *this* shape?) → sometimes
  build-time with the geometry in hand, often only **runtime**;
- assumptions about **regularity and conditioning** (is the Jacobian non-singular here? is
  the reaction too stiff for this `dt`?) → almost always **runtime**.

## 3. The validation pipeline

Validation is a pipeline of layers, run in order, each catching a class the earlier ones
cannot. The guarantee weakens as we move down — and, crucially, **down the list is also the
direction of increasing expressiveness**, which is why the escape hatch forfeits the upper
guarantees and must be served by the lower (runtime) layers.

| Layer | Catches | When | Status |
|---|---|---|---|
| Schema / parser | malformed documents, syntax errors | build | **have** (`formalism/schema`, `expr`) |
| Typed AST + vocabulary | undefined names, arity, type mismatch, reserved-name collisions | build | **have** (`formalism/validator.py` §2.5) |
| Template structural consistency | the VCell-style guarantee, *for templated equations* | build | **partial** — T1–T5 and `cahn_hilliard` are consistent by construction; the mechanics templates T6–T8 do not exist |
| **Dimensional / units** | unit mismatches (a huge class of silent modeling errors) | build | **not built** |
| **Discretization compatibility** | inf-sup, element/BC mismatch, missing conservation term, BC ↔ solution-class consistency | build (+ geometry) | **partial** (2026-10-08) — validator warnings for a moving weak-form density without a dilution term and for a non-inf-sup velocity/pressure pair (§6 rows 1, 6); `NonlinearTermError` for a nonlinear source under backward Euler (row 7); the templates add the dilution term themselves. Element/BC mismatch and BC ↔ solution-class checks are still by hand |
| **Runtime-failure translation** | SNES divergence, singular/ill-conditioned Jacobian, mesh tangling, NaN/Inf, step collapse | solve | **built for the single-mesh paths** (2026-10-08) — `backend/diagnostics.py` (`SolveError`, `preflight_failure_message`, `ts_failure_message`, `linear_step_failure_message`), used by `solver.run` and `reaction_diffusion`; mesh tangling is `MeshQualityError` / `StepTooLarge` in the ALE driver. The multi-compartment and interface-coupled solvers and the CLI's own backward-Euler loop still surface raw PETSc errors |

The first two layers exist and are solid (the `Diagnostic` / `validate_or_raise`
machinery). The three bold rows are the work this note is about; two of them are partly built.

## 4. Build-time validation — extending the existing pass

The foundation is already the right shape: `formalism/validator.py` emits
`Diagnostic(severity, path, message)` objects keyed to a path into the model
(`equations[1].terms`), with `error` / `warning` severities and a `validate_or_raise`
gate. New static checks are **new methods on `_Validator`** producing diagnostics — no new
machinery, just more rules. Two new layers belong here.

**Dimensional / units (highest a-priori value).** Most silent modeling errors are unit
mismatches — a rate constant in the wrong units, a diffusivity that is off by a length²
factor, a flux that does not balance. VCell carries units on every quantity and checks them;
we do not yet. The increment: attach (optional) units to parameters and variables, infer
units bottom-up through the expression AST (the same walk that already does type inference,
§2.5), and emit a diagnostic when an operator combines incompatible units or a term's units
do not match its slot. This is **symbolic and template-agnostic** — it works on escape-hatch
weak forms too, because units compose through `+`, `*`, `grad`, `∫…dx` regardless of what
the term *means*. High value, self-contained, does not need the geometry or a solve.

**Discretization compatibility (the "assumed solution class" checks).** These encode the
structural assumptions of §2. Many are cheap symbolic checks once we know the chosen
elements/scheme:

- a moving-surface density equation **must** carry the `ρ ∇_Γ·v_Γ` dilution term (its
  absence is the single most common subtle bug in this problem class — CLAUDE.md already
  flags it for humans; make it a diagnostic);
- a saddle-point system (incompressible Stokes) **must** use inf-sup-stable elements;
  a like-order velocity/pressure pair is a build-time error, not a runtime surprise;
- a pure no-penetration / all-Neumann problem on a symmetric domain has a **rigid-body null
  space**; warn unless a screening / pin / constraint removes it;
- a prescribed boundary motion into an incompressible bulk must satisfy `∮ w·n = 0` on the
  *current* geometry — checkable once the geometry is in hand;
- a nonlinear (non-affine) `source` is incompatible with the backward-Euler `lhs/rhs`
  split; the model is fine, but the **integrator choice** is not — emit a diagnostic that
  names the fix (method-of-lines, or lag the term) rather than letting `ufl.lhs` throw an
  arity mismatch deep in form compilation.

The last point generalizes: several "errors" are really **model × discretization**
incompatibilities. The diagnostic should name *both* sides and the resolution.

## 5. Runtime diagnostics — translating the unanticipated

For escape-hatch models we *cannot* rule out failure up front, so the second pillar is
**catching solver failures and re-expressing them in model terms.** Today a bad model
surfaces as `petsc4py.PETSc.Error: error code 91`, `SNES_DIVERGED_LINE_SEARCH`, an FFCx
`ArityMismatch`, or a `MeshQualityError` — none of which mention the model. The plan:

1. **Wrap the solve.** A thin layer around the `run` / integrator entry points catches the
   known failure types and maps each to a model-level message with as much localization as
   we can afford:
   - `SNES`/`TS` non-convergence → *"the nonlinear solve failed at t ≈ 0.31; the reaction
     `k·A·B` is likely stiff — try `method_of_lines`, reduce `dt`, or check rate
     constants."* (We have the time, the residual norm history, the SNES reason.)
   - singular / ill-conditioned Jacobian → *"the operator has a null space — a likely
     rigid-body or constant mode is unconstrained; add a screening term, a Dirichlet pin, or
     a constraint."*
   - `MeshQualityError` → already a good model-level message; keep and enrich it with the
     step and the offending cell.
   - `NaN`/`Inf` in the residual → *"a term produced a non-finite value at step k — a
     division by a quantity that reached zero, or `**` of a negative base?"*
2. **Cheap diagnostic probes on failure.** When a solve fails, run inexpensive probes to
   localize before reporting: assemble each tagged term's contribution and report which term
   carries the blow-up; evaluate the residual at the IC to catch an inconsistent start;
   check the Jacobian's smallest singular value / a near-null vector to *name* the
   unconstrained mode. The IR's tagged-term structure (ADR 004) is exactly what makes
   per-term attribution possible. *(Done for the pre-flight: `_localize_nonfinite_term`
   assembles each `DiscreteProblem` term at the IC and names the non-finite one — "the source
   term is non-finite at t=0" rather than "the residual is non-finite" — or "the initial
   condition" if the state itself is bad. Wired as the `localize` hook of the pre-flight; the
   Jacobian-null-vector probe waits for the steady solver where the Jacobian can be singular.)*
3. **Pre-flight checks at `t = 0`.** Before stepping, evaluate the residual and Jacobian
   once and run the geometry-dependent §4 checks (null space, BC-data consistency) that
   could not run without the assembled operator. Catch the inconsistent problem *before* the
   first expensive step, and report it as a model issue. *(Done for the time-dependent
   integrators: a **non-finite residual at the initial condition** — a divide-by-zero, or a
   fractional power / log of a non-positive value at the IC — is caught before any step and
   reported as `SolveError(preflight_failure_message())`. The **null-space** classes (rows
   3/4/5) belong to a **steady / saddle-point** solver: a time-dependent operator is
   mass-regularised — `σM + K` is non-singular even when `K` has a constant or rigid-body null
   space — so those singularities do not arise on the integrator path, and the pre-flight
   null-space check lands when steady solves do.)*

Runtime diagnostics are where the escape hatch is *paid for*: because we let users write
arbitrary physics, the discipline is that when it goes wrong, the failure is **explained in
their terms**, not the solver's.

## 6. A registry of common problems

A structured, growing catalog of known failure modes — the user's suggestion, and the
connective tissue between §4/§5 and reality. Each entry is **both documentation and the
spec for a check**: the *Detection* column says whether it is an a-priori diagnostic or a
runtime signature, which is exactly what an implementer needs. Seeded from the failures
already hit while building the backend (so it is real, not hypothetical); it should grow
every time a new class is diagnosed.

| # | Problem | Assumption violated (solution class) | Detection | Message / fix |
|---|---|---|---|---|
| 1 | Missing surface dilution `ρ ∇_Γ·v_Γ` on a moving membrane | density on a stretching surface must dilute (mass conservation) | **a-priori** — *done* for the escape hatch (a validator **warning**: a time-dependent scalar weak form on a moving subdomain with no `div`/`div_surf`). Templated equations get it automatically (the backend adds it whenever the subdomain moves), so the gap is weak-form-only | "add the dilution term; a moving surface density without it does not conserve mass" |
| 2 | Missing bulk dilution `c ∇·v_carrier` on a compressible carrier | a species on a compressing phase must concentrate | **a-priori** for the FSI species path | bulk analogue of #1 |
| 3 | No-penetration / all-Neumann BC on a symmetric domain | the operator has a rigid-body / constant **null space** | **runtime** (singular Jacobian / near-null vector) → could be **a-priori** with geometry | "unconstrained rigid-body or constant mode — add screening, a pin, or a constraint" |
| 4 | Forcing aligned with the null mode | RHS must be in the range (orthogonal to the null space) | **runtime** (divergence grows under refinement) | "forcing is inconsistent with the constrained problem's null space" |
| 5 | Prescribed boundary motion with `∮ w·n ≠ 0` into an incompressible bulk | incompressibility forces `∮ v·n = 0`; the motion over-determines it | **a-priori with geometry** (`∮ w·n` on the current shape) | "prescribed motion is not volume-consistent on this shape; use a divergence-free field or the force-balance closure" |
| 6 | Like-order velocity/pressure for incompressible Stokes | saddle point needs an **inf-sup-stable** pair | **a-priori** — *done* (a validator **warning**): a scalar weak form with a bare `div(v)` of a vector variable and no `partial_t` is a pressure multiplier; its `space` must be a higher order than the velocity's | "use Taylor–Hood (P2/P1) or another inf-sup-stable pair" |
| 7 | Nonlinear `source` under backward Euler | the `lhs/rhs` split assumes the residual is **affine** in the unknown | **a-priori** — *done* (`NonlinearTermError`): a pure-UFL `check_form_arity` on the composed bilinear form, before FFCx, so no cache-poisoning arity traceback | "this reaction is nonlinear; use `method_of_lines` or lag the term — backward Euler cannot split it" |
| 8 | Stiff reaction with an explicit / lagged treatment | step bounded by the reaction time scale; lag can go negative | **runtime** (step rejection / negative concentration) | "stiff kinetics — use the implicit (method-of-lines) path" |
| 9 | Crank–Nicolson on a stiff nonlinear problem | CN is A- but not **L-stable** → rings / diverges | **runtime** (`DIVERGED_NONLINEAR_SOLVE`) | "use BDF (the L-stable default) for stiff problems" |
| 10 | Prescribed motion that tangles the mesh | node-moving with no remeshing assumes the motion keeps cells valid | **runtime** (`MeshQualityError`, already good) | reduce the step / motion, or enable remeshing |
| 11 | BC built from a *smooth* normal instead of the discrete facet normal | Nitsche consistency assumes `g` and `v·n` use the **same** normal | **runtime** (penalty error grows with β) | "build boundary data from the discrete `FacetNormal`" |
| 12 | Time-dependent algebraic BC with symmetric (row+column) elimination | the Jacobian must stay consistent with the un-lifted residual when `x_bc ≠ g` | **a-priori** (a backend invariant — fixed: zero rows only) | internal; the backend keeps the columns |
| 13 | IC inconsistent with a Dirichlet boundary | a DAE needs a **consistent** initial state | **a-priori** (seed the IC to `g`, already done) | internal; the backend seeds the boundary |
| 14 | Sub-inf-sup-threshold Nitsche penalty `β` | symmetric Nitsche needs `β` above a coercivity threshold | **runtime** (loss of coercivity) → heuristic a-priori bound | raise `β`, or use the penalty-free non-symmetric variant |

Patterns worth seeing across the table: most rows are an **assumption of the
discretization** (null space, inf-sup, affinity, L-stability, regularity); the *Detection*
column splits cleanly into **a-priori** (structural, often template/geometry checks) and
**runtime** (the failure has a recognizable signature). Several rows (3, 5, 8) are
**a-priori-with-geometry** — checkable at the `t = 0` pre-flight (§5.3) but not from the
model alone — which is exactly the boundary between VCell's guarantee and ours.

## 7. A staged plan

Highest value first, each increment self-contained:

1. **The registry itself** (this note) — near-zero cost, immediate value as documentation,
   and the spec for everything below. Keep it growing.
2. **Runtime-failure translation** (§5.1) — wrap the solve, map the handful of known
   failure types to model-level messages. High value for **template authors** hardening new
   physics (and the rare expert-mode user), and it needs no new theory — just catching and
   re-phrasing. **Build this first after the registry.** *(Done — `backend/diagnostics.py`.)*
3. **A-priori discretization checks** (§4, rows 1, 6, 7) — the cheap symbolic ones
   (require-dilution, inf-sup element pair, nonlinear-source-vs-backward-Euler). They turn
   runtime surprises into build-time errors with named fixes. *(**All done.** Row 7 —
   `NonlinearTermError` via a pre-FFCx UFL arity check at backward-Euler lowering; Row 1 — a
   `_Validator` warning for a moving-subdomain weak form without a divergence operator; Row 6 —
   a `_Validator` warning for an equal-/low-order velocity/pressure saddle point.)*
4. **The `t = 0` pre-flight** (§5.3) — assemble once, run the geometry-dependent
   consistency checks before stepping. *(The non-finite-residual-at-IC check is done in the
   time-dependent integrators; the null-space checks (rows 3/4/5) wait for a steady /
   saddle-point solver, where the operator can actually be singular.)*
5. **Dimensional / units** (§4) — the largest single class of silent errors, but the biggest
   build (a units representation + inference). Sequence it when the modeling surface is
   stable enough to be worth annotating.

## 8. Non-goals — the honest boundary

- We will **not** recover VCell's a-priori guarantee for **raw weak-form** authoring — but per
  §1.1 that is an explicit *expert mode*, not the user surface. At the **templated** surface
  (the front door — VCell import, the template library) the guarantee *is* recoverable, and
  that is where we aim for "parses ⇒ well-posed." The escape hatch is the compiler's input
  language, not the modeler's.
- Validation will **not** prove well-posedness, stability, or convergence for an *arbitrary*
  weak form — those are undecidable in general. The discipline that makes this acceptable is
  §1.1: arbitrary weak forms are written by template *authors*, once, with the runtime
  diagnostics (§5) and the registry (§6) as their tools — not by end modelers on every model.
- The cost is borne at **template-authoring** time, not modeling time. A new template must be
  shown well-posed by its author (helped by §4–§6); once solidified (§1.1) it carries that
  guarantee for everyone who uses it.

Related: `docs/modeling/declarative-formalism.md` (§1.5 escape hatch, §2.5 validation),
`docs/decisions/004-discreteproblem-ir.md` (the tagged-term IR that makes per-term runtime
attribution possible), `docs/modeling/approaches.md` (where the discretization assumptions
of each approach live).

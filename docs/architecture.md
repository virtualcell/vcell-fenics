# vcell-fenics — architecture

> **Documentation review (2026-10-08):** See the [documentation map](README.md) and
> [discrepancy register](reviews/2026-10-08-documentation-discrepancies.md) for distinctions
> between historical plans and current implementation. New modeling discussion lives in the
> [cell-mechanics workspace](modeling/cell-mechanics/README.md).

How the code is organised and how a model flows from text to a solve. Diagrams
are [Mermaid](https://mermaid.js.org/) (renders on GitHub and most viewers). For
the *why*, see `docs/decisions/` (ADR 004 = the DiscreteProblem IR, ADR 005 =
typing) and the spec `docs/modeling/declarative-formalism.md`. For a gentler
intro see [`overview.md`](overview.md).

## Two layers

The code splits cleanly in two, and the boundary is load-bearing: **the
`formalism` layer is pure-Python with no FEniCSx dependency** (it only parses and
validates), and **DOLFINx enters only in the `backend` layer**. A different
backend (finite volume, a VCell wrapper) would reuse `formalism` unchanged.

```mermaid
flowchart LR
    subgraph user[" "]
        Y["MathDescription<br/>(YAML / JSON / dataclass)"]
        G["Geometry<br/>(mesh + region map)"]
        S["SolverConfiguration<br/>(dt, t_final, fe_degree)"]
    end

    subgraph formalism["vcell_fenics.formalism — pure Python, no FEniCSx"]
        L["loader / dumper"]
        P["parser → typed AST"]
        V["validator (§1.11)"]
        SC["schema dataclasses"]
        T["templates (T1–T4)"]
    end

    subgraph backend["vcell_fenics.backend — DOLFINx"]
        C["compiler: AST → UFL"]
        A["assemble"]
        IR["DiscreteProblem (IR)"]
        D["lowering → dolfinx LinearProblem"]
        R["run / step loop"]
    end

    Y --> L --> SC
    L --> V
    P --> V
    V --> A
    G --> A
    S --> R
    A --> C
    A --> IR
    C --> IR
    IR --> D --> R
    R --> OUT["solution field(s)"]
```

## The translation pipeline

The heart of the backend is `AST → UFL → DiscreteProblem → lowering → dolfinx`
(ADR 004). The `DiscreteProblem` IR is the FEniCSx-backend-internal representation
of one (possibly coupled) solve — it makes the assembly decisions explicit and
testable, rather than burying them in imperative assembly code.

```mermaid
flowchart TD
    MD["MathDescription + Geometry + SolverConfiguration"]
    MD --> VAL["validate_or_raise (§2.5)<br/>+ geometry cross-check (§1.11.10)"]
    VAL --> RES["resolve equation(s) → subdomain mesh,<br/>function space (scalar, or vector if coupled)"]
    RES --> CMP["compile each slot expression to UFL<br/>(parser → AST → compiler)"]
    CMP --> TERMS["build tagged Terms:<br/>TIME_DERIVATIVE · DIFFUSION · DILUTION · SOURCE"]
    TERMS --> DP["DiscreteProblem(V, terms, scheme, motion)"]
    DP --> LOWER["BackwardEuler.compose:<br/>residual F = mass + dt·(diffusion+dilution) − dt·source<br/>a = ufl.lhs(F), L = ufl.rhs(F)"]
    LOWER --> LP["dolfinx LinearProblem (P1, LU)"]
    LP --> STEP["step loop: (move mesh) → solve → roll u_old"]
```

Why `lhs`/`rhs`: building the residual `F` and letting UFL split it means a
`source` *linear in the unknowns* — including cross-variable coupling — lands in
the implicit bilinear form automatically, with no per-term bookkeeping.

## The DiscreteProblem IR

A `DiscreteProblem` carries a function space, the solver state (`unknown`,
`previous`), a closed enum of **tagged terms** (each a UFL integrand), a **time
scheme**, and optional **prescribed motion**. Lowering happens once at
construction; `step()` advances it.

```mermaid
classDiagram
    class DiscreteProblem {
        V : FunctionSpace
        trial, test, dx
        unknown, previous : Function
        dt : Constant
        terms : tuple~Term~
        scheme : BackwardEuler
        motion_velocity : UflExpr?
        +term_kinds() set~TermKind~
        +step()
        +total_mass() float
    }
    class Term {
        kind : TermKind
        integrand : UflExpr?
    }
    class TermKind {
        TIME_DERIVATIVE
        DIFFUSION
        ADVECTION
        DILUTION
        SOURCE
    }
    class BackwardEuler {
        +compose(problem) (a, L)
    }
    class _MeshMotion {
        +advance()  // move nodes + quality guard
    }
    DiscreteProblem "1" --> "*" Term
    Term --> TermKind
    DiscreteProblem --> BackwardEuler
    DiscreteProblem --> "0..1" _MeshMotion
```

Because the terms are tagged and the assembly assumptions are explicit, a test
can assert *structure* with no solve — e.g. a static membrane has
`{TIME_DERIVATIVE, DIFFUSION}` while a moving one gains `DILUTION`. Correctness is
verified in layers: IR structure → compiler↔UFL equivalence → operator invariants
(`K·1 ≈ 0`, `A·1 = M·1`) → analytical/manufactured solutions.

## Validation flow

Validation runs on the parsed dataclass tree before any FEniCSx work. The
geometry cross-check and BC-expression contents are deferred to the backend
boundary because they need a concrete `Geometry`.

```mermaid
flowchart TD
    MD["MathDescription (dataclasses)"] --> STR["structural checks<br/>reserved-name shadowing · duplicates ·<br/>coverage (one eq per variable) ·<br/>temporality ↔ IC · template conformance · BC consistency"]
    STR --> EXPR["per-expression checks<br/>parse → AST walk:<br/>name resolution · trace direction ·<br/>parameter cycles · IC content ·<br/>operator narrow/smoothness rules"]
    EXPR --> TY["type inference (§1.11.5)<br/>each slot expr produces its declared type"]
    TY --> OUT["Diagnostics (errors + warnings)"]
    OUT -.->|at solve time| GEO["geometry cross-check (§1.11.10)<br/>+ BC-expression resolution"]
```

## Coupled multi-species assembly

Several equations sharing a subdomain become **one solve over a vector space**
(component k ↔ equation k). Each equation's terms use its component trial/test; a
`source` is compiled with every variable bound to its component trial, so
cross-variable coupling becomes off-diagonal blocks the `lhs`/`rhs` split
resolves. This is how the §1.4.5 two-species model runs as a single backward-Euler
step. (Equations spanning *different* subdomains — bulk↔surface `trace` coupling —
are deferred.)

## Module map

| Module | Responsibility |
|---|---|
| `formalism/schema.py` | frozen dataclasses for the MathDescription (the in-memory model) |
| `formalism/loader.py`, `dumper.py` | YAML/JSON ↔ dataclasses, structural well-formedness |
| `formalism/expr.py`, `parser.py` | typed expression AST + hand-written recursive-descent parser |
| `formalism/validator.py` | the §2.5 validation pass (most of §1.11) |
| `formalism/templates.py`, `vocabulary.py` | T1–T4 slot specs; reserved-name / function vocabulary |
| `backend/compiler.py` | expression AST → UFL (`CompileContext` binds names to UFL objects) |
| `backend/discrete.py` | `DiscreteProblem` IR, `BackwardEuler` lowering, `_MeshMotion`, mesh-quality guard |
| `backend/geometry.py` | `Geometry` adapter, name registry, §1.11.10 cross-check, disk builders |
| `backend/assemble.py` | MathDescription + Geometry → `DiscreteProblem` (scalar or coupled) |
| `backend/solver.py` | `SolverConfiguration` + the `run` driver |
| `backend/_typing.py` | the one irreducible `Any` alias (`UflExpr`); real DOLFINx/UFL types used directly elsewhere |

## Key design decisions (ADRs)

- **ADR 004** — the `DiscreteProblem` IR: semi-discrete residual (tagged terms +
  separate time scheme), FEniCSx-backend-internal, UFL as the expression IR, the
  build-once + per-step lifecycle, monolithic-linear coupling.
- **ADR 005** — use the FEniCSx stack's real (`py.typed`) types instead of
  `follow_imports = "skip"`, with `untyped_calls_exclude` for the unannotated-call
  noise.
- **001–003** — Pixi + pyproject, DOLFINx 0.10 pin, MPICH (not OpenMPI).

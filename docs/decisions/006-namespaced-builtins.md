# ADR 006 — Namespaced built-ins: the bare identifier namespace belongs to the user

**Date:** 2026-06-16
**Status:** Accepted (implemented)

## Context

The declarative formalism resolves a bare identifier in an expression through a single flat
**reserved-name set** (`formalism/vocabulary.py`). That set has grown to cover several
genuinely different kinds of name, all flattened together:

- **coordinates / time** — `x`, `t`;
- **calculus operators** — `grad`, `div`, `lapl`, `grad_surf`, `div_surf`, `lapl_beltrami`;
- **tensor algebra & trace** — `inner`, `outer`, `cross`, `trace`;
- **the time-derivative marker** — `partial_t`;
- **standard functions** — `sin`, `cos`, `exp`, …;
- **integration measures** — `dx`, `ds`, `dS`, `dx_Gamma`, …;
- **geometric-quantity helpers** — `n`, `H`, `kappa1`, `kappa2`, `tangent`, `theta`, `phi`, `r`.

Every name in that union is forbidden as a user variable or parameter. The problem is that the
last group — the geometric helpers — are **short, physically natural identifiers that modellers
want for their own quantities**, and they sit in the *same value namespace* as user variables.
Collisions are therefore not occasional; they are structural and will recur indefinitely. The
trigger for this ADR: the Cahn–Hilliard template's order parameter could not be named `phi`,
because `phi` is reserved (the azimuthal angle). `r`, `theta`, `n`, `H` are equally desirable and
equally lost.

The reason the flat set feels necessary is a category error. Two of the groups **do not actually
compete with variable names at all**:

- **Operators and functions** (`grad`, `sin`, `inner`, …) only ever appear in **call position**,
  `name(...)`. They are syntactically distinguishable from a value: a variable is read as
  `name`, an operator only as `name(`. They occupy a *call* namespace, not the *value* namespace.
- **Measures** (`dx`, `ds`) appear only in weak-form residuals (the expert escape hatch), a tiny,
  domain-specific syntactic vocabulary.

Only the **geometric helpers** are bare identifiers competing with user variables in the value
namespace. They — together with `x` and `t` — are the entire collision surface.

How other tools handle this:

- **Virtual Cell** keeps the reserved set tiny: `x, y, z, t`, common functions, and operators
  that are syntactically special (`grad` is output-only). Few bare-name *quantities* ⇒ few
  collisions.
- **UFL / FEniCS** has no reserved *value* names at all: every quantity is a Python object the
  user binds to a name of their choosing; operators are functions (`ufl.grad`).
- **Modelica / SBML** reserve keywords and operators only; built-in quantities are namespaced
  (`time`) or function-form (`Constants.pi`).

The consistent lesson: **operators and functions need no protection in the value namespace; the
hazard is bare-name built-in *values*, and the fix is to not put built-in values in the bare
namespace.**

## Decision

**The bare identifier namespace belongs entirely to the user.** No built-in *quantity* occupies a
bare name. Concretely:

1. **Built-in quantities are namespaced** under two roots, with **explicit, spelled-out member
   names** (not the terse single letters of the old flat set):
   - **`geom.*`** — geometry-derived quantities: `geom.x` (the position vector, indexable
     `geom.x[0]`/`geom.x[1]`/`geom.x[2]`), `geom.normal` (boundary/surface normal),
     `geom.mean_curvature`, `geom.curvature1`/`geom.curvature2` (principal curvatures),
     `geom.tangent`, and the curvilinear coordinates `geom.radius` / `geom.azimuth`.
   - **`sim.*`** — simulation quantities: `sim.t` (time), `sim.dt` (the step), and room for
     future run-level quantities.

   So `x` becomes `geom.x` and `t` becomes `sim.t`; the former terse helpers map to explicit
   names — `n`→`geom.normal`, `H`→`geom.mean_curvature`, `r`→`geom.radius`, `theta`→`geom.azimuth`.
   Even the coordinates and time leave the bare namespace, so a modeller may now use `x`, `t`,
   `phi`, `r`, `theta`, `n`, `H`, `c`, … as their own variable or parameter names without conflict.
   The explicit names also read unambiguously at the call site (`geom.normal`, not `n`) — the terse
   forms were a frequent source of "is this the user's `n` or the normal?" confusion.

2. **Operators and functions stay bare and call-position** — `grad(...)`, `div(...)`,
   `inner(...)`, `trace(...)`, `partial_t(...)`, `sin(...)`, etc. They are reserved as
   *callables* (a modeller cannot define a function, and the names are not redefinable), but they
   do **not** consume value-namespace names: the parser distinguishes `name` (a value) from
   `name(` (a call), and a bare-value use of an operator name is simply a `CompileError`. The
   reserved-callable list is small and standard (the same spirit as VCell reserving `sin()`).

3. **Measures** (`dx`, `ds`, …) remain a small reserved vocabulary usable only inside weak-form
   residuals (the expert surface), not in templated-slot or IC expressions.

The net reserved-as-bare-name set shrinks to essentially the two roots **`geom`** and **`sim`**
(plus the small reserved-callable list, which never collides with values anyway).

## Consequences

**Positive:**

- **Collisions are eliminated by construction.** A user variable can be *any* identifier except
  `geom`/`sim`; the short physical names (`phi`, `r`, `theta`, `n`, `x`, `t`, …) are theirs. The
  `phi` problem disappears, and no future template can collide with a built-in quantity.
- **Built-ins become explicit and discoverable.** `geom.x`, `sim.t`, `geom.normal` read
  unambiguously as "a quantity the system provides", visually separate from model quantities.
  Autocomplete/validation can list `geom.`/`sim.` members.
- **Aligns with the template-surface architecture (ADR-pending §1.1 / PR #31).** End users mostly
  do not author raw expressions — templates do. The people who write `geom.x` are template
  authors, a small expert population for whom a clear, stable namespace convention is a feature,
  not a tax.

**Negative / costs:**

- **Migration is real and touches existing models.** Every current use of a bare built-in must be
  rewritten: `x[0]` → `geom.x[0]`, `cos(2*theta(x))` → `cos(2*geom.azimuth)`, `r_dot * x / r(x)` →
  `r_dot * geom.x / geom.radius`, `n(x)`/`H(x)` → `geom.normal`/`geom.mean_curvature`, etc. The
  conformance fixtures, the surface-diffusion and moving-membrane tests, and the worked-example
  YAML all reference these.
- **The parser gains a qualified-name form.** `geom.x` is a member access (`root . member`), a new
  syntactic construct — a small extension to the tokenizer/grammar and a `MemberAccess` (or
  qualified-`Name`) AST node, resolved by the compiler exactly as the bare names are today.
- **A one-time docs/teaching cost.** The `geom.`/`sim.` convention must be documented and learned;
  expressions are marginally longer.

## Alternatives considered

- **Sigil-prefix the built-ins** (`$phi`, `$grad`). Frees the bare namespace like this proposal,
  but `$` is syntactic noise on *every* built-in use and reads less like math. Namespacing
  (`geom.phi`) carries the same benefit with a more familiar, self-documenting form. Rejected.
- **Force *user* names to a prefix** (e.g. `my.phi`). Burdens the common case — model quantities
  should be the plain, unadorned names, since they are what the model is *about*. Rejected.
- **Keep the flat set and just add names as needed.** This is the status quo; it loses a short
  physical name to users permanently each time a built-in is added, and collisions recur forever.
  Rejected.
- **Context-scoped reservation** (a helper is reserved only in expressions that reference
  geometry). Removes some collisions but is fuzzy and magical — whether `phi` means the user's
  variable or the angle depends on surrounding context. Rejected for predictability.
- **Namespace only the geometric helpers, keep `x`/`t` bare.** A smaller migration, and `x`/`t`
  are the least collision-prone. But the user's quantities still cannot be `x` or `t`
  (concentration `c`, position `x`…), and "everything system-provided is under a root" is a
  cleaner, more teachable rule than "everything except `x` and `t`". Adopted the fuller form.

## Migration (implemented)

The policy was rolled out in one change:

1. Parser: the name token accepts a dotted qualified form (`root.member`, with indexing
   `geom.x[0]`) — `formalism/parser.py`, `_NAME_RE`.
2. Vocabulary/validator: the flat reserved set was replaced with the `geom`/`sim` member tables
   (`GEOMETRY_MEMBERS`, `SIMULATION_MEMBERS`, `QUALIFIED_BUILTINS`, `SCOPED_GEOMETRY_NAMES`) plus
   the reserved-callable list (`RESERVED_CALLABLES`); `RESERVED_NAMES` shrank to the two roots and
   the measures. Bare-name resolution falls through to user variables/parameters
   (`formalism/vocabulary.py`, `formalism/validator.py`).
3. Compiler: `geom.*` / `sim.*` resolve to the UFL objects the bare helpers resolved to before —
   `geom.x`→`SpatialCoordinate`, `geom.radius`/`geom.azimuth` as functions of position,
   `geom.normal`/`geom.mean_curvature` from the bound curvature projection, `sim.t`→the time
   Constant (`backend/compiler.py`, `_resolve_qualified`; assemblers bind `geom.x`/`sim.t`).
4. The worked-example fixtures, tests, and docs were migrated to the namespaced forms.
5. `RESERVED_NAMES` and the formalism reference (`docs/modeling/declarative-formalism.md`) were
   updated.

## Notes

- Prompted by the Cahn–Hilliard template (PR #38), where `phi` collided with the reserved
  azimuthal angle — the concrete instance that showed the collision surface is structural.
- The template-surface direction (PR #31) is what makes the migration cost acceptable: the
  namespaced forms are written mostly by template authors, not end modellers.

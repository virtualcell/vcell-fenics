# ADR 009 — Fold the FSI co-moving species transport into the declarative formalism

**Date:** 2026-07-17
**Status:** Proposed. **Step A implemented** (2026-07-17, the compressible relative-advection dilution fix);
steps B (route FSI through `assemble()`) and C (migrate the MMS to YAML) not yet done.

## Context

FSI (`backend/fsi.py`, the multiphase ALE path — `docs/modeling/multiphase-cytoplasm-ale.md`) carries an
Eulerian **volume species** through the moving cell (`step_two_phase_fsi_with_species` and its reacting /
nonlinear siblings). The transport it solves each backward-Euler step is

```
∂c/∂t|_mesh + (v_carrier − w)·∇c + c ∇·v_carrier = D ∇²c
```

where `w` is the ALE **mesh (frame) velocity** and `v_carrier` is the **physical phase** the species rides
(mixture / network / solvent). The species is *Eulerian* — it has no material points — so it dilutes by the
**carrier's** compression `∇·v_carrier`, while the mesh moves at the *bookkeeping* frame `w`. When
`carrier = frame` (a Lagrangian species riding the mesh) these coincide; the FSI-distinctive case is
**`carrier ≠ frame`**, where the species drifts relative to the mesh *and* dilutes by a divergence that is
not the mesh's.

Two facts surfaced while adding MMS convergence coverage for this operator (`tests/test_fsi_species_mms.py`,
PRs #141 / #142):

1. **The formalism cannot express `carrier ≠ frame` dilution.** `bulk_radv_diff` (declarative-formalism §1.4,
   T1) has a `relative_advection` slot — `assemble.py` adds `dot(drift, ∇u)·w` for it (line ~189) — but its
   dilution is **hard-wired to the mesh velocity**: `motion.effective_dilution_rate() * u * w`
   (`assemble.py:197`), the GCL swept-volume rate `ln(|Kⁿ⁺¹|/|Kⁿ|)/dt → ∇·v_mesh`. The **divergence of the
   relative-advection velocity is dropped**. So a compressible drift dilutes by `∇·v_mesh`, never `∇·v_carrier`.

2. **`fsi.py` reimplements the transport inline.** `_transport_co_moving_species` builds its own UFL weak
   form; it does **not** route through `assemble()`. So the species transport exists in two places (the
   formalism assembler *and* fsi.py), and the FSI MMS had to be a pytest test driving the inline operator —
   an `mms/*.yaml` case would exercise the formalism assembler (already covered by the `bulk_expansion`
   cases), not fsi.py's real code.

There is a clean identity linking the two. `v_carrier = w + v_rel` where `v_rel = v_carrier − w` is exactly
the `relative_advection` velocity, so

```
∇·v_carrier  =  ∇·v_mesh  +  ∇·v_rel
              └── GCL swept-volume ──┘   └─ currently dropped ─┘
```

The formalism already computes the first term (the mesh part, discretely, for exact mass conservation). It
simply **omits the second** — the divergence of the relative-advection velocity. For the incompressible
fluid drift the slot was designed around (`∇·v_rel = 0`) that omission is harmless; for a *compressible*
carrier (FSI's network phase, or any `∇·v_rel ≠ 0` drift) it is a **latent under-dilution**, independent of
FSI. (Concretely, the FSI MMS uses `v_carrier = [0.3x, 0.1y]` (∇· = 0.4) and `w = [0.25x, 0.25y]` (∇· = 0.5):
the dropped `∇·v_rel = −0.1` is exactly the gap between the formalism's 0.5 and the correct 0.4.)

## Proposal

**(1) Extend the relative-advection term to its conservation form** (implemented — step A landed). The
drift's divergence must join the transport so the total dilution becomes `c ∇·v_carrier = c(∇·v_mesh +
∇·v_rel)`. It goes into the **advection** term as `∇·(u·v_rel) = v_rel·∇u + u·(∇·v_rel)`, **not** a separate
`DILUTION` term:

```python
# assemble.py, the relative_advection branch (was: only ufl.dot(drift, ufl.grad(u)) * w)
terms.append(Term(TermKind.ADVECTION, (ufl.dot(drift, ufl.grad(u)) + ufl.div(drift) * u) * w))
```

Why not a `TermKind.DILUTION` term (the ADR's first sketch): the backward-Euler scheme **drops** `DILUTION`
on a moving mesh (`discrete.py`), conserving via the swept-volume time term instead — but the drift's
divergence is *not* the mesh's swept volume and must survive. Keeping it in `ADVECTION` (which BE always
keeps) is correct on static and moving meshes alike; on a moving mesh it rides on top of the mesh's GCL
dilution. This is a **correctness fix** for any compressible relative-advection velocity, not only FSI —
existing incompressible-drift cases are unchanged (`∇·v_rel = 0`). Verified by a discriminating MMS
(`mms/cases/bulk_compressible_advection_{box,disk}.yaml`): order 0.0 → collapse before the fix, ≈2.0 (box) /
≈1.8 (disk) after.

**(2) Route the FSI species transport through `assemble()`.** Replace `_transport_co_moving_species`'s inline
weak form with a formalism-driven assembly: a one-equation `bulk_radv_diff` problem with `motion = w` (frame),
`relative_advection = v_carrier − w`, `diffusion = D`, on the FSI mesh. The reacting / nonlinear siblings map
onto the reaction slot / weak-form escape hatch respectively. FSI then holds **one** transport implementation,
shared with every bulk case.

**(3) The species MMS becomes `mms/cases/*.yaml`.** Once (2) lands, the four `test_fsi_species_mms` variants
are ordinary manufactured-solution cases in the persistent suite (box/disk × steady/decaying), testing the
*shared* assembler — the duplication and the pytest-only status both go away.

## Scope / work breakdown

- **A. Formalism dilution fix** — **DONE** (2026-07-17). The conservation-form advection in `assemble.py`
  (`+ u·∇·v_rel`); a discriminating MMS on a static mesh (`bulk_compressible_advection_{box,disk}`, order
  0.0 → ≈2.0/1.8). Audited all existing `relative_advection` usages — every one is a divergence-free
  (constant) drift, or bypasses `assemble` (`test_backend_slip_moving` builds the term by hand) — so nothing
  regressed. This step stood alone and closed a latent correctness gap independent of FSI.
- **B. FSI transport → `assemble()`**: refactor `_transport_co_moving_species` (and, if desired, the reacting
  variants) onto the formalism path; keep behavior identical (the existing `test_backend_fsi.py` conservation
  / equilibrium tests are the guard).
- **C. Migrate the MMS**: port the four `test_fsi_species_mms` variants to `mms/cases/*.yaml`; retire the
  pytest module. Net: −1 bespoke transport implementation, +4 persistent gate cases.

Steps are independent and land in order A → B → C; A is valuable even if B/C are deferred.

## Non-goals

- **The FSI fluid solve stays out of the formalism.** The coupled Stokes system (momentum + tension + drag +
  pressure / incompressibility, `multiphase.py` / `stokes_hdiv.py`) is a vector momentum balance with a
  constraint — not scalar reaction-advection-diffusion — and has no place in the math-description language.
  It remains solver code, verified by pytest (conservation, Laplace-law equilibria, and the planned
  fixed-domain Stokes-order MMS — "step 1" of the FSI verification plan). Only the **scalar species
  transport** is in scope here.

## Alternatives considered

- **Status quo — keep fsi.py's inline transport + pytest MMS** (what PRs #141/#142 do). Correct and shipped;
  the cost is a duplicated transport operator and MMS coverage that lives outside the persistent suite. Fine
  as long as the two implementations don't drift — but "two implementations of the same operator" is exactly
  the shape of the interface-flux over-count that motivated the MMS work in the first place.
- **Fix only the dilution (A), leave FSI inline.** Closes the correctness gap without the refactor; FSI keeps
  its own copy. A reasonable stopping point if B/C aren't worth the churn while the multiphase API is still
  moving.
- **A dedicated "carrier velocity" concept** (a first-class second velocity on the subdomain, distinct from
  both `motion` and `relative_advection`). More explicit than reusing `relative_advection`, but heavier — a
  new schema field, validator rules, dumper support — for no expressive gain over `v_rel = v_carrier − w`.
  Rejected in favor of reusing the existing slot.

## Open questions

- Does the discrete **mass-conservation** guarantee still hold once dilution is `GCL(mesh) + ∇·v_rel`? The
  mesh part telescopes exactly (conservative time term); the continuous `∇·v_rel` part is O(dt) like any
  advective term — so conservation should match the existing MOL/TS story (near-exact, not machine-exact)
  for a compressible carrier. Worth a mass-drift measurement in step A.
- Is the multiphase API stable enough for B/C now, or should they wait? (The design note is still evolving;
  A does not depend on this.)

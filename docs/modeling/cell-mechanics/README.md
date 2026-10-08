# Cell kinematics and mechanics workspace

**Status:** proposed organization and initial discussion draft, 2026-10-08; reviewed the same day in
PR #213 (corrections applied in the framework document). The physics and representation choices remain
proposals until a worked example is accepted. The repository ownership direction below records the
user's architectural clarification; it is not a new solver capability.

The goal is to identify biological modeling goals and abstractions, derive the mathematical problem
families they require, verify suitable solution methods, and map the proven concepts into VCell and
vcell-fenics. The existing solver is a foundation, not a constraint on how biology must be described.

## Repository ownership and the purpose of this workspace

**Architectural direction, clarified 2026-10-08:** the modeling representation and transformation from
biological models into a declarative math description will likely live in `virtualcell/vcell`,
consistent with the current VCell architecture. The conceptual pipeline is:

```text
Biological modeling representation → declarative mathematical problem → numerical solver
Expected production home: VCell     → shared model/solver contract    → vcell-fenics backend
```

For now, centralize the full development story here: biological goals, candidate representations,
model-to-math transformations, mathematical closure, numerical methods and verification. Keeping these
together lets a worked example test the entire chain before deciding how to integrate it into VCell.
Prototypes of modeling representations and transformations may live here when they help settle a
concrete question. Their location does not decide the eventual production architecture.

A useful prototype should keep the biological representation, transformation and backend distinct,
record assumptions and unsupported cases, and pair an input model with its expected declarative math
and a verification example. Preserve that input/output contract and its tests for eventual integration
into VCell; implementation language and packaging can be decided then. Do not require a production
model representation to depend on DOLFINx, UFL or ALE mesh objects merely because its prototype is here.

This direction distinguishes two transformations: **biological model → mathematical description**,
expected in VCell, and **mathematical description → discrete numerical problem**, performed by the
solver backend. The latter already has its own architecture and verification responsibilities.

**Second clarification (2026-10-08, after the [worked use cases](use-cases.md)):** VCell will supply
geometry, kinematics and membrane velocity laws *and* the high-level description of all modeling,
cell mechanics included, and will own the **math-generation phase** that translates modeling
concepts into **templated equations** whose solutions this repository verifies. The weak-form escape
hatch is for developing numerics here; VCell's external interface should map to well-defined,
somewhat modular templates with verified forms. Realizing new high-level modeling therefore needs a
representation in VCell. Whether equation templates (equation-based modeling) will suffice for the
mechanics, or a physics code will be needed, is not yet known; the preference is equation-based
modeling with particular, well-verified forms. Drivers built here are stepping stones to such
templates, not the deliverable.

## Reading and writing here

Start with the [modeling framework](modeling-framework.md). It is the working discussion document,
containing the initial vocabulary, problem families, verification ladder and unresolved decisions.
Then read the [worked use cases](use-cases.md): three published models (Nickaeen–Novak–Mogilner
2017, Novak–Slepchenko 2014, Nickaeen et al. 2019/2022) mapped onto the framework, the formalism and
the backend, with the missing constructs and the impedance mismatches they expose — the input for
hardening the plan. For limitations in the inherited documentation, see the
[discrepancy register](../../reviews/2026-10-08-documentation-discrepancies.md).

Keep one working document initially. Split it only when a topic has enough reviewed substance to stand
alone, using the following responsibilities:

| Future document, if needed | Owns |
|---|---|
| `kinematics-and-balances.md` | Configurations, velocities, measures, conservation and interface balances |
| `constitutive-and-interface-laws.md` | Rheology, active stresses, friction, permeability, reference/history state |
| `examples-and-verification.md` | Biological examples, closed mathematical problems, expected results and evidence |
| `formalism-mapping.md` | Mapping accepted concepts to existing constructs and proposed extensions |

These are proposed filenames, not existing deliverables. Until a split occurs, link to sections of the
framework. After a split, move the substantive text and leave a pointer; avoid parallel copies.

## Deferred context from earlier modeling discussions

Earlier planning also considered mechanical representations of cytoskeletal networks and molecular
motors; zero-dimensional fiduciaries or domains such as vesicles and possible adhesion sites; and a
deformable atlas or geometric metrics that expose distance/vector fields to the nearest membrane,
cell centroid, or fiduciary point. These ideas remain useful context for the eventual VCell modeling
layer, but they are deliberately deferred from the current physics work. They should not expand the
first continuum model's scope or its acceptance criteria. Revisit them when a concrete biological use
case requires point-domain coupling, motor mechanics, adhesion geometry, or motion-aware spatial
queries.

## Boundaries with existing documentation

- [Declarative formalism](../declarative-formalism.md): existing mathematical language and proposed
  language extensions. This workspace develops the modeling rationale before a language change.
- [Multiphase ALE note](../multiphase-cytoplasm-ale.md): preserve its implementation history and reuse
  its carrier/frame distinction; re-examine its proposed constitutive swaps before generalizing them.
- [Active migration note](../active-protrusion-migration.md): retain the planned actin–myosin model as
  a candidate worked example, clearly separate from passive surface-tension relaxation.
- [Approaches](../approaches.md): numerical interface representations. A/B/C/D and CG/DG choices do
  not define the biological material model.
- [Validation strategy](../validation-and-diagnostics.md): templates and model-level diagnostics;
  [MMS](../../../mms/README.md) and [cross-validation](../../../cross_validation/README.md) own executable
  verification and numerical results.
- [ADRs](../../decisions/): record consequential accepted implementation decisions. Ordinary draft
  discussion belongs here rather than creating an ADR for every open question.

## Review workflow

1. Draft a model family with assumptions, unknowns, closure laws, interface/initial/boundary data,
   observables and a known-answer or manufactured test.
2. Mark each capability as **proposed**, **source-inspected**, **previously reported verified**, or
   **verified in this work**; include the relevant entry point and evidence. Test existence alone does
   not establish a passing result at the current revision.
3. Review the modeling and mathematical closure independently of the author (first review: PR #213).
4. Prototype representations or transformations here when useful, in a scoped change with explicit
   input/output examples and tests. Record what should eventually move into VCell and what remains
   backend-specific.
5. Record accepted modeling decisions with date and PR. Implement and verify production support in
   a separate change before advertising new formalism or solver capabilities.

The repository ownership direction above is recorded from the user. The proposed organization,
physics and representation choices remain review items. A later implementation handoff should use
the reviewed document and acceptance tests, rather than reconstructing intent from chat history.

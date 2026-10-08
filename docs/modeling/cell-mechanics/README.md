# Cell kinematics and mechanics workspace

**Status:** proposed organization and initial discussion draft, 2026-10-08. Review with Claude Code
Fable before treating any new modeling choice as accepted. Implementation is a later, separate step.

The goal is to identify biological modeling goals and abstractions, derive the mathematical problem
families they require, verify suitable solution methods, and map the proven concepts into VCell and
vcell-fenics. The existing solver is a foundation, not a constraint on how biology must be described.

## Reading and writing here

Start with the [modeling framework](modeling-framework.md). It is the working discussion document,
containing the initial vocabulary, problem families, verification ladder and unresolved decisions.
For limitations in the inherited documentation, see the
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
3. Review the modeling and mathematical closure independently (initial reviewer: Claude Code Fable).
4. Record accepted decisions with date and PR. Implement and verify in a separate change before
   advertising new formalism or solver support.

No decisions are accepted by this initial scaffold. Its organization and starting questions are the
first review items. A later implementation handoff (including to Opus) should use the reviewed document
and acceptance tests, rather than reconstructing intent from chat history.

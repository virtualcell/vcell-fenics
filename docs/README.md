# Documentation map

The documentation contains both design history and descriptions of implemented behavior. A design's
acceptance, a backend implementation, and availability through the CLI are separate milestones.

| Read for | Start here |
|---|---|
| Project orientation | [Overview](overview.md) (with the dated capability matrix by entry point) and [architecture](architecture.md) (current module map); both rewritten 2026-10-08 |
| Documentation discrepancies and evidence | [2026-10-08 review](reviews/2026-10-08-documentation-discrepancies.md): ten findings, their evidence, and what PR #213 corrected |
| New cell kinematics and mechanics work | [Cell-mechanics workspace](modeling/cell-mechanics/README.md) |
| Mathematical language and geometry | [Declarative formalism](modeling/declarative-formalism.md) and [geometric formalism](modeling/geometric-formalism.md) |
| Numerical representations | [Approaches](modeling/approaches.md), [surface remap](modeling/conservative-surface-remap.md), [ALE remeshing](modeling/ale-remesh-driver.md) |
| Existing mechanics designs | [Multiphase cytoplasm](modeling/multiphase-cytoplasm-ale.md) and [active migration](modeling/active-protrusion-migration.md) |
| Validation strategy | [Validation and diagnostics](modeling/validation-and-diagnostics.md) |
| Implementation decisions | [ADRs](decisions/) (read amendments and dates as well as status headings) |
| VCell integration and recent progress | [Integration tracker](integration/vcell-solver-integration.md) (its header dates the current status; the opening design sections are from 2026-09-22 and the progress log supersedes them where it says so) |
| Numerical verification and comparisons | [MMS suite](../mms/README.md) and [cross-validation](../cross_validation/README.md) |
| Results interchange | [Bundle decision](decisions/010-results-bundle-vtu-zarr.md) and [JSON schema](results-bundle.schema.json) |
| Original demonstrations | [Notebooks](notebooks/), re-executed 2026-10-08 in the pixi dev env; they show the single-mesh Python API, not the CLI |
| Background research | [May ecosystem snapshot](research/2026-05-21-fenicsx-ecosystem.md) and [June conversation capture](research/2026-06-06-cutcell-fronttracking-chatgpt.md); dated leads, not current dependency guidance |

## Keeping the documents useful

- For this development effort, biological modeling concepts, mathematical mapping and solver examples
  are collected in the cell-mechanics workspace. The expected production home of the modeling
  representation and model-to-math transformation is `virtualcell/vcell`, consistent with the current
  VCell architecture; local prototypes here support that eventual integration.
- Accepted language changes belong in the formalism; consequential implementation decisions belong in ADRs.
- Reproducible numerical results belong with the MMS or cross-validation harness that produced them.
- Keep old rationale in place and link to its replacement. Label historical scope instead of silently
  treating every old statement as a current capability claim.
- For a capability claim, name the entry point, supported problem envelope, and verification evidence.
  Record whether evidence is source inspection, an existing test, or a test actually run at a named revision.

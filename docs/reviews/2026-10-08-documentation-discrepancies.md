# Documentation discrepancies identified on 2026-10-08

**Status:** initial review for independent review by Claude Code Fable; findings remain open unless
explicitly marked otherwise. This PR records discrepancies and adds navigation, rather than rewriting
all historical documents or changing solver behavior.

**Baseline:** `4dab281e1ed3f7c2b2bb60c6c2d1a8ccb251ca17` (local `main`, merge of PR #211).
Evidence below is documentation and source inspection. Test files identify existing checks; **the
numerical tests were not executed for this review**. This is a targeted orientation audit, not a complete
conformance or scientific-correctness audit.

Use the [documentation map](../README.md) for navigation and the
[cell-mechanics workspace](../modeling/cell-mechanics/README.md) for new modeling work.

## D01 — The overview and architecture describe an early subset as the current backend

**Locations:** [overview](../overview.md), “What v1 implements” and “What's deferred”;
[architecture](../architecture.md), “Coupled multi-species assembly” and module map.

They describe zero-Neumann-only boundaries, deferred bulk–surface coupling, no remeshing, and a
largely backward-Euler-only pipeline. Those descriptions are useful history but understate the checkout.
[assemble.py](../../src/vcell_fenics/backend/assemble.py) builds Dirichlet/Neumann/Robin conditions and
transport terms; [reaction_diffusion.py](../../src/vcell_fenics/backend/reaction_diffusion.py) supplies
method-of-lines integration; [ale.py](../../src/vcell_fenics/backend/ale.py) supplies remeshing.
[runner.py](../../src/vcell_fenics/runner.py) routes fixed multi-domain models to
[multi_compartment.py](../../src/vcell_fenics/backend/multi_compartment.py).

**Action:** replace the introductory capability list with a dated, entry-point-specific matrix;
keep the early translation diagram as a labelled single-mesh example. Do not imply that all dedicated
mechanics drivers are reachable through the CLI. Navigation pointers are added in this PR; the rewrite
is still open.

## D02 — The formalism's implementation summary contradicts its newer sections

**Location:** [declarative formalism](../modeling/declarative-formalism.md), §3.6, especially
“Punted in v1.” It lists advection, nonlinear sources, time-dependent BCs, output snapshots, and T4 as
deferred despite later implementations and, in some cases, implemented bullets in the same section.

**Evidence:** [assemble.py](../../src/vcell_fenics/backend/assemble.py), `_resolve_equations` and
`_build_boundary_conditions`; [reaction_diffusion.py](../../src/vcell_fenics/backend/reaction_diffusion.py);
[output_times.py](../../src/vcell_fenics/backend/output_times.py);
[reaction-diffusion tests](../../tests/test_backend_reaction_diffusion.py).
T4 field equations and T5 region equations are described in the formalism's own §1.4.2;
[region-variable tests](../../tests/test_backend_region_variables.py) cover the latter.

**Action:** separate language expressibility from each backend's supported subset. Preserve restrictions:
spatial T4 is a field without transport, not necessarily a well-mixed pool; region variables have
path and connected-region restrictions; the single-mesh affine BE lowering still rejects nonlinear terms.
“Nonlinear backward Euler is impossible” would be incorrect: the dedicated FSI driver has a Newton BE path.

## D03 — T5 has two meanings in the same formalism

**Location:** [declarative formalism](../modeling/declarative-formalism.md), §1.4.2 names T5
`region_ode`, while §1.4.3 names proposed Stokes/Navier–Stokes momentum T5. Other references to
“mechanics templates T5–T7” inherit the ambiguity.

**Evidence:** [templates.py](../../src/vcell_fenics/formalism/templates.py) registers `region_ode` as T5;
the proposed mechanics templates are not registered there.

**Action:** use stable semantic names for proposed mechanics families until numbering is settled; then
update all numeric references together. This review does not rename schema keys or allocate new IDs.

## D04 — Production meshing descriptions still name gmsh

**Locations:** [geometric formalism](../modeling/geometric-formalism.md), realization sections;
[approaches](../modeling/approaches.md), remesher descriptions;
[ALE driver note](../modeling/ale-remesh-driver.md), pseudocode and dependency table.
[ADR 007](../decisions/007-geometry-source-of-truth.md) is an earlier decision; even
[ADR 008](../decisions/008-gmsh-license-isolation.md) retains intermediate migration statements.

**Evidence:** [realize.py](../../src/vcell_fenics/backend/realize.py) and
[region_remesh_netgen.py](../../src/vcell_fenics/core/region_remesh_netgen.py) use Netgen;
[ale.py](../../src/vcell_fenics/backend/ale.py), `_remesh`, calls it. [pyproject.toml](../../pyproject.toml)
places gmsh in the dev feature, and [LICENSING.md](../../LICENSING.md) describes the repository policy.
The former gmsh region mesher lives under [tests/gmsh_meshers](../../tests/gmsh_meshers/).

**Action:** update operational recipes and source paths to Netgen; annotate historical ADRs with dated
implementation follow-ups. Retain the specific `fix_boundary_nodes` limitation rather than promising
all gmsh fast paths. This finding concerns repository dependency policy, not a new legal assessment.

## D05 — Remeshing scope has outgrown the original notes

**Locations:** opening status of [ALE driver note](../modeling/ale-remesh-driver.md) says membrane
only and bulk still a sketch, while its later table says both are built;
[surface-remap note](../modeling/conservative-surface-remap.md) still defers the driver.

**Evidence:** [ale.py](../../src/vcell_fenics/backend/ale.py) dispatches by dimension/codimension;
[remesh_3d.py](../../src/vcell_fenics/backend/remesh_3d.py) handles 3D bulk rebuilds;
[ALE tests](../../tests/test_backend_ale.py) and
[moving-boundary tests](../../tests/test_moving_boundary_run.py) cover relevant paths.
The [integration tracker](../integration/vcell-solver-integration.md), “3D moving boundaries,” records
3D results and limitations.

**Action:** distinguish three capabilities: 2D surface remap, 2D bulk remap, and 3D bulk remeshing with
interpolation plus global mass correction. The 3D path is not a 3D conservative surface-supermesh
implementation. `_remesh` explicitly refuses a moving 3D surface mesh; genuine topology changes and
Approach-A trace physics must not be inferred from bulk remeshing support.

## D06 — The multiphase note calls implemented dynamic FSI work outstanding

**Location:** [multiphase cytoplasm](../modeling/multiphase-cytoplasm-ale.md), opening “not implemented”
status and §10 step 4's “remaining” dynamic coupling. Its own §9 table records dynamic FSI as done.

**Evidence:** [fsi.py](../../src/vcell_fenics/backend/fsi.py) exposes prescribed, single-phase and
two-phase force-balance steps, plus single, reacting and nonlinear species variants.
[FSI tests](../../tests/test_backend_fsi.py) include Laplace equilibrium, ellipse relaxation, frame
selection, drag locking and species invariants. [stokes_hdiv.py](../../src/vcell_fenics/backend/stokes_hdiv.py)
and [its tests](../../tests/test_backend_stokes_hdiv.py) cover a distinct strong-normal-BC flow path.

**Action:** update status by component and driver. Keep open limitations explicit: reference-configuration
and hyperelastic machinery are deferred; the natural-traction two-phase closure does not impose each
phase's closed-cell normal matching. Do not equate existing viscous FSI with a general poroelastic solver.

## D07 — Transport descriptions omit newer carrier/frame semantics

**Location:** [declarative formalism](../modeling/declarative-formalism.md), §3.6's description of
`relative_advection` as only `w_rel·grad(c)` and of dilution as mesh-only.

**Evidence:** [ADR 009](../decisions/009-fsi-species-transport-in-formalism.md) records the compressible
relative-drift correction and the later lab-frame `advection` slot.
[assemble.py](../../src/vcell_fenics/backend/assemble.py) includes both `drift·grad(u)` and `div(drift)*u`
for relative advection, and an integrated-by-parts lab-frame flux using carrier minus mesh velocity.
[Lab-frame tests](../../tests/test_backend_lab_frame_advection.py) distinguish swept from carried species.

**Action:** specify carrier velocity, mesh velocity, concentration measure and boundary flux convention
together. Preserve ADR 009's decision to defer unifying FSI's inline transport with `assemble()`;
conceptual agreement does not mean the discrete time terms or drivers are interchangeable.

## D08 — Diagnostics are described as unbuilt despite implemented checks

**Location:** [validation strategy](../modeling/validation-and-diagnostics.md), opening status and §3
status table, versus its later “Done” annotations.

**Evidence:** [diagnostics.py](../../src/vcell_fenics/backend/diagnostics.py) provides `SolveError`,
`NonlinearTermError`, preflight and solver-failure messages;
[validator.py](../../src/vcell_fenics/formalism/validator.py) contains moving-weak-form dilution and
velocity/pressure-space warnings; [diagnostic tests](../../tests/test_backend_diagnostics.py) exist.

**Action:** reconcile the status table with those implemented checks. Keep units inference and remaining
null-space checks separate. A structural warning is not a proof of conservation or inf-sup stability;
a parsed template still needs admissible coefficients and compatible boundary data.

## D09 — Integration summaries and the original solver contract retain superseded refusals

**Locations:** [integration tracker](../integration/vcell-solver-integration.md), opening “Next: moving
boundaries,” CLI exit-code description, and historical two-compartment descriptions;
[ADR 011](../decisions/011-vcell-solver-contract.md), initial blanket MovingB rejection.

**Evidence:** the tracker's M1–M5, 3M1–3M3 and September 28 entries;
[simtask.py](../../src/vcell_fenics/pyvcell_bridge/simtask.py), moving-task checks;
[runner.py](../../src/vcell_fenics/runner.py), `_run_moving` and `_run_multi_compartment`;
[multi-compartment tests](../../tests/test_multi_compartment.py).

**Action:** update current summaries, retain dated discovery logs, and amend the old contract's refusal
list. Supported moving tasks are scoped: one front, interior species, and geometry/BC restrictions;
this does not make the mechanics/FSI drivers general CLI modes. VCell Java deployment claims remain
reported history here; this audit did not inspect the sibling checkout or a running deployment.

## D10 — Notebook snapshots use old names and capability claims

**Locations:** [notebook 01](../notebooks/01_surface_diffusion.ipynb) uses `theta(x)`;
[notebook 03](../notebooks/03_validation_and_the_ir.ipynb) uses `x/r(x)` and presents weak forms as
unsupported. [ADR 006](../decisions/006-namespaced-builtins.md) establishes `geom.*` and `sim.*`.
[weakform.py](../../src/vcell_fenics/backend/weakform.py) and
[unknown_motion.py](../../src/vcell_fenics/backend/unknown_motion.py) provide dedicated entry points;
that does not mean every weak form can be passed to the generic `assemble()` call shown in notebook 03.

Notebook 02's saved output labels total mass `12.562 -> 12.616` as conserved: approximately 0.43% drift,
not conservation to round-off. This is evidence about its saved snapshot, not a new measurement of the
current conservative time stepper.

**Action:** migrate expressions, make the rejected-example lesson entry-point-specific, rerun notebooks
in the pinned dev environment, and report numerical tolerances explicitly. Do not edit saved results
without executing the revised examples.

## Questions to carry into the new modeling work

These are design questions, not verified defects:

- Is subdomain motion a physical material velocity or geometric boundary motion in each model family?
- Which constitutive models need general tensor state, evolving natural configurations or history
  transport, including through remeshing?
- Which interface conditions are physical assumptions, and which are numerical enforcement choices?
- How should supported model families be exposed without promising arbitrary weak-form well-posedness?

## Follow-up and closure

First review this register and the proposed organization. Then make focused corrections: orientation and
status summaries; transport semantics and template identifiers; meshing/remeshing scope; executable
notebooks. Close each D-number with the correction commit/PR and validation performed. This initial PR
only supplies the register and navigation, so all substantive D01–D10 corrections remain open.

Fable's review should challenge the evidence and scope of each finding, check that historical decisions
are not mistaken for code defects, and assess whether the proposed modeling workspace preserves the
separation between biological assumptions, mathematical problems and numerical methods.

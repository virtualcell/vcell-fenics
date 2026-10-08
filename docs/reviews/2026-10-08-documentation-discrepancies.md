# Documentation discrepancies identified on 2026-10-08

**Status:** reviewed 2026-10-08 (PR #213, second session). Every finding's cited evidence was checked
against the source at the baseline below; all ten hold in substance. Corrections to the register's own
wording are applied in place, additional stale spots the first pass missed are listed under
[Further stale spots](#further-stale-spots-found-in-review), and each finding carries a **Closure** line
recording what this PR corrected and what remains open.

**Baseline:** `4dab281e1ed3f7c2b2bb60c6c2d1a8ccb251ca17` (local `main`, merge of PR #211).
Evidence below is documentation and source inspection. Test files identify existing checks; **the
numerical tests were not executed for this review**. This is a targeted orientation audit, not a complete
conformance or scientific-correctness audit.

Use the [documentation map](../README.md) for navigation and the
[cell-mechanics workspace](../modeling/cell-mechanics/README.md) for new modeling work.

## D01 — The overview and architecture describe an early subset as the current backend

**Locations:** [overview](../overview.md), “What v1 implements” and the “What's deferred” bullet under
“Where to go next” (it is a bullet, not a section); [architecture](../architecture.md), the formalism
box (“templates (T1–T4)”), “Coupled multi-species assembly” and the module map.

They describe zero-Neumann-only boundaries, deferred bulk–surface coupling, no remeshing, the weak-form
escape hatch and unknown motion as deferred, and a backward-Euler / direct-LU-only pipeline. Those
descriptions are useful history but understate the checkout.
[assemble.py](../../src/vcell_fenics/backend/assemble.py) builds Dirichlet/Neumann/Robin conditions and
transport terms; [reaction_diffusion.py](../../src/vcell_fenics/backend/reaction_diffusion.py) supplies
method-of-lines integration; [ale.py](../../src/vcell_fenics/backend/ale.py) supplies remeshing.
[runner.py](../../src/vcell_fenics/runner.py) routes fixed multi-domain models to
[multi_compartment.py](../../src/vcell_fenics/backend/multi_compartment.py).

**Action:** replace the introductory capability list with a dated, entry-point-specific matrix;
keep the early translation diagram as a labelled single-mesh example. Do not imply that all dedicated
mechanics drivers are reachable through the CLI.

**Closure (2026-10-08, PR #213):** both documents rewritten. The overview's “What runs today” matrix
separates the three CLI routes (single mesh, multi-compartment, moving front) from the backend-only
drivers and names the refusals; the architecture's diagrams, pipeline (backward Euler *and* method of
lines), validation flow and module map now cover the whole `src/vcell_fenics` tree. Still true and now
stated: the CLI's moving path is backward-Euler only, and the diagnostics translation covers only the
single-mesh paths.

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
[remesh_3d.py](../../src/vcell_fenics/backend/remesh_3d.py) rebuilds the 3D *mesh* (lattice, signed
distance, SurfaceNets, Netgen fill, exact-volume restore), while the 3D field transfer is
`remap_bulk_function_3d` in [bulk_remap_mesh.py](../../src/vcell_fenics/core/bulk_remap_mesh.py)
(non-matching interpolation plus one global rescale, called from `assemble._remap_scalar`).
[Membrane ALE tests](../../tests/test_backend_ale.py), [bulk ALE tests](../../tests/test_backend_ale_bulk.py),
[3D ALE tests](../../tests/test_backend_ale_3d.py) and
[moving-boundary tests](../../tests/test_moving_boundary_run.py) cover the three paths.
The [integration tracker](../integration/vcell-solver-integration.md), “3D moving boundaries,” records
3D results and limitations.

**Action:** distinguish three capabilities: 2D surface remap, 2D bulk remap, and 3D bulk remeshing with
interpolation plus global mass correction. The 3D path is not a 3D conservative surface-supermesh
implementation. `_remesh` explicitly refuses a moving 3D surface mesh; genuine topology changes and
Approach-A trace physics must not be inferred from bulk remeshing support.

## D06 — The multiphase note calls implemented dynamic FSI work outstanding

**Location:** [multiphase cytoplasm](../modeling/multiphase-cytoplasm-ale.md), opening “not implemented”
status and §10 step 4's “remaining” dynamic coupling. Its own §9 table records dynamic FSI as done.

**Evidence:** [fsi.py](../../src/vcell_fenics/backend/fsi.py) exposes prescribed-motion, single-phase
force-balance and two-phase force-balance steps, plus three *two-phase* species variants (one or more
linear species, linearly reacting species, nonlinear reactions via Newton); there is no single-phase
species step.
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

**Evidence:** the tracker's M1–M5 and 3M1–3M3 status-table rows and its September 28 progress entry
(the progress log itself has no moving-boundary entries);
[simtask.py](../../src/vcell_fenics/pyvcell_bridge/simtask.py), `_check_moving_boundary` (refuses: no
`<Velocity>`, a dimension other than 2 or 3, image subvolumes, more than one front, membrane species,
species outside the moving compartment; warns on a species-dependent velocity);
[runner.py](../../src/vcell_fenics/runner.py), `_run_moving` and `_run_multi_compartment`;
[multi-compartment tests](../../tests/test_multi_compartment.py). The tracker's M1 row also lists “3D”
(lifted by 3M1) and an up-front “Dirichlet on the moving interior” refusal that has no counterpart in
the bridge: boundary conditions on a moving model fail only when a remesh occurs
(`assemble.rebuild_on_mesh` raises `NotImplementedError`), so a run that never remeshes is not refused.

**Action:** update current summaries, retain dated discovery logs, and amend the old contract's refusal
list. Supported moving tasks are scoped: one front, interior species, and geometry restrictions, with
BCs unsupported across a remesh; this does not make the mechanics/FSI drivers general CLI modes. VCell Java deployment claims remain
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

## Further stale spots found in review

Found while verifying D01–D10 against the source (2026-10-08); each is corrected in this PR unless marked
*open*.

- [runner.py](../../src/vcell_fenics/runner.py)'s module docstring says two-compartment models route to
  `integrate_interface_coupled` / `integrate_membrane_coupled`; the code routes every fixed model whose
  equations span two or more subdomains to `integrate_multi_compartment`. The formalism's §1.4.2 T5 and
  Appendix B region-variable notes (“both coupled solvers”) omit that solver too.
- Source docstrings carry the old template set: [templates.py](../../src/vcell_fenics/formalism/templates.py)
  (“T1–T4”, while it registers T1–T5 and `cahn_hilliard`), [schema.py](../../src/vcell_fenics/formalism/schema.py)
  and [weakform.py](../../src/vcell_fenics/backend/weakform.py) (“T5–T7 are v2”, the mechanics meaning of
  the D03 clash). `cahn_hilliard` is a registered template with no T-number and no entry in the formalism's
  template list.
- [ale.py](../../src/vcell_fenics/backend/ale.py)'s docstring says “a moving 2D region” and names the gmsh
  `mesh_region`; `_remesh` handles 3D and calls `mesh_region_netgen`.
  [region_remesh_netgen.py](../../src/vcell_fenics/core/region_remesh_netgen.py)'s `fix_boundary_nodes`
  error tells the user to “use the gmsh mesh_region”, which lives only under `tests/`.
- [pyproject.toml](../../pyproject.toml)'s gmsh comments cite `approaches/*` prototypes; that package no
  longer exists in `src/`.
- [ADR 008](../decisions/008-gmsh-license-isolation.md) §4 still says the `approaches/*/geometry.py`
  prototypes use gmsh and that two meshers coexist “until fully migrated”; §6 gates the ALE remesher's
  switch to Netgen on a pinch check (the switch happened); the 3D path is called “spiked, not yet
  productionized” (it is `remesh_3d.py`, with `PinchOffError` and a fallback ladder of surfaces).
- [approaches.md](../modeling/approaches.md) lists “3D tetrahedra” as deferred for the bulk remap; the
  interpolation-plus-rescale transfer exists (an exact 3D supermesh remains deferred). Its package tree
  shows an `approaches/` layout (ale / submesh / phase_field / cut_fem) that was never built; the
  drivers live in `backend/`.
- The formalism's §3.6 says the backend has “no `t` handle”; `assemble()` binds `sim.t` to a time
  Constant that `solver.run` and the method-of-lines driver advance, and Dirichlet data refresh from it.
- [overview.md](../overview.md) says “direct LU”; the method-of-lines paths default to GMRES + ILU
  ([linear_solvers.py](../../src/vcell_fenics/backend/linear_solvers.py)).
- The runtime-failure translation in [diagnostics.py](../../src/vcell_fenics/backend/diagnostics.py) wraps
  only the single-mesh backward-Euler and method-of-lines paths; the multi-compartment and
  interface-coupled solvers call `ts.solve` untranslated, and the CLI's own backward-Euler loop in
  `runner.py` bypasses `solver.run`'s `SolveError` wrapper. *Open* (code change).
- The CLI's moving path is hard-wired to backward Euler (`runner._run_moving`) although a method-of-lines
  moving stepper exists; that part of the “BE-only” description is still true for the CLI. *Open*.
- [stokes_hdiv.py](../../src/vcell_fenics/backend/stokes_hdiv.py)'s docstring says a manufactured
  divergence-free solution is recovered; its two tests check `div` at round-off on a bulging boundary
  and that the fluid still slips, not a manufactured solution.
- Nonlinear sources under backward Euler are rejected at two sites (`discrete._compose_backward_euler`
  and the membrane-coupled assembler in `interface_coupled.py`), not one; the method of lines accepts them.
- Region-variable connectivity is enforced by the solvers (`connected_region_count`), not the validator,
  which checks only the region/template pairing. Spatial T4 is also refused on a moving subdomain.

## Questions to carry into the new modeling work

These are design questions, not verified defects:

- Is subdomain motion a physical material velocity or geometric boundary motion in each model family?
- Which constitutive models need general tensor state, evolving natural configurations or history
  transport, including through remeshing?
- Which interface conditions are physical assumptions, and which are numerical enforcement choices?
- How should supported model families be exposed without promising arbitrary weak-form well-posedness?

## Follow-up and closure

The register was reviewed and the corrections were made in the same PR (#213), one commit per group of
findings: orientation (D01), the formalism (D02, D03, D07), meshing and remeshing (D04, D05), the
multiphase note (D06), the validation strategy (D08), the integration tracker and solver contract (D09),
the notebooks (D10), and the source docstrings listed above. Each finding's **Closure** line names what
was corrected and what stays open; open items are design or code work, not documentation.

Historical decisions were not treated as defects: ADR 007's gmsh choice and ADR 011's original refusal
list were right when written and are annotated with dated follow-ups rather than rewritten. The
modeling workspace keeps biological assumptions, mathematical problems and numerical methods separate;
the review's corrections to it are recorded in the framework document itself.

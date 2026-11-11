# ADR 014 — Membrane species on a moving front

**Date:** 2026-11-09
**Status:** Accepted (Jim, 2026-11-09): decisions 1 and 3–8 as recommended; decision 2 as **(a)**, the
construct specified below, with the MovingBoundary solver's support tracked as a vcell-mbsolver issue.
Revised after an independent review (9 findings, all addressed below).
**Builds on:** ADR 013 (accepted 2026-11-09): decision 1 (a surface's prescribed motion is its material
velocity, tangential part included), decision 3 (a surface-driven bulk is at rest; transport relative to the
mesh; flux relative to the moving boundary) and decision 5(a) (the same reading in the coupled backend).
**Would amend:**
- formalism §1.4.2 T2 (assembly on a mesh whose tangential motion differs from the material's) and §1.10.1
  (the outside compartment is surface-driven too);
- the multi-compartment solver (`backend/multi_compartment.py`), which gains prescribed motion;
- the runner's moving dispatch (`runner._run_moving`, `_is_multi_compartment`);
- the VCell import's refusals (`pyvcell_bridge/simtask._check_moving_boundary`);
- the membrane-coupled solver's prescribed moving mode (`backend/interface_coupled.py`, decision 7).

Notation: `v` is the membrane's prescribed (material) velocity, `d` a species' own drift (T2's
`relative_advection`), `w_mesh` the mesh velocity. (Formalism T1/T2 call the drift `w`; this note avoids that
letter for it.)

## Context

**What VCell offers.** VCell's moving-boundary math can carry species on the moving membrane, but no
released solver has run them:
- the MovingBoundary solver (mbsolver) has no membrane state; VCell's `MovingBoundaryFileWriter` drops a
  membrane species' equation and every jump condition without a message (vcell-core
  `spec/semantics/motion-and-mechanics.md` §1, checked in VCell's source 2026-11-08);
- VCell's math generator writes a moving membrane's species **without a dilution term** (same note, §2;
  stated there, not re-checked here), and VCell's math has no curvature quantity (same note, §1), so it
  cannot state the dilution of a front that moves along its normal;
- the integrated FEniCSx solver refuses them today (`simtask._check_moving_boundary` 360: any PDE *or* ODE
  on the moving membrane), and so does vcell-core: its vcell-fenics target (class
  `prescribed-motion-membrane-species`, 3 corpus applications, D99 and `tools/oracle/README.md`) and its
  VCell target (`targets/vcell.py` 59–62: "VCell's math cannot carry its dilution").

So there is no legacy result to reproduce, and no reference solver for a membrane carrying state while it
moves. The roadmap records the consequence as Jim's call (vcell-core `ROADMAP.md`, "Dilution in VCell's
moving-boundary math").

**What vcell-fenics has** (*checked in the code, 2026-11-09, on `main` at `f72b1f4`*):
- **T2** (`surface_pde_with_dilution`, formalism §1.4.2) states a membrane species in conservation form with
  the substrate velocity from the subdomain's motion, so the dilution is the template's, not the user's. The
  VCell import writes a membrane **PDE** as T2 and a membrane **ODE** as `lumped_ode` (T4), which has no
  dilution or transport (`importer.py` 298–310).
- **The single-mesh moving path** (ADR 013) solves one subdomain: equations on a moving membrane *and* its
  volume are refused (`runner.py` 370–375).
- **The multi-compartment solver** is the path VCell's static models with membranes take: N compartments and
  their membranes on one parent mesh with submeshes and `EntityMap`s, jump-condition fluxes through a
  side-masked trace on `dS`, membrane species on codim-1 submeshes, method of lines with a matrix-free
  Jacobian when membrane species are present (`multi_compartment.py` 1–36, 349, 632–639). It is "fixed
  geometry only" (36): region sizes are constants fixed at realize time (236–251, 327–338), and one
  `ts.solve` runs from 0 to `t_final` with no outer stride loop (692). Its forms reference the mesh
  coordinates, so they re-evaluate on a moved mesh; its matrices are created once because "their sparsity
  is fixed" (562–565), which holds while the topology does.
- **The membrane-coupled solver** (`interface_coupled.py`) already moves a membrane that carries species, for
  two bulks and one membrane, with prescribed or force-balance velocity. It reads both bulks as
  **co-moving with the mesh**: the backward-Euler path carries the ratio `(u − r uⁿ)` (1511–1527) and the
  method-of-lines path the swept rate `effective_dilution` (1790–1808), with no transport relative to the
  mesh for a bulk at rest (a `relative_advection` drift is assembled as `d·∇c`, 1838–1842); docstring
  308–320: "Both bulk and surface fields co-move with the deforming domain". That is the reading ADR 013
  decision 5 replaced in `coupled.py`; ADR 013 did not list this file.
- **No multi-region remesher exists.** `ale._remesh` remeshes one region: a 2D bulk from its boundary loop, a
  3D bulk (`remesh_region_3d`), or a 2D membrane (`ale.py` 255–300); only the initial realizers build a
  tagged multi-subvolume partition.
- **Tangential node motion** (vcell-fenics #221) slides a membrane-driven volume's boundary nodes along the
  membrane each step, back to their fractions of its length. It works on a single-mesh volume's *exterior*
  loop, in 2D and serial only (`discrete._boundary_loop` 218–219).

## Decisions

1. **A diffusing membrane species on a moving membrane rides the membrane's material** (T2 as written):
   `∂_t ρ + ∇_Γ·(ρ (v + d)) = ∇_Γ·(D ∇_Γ ρ) + s`, with `v` the membrane's prescribed velocity, tangential
   part included (ADR 013 decision 1), and `d` the species' own drift along the membrane, if any. For VCell's
   fronts, which prescribe a full velocity vector, a rigid translation carries a membrane pattern rigidly; a
   front moving along its normal (`v = V_n n`, RacRho) moves no species along the membrane, and they dilute
   at the rate `V_n ∇_Γ·n = (d−1) H V_n` (`n` out of the inside compartment, `H` the mean curvature, positive
   on a circle; formalism §1.8.4).

   *Recommendation: yes.* It is what T2 states and what vcell-core's clean model states (motion note §2:
   "membrane dilution is not optional").

2. **VCell's math for such a species omits the dilution; vcell-fenics dilutes anyway.** The import keeps
   writing VCell's membrane PDE as T2, so the dilution comes from the membrane's motion, not from VCell's
   equation. Options for the legacy side (Jim's call, recorded on vcell-core's roadmap):
   - **(a) VCell's math states it.** Not one more generated term: a normal front's dilution needs the
     membrane's curvature (or a surface divergence of its velocity), which VCell's math has no quantity for.
     So (a) is a new construct in VCell's math, for example a flag on a moving membrane's species saying they
     dilute with it, interpreted by the solvers that move membranes (today only FEniCSx);
   - **(b) VCell unchanged**: vcell-fenics warns where it imports such a model ("membrane species on a moving
     membrane dilute, which VCell's math does not state"), and vcell-core keeps the difference a named class
     in its oracles (its VCell target already refuses it by name);
   - **(c) follow VCell's math literally** (no dilution): rejected, since a uniformly stretching membrane
     would then gain mass.

   *Recommendation was (b) now; **Jim chose (a)**: VCell's math states it, so VCell, vcell-fenics and
   vcell-core agree, and the MovingBoundary solver can gain it (a vcell-mbsolver issue).*

   **The construct (decided with (a)).** A per-membrane boolean, `Dilution`:
   - VCML: the token `Dilution true` inside a `MembraneSubDomain` block; XML: the attribute
     `Dilution="true"` on `<MembraneSubDomain>`. Absent means false.
   - Meaning: every species on that membrane is a density carried by the membrane's velocity `v`
     (`velocityX/Y/Z`): `∂_t ρ + ∇_Γ·(ρ v) = …` for a PDE (the dilution and the tangential carry), and
     `d_t ρ + ρ ∇_Γ·v = rate` along the membrane's material points for an ODE. A membrane PDE's own
     `<Velocity>` is its drift relative to the membrane. It states nothing for a membrane without a velocity.
   - VCell's math generator (`DiffEquMathMapping`) writes it for a membrane that has a velocity *and*
     species, and only then: every other model's math is unchanged, and VCell clients that predate the
     token (the VCML reader rejects unknown tokens; `MembraneSubDomain.FUTURE_VCML_SYMBOLS` lists only
     `Name`) are affected only by models they could not run anyway.
   - vcell-fenics: a moving membrane whose species have `Dilution` imports as T2 (decision 1). One that
     carries species **without** it is math from a VCell that predates the construct, and is refused by
     name ("regenerate the math with a current VCell"); VCell regenerates a simulation's math when it runs
     it, so a current VCell always states it.
   - The MovingBoundary solver has no membrane state at all; supporting membrane species with the dilution
     is an issue on vcell-mbsolver. Until then VCell keeps such a model off that solver.
   - vcell-core captures the flag in its legacy layer and its VCell target writes it, lifting its
     `moving-membrane-dilution` refusal (a follow-up with its own D-row).

3. **The bulk on each side is at rest** (ADR 013 decisions 3 and 5): a compartment with `motion: none`
   beside the moving membrane is surface-driven — the enclosed volume and, unlike ADR 013's single-mesh path,
   the compartment outside. Every T1 species there is transported relative to the mesh,
   `−u (c − w_mesh)·∇q` (carrier `c` = its drift, 0 by default), with the conservative time term, and a
   jump-condition flux at the moving membrane is a total flux relative to the membrane.
   - **The box and every other membrane do not move** (decision 5). So an expanding or translating cell
     compresses the outside compartment between the membrane and the walls, and with no-flux walls its
     species are swept and concentrated there: total conserved, concentration rising as its volume shrinks.

   *Recommendation: yes.* Nothing in the sources contradicts it (Addendum C: for the outer volume "the signs
   and roles swap"); ADR 013 decision 4's "membrane alone" restriction came from the single-mesh remesh, not
   from the physics.

4. **The membrane's mesh may slide along it; its species are transported relative to that slide.** The
   membrane mesh moves with `w_mesh`, whose normal part is the front's and whose tangential part is a mesh
   choice. T2 is then assembled as on any moving mesh, and nothing else:
   - the time term on the moving membrane mesh: the carried facet ratio `|Kⁿ|/|Kⁿ⁺¹|` under backward Euler,
     the swept rate `effective_dilution` under the strided method of lines (decision 5) — either supplies
     `ρ ∇_Γ·w_mesh`;
   - the transport relative to the mesh, `−ρ (v + d − w_mesh)·∇_Γ q`, integrated by parts (a closed membrane
     has no boundary term) — it supplies `∇_Γ·(ρ (v + d − w_mesh))`; under the strided method of lines,
     `w_mesh` is the stride's displacement over its length, a splitting error of one stride;
   - **no separate dilution term**: the two above add up to `∇_Γ·(ρ (v + d))`. The facet ratio alone *is*
     `∇_Γ·v` only when the mesh follows the material (`interface_coupled.py` 1517–1518 assumes so).
   - **The slide itself:** #221's is for a single-mesh volume's exterior loop. Here the membrane is an
     interior facet set of the parent mesh with its own codim-1 submesh, so the slide is applied to the
     membrane's ordered loop and the parent takes it through the harmonic extension. 2D and serial only; in
     3D, `w_mesh = v` on the membrane (no slide).

   *Recommendation: yes.* Without the slide a normal-only front gathers the membrane's nodes where it
   retracts (#221 measured the spacing spread growing from 2.8 to 4.7 on a one-sided front). Option rejected:
   turning the slide off when a membrane carries species, which brings the gathering back for exactly the
   models this ADR enables.

5. **The multi-compartment solver gains prescribed motion of one membrane.**
   - **Mesh motion:** the parent mesh moves by a harmonic extension of the membrane's displacement, with the
     box and every *other* membrane held fixed (a membrane without a velocity is static, as in VCell); the
     compartment and membrane submeshes take the same displacement at their coincident nodes, so the
     topological `EntityMap`s stay valid. This is the scheme of `MembraneCoupledMeshMotion`
     (`interface_coupled.py` 307–407), generalized to N compartments and extended with what VCell's fronts
     need and it lacks: the time bound in the velocity (it compiles it without `sim.t`, 329/367 and 235, so a
     front like `(sin t, cos t)` would see t = 0), `geom.normal` and `trace(·)` of the bulk species
     (ADR 013 decision 2), and decision 4's slide (it moves the membrane by `dt·v`, 362–372, 402–405).
   - **Time:** the method of lines in strides, as the moving runner does: strides of at most `dt`, ending on
     every output time (`runner._run_moving` 379–381, 400–403). Each stride moves the mesh first, with the
     velocity evaluated at the stride's end (as `ale.stride_with_remeshing` does), then integrates the stride
     with adaptive BDF on the moved mesh. The swept-measure rates (`_SweptMeasures`, `interface_coupled.py`
     269–305) supply the GCL-consistent time terms per compartment and membrane.
     *Amended by ADR 015 (2026-11-10):* the swept rates are replaced: each move carries every species' nodal
     moments exactly, and each stride moves half, integrates on the midpoint mesh, and moves the other half.
   - **Refused for now, by name:** non-diffusing species (T4) on the moving membrane or in a compartment the
     membrane drives (as today: "a non-diffusing species … on a moving subdomain is not supported yet");
     `region_size` and region variables on a moving partition (vcell-core's M2; `region_size` lifted
     2026-11-10, re-measured after every move); several moving membranes;
     lab-frame `advection` (as today).

   *Recommendation: yes.* It is the solver VCell's models with membranes already take (jump conditions,
   multi-species membranes, the matrix-free Jacobian), so a moving model reaches it unchanged. Options
   rejected: extending the membrane-coupled solver (two bulks only, not on VCell's path), or the single-mesh
   path (one subdomain by design).

6. **Remeshing comes second.** First stage: no remesh. A run whose mesh degrades past the quality limit, or
   whose moving membrane approaches the box or another membrane, stops with a named error and the time.
   Second stage: a 2D multi-region remesher (the box, the moving loop and any fixed loops meshed together,
   as the realizer's contour mesher does), with each compartment's fields and each membrane's species
   remapped conservatively. 3D after.

   *Recommendation: yes.* Without a remesh, the first stage covers motions small against the gaps between the
   moving membrane and the box or another membrane: the fixed walls make the outside compartment's cells
   compress (decision 3), unlike ADR 013's single mesh, which translated rigidly. The slide (decision 4)
   removes the membrane's own node gathering in 2D; 3D has neither the slide nor a multi-region remesh yet.

   *First stage's stop, built late (2026-11-09).* The first stage shipped without it: the moving partition
   checked nothing, and moving_bleb (BioModel 294599465) at VCell's mesh (h = 0.04) kept integrating on
   inverted cells, losing 2 % of its membrane species by t = 2.6 (an analytic recast of its 35-tooth cell
   tangled the same way at t ≈ 1.4, an 8-tooth one at t = 2). `CellOrientation` (`backend/discrete.py`) now
   checks the moved parent after every stride, by **signed** cell volumes against each cell's orientation
   as built (an assembled `1·dx` is `|det J|` and cannot see an inverted cell), and stops on inversion or
   past the hard growth limit. The single-mesh path (ADR 013) had the same blind spot: the 3D furrow
   inverted a tetrahedron near t = 2 unseen. There the move is now undone and the step retried once on a
   fresh mesh (`MeshTangled`).

7. **The membrane-coupled solver's prescribed moving mode**, which reads `none` bulks as co-moving
   (decision 3 says the opposite):
   - **(a) align it** with decision 3 (transport relative to the mesh, total relative flux);
   - **(b) refuse a `none` bulk beside its prescribed moving membrane**, pointing to the multi-compartment
     solver, and move its tests there;
   - **(c) leave it and document the difference.**

   Its force-balance mode (`ForceBalanceMeshMotion`, 409–623) is not covered here: there the cytoplasm has a
   solved Stokes velocity, so neither "at rest" nor "co-moving with the mesh" is its material velocity.

   *Recommendation: (b)* once decision 5 lands: one implementation of the semantics, and the prescribed mode
   is not on VCell's path. The force-balance mode's bulk transport is recorded as an open question for the
   mechanics track, not changed here.

8. **The VCell import lifts its refusals** of diffusing species on the moving membrane and of species outside
   the moving front (`simtask._check_moving_boundary`), keeping the refusal of non-diffusing ones
   (decision 5); the runner sends a moving model with equations on more than one subdomain to the
   multi-compartment solver. vcell-core's vcell-fenics target lifts `prescribed-motion-membrane-species` and
   `-outside-species` in step (a cross-repo follow-up with its own D-row), keeping a non-diffusing membrane
   species refused: it lowers one as T2 (D99), the import as T4, so the two would disagree.

   *Recommendation: yes.*

## Verification

There is no reference solver for a membrane that carries state while it moves, so the checks are exact
solutions, invariants that discriminate the conventions, and agreement with ADR 013's path:
- **Uniform stretch:** uniform membrane density on an expanding circle and sphere: total conserved to solver
  precision, density `ρ₀ |Γ₀| / |Γ(t)|`. (Catches a missing or doubled dilution.)
- **Relative transport, `w_mesh ≠ v`:** a fixed circle whose membrane flows tangentially, `v = f(θ) t̂` with
  `f` non-uniform: the slide keeps the nodes near their places while the species is advected and diluted at
  `f′(θ)/R` — exact by characteristics. (A rigid translation does not test this: there the slide moves
  nothing and `w_mesh = v`.)
- **Normal-only front:** a non-uniform membrane pattern under `V_n n`: the same solution with the slide on and
  off, to discretization error; no motion along the membrane.
- **Bulk at rest, both sides:** a rigidly rotating circular membrane: the domain does not change, the interior
  mesh rotates and the outside annulus shears, and every bulk species must equal the static run's (a
  co-moving reading on either side would rotate or smear it). Plus an asserted outside case: an expanding
  membrane in a closed box, outside species well mixed, concentration `M / |Ω_out(t)|`.
- **Agreement with ADR 013:** a model with species inside the front only, through the new solver, matches the
  single-mesh moving path and `cross_validation/mb_swept.py`'s numbers.
- **Flux convention:** bulk–membrane binding under an expanding membrane: total (bulk + bound) conserved, and
  the bulk at rest piles up ahead of the membrane by the boundary condition `D ∂u/∂n = v·n u` checked
  quantitatively (no binding) — conservation alone does not distinguish the co-moving reading.
- **The corpus:** the 3 applications with species on a moving membrane import and run (all PDEs in
  regenerated VCell math); vcell-core's bridge oracle compares their math once its target lifts the refusal.
- **Unchanged:** every static multi-compartment test and cross-validation, and ADR 013's moving tests.

## Staging

1. This ADR, accepted.
2. Formalism text: T2 on a sliding mesh (decision 4); §1.10.1 (the outside compartment is surface-driven).
3. The multi-compartment solver's motion without remeshing (decision 5), with decisions 3 and 4's assembly;
   the runner's dispatch and the import's lifted refusals (decision 8); the verification above.
4. Decision 2: the `Dilution` construct in VCell's math (reader, writer, comparison, `DiffEquMathMapping`,
   and an error issue keeping the MovingBoundary solver off such models), a VCell PR; the vcell-mbsolver
   issue (virtualcell/vcell-mbsolver#35). vcell-fenics reads the flag from the SimulationTask's raw XML (as it
   reads the front's velocity) and refuses its absence. pyvcell follows: `libvcell` rebuilt from that VCell
   (so pyvcell's regenerated math states it), and pyvcell's math model (`models_math.MembraneSubDomain`)
   capturing `Dilution` and the membrane velocity on read and write; vcell-fenics then reads them from
   pyvcell. Last, VCell's pinned vcell-fenics image is bumped to the release that runs these models, and
   VCell's FEniCSx refusals of species on and outside a moving membrane are lifted.
5. Decision 7.
6. The 2D multi-region remesher (decision 6).
7. vcell-core: the target's refusals lifted, the bridge oracle over the corpus.

## Not decided here

- Non-diffusing species on a moving membrane or in a membrane-driven compartment (T4 has no dilution or
  transport; pure advection needs stabilization).
- Several moving membranes; a moving membrane approaching or crossing a fixed one, or touching the box; a
  moving membrane meeting another at a triple junction (conflicting mesh-motion data). Each is refused by
  name in the first stage.
- Region variables on a moving partition (vcell-core M2). (`region_size` landed 2026-11-10: re-measured after
  every move.)
- Mean curvature in a front velocity.
- The force-balance mode's bulk transport (decision 7).
- A membrane whose material flows tangentially on its own (cortical flow): T2's `relative_advection` covers a
  prescribed drift; a solved flow is the mechanics track.

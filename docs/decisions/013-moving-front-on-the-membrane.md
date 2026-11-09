# ADR 013 — A moving front is the membrane's motion; the enclosed volume's material stays at rest

**Date:** 2026-11-09
**Status:** Accepted (Jim, 2026-11-09): option B, over a near-term "normal inside a volume frame"
fix; decisions 1–6 as recommended, decision 5 as (a). Revised after an independent review (two
passes: 12 findings, then 6).
**Amends:**
- **formalism:**
  - §1.3.5 and §1.10.1 (the `none` row; geometry on a moving domain);
  - §1.4 T1 (assembly on a surface-driven domain; the "VCell moving boundary" bullet);
  - §1.6.4 (flux at a moving boundary);
  - §1.8.4 (`geom.normal` in a prescribed surface motion);
  - §1.10.6 (moving labelled boundaries);
- **the VCell import** (`pyvcell_bridge/importer.py`, `simtask.py`);
- **the single-mesh moving path** (`assemble.py`, `discrete.py`, `ale.py`, `runner._run_moving`);
- **the coupled backend's reading of `none`** (`coupled.py`).

## Context

VCell's moving-boundary math states a **front velocity on the membrane**
(`MembraneSubDomain/<Velocity>`). Every species is at rest in the lab frame unless it has a velocity
of its own, so the front sweeps them.
- **How often fronts read the normal:** 20 of the 43 VCell front velocities in the public corpus read
  the normal (vcell-core's survey, motion note §1).
- **Blocked today:** vcell-core's bridge oracle refuses 15 applications for this alone (vcell-core
  D100, class `prescribed-motion-normal`).

**Today** (*checked in the code, 2026-11-09; line numbers on `origin/main`*):
- **The import puts the front on the volume.** `importer._front_motion` attaches the front's
  velocity to the **enclosed volume** as its prescribed `motion` (360–405). Every membrane gets
  `MotionNone` (185). The volume's equations take the lab-frame `advection` slot, zero when a species
  has no velocity of its own (`_lab_frame`, 212–222).
- **`normalX()` is never translated,** so validation fails with "unknown function 'normalX'"
  (checked on `biomodel_104122918`).
- **Mesh motion follows only the motion of the subdomain the equations live on**
  (`assemble._motion_velocity`, 504–526). `_MeshMotion` interpolates `dt·v` at the nodes, pins every
  exterior facet to those values, and fills the interior harmonically (`discrete.py` 164–209,
  263–289). So the volume's motion is a frame (only boundary values matter), and today it also moves
  a box wall the volume touches, or the membrane of a region inside it.
- **`geom.normal` is bound only in a mechanics solve** (`compiler._resolve_qualified`, 229–237).
- **The coupled backend reads `none` beside a moving membrane as co-moving.** The bulk gains
  `∇·v_mesh` dilution (`coupled.py` 194–199). Its docstring says a relative-flux correction "would be
  needed only for a lab-fixed bulk field".

The formalism says otherwise (§1.3.5, §1.10.1). A subdomain's `motion` is its substrate velocity at
every point, and the mesh is solver-side (§1.10.7). vcell-core states the same physics with material
velocities per compartment (`motion-and-mechanics.md`, Addendum C, accepted 2026-11-08). The swept
front is the **membrane** moving while the **bulk's material is at rest**.

## Decisions

1. **A surface's prescribed motion moves its boundary, and the domains it bounds follow.**
   - **The formalism text.** A subdomain's own `motion` is its substrate velocity. Its *domain*
     moves whenever a surface bounding it moves. For `none`, the substrate is at rest, but the
     domain changes. Geometric quantities and labelled boundaries then follow the current
     configuration. This amends the "static … initial (and only) configuration" wording of
     §1.3.5, the §1.10.1 `none` row, and §1.10.6's "the subdomain's motion map".
   - **Literal-zero velocities.** A prescribed surface velocity that is a literal zero does not
     count as moving (as `coupled._surface_velocity` already treats it).
   - **Kinematic only, for now.** A prescribed surface motion is the surface's material velocity,
     tangential part included (§1.10.2). VCell's front is kinematic, and the difference matters only
     once there are membrane species (1c), which will state it.

   *Recommendation: yes.*
2. **The front's velocity, including `geom.normal`, is stated on the membrane.**
   - **The import.**
     - The front is attached to the membrane subdomain as `MotionPrescribedVelocity`.
     - `normalX()` is translated to `geom.normal[0]`, and so on.
     - Bulk species are wrapped in `trace(·)`, which the validator requires on a surface
       (`validator.py` 845–853), using the importer's existing helper (741–748).
   - **The normal's meaning.** `geom.normal` there is the membrane's normal pointing out of its
     inside compartment. That VCell's `normalX` points the same way is a **hypothesis**, which the
     importer relies on. It is settled by the mbsolver sign check (staging step 2), which runs before
     the importer emits `geom.normal`. If the sign is opposite, the importer negates the
     translation.
   - **How the single-mesh path evaluates the velocity.** The membrane is the enclosed volume's
     boundary (decision 4), so the backend evaluates the velocity at that boundary's nodes:
     - the normal is the area-weighted outward facet normal of the volume's boundary (`∫_∂Ω n·φ_i ds`,
       normalized at boundary nodes only, so interior nodes are 0 and no NaN arises);
     - `trace(X)` compiles to the bulk field (compiler 288), read at those nodes, with today's
       explicit one-step lag for a species-dependent velocity;
     - the normal field is bound only in the velocity's compile context, recomputed after each
       `advance()`, and rebuilt with a remesh.
   - **Deferred:** mean curvature in a front velocity.

   *Recommendation: yes.*
3. **On a surface-driven domain, every T1 species' ALE convective term uses its carrier velocity
   relative to the mesh velocity, and its flux at the moving boundary is relative to the
   boundary.** The species does not move with the mesh. The mesh nodes move with `w` (inside the
   volume, the harmonic extension of the membrane's motion), and the term `−u (c − w)·∇q`
   compensates exactly for that node motion. With the material at rest (`c` = the species' own
   drift, 0 by default), the solution stays fixed in the lab frame and does not depend on how the
   interior nodes move. vcell-fenics verifies this today: `tests/test_backend_lab_frame_advection.py`
   converges a rotating-mesh run to the static-mesh solution. Only the front's normal motion, which
   the mesh shares, enters the physics. A surface-driven domain is a volume whose
   own `motion` is `none` and whose bounding surface moves.
   - **How the backend knows the adjacency.** A fresh `assemble` reads it from the realized
     geometry (`geometry.boundaries[*].subdomains`). A rebuild after a remesh reads it from the
     stored moving surface (decision 4). On the single-mesh path, only the enclosed (inside) volume
     may carry equations; equations on an outside volume hit decision 4's refusal.
   - **The transport term.** Every `bulk_radv_diff` equation on it is assembled as `advection` is
     today: `−u (c − w)·∇q`, integrated by parts. The carrier `c` is the substrate velocity (0) plus
     the species' `relative_advection` drift, 0 when there is none. An explicit `advection` slot
     there keeps its meaning: `c` is that lab-frame velocity. The two slots stay exclusive.
   - **The time term.** The DILUTION term, and with it the conservative `volume_ratio` time term,
     is added from the mesh motion, as today. It is the mesh's swept-volume (GCL) bookkeeping, not
     the substrate's compression.
   - **Where the rule applies.** Only on a surface-driven domain. Elsewhere `relative_advection`
     keeps its present assembly: `(drift·∇u + u∇·drift)·q`, not integrated by parts, on static
     meshes and on volumes whose own motion moves the mesh (decision 7). This keeps the static MMS
     cases, FSI and the Lagrangian tests unchanged. On a domain whose substrate is at rest, drift
     relative to the substrate *is* the lab-frame velocity, so the lab-frame assembly is the right
     one there.
   - **The boundary condition.** The natural condition is zero total flux relative to the moving
     boundary, the Rankine–Hugoniot condition. A user's Neumann flux on such a boundary is likewise a
     total flux relative to the boundary. §1.6.4 gains both.
   - **The numbers.** For VCell's fronts this is today's swept assembly term for term: the same `w`,
     the same carrier, the same conservative time term. A species with its own velocity gets the same
     equation, with `c` the drift.

   *Recommendation: yes.*
4. **The single-mesh path drives the volume mesh from its bounding membrane, for an enclosed volume
   only.**
   - **Selecting the domain.** The runner distinguishes three moving cases:
     - **a prescribed volume** (decision 6): its own motion moves its mesh, as today;
     - **a moving surface with its equations on the surface:** membrane-only models (T2), as today;
     - **a moving surface driving the volume it encloses,** which carries the equations: the new
       case.

     `_moving_subdomains` keeps prescribed subdomains with a non-zero velocity, as today, and the
     runner chooses among the three by where the equations live.
   - **Its boundary is only the membrane.** The volume's boundary must be that membrane alone: not
     the box, and no other surface, such as a nucleus or other region inside it. Both are
     **refused by name** in `cross_validate`, after realization, where adjacency is known
     (`realize`'s `BoundaryGeometry`). `cross_validate` covers the tests and scripts that call
     `assemble` directly, as well as the runner. The rule applies only when a surface-driven volume
     carries equations, so a static model with a nucleus is not refused. Then every exterior
     facet of the volume mesh is the membrane, as `_HarmonicExtension` already assumes. Today such
     volumes move the box wall or the inner membrane, which is wrong.
   - **Surviving a remesh.** `rebuild_on_mesh` has only `md` and the new mesh, and `Subdomain`
     carries no adjacency. So the resolved moving surface is stored in a new `DiscreteProblem` field,
     set in `_build_problem` and carried by `rebuild_on_mesh`. The rebuild takes the velocity from
     that surface's record; the geometry checks of decisions 4 and 6 already ran and are not
     repeated. A remeshed enclosed
     volume again has only membrane facets, in 2D (one boundary loop) and 3D alike.

   *Recommendation: yes.* Lifting the restriction needs boundary tags carried across a remesh in
   2D and 3D, which is a later step.
5. **The coupled backend adopts the same reading:** a `none` bulk beside a moving membrane has its
   material at rest. Two options:
   - **(a) Implement it in `coupled.py`.** It needs:
     - the frame term `+u w·∇q`;
     - the conservative time term (it now uses the plain term plus `div(dt·w)·u`);
     - the binding flux redefined as a total flux relative to the boundary;
     - `geom.normal` bound on the surface mesh, where `_CoupledMeshMotion` compiles the velocity;
     - and either an implementation or a refusal of a bulk `relative_advection` or `source`, which
       it currently **drops silently** (it reads only `diffusion`, line 185) and should refuse today.
   - **(b) Refuse `none` beside a moving membrane in the coupled backend until (a),** and keep its
     tests by declaring the co-moving bulk they mean.

   The test expectations under (a) must be re-derived, not assumed:
   - total ligand stays conserved;
   - receptor totals are unchanged;
   - `consumption_keeps_ligand_below_reservoir` may not survive. Ligand at rest ahead of an expanding
     membrane piles up (`D ∂L/∂n = v·n L`), so the maximum-principle bound can fail.

   *Recommendation: (a)*, because membrane species on a moving front (1c) need a bulk at rest coupled
   to a moving membrane, which is this. It is also the larger part of this ADR's work. If you'd rather
   land the single-mesh part first, (b) now and (a) with 1c is the smaller step.
6. **A prescribed volume motion with no moving bounding surface keeps today's meaning and
   assembly:** its mesh follows its substrate (`mb_translation`, `mb_expansion`, the lab-frame tests).
   **A prescribed volume bordered by a prescribed membrane is refused for now,** because the two
   normal velocities must agree. This is a geometry cross-check (it needs adjacency), not a
   `MathDescription` validator rule. *Recommendation: yes.*

## Consequences

- **VCell's fronts are stated as VCell states them.** The membrane moves, the bulk is at rest, and
  the normal is defined where it is read. The 15 normal-reading fronts become importable, and the
  integrated FEniCSx solver can run them if they pass its other checks.
- **The swept numbers are unchanged** (decision 3). The code and tests that change with the import:
  - `mb_swept.py`'s `_carried` control writes `advection: front`. It becomes
    `relative_advection: front` (on a domain at rest, drift equal to the front velocity carries the
    species), since both slots together are refused.
  - `tests/test_simtask.py` 339–353 (motion on `cell`, `advection == [0, 0]`) is rewritten for the
    membrane motion.
  - `runner._run_moving`'s "the one moving volume" selection and its `motion_velocity is None` check.
- **The validator** needs no change to accept a `none` volume with equations or
  `relative_advection`. Its weak-form dilution warning (`validator.py` 529), which keys on the
  subdomain's own motion, should also key on a surface-driven domain.
- **vcell-core's translation simplifies** (cross-repo follow-up):
  - the front becomes the membrane's motion;
  - an enclosed volume's material at rest needs no nominal motion;
  - drift goes in `relative_advection`.
- **Membrane species on a moving front (1c)** build on decisions 1, 3 and 5.

## Staging and verification

1. **Formalism text** (the Amends list).
2. **Import:**
   - the front moves onto the membrane;
   - `normalX()` is translated;
   - species are wrapped in `trace`;
   - species velocities go in `relative_advection`.
3. **Single-mesh backend:**
   - motion found through the stored moving surface;
   - the boundary normal bound, refreshed and rebuilt;
   - decision 3's assembly;
   - the refusals of decision 4 and decision 6;
   - the rebuild.
4. **Coupled backend:** decision 5 as chosen.
5. **Verification:**
   - `test_moving_boundary_run.py` reproduces its assertions exactly, and the updated `mb_swept.py`
     (translate, deform, remesh) reproduces its errors. It prints rather than asserts, so the
     comparison is against today's printed numbers, recorded first.
   - **The normal's sign against mbsolver, first, before step 2's import emits `geom.normal`:** one
     normal-reading front cross-validated through `cross_validation/`. It expands, so a wrong sign
     shows as contraction.
   - A normal front, an expanding circle and sphere with `V_n = c` and exact radius.
   - The RacRho front with the true normal.
   - The coupled tests, re-derived.
   - vcell-core's bridge oracle over the 15 fronts.
6. **Docs:**
   - this ADR, accepted;
   - the formalism;
   - the integration tracker's moving-boundary section and progress log;
   - `docs/overview.md`;
   - `CLAUDE.md`'s ADR list.

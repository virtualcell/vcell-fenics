# ADR 013 — The outward normal in a moving front's velocity

**Date:** 2026-11-08
**Status:** Proposed
**Amends:** formalism §1.8.4 (where `geom.normal` is available) and the VCell import of moving
boundaries (`pyvcell_bridge/importer.py`, `_front_motion`)

## Context

VCell's moving-boundary math states a front velocity on the membrane (`MembraneSubDomain/<Velocity>`),
and about a third of the corpus's front velocities read the surface normal, as in
`(a − 1)|a − 1| · normalX`.

vcell-fenics imports a front by attaching its velocity to the **enclosed volume** as a prescribed
`motion`, and giving every species there the lab-frame `advection` slot (`importer.py`, `_front_motion`
and `_lab_frame`). The importer calls that motion "only the frame". The backend keeps only its boundary
values and fills the interior by harmonic extension (`discrete.py`, `_MeshMotion`). Because every
species is lab-frame, the interior values never matter.

**The normal is not available there** (*checked 2026-11-08*):
- `import_model` passes VCell's `normalX()` through untranslated, and validation fails with "unknown
  function 'normalX'" (`biomodel_104122918`, application `moving_boundary`).
- `geom.normal` is bound only where a curvature projection exists: a mechanics solve (`compiler.py`,
  `_resolve_qualified`; `unknown_motion.py`).
- VCell's integrated FEniCSx solver does not refuse such fronts up front (vcell
  `FenicsSolver.unsupportedReasons`, `simtask._check_moving_boundary`). A VCell user gets the
  validation error.
- vcell-core's bridge oracle counts 15 corpus fronts that it refuses for this alone, after every
  other check (vcell-core D100).

The well-founded fix is to make the front the **membrane's** motion. The enclosed volume would then be
a domain that follows its boundary while its material stays at rest, as vcell-core states it
(`motion-and-mechanics.md`, Addendum C). An independent review (2026-11-08) showed that this is a
substantial backend change, not a relabelling:
- a volume with `motion: none` gets no mesh motion, no dilution and no conservative time term;
- the coupled backend reads `none` beside a moving membrane as co-moving;
- `relative_advection` would have to be reassembled relative to the mesh, which changes its natural
  boundary condition;
- the single-mesh moving path has no membrane mesh to evaluate a normal or a trace on;
- remeshing rebuilds the motion from the volume;
- the formalism has no boundary-condition convention for flux relative to a moving boundary.

That work belongs with **membrane species on a moving front**, which needs a membrane mesh in the
moving path anyway. See "Deferred" below.

## Decision (proposed)

1. **`geom.normal` in a volume's prescribed motion** is the outward unit normal of that volume's
   boundary, defined at its boundary nodes only. Such a motion is a frame: only its boundary values
   move the mesh, and the backend extends them inside (§1.10.7).
   - §1.8.4 gains a "prescribed volume motion" row.
   - §1.10.1 says that a volume's prescribed motion which reads `geom.normal` states its boundary's
     motion only; otherwise a prescribed volume motion is its substrate velocity, as today. This dual
     reading is what the deferred design removes.

   The validator enforces four rules:
   - every `bulk_radv_diff` equation on that volume uses the lab-frame `advection` slot, so that no
     species rides a velocity that has no meaning inside the volume;
   - a `weak_form` equation on that volume is refused (`lumped_ode` is refused there by the backend
     already);
   - `geom.normal` is refused anywhere else on a volume, in a term, an initial condition or a
     parameter it reaches;
   - the check is transitive through parameters, so a motion that reads a parameter whose body reads
     `geom.normal` counts.
2. **The backend binds it** as a nodal outward normal on the volume mesh: per boundary node, the
   area-weighted average of its facets' outward normals (`∫_∂Ω FacetNormal·φ_i ds`, ghost-scattered,
   then normalized).
   - It is outward by construction: `FacetNormal` on the volume's own exterior facets.
   - It works the same in 2D and 3D.
   - Interior nodes get 0. The velocity there stays finite and is overwritten by the harmonic
     extension.
   - It is bound **only** in the motion velocity's compile context (`assemble._motion_velocity`'s
     `velocity_ctx`), never in the shared one, so no other term can see it.
   - It is recomputed at the start of each `_MeshMotion.advance()`, before the velocity is
     interpolated. A remesh rebuilds it with the problem (`rebuild_on_mesh` → `_motion_velocity`).
   - **A normal-reading front on a volume that touches the box is refused by name.** The harmonic
     extension moves every exterior node by `dt·v`, so a box-face node would move along the face's
     normal and push the wall. Today's normal-free velocities only translate the wall, and they keep
     doing so.
3. **The VCell import translates `normalX()` / `normalY()` / `normalZ()` to `geom.normal[0..2]`** in the
   front velocity.
4. **Nothing else changes:**
   - the front stays the enclosed volume's frame;
   - species stay lab-frame;
   - a species-dependent velocity keeps its explicit one-step lag (tracker, M1);
   - `geom.mean_curvature` stays unavailable in a prescribed motion.

## Consequences

- **The 15 normal-reading corpus fronts validate.** They run once every other check passes; "validate"
  is not "run".
- **Fronts that do not read the normal assemble exactly as before** (the normal is bound only when the
  motion reads it, and UFL drops unused coefficients), so the mbsolver cross-validation is untouched. A new test pins the normal path: an expanding circle `V_n = c` with exact
  `r = r₀ + c t`, in 2D and in 3D (a sphere).
- **This normal is the boundary's facet normal on a volume mesh.** It is distinct from the membrane-loop
  normal of the curvature projection (formalism §1.8.4, the 2026-11-08 conventions, PR
  `fix/curvature-convention`). Both are outward.
- **vcell-core (cross-repo):** its vcell-fenics target lifts the `prescribed-motion-normal` refusal and
  translates `normal_x()` to `geom.normal[0]`.

## Deferred: the front as the membrane's motion

This belongs with membrane species on a moving front, and its own ADR. The review's findings are its
work list:
- **formalism:**
  - §1.10.1's `none` row, and §1.3.5, §1.8.4, §1.10.6, which key the current configuration on `motion`;
  - a convention for flux relative to a moving boundary, beyond §1.6.4's diffusive no-flux;
- **backend:**
  - a moving domain with its substrate at rest: mesh motion and the conservative time term follow the
    mesh, not the substrate;
  - `relative_advection` assembled relative to the mesh and integrated by parts (amending ADR 009's
    addendum and the §1.4 row);
  - the coupled backend's co-moving reading of `none`;
  - a membrane mesh in the moving path, with its entity map, a nodal normal, traces, a box-touching
    volume, and remeshing;
  - what drives the mesh when a prescribed volume borders a prescribed membrane;
- **the cross-validation's carried control.**

## Alternatives

- **The full membrane-motion change now.** It is right in the long run, but too large for what it
  unblocks today (see Deferred).
- **Approximating the normal in the importer**, for example radially, as the RacRho cross-validation
  does. Not general.

## Staging and verification

1. Formalism §1.8.4 and §1.10.1, and the validator rules of decision 1.
2. The nodal boundary normal, bound in the motion's compile context only; refreshed per step, rebuilt per remesh; the box-touching refusal.
3. The importer translation.
4. Tests: the expanding circle and sphere, a normal front that remeshes, the validator rules (including the transitive case), and the box-touching refusal.
5. Docs:
   - this ADR, accepted;
   - formalism §1.8.4 and the T1 VCell bullet;
   - the integration tracker's moving-boundary section and progress log;
   - `docs/overview.md`.

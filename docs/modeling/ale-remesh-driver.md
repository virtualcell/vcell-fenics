# ALE remesh driver (design sketch)

**Status: implemented for the moving *membrane* (1D); the bulk Approach-A path is still a
sketch.** As of 2026-06-11 the driver loop exists as `backend/ale.py` —
`ALEState`, `step_with_remeshing`, `run_with_remeshing` — and runs the remesh-and-continue
loop for a moving 1D membrane: it remeshes the deformed membrane as the boundary of a
freshly meshed region and carries ρ across via the conservative surface remap. What is
still a sketch is the genuine Approach-A case where ρ is a bulk *trace* on a 2D mesh whose
interior nodes move by harmonic extension (needs bulk mesh-motion, not built, and would
route the transfer through `correct_surface_trace`). The pseudocode below is the original
design shape; the [Dependencies](#dependencies-what-this-rides-on) table and the closing
note record what is built vs. outstanding. Treat the pseudocode as shape, not API — the
real signatures are `step_with_remeshing(state, *, quality_limit, target_h)` etc.

## Where it sits, and what changes vs. today

The v1 backend (`backend/discrete.py`) advances a membrane by moving nodes `dt·v` each
step (`_MeshMotion.advance()`) and **raises `MeshQualityError`** once the cell-size ratio
blows past a hard limit — it moves nodes but never remeshes. The driver inserts a layer
above the per-step solve: monitor mesh quality and, *before* a step would tangle, swap to
a fresh mesh and re-interpolate all state. Everything else is unchanged — the
backward-Euler residual and the `ρ ∇_Γ · v_Γ` dilution term are orthogonal to remeshing.

So the driver turns "fail loudly on tangling" into "remesh at the step boundary and
continue."

## The state object and the step loop

```python
@dataclass
class ALEState:
    bulk_mesh                 # the deforming 2D mesh; its boundary facets are Γ(t)
    V_bulk                    # bulk function space(s); ρ is the boundary trace of u
    u, u_prev                 # current and previous-step solution (uⁿ⁺¹, uⁿ)
    velocity                  # mesh velocity field (prescribed, or from mechanics)
    problem                   # assembled DiscreteProblem bound to THIS mesh
    t

def step_with_remeshing(state, dt, *, quality_limit, target_h):
    # Predictive: would advancing this mesh by dt·v degrade it past the limit?
    if tentative_motion_degrades(state, dt, quality_limit):
        state = remesh(state, target_h)          # swap at tⁿ, on the still-valid geometry
        if tentative_motion_degrades(state, dt, quality_limit):
            raise StepTooLarge(...)              # fresh mesh still tangles ⇒ sub-step dt
    advance_mesh(state, dt)                       # commit the node displacement
    state.problem.step()                         # one backward-Euler solve; updates u_prev
    state.t += dt
    return state
```

The remesh fires **at the step boundary `tⁿ`, on the last-known-good geometry** — a clean
fixed-time swap. The fresh mesh then takes the `dt` step safely. (A reactive alternative —
attempt the step, catch `MeshQualityError`, roll back the node coordinates, remesh, retry
— also works; the predictive form avoids the rollback.)

## The remesh routine — where the trace correction plugs in

```python
def remesh(state, target_h):
    # a. recover the current (deformed) boundary loop from the bulk mesh
    boundary = BulkBoundaryTrace(state.V_bulk).boundary_loop()   # reuse the built extractor
    guard_against_self_intersection(boundary)     # pinch-off ⇒ Approach A genuinely breaks → fail

    # b. generate a fresh bulk mesh of the region the loop encloses (gmsh)
    new = build_state_on(mesh_region(boundary, target_h), like=state)

    # c. transfer BULK fields conservatively (∫_Ω c dx preserved)
    transfer_bulk(state.u,      new.u)            # bulk supermesh / interp + mass correction
    transfer_bulk(state.u_prev, new.u_prev)       # u_prev too, or backward Euler is inconsistent

    # d. correct the SURFACE trace so ∫_Γ ρ ds is preserved (bulk transfer does NOT)
    correct_surface_trace(state.u,      new.u)    # <-- the built piece
    correct_surface_trace(state.u_prev, new.u_prev)

    # e. rebuild the assembled problem on the new mesh, re-derive the velocity field
    new.problem  = assemble_on(new)               # forms/matrices were bound to the old mesh
    new.velocity = rebuild_velocity(new)
    new.t = state.t
    assert_conservation(state, new)               # surface AND bulk mass invariant across the swap
    return new
```

## The subtleties that bite

1. **`u_prev` must be transferred and corrected too.** Backward Euler's residual references
   `uⁿ` on the *new* mesh. Forgetting it silently corrupts the first post-remesh step —
   easy to miss because `u` looks right.
2. **The `DiscreteProblem` must be rebuilt, not mutated.** Its `LinearProblem`, forms, and
   matrices are bound to the old mesh (ADR 004's build-once / mutable-handles lifecycle
   assumes a fixed mesh). Remesh = teardown + reassemble. This is the biggest structural
   change the driver forces on the existing backend.
3. **Interior-only fast path.** If only the *interior* degraded (the harmonic-extension
   fill) and Γ's nodes are still fine, regenerate the interior with gmsh holding the
   boundary points fixed → Γ_new ⊂ Γ_old exactly → **the trace is unchanged and
   `correct_surface_trace` is skipped**. For a Lagrangian membrane the boundary usually
   *does* degrade under non-uniform stretch, so the full path (with the trace correction)
   is the common case — but the optimization is real and worth the branch.
4. **Dilution never double-counts.** Motion + `ρ ∇_Γ · v_Γ` live in the per-step residual;
   the remap is a pure geometry swap with no dilution. The two are cleanly separated by
   construction (same point as the remap note: the swap conserves *mass*, not concentration).
5. **Remesh-still-tangles ⇒ sub-step.** If even a fresh mesh can't survive one `dt` (steep
   velocity, big `dt`), remeshing won't save it; reduce `dt` or sub-step. Don't loop
   remeshing.

## Dependencies: what this rides on

| Needed | Status |
|---|---|
| `correct_surface_trace` (boundary post-pass) | **built** — `core/surface_remap_trace.py` |
| `BulkBoundaryTrace` boundary extraction | **built** — `boundary_loop()` accessor added |
| **Bulk-conservative interpolation** `transfer_bulk` | **built** — `core/bulk_remap.py` (supermesh kernel) + `core/bulk_remap_mesh.py` (`remap_bulk_function`, the DOLFINx-Function bridge with optional global mass correction) |
| **Region remesher** `mesh_region(loop, h)` | **built** — `core/region_remesh.py`; meshes an arbitrary deformed polyline, with `fix_boundary_nodes` for the interior-only fast path |
| **`DiscreteProblem` rebuild path** | **built** — `backend.rebuild_on_mesh(problem, md, new_mesh)`; teardown + reassemble on the new mesh with conservative transfer of **both** `unknown` and `previous` (scalar / vector, bulk / surface by tdim). This is `assemble_on(new)` + steps (c)+(d) of `remesh()` fused into one call. |
| **Approach-A bulk mesh-motion** (harmonic-extension displacement PDE writing `geometry.x`) | not built — current motion is on the membrane submesh, not a bulk mesh |
| **The driver loop** `step_with_remeshing` / `run_with_remeshing` | **built (membrane)** — `backend/ale.py`; the 1D-membrane case (remesh = `mesh_region` boundary; transfer = `rebuild_on_mesh`). The bulk Approach-A wiring waits on bulk mesh-motion. |

As of 2026-06-11 the three `core/` field-transfer + meshing prerequisites and the
`DiscreteProblem` rebuild path are all built — the surface-trace correction (step d), the
bulk-conservative interpolation `transfer_bulk` (step c; `remap_bulk_function`, the FEM
form of the 2D overlap remap the `../vcell-mbsolver` baseline does on Voronoi cells), the
region remesher `mesh_region` (step b), and `backend.rebuild_on_mesh` (steps c+d fused:
reassemble on the new mesh + conservatively transfer `unknown` and `previous`; subtleties
1 and 2). `rebuild_on_mesh` currently calls the `core` remaps *directly* on the subdomain
field rather than going through the bulk-remap-then-`correct_surface_trace` two-step — that
two-step is the Approach-A path where ρ is a bulk *trace*; the v1 backend's fields are the
subdomain's own DOFs (Approach-B-flavoured), so the direct remap is correct and conservative
for it. The driver loop itself is now built (`backend/ale.py`) for the moving membrane — it polls
`DiscreteProblem.mesh_quality_growth()` and, once distortion crosses `quality_limit`,
remeshes the deformed membrane (order its loop → `mesh_region` → extract the boundary
submesh) and `rebuild_on_mesh`es onto it; a step that tangles even a fresh mesh raises
`StepTooLarge` (subtlety 5). The remesh trigger is *proactive on accumulated distortion*
rather than the tentative look-ahead the pseudocode shows — simpler, and equivalent given
the margin below the hard limit. The single **remaining gap** is **Approach-A bulk
mesh-motion**: a harmonic-extension displacement PDE that moves *interior* nodes with the
boundary (today's `_MeshMotion` moves a membrane submesh only). With it, ρ becomes a bulk
trace, `_remesh_membrane`'s 1D guard generalises to the bulk region, and the transfer in
`rebuild_on_mesh` routes through `correct_surface_trace` (built, waiting) — at which point
the same driver loop runs the full Approach-A case.

## Verification plan (when built)

Same analytical-check + negative-control pattern the project uses elsewhere:

1. **Remesh is transparent** — a problem that does *not* need remeshing but is *forced* to
   remesh every step matches a never-remesh reference to remap accuracy.
2. **Mass invariant across the event** — force one remesh mid-run; assert ∫_Γ ρ ds and
   ∫_Ω c dx are unchanged across it (round-off), and the post-remesh trajectory tracks the
   reference.
3. **Tangling motion now completes** — a prescribed motion the v1 backend rejects with
   `MeshQualityError` runs to `t_final`.
4. **`u_prev` control** — a variant that skips correcting `u_prev` visibly drifts on the
   first post-remesh step, proving step (d)-on-`u_prev` is load-bearing.

## References

- `docs/modeling/conservative-surface-remap.md` — the surface-trace correction this driver
  calls (step d), and the bulk-vs-surface conservation argument.
- `docs/modeling/approaches.md` — Approach A (ALE explicit membrane) and its weaknesses;
  the `core/` architecture; the FV / front-tracking comparison baseline.
- `docs/decisions/004-discreteproblem-ir.md` — the build-once IR lifecycle the rebuild path
  (subtlety 2) must change.

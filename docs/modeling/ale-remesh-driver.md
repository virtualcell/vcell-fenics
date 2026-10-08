# ALE remesh driver (design sketch)

**Status (2026-10-08): built for three cases.** The driver loop is `backend/ale.py` —
`ALEState`, `step_with_remeshing`, `run_with_remeshing`, `run_moving_with_remeshing` — and
`_remesh` dispatches on dimension and codimension:

- a moving **2D membrane** (codim 1, since 2026-06-11): the deformed membrane is remeshed as
  the boundary of a freshly Netgen-meshed region and ρ carried across by the conservative
  surface remap;
- a moving **2D bulk** region (codim 0, since 2026-06-11): interior nodes move by harmonic
  extension (`_HarmonicExtension`); the deformed boundary loop is meshed directly and the
  field transferred by the conservative 2D supermesh remap;
- a moving **3D bulk** region (since 2026-09-24, `backend/remesh_3d.py`): the deformed region
  is rebuilt implicitly (signed distance on a lattice at h/2 → SurfaceNets → Netgen volume
  fill → projection and exact-volume restore, with a fallback ladder of surfaces and a
  `PinchOffError` for unresolvable necks) and the field transferred by non-matching
  interpolation plus one global mass correction — globally, not locally, conservative; a 3D
  supermesh is deferred.

Not built: remeshing a moving 3D *surface* mesh (`_remesh` raises `NotImplementedError`),
boundary conditions across a remesh (`rebuild_on_mesh` refuses them), and the genuine
Approach-A case where ρ is a bulk *trace* whose transfer would route through
`correct_surface_trace` (built, waiting). The CLI drives the 2D and 3D bulk cases for VCell
moving-boundary models (the integration tracker, "Moving boundaries"). The pseudocode below
is the original design shape; the [Dependencies](#dependencies-what-this-rides-on) table and
the closing note record what is built. Treat the pseudocode as shape, not API — the real
signatures are `step_with_remeshing(state, *, quality_limit, target_h)` etc.

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

    # b. generate a fresh bulk mesh of the region the loop encloses (Netgen; remesh_3d in 3D)
    new = build_state_on(mesh_region_netgen(boundary, target_h), like=state)

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
   fill) and Γ's nodes are still fine, regenerate the interior holding the
   boundary points fixed → Γ_new ⊂ Γ_old exactly → **the trace is unchanged and
   `correct_surface_trace` is skipped**. For a Lagrangian membrane the boundary usually
   *does* degrade under non-uniform stretch, so the full path (with the trace correction)
   is the common case — but the optimization is real and worth the branch. *Not available
   with the production mesher:* Netgen's high-level 2D mesher always resamples the boundary
   at `h`, so `mesh_region_netgen(fix_boundary_nodes=True)` raises (ADR 008 §5); only the
   test-only gmsh `mesh_region` has it.
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
| **Region remesher** `mesh_region_netgen(loop, h)` | **built** — `core/region_remesh_netgen.py` (LGPL Netgen, ADR 008); meshes an arbitrary deformed polyline. The `fix_boundary_nodes` interior-only fast path is *not* available (Netgen resamples the boundary); the gmsh `mesh_region` that has it is test-only (`tests/gmsh_meshers/`) |
| **3D region remesher** `remesh_region_3d` | **built** (2026-09-24) — `backend/remesh_3d.py`: implicit rebuild of the deformed tetrahedral region (lattice signed distance → SurfaceNets → Netgen fill → projection, exact-volume restore, fallback surfaces, `PinchOffError`) |
| **3D bulk transfer** `remap_bulk_function_3d` | **built** — `core/bulk_remap_mesh.py`: non-matching interpolation + one global mass rescale (globally conservative; a 3D supermesh is deferred) |
| **`DiscreteProblem` rebuild path** | **built** — `backend.rebuild_on_mesh(problem, md, new_mesh)`; teardown + reassemble on the new mesh with conservative transfer of **both** `unknown` and `previous` (scalar / vector, bulk / surface by tdim). This is `assemble_on(new)` + steps (c)+(d) of `remesh()` fused into one call. |
| **Approach-A bulk mesh-motion** (harmonic-extension displacement PDE writing `geometry.x`) | **built** — `_HarmonicExtension` in `backend/discrete.py`; `_MeshMotion` selects it for a codim-0 (bulk) mesh: the prescribed velocity sets the boundary displacement, ∇²d=0 fills the interior, `geometry.x += d`. A codim-1 membrane keeps the direct-interpolation move. |
| **The driver loop** `step_with_remeshing` / `run_with_remeshing` / `run_moving_with_remeshing` | **built (2D membrane + 2D bulk + 3D bulk)** — `backend/ale.py`; `_remesh` dispatches on dimension and codimension. Codim-1 membrane: remesh = `mesh_region_netgen` boundary. Codim-0 2D bulk: `BulkBoundaryTrace.boundary_loop()` → `mesh_region_netgen` (the bulk mesh directly), transfer via `remap_bulk_function`. 3D bulk: `remesh_region_3d` + `remap_bulk_function_3d`. A moving 3D *surface* mesh raises `NotImplementedError`. Tests: `test_backend_ale.py` (membrane), `test_backend_ale_bulk.py`, `test_backend_ale_3d.py`, `test_moving_boundary_run.py` (through the CLI). |

As of 2026-06-11 the three `core/` field-transfer + meshing prerequisites and the
`DiscreteProblem` rebuild path are all built — the surface-trace correction (step d), the
bulk-conservative interpolation `transfer_bulk` (step c; `remap_bulk_function`, the FEM
form of the 2D overlap remap the `../vcell-mbsolver` baseline does on Voronoi cells), the
region remesher `mesh_region` (step b; since ADR 008 the Netgen `mesh_region_netgen`), and `backend.rebuild_on_mesh` (steps c+d fused:
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
the margin below the hard limit.

Approach-A bulk mesh-motion is also built (`_HarmonicExtension` in `backend/discrete.py`):
a moving codim-0 mesh moves its interior nodes by harmonic extension of the boundary
displacement, so a prescribed boundary velocity gives a smooth interior fill (and interior
singularities of the velocity formula, e.g. `x/r(x)` at the centre, are sidestepped). And
the driver now drives it: `_remesh` dispatches on codimension, so a moving **bulk** region
runs remesh-and-continue — the deformed boundary loop (`BulkBoundaryTrace.boundary_loop()`)
is meshed directly into a fresh bulk mesh and the field transferred conservatively
(`remap_bulk_function`). A moving **bulk-diffusion** problem now completes through
`run_with_remeshing` with mass conserved across each remesh.

The 3D bulk case followed on 2026-09-24 for VCell's 3D moving boundaries (the integration
tracker, "3D moving boundaries"): `backend/remesh_3d.py` rebuilds the deformed tetrahedral
region implicitly and `remap_bulk_function_3d` transfers the field by interpolation with a
global mass correction. Mass is conserved to 1e-12 across remeshes and the enclosed volume
restored exactly at each; the known limits are cost (a remesh grades the volume mesh at h/2
near the boundary) and necks the mesh cannot resolve (`PinchOffError`, reported as a
resolution limit rather than a topology change).

What remains is the genuine Approach-A *physics*: ρ as a bulk boundary **trace** — a surface
PDE living on bulk boundary facets, rather than the current independent-submesh T2 or plain
bulk T1. That needs trace-restricted function spaces / mixed-dimensional assembly (a new
modelling capability, v2-scale). Its remesh transfer would then route through
`correct_surface_trace` (built, waiting) on top of the bulk transfer. The mesh-motion,
remeshing, and conservative transfer machinery it would ride on are all in place.

## Verification plan

Same analytical-check + negative-control pattern the project uses elsewhere. Items 1–4 are
implemented in `tests/test_backend_ale.py` (membrane), `tests/test_backend_ale_bulk.py` and
`tests/test_backend_ale_3d.py`; the moving MMS cases in `mms/` (static/moving twins) and
`cross_validation/mb_*.py` / `furrow3d_midplane.py` (against VCell's mbsolver and an exact
waist motion) cover the end-to-end moving-boundary runs.

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

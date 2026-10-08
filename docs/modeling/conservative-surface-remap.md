# Conservative surface-density remap on remeshing

When a moving-membrane simulation remeshes, the surface density ρ must be carried
from the old membrane discretization to the new one **without creating or destroying
surface mass**. This note sketches the algorithm, its conservation guarantee, the
Approach-A–specific wrinkle that makes it subtler than it first looks, and a
verification plan.

This is an **approach-agnostic primitive**: Approach A needs it whenever ALE node
motion degrades element quality and the boundary is re-tessellated; Approaches B and D
need the same operation on their own surface representations once remeshing /
re-cutting enters the picture. It lives in `core/` (it is in fact `core/`'s first
occupant), not in any one approach.

Background and the comparison baseline that motivated this note: the FronTier-based
`../vcell-mbsolver` implements the *2D bulk* version of this remap (overlap-area-weighted
between Voronoi control volumes, Clipper polygon clipping). See the "Related
finite-volume / front-tracking work" section of `approaches.md` and
`docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md`.

## Implementation status

The serial / P1 / single-closed-2D-membrane envelope of this design is **implemented** in
`src/vcell_fenics/core/` as three composable pieces. Sections below note `[implemented]`
against the parts that are built and `[deferred]` against the rest; the prose remains the
design rationale.

| Piece | Module | Tests |
|---|---|---|
| Supermesh kernel (pure NumPy) | `core/surface_remap.py` | `tests/test_core_surface_remap.py` |
| DOLFINx `Function` bridge | `core/surface_remap_mesh.py` | `tests/test_core_surface_remap_mesh.py` |
| Approach-A trace correction | `core/surface_remap_trace.py` | `tests/test_core_surface_remap_trace.py` |

**Deferred:** P0 variant, MPI / multi-rank, higher-order spaces, open arcs, and the 3D
triangle-surface supermesh. The ALE remesh *driver* (`backend/ale.py`, built 2026-06-11; see the
driver note) calls `remap_surface_function` directly on the membrane's own DOFs; the trace
correction stays built-and-waiting for the Approach-A trace physics (ρ as a bulk boundary trace),
which is not implemented.

## Why the 2D-bulk trick does not port directly

The mbsolver computes overlap as a literal polygon **intersection**: two area control
volumes that occupy the same region overlap on a sub-area, and Clipper returns it. Two
nearby **polylines do not** — they meet only at isolated crossing points, so there is no
"overlap length" to clip out. The 1D surface remap therefore cannot be "clip and take
the overlap." It must lift both curves onto a **common 1D coordinate** — arc length `s`
along the closed membrane — and do the remap there.

In 3D the membrane is a triangulated surface and the polygon-clipping flavor *returns*
(triangle–triangle intersection on the surface, e.g. via `libsupermesh` or a CGAL
backend). The interface below is written so the 1D (2D-problem) and 2D-surface (3D-problem)
cases share a signature.

## What the remap must do

A remesh is a **fixed-time geometry swap**: Γ_old and Γ_new are two discretizations of
the *same physical curve at the same instant*. The target:

```
conservation:  ∫_{Γ_new} ρ_new ds  =  ∫_{Γ_old} ρ_old ds     (total surface mass held fixed)
accuracy:      ρ_new ≈ ρ_old at corresponding material points, linear-preserving
```

**No dilution is applied.** The `ρ ∇_Γ · v_Γ` dilution term is the *time-evolution* term
owned by the time stepper (see `approaches.md` §"Governing surface PDE"); remeshing does
not advance time and does not move the membrane. So the remap conserves **mass**
(∫ρ ds), and lets ρ adjust if the new polyline's length differs slightly from the old.
Conserving *concentration* across a remesh instead would be exactly the mass-balance bug
this whole machinery exists to prevent.

## Algorithm — 1D supermesh / Galerkin-conservative remap  `[implemented]`

Built as `supermesh_remap_1d(s_old, rho_old, s_new, length)` in `core/surface_remap.py`,
with `arclength_parameterization` for step 1's `cumulative_arclength` and
`project_points_to_polyline_arclength` for `arclength_of_closest_point`. The pseudocode
below is the shape of the real code:

```python
def remap_surface_density(gamma_old, rho_old, gamma_new):
    # 1. lift both loops onto a common 1D coordinate: arc length s in [0, L)
    s_old = cumulative_arclength(gamma_old.nodes)            # exact: nodes lie on the curve
    s_new = [arclength_of_closest_point(p, gamma_old)        # project new nodes onto old curve
             for p in gamma_new.nodes]

    # 2. supermesh = merge both node sets in s
    #    each sub-segment lies inside exactly one old AND one new segment
    breaks = sorted(set(s_old) | set(s_new))                 # closed loop: wrap past L -> 0
    subsegs = cyclic_pairs(breaks)

    # 3. assemble the two mass matrices over the supermesh
    M = zeros(n_new, n_new)        # new-mesh surface mass matrix   ∫ φ_new_j φ_new_k ds
    B = zeros(n_new, n_old)        # mixed mass matrix              ∫ φ_new_j φ_old_i ds
    for (a, b) in subsegs:
        j = new_seg_of(a, b);  i = old_seg_of(a, b)          # parents unique by construction
        local_assemble(M, j, j, a, b)                        # P1 basis products, exactly integrable
        local_assemble(B, j, i, a, b)

    # 4. conservative L2 projection
    rho_new = solve(M, B @ rho_old)
    return rho_new
```

```
old:  *---------*----------*--------*        (Γ_old segments, ρ_old nodal)
new:  *------*------*------*------*           (Γ_new segments)
s:    |--|---|--|---|--|----|--|----|         supermesh sub-segments
            ^ each sub-seg has exactly one old parent and one new parent
```

**Why it conserves.** With `ρ_new = M⁻¹ B ρ_old`,

```
∫_{Γ_new} ρ_new ds = 𝟙ᵀ M ρ_new = 𝟙ᵀ M (M⁻¹ B ρ_old) = 𝟙ᵀ B ρ_old.
```

Because the new P1 basis is a partition of unity (Σ_j φ_j ≡ 1), the column sums of `B`
equal `(∫ φ_old_i ds)` = `𝟙ᵀ M_old`, so total mass transfers **exactly**. This is the
Farrell–Maddison–Pelletier supermesh conservation property, specialized to 1D.

The cheaper **P0 (cell-average)** variant is the donor-cell special case `[deferred]`:

```
avg_new[j] = (1 / len_j) · Σ_{sub ⊂ j} ρ_old[parent(sub)] · len(sub)
```

conservative and exact, but only piecewise-constant accuracy.

## The Approach-A–specific wrinkle  `[implemented]`

Built as `correct_surface_trace(u_bulk_old, u_bulk_new)` + `BulkBoundaryTrace` in
`core/surface_remap_trace.py`.

In Approach A, ρ is **not an independent field** — it is the *trace* of a bulk function
space on the boundary facets (`approaches.md` §A). A remesh regenerates the *bulk* mesh
and conservatively interpolates the *bulk* fields `c` (a 2D area supermesh, or
non-matching interpolation plus a mass-correction step). The boundary DOFs of the new
bulk space then carry whatever that bulk remap left on them.

**A conservative bulk (area) remap does not imply a conservative surface (arc-length)
remap.** The bulk interpolation preserves ∫_Ω c dx, not ∫_Γ ρ ds — different integrals
over different-dimensional measures. So even though ρ "lives in the bulk space," surface
mass is *not* preserved for free; the 1D remap above must still run as a **separate,
explicit correction on the boundary DOFs** after the bulk transfer.

Design consequence worth stating plainly: **the remap difficulty is another force
pushing toward Approach B.** With independent surface DOFs (B), the membrane has its own
1D mesh and the remap is clean and self-contained — no gather/correct/scatter. In A you
either (a) keep ρ as a trace and bolt on the boundary-DOF correction, or (b) carry ρ as a
quasi-independent boundary field — which is really drifting into B. This compounds the
trace/resolution coupling already noted as an Approach A weakness.

## Two regimes, and the honest failure mode

The bridge (`remap_surface_function`) always uses closest-point projection onto the old
polyline, so both regimes are handled by the same code path:

- **Co-located remesh.** New nodes are inserted / removed / redistributed *along the
  existing polyline*, so Γ_new ⊂ Γ_old as a curve. The projection is then exact and the
  remap is exact to round-off. The well-behaved path.
- **Smoothed / non-co-located remesh.** If remeshing also moves the curve (smoothing it
  off the old polyline), the projection is a genuine normal projection and the supermesh
  is approximate in the new mesh's own metric. `remap_surface_function(..., conserve=True)`
  (the default) applies the global rescale `ρ_new *= M_old_total / M_new_total` so total
  mass on the new mesh is exact regardless; `conserve=False` leaves the raw projection-frame
  remap (constants still exact). This is the lossy case flagged in the front-tracking
  research note.

## Verification plan  `[implemented]`

Mirrors the project's established pattern (analytical check + negative control; see the
verification-patterns memory and `tests/test_backend_convergence.py`). As built across the
three test files:

1. **Conservation (round-off).** Remap a known ρ(s); assert `∫_{Γ_new} ρ_new ds ==
   ∫_{Γ_old} ρ_old ds` to ~1e-12 — at the kernel level, on real membrane meshes, and on
   the bulk trace.
2. **Accuracy (refinement).** For a smooth ρ(s), the L² remap error → 0 at second order
   (P1) as both meshes refine. *(Refining only the target floors the error at the source
   interpolation error — accuracy is bounded by the coarser mesh, so both must refine.)*
3. **Constant / refinement exactness.** A constant ρ remaps to the same constant exactly
   (partition of unity); a co-located refinement reproduces ρ exactly at shared nodes.
   *(The original sketch said "linear-preservation"; a globally linear field is
   discontinuous across the closed-loop seam, so constant preservation is the clean exact
   invariant.)*
4. **Negative control.** The naive nearest-node copy is *not* conservative — it drifts
   ∫ρ ds, confirming the supermesh step is load-bearing (checked for both the standalone
   membrane and the bulk trace).

## Implemented interface

The original sketch imagined a `SurfaceRemap` class; the built API is three small modules
of functions (`core/__init__.py` re-exports all of them):

```python
# core/surface_remap.py — pure-NumPy kernel
def arclength_parameterization(points, *, closed=True) -> (s, length): ...
def supermesh_remap_1d(s_old, rho_old, s_new, length) -> rho_new: ...     # conserves ∫_Γ ρ ds
def project_points_to_polyline_arclength(points, loop_coords, s_loop, length) -> s: ...

# core/surface_remap_mesh.py — DOLFINx Function bridge
def ordered_membrane_loop(V) -> (coords, dof_order): ...
def remap_surface_function(u_old, V_new, *, conserve=True) -> fem.Function: ...

# core/surface_remap_trace.py — Approach-A trace correction
class BulkBoundaryTrace:  # gather()/scatter() between bulk-boundary and surface DOFs
def correct_surface_trace(u_bulk_old, u_bulk_new, *, conserve=True) -> None: ...   # in place
```

Both A (on remesh) and the eventual B / D remeshing paths call the same primitive. Applies
no dilution. The 3D case (triangle-surface supermesh) is deferred but the function-level
signatures generalize.

## References

- Farrell, Maddison (2011), *Conservative interpolation between volume meshes by local
  Galerkin projection* — the supermesh method this specializes; `libsupermesh` is the
  reference implementation for the 2D/3D cases.
- `../vcell-mbsolver` — the 2D-bulk overlap remap baseline (Clipper-based); see
  `approaches.md` "Related finite-volume / front-tracking work".
- `docs/research/2026-06-06-cutcell-fronttracking-chatgpt.md` — the cut-cell /
  front-tracking survey that surfaced the conservative-remap-on-remeshing problem.
- `docs/modeling/ale-remesh-driver.md` — the ALE remesh loop (built), whose Approach-A
  variant would call `correct_surface_trace` as its boundary post-pass.

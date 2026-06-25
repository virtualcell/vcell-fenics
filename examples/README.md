# Examples

Runnable demos that exercise the backend end-to-end. Run any of them with the dev env, e.g.

```bash
.pixi/envs/dev/bin/python examples/cahn_hilliard_phase_separation.py
```

- **`cahn_hilliard_phase_separation.py`** — declarative spinodal decomposition (a `cahn_hilliard`
  template + a `normal(...)` random IC), driven through `iter_cahn_hilliard`. The solver is verified
  against exact solutions (interface width, dispersion relation, droplet curvature) in
  `tests/test_backend_cahn_hilliard.py`.
- **`legi_sustained_chemotaxis.py`** — LEGI gradient **sensing** with a surface-tension force balance.
  See the note below: the sensing works; net migration does **not** (yet).
- **`legi_diagnostic.py`** — the analysis that established that (writes `legi_diag.png`).
- **`self_organizing_chemotaxis.py`** — an earlier receptor-binding chemotaxis demo. ⚠️ Its "migration"
  uses the same membrane node-mean metric the LEGI note below shows is misleading; treat its migration
  claim as unverified pending the same CoM check.
- **`deforming_membrane_mol_remeshing.py`** — large-deformation moving-surface PDE: a receptor density
  `rho` on a cell membrane (surface diffusion + the mandatory dilution `rho ∇_Γ·v_Γ`) driven through a
  big shape change (a circle elongating into a dumbbell), solved by **method-of-lines + remeshing**
  (`run_moving_with_remeshing`). Writes `deforming_membrane_mol_remeshing.png`: the membrane snapshots
  coloured by `rho` (top) and the mesh-quality-growth trace (bottom) — held bounded by remeshing
  (sawtooth, resetting at each remesh) versus climbing unbounded without it. The driver is verified in
  `tests/test_backend_ale_mol.py`.

---

## Note: the LEGI "migration" is treadmilling, not migration (2026-06)

This records a real finding from revisiting the moving-boundary force balance, so it isn't re-discovered later.

### What the demo does and doesn't do

The LEGI **sensing** chain is correct and sustained: an external chemoattractant gradient (a Dirichlet
on the box wall) → membrane receptor occupancy `R` → a cytosolic local activator `a` / global inhibitor
`h` → a relative-gradient response `a − h` that polarises up-gradient and is held by the external cue.
That response sets a membrane tension `γ = base + α·(a − h)`, and `ForceBalanceMeshMotion` solves a
surface-tension Stokes flow on the cell and moves the membrane by it.

**The cell does not migrate.** Its true centre of mass — the area centroid `∫x dx / ∫dx` — stays at the
origin (drifts to about −0.013, i.e. slightly *backward*) over the whole run, and the membrane outline
stays a centred circle (`legi_diag.png`, left panel). The "+0.2 migration Δx" an earlier version printed
was the membrane **node-mean** sliding along that fixed circle with the surface flow — a misleading
diagnostic, not translation.

### Why — and what migration would require

`legi_diag.png` (right panel) shows the velocity at a polarised time: a recirculating **Marangoni flow**
(tangential along the membrane toward the high-tension side, returning through the interior),
`max|u| ≈ 0.12`, divergence-free, zero net drift. This is exactly what the physics demands: a closed,
incompressible cell under a *pure surface-tension* balance has **no net propulsive force**, so a tension
gradient drives a surface circulation that treadmills the membrane without moving the CoM. Genuine
migration needs an active, front–back-asymmetric ingredient the force balance lacks — a **normal**
protrusion/retraction velocity (actin-driven), an **active cortical stress**, or **asymmetric substrate
adhesion** (spatially varying friction). That is a follow-up.

The long run is also not indefinitely stable: past ~t=12 the polarised response grows a leading-edge
feature that blows the explicit ALE update up — a residual instability that neither the area correction
nor a tension-smoothing filter resolves.

### Area conservation (a clean by-product)

`ForceBalanceMeshMotion(area_correction=True)` is **not** a geometric area rescale. The P2 Taylor–Hood
velocity is divergence-free (`∮u·n = 0` over the cell boundary), but its **P1 interpolation** — which
actually moves the mesh — is not: interpolating drops the P2 edge-midpoint velocity that a straight-edged
polygon cannot carry, leaving a spurious inward flux `∝ h` (~0.13 %/step at `h=0.06`). The correction
re-projects that interpolated velocity back to divergence-free, *at the interpolation*, by removing the
net flux as a radial field `α(x − c)` with `α = −∮v1·n / (2A)` (since `∮α(x − c)·n = 2αA`). Area is then
conserved to `O(dt²)` — a principled velocity projection, not a band-aid.

### Reproduce

```bash
.pixi/envs/dev/bin/python examples/legi_diagnostic.py   # writes legi_diag.png
```

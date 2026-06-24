# Plan: active-protrusion migration (making the cell actually translate)

**Status:** planned, not started. Deferred behind the kinematics / mbsolver-comparison work.

## Why

The surface-tension force balance (`ForceBalanceMeshMotion`) **does not migrate the cell** — it
treadmills the membrane without translating the centre of mass. Diagnosed and recorded in
`examples/README.md` and `examples/legi_diagnostic.py`: under a tension gradient `γ = base + α·(a−h)`
the cell's area-centroid CoM stays at the origin (drifts ~−0.013, slightly *backward*) while the
membrane node-mean drifts +0.2 — the latter is the misleading "migration Δx". The velocity is a
recirculating **Marangoni surface flow** (`max|u|≈0.12`), divergence-free, zero net drift. This is
correct physics: a closed, incompressible cell under a *pure* surface-tension balance has **no net
propulsive force**.

## The mechanism to add

The standard active-gel / keratocyte model: a **graded normal protrusion velocity** on the membrane —
actin pushes the front out, the back retracts — that is **volume-conserving** but **front–back
asymmetric**, so the *shape* translates (real boundary treadmilling, not just the parametrization):

```
v_n = β · (s − s̄) · n̂,    s̄ = (1/L) ∮ s dl
```

- `s` — the LEGI polarity signal on the membrane (the response `a − h` traced to the membrane, or the
  receptor `R`, already a membrane field);
- `s̄` — its arc-length mean, so `∮ v_n·n = β ∮(s − s̄) dl = 0` ⇒ **volume conserved by construction**
  (high-signal arcs protrude, low-signal arcs retract);
- `n̂` — the outward membrane normal; `β` — a protrusion-rate constant.

The CoM translates toward the high-signal side. This is the active-surface direction of the project's
Contri–Massing–Rangamani North Star.

## Implementation pieces

1. **Outward nodal normals on the membrane** — the one genuinely new bit. Geometric, from the ordered
   membrane loop: tangent ≈ `x_{i+1} − x_{i−1}`, rotate 90°, orient away from the centroid. Robust for a
   smooth membrane; may be noisy near the leading-edge instability.
2. **Signal on the membrane, mean-subtracted** — reuse the existing cyto-boundary→membrane transfer
   (`_mem_to_cyto` in `ForceBalanceMeshMotion`) to get `a − h` on the membrane; or use `R` directly.
3. **Inject `dt·v_n` into the interface/membrane velocity** before the existing harmonic extension +
   move in `advance()`. Everything downstream (harmonic extension into the bulk, EntityMap consistency)
   is unchanged. The flux-projection area correction already enforces `∮v·n=0`, mopping up any residual
   from the mean-subtraction.

Cleanest packaging: a `protrusion_rate` (+ a signal handle) on `ForceBalanceMeshMotion`, or a small
subclass. `β = 0` recovers today's passive behaviour exactly.

## Verification (the decisive check)

A test contrasting `β > 0` vs `β = 0`:

- **area-centroid CoM Δx grows up-gradient** for `β > 0`, ≈ 0 for `β = 0` (the decisive one — the test
  the treadmilling demo *fails*);
- **`lead_x` and `trail_x` both advance** (the whole shape translates, not the node-mean);
- **area conserved** (volume held).

## Honest limits

- **Mesh tangling caps the distance.** As the cell translates, the harmonic-extended bulk meshes distort;
  past some migration the ext mesh between cell and wall inverts (this is what blew up the bigger-cell
  probe). *Sustained* migration to a boundary needs **remeshing** — see the ALE remesh-driver sketch in
  `docs/modeling/`. A prototype that moves the CoM ~a body-length needs none.
- The **leading-edge instability** (explicit-ALE blow-up ~t=12) may still cap the horizon; `β` must be
  moderate.

## Effort

~2–3 hrs to prototype as a throwaway probe (subclass + protrusion + measure the area-centroid CoM) and
confirm the CoM translates. If it works, ~a day to formalise: the `protrusion_rate` API, the verification
test, and an honest migrating-cell demo.

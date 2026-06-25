# Plan: active actin–myosin gel migration

**Status:** planned, not started. Grounded in the VCell moving-boundary cell-migration model
(Nickaeen, Novak, Pulford, Rumack, Brandon, Slepchenko, Mogilner, *PLoS Comput Biol* 2017 — in
`docs/papers/`, gitignored), which is exactly what `../vcell-mbsolver` runs. Supersedes the earlier
"graded normal protrusion" placeholder (kept below as the crude approximation it was).

## Why the surface-tension force balance does not migrate

`ForceBalanceMeshMotion` solves an **incompressible** surface-tension Stokes flow. Diagnosed in
`examples/README.md` / `examples/legi_diagnostic.py`: a tension gradient drives a divergence-free
**Marangoni surface circulation** that treadmills the membrane, but the centre of mass does not move —
a closed incompressible cell under pure surface tension has **no net propulsive force**. Real migration
needs a fundamentally different model.

## The real model (Nickaeen–Novak–Mogilner 2017)

A 2D **compressible visco-active actin–myosin gel** with a free boundary, all in the lab frame:

- **Force balance for the actin flow velocity `U`:**
  ```
  η ΔU  +  σ ∇M  −  ξ U  =  0
  ```
  - `η ΔU` — passive viscous stress of the deforming actin network. **The network is compressible**
    (cytoplasm squeezes out dorsally) — there is *no incompressibility condition*. This is the central
    departure from our incompressible Stokes `ForceBalanceMeshMotion`.
  - `σ ∇M` — divergence of the **myosin contractile stress** (isotropic, ∝ myosin density `M`). The motor.
  - `ξ U` — **viscous drag of F-actin against the substrate**, mediated by adhesions. **This is the
    propulsion.** The actin flows centripetally; the adhesions transmit the reaction as a forward traction
    on the cell, giving net CoM translation. (It is the "asymmetric substrate adhesion" ingredient flagged
    as missing in the surface-tension diagnosis — and it is *central*, not an add-on.)
- **Myosin transport:** `∂_t M = ∇·(D_eff ∇M − U M)` — myosin diffuses and is advected by the actin flow.
- **Boundary (membrane) velocity:** `V_f = V_p n + U|_∂Ω` — locally-normal protrusion `V_p` (actin
  polymerization at the edge) superposed on the actin flow `U` at the boundary (which retracts it).
- **Protrusion `V_p`:** actin-growth rate with an area-preserving membrane-tension term; in the
  zero-velocity (ZV) variant it decreases with local myosin (myosin bundles actin, impeding growth).
- **Force-balance BCs:** zero actin velocity `U|_∂Ω = 0` (a sticky adhesion band — ZV) **or** zero stress
  `n·(η∇U + σM I)|_∂Ω = 0` (ZS). Myosin: no-flux Rankine–Hugoniot `n·(−D_eff∇M + (U−V_f)M)|_∂Ω = 0`.

**Migration is emergent, not imposed.** Above a critical myosin contractility, a positive feedback —
contraction → centripetal actin flow → advects/concentrates myosin → reinforces contraction — breaks the
radial symmetry; myosin piles at the rear and the cell translates steadily (or rotates). The CoM moves
because of the substrate traction `ξU` combined with the myosin asymmetry.

## What this means for vcell-fenics

This is a different (and bigger) backend than the incompressible force balance:

1. **Compressible visco-active force balance** for `U` (`η ΔU + σ∇M − ξU = 0`) — a vector elliptic solve,
   *no pressure/incompressibility*. Reuses the moving-mesh + harmonic-extension machinery.
2. **Myosin transport** `∂_t M = ∇·(D_eff∇M − UM)` on the moving domain — a `bulk_radv_diff`-style equation
   with advection by `U` and the dilution from `∇·U ≠ 0` (now nonzero — the compressible case the dilution
   term was built for).
3. **Boundary motion** `V_f = V_p n + U|_∂Ω` — a new motion mode that reads `U` at the boundary and adds a
   normal protrusion; folds into the existing harmonic extension.
4. **Substrate drag** `ξU` in the force balance — the propulsion term.

The same model is what the mbsolver runs (FronTier front-tracking + conservative Voronoi FV), so once
both exist this is the apples-to-apples **ALE-FEM vs FV cross-validation** of cell migration — the real
goal of the comparison track.

## Verification

- **CoM translates** (area-centroid `∫x dx/∫dx`), not just the node-mean — the test the treadmilling demo
  fails — and `lead_x`/`trail_x` both advance.
- **Symmetry-breaking threshold:** stationary below a critical myosin contractility, motile above it
  (reproduce the 2017 stationary/translating/rotating modes).
- Cross-validate the steady speed + shape + myosin profile against the mbsolver on the same parameters.

## Honest limits

- **Mesh tangling** still caps the distance; sustained migration needs remeshing (ALE remesh-driver sketch
  in `docs/modeling/`). The mbsolver sidesteps this with its fixed Cartesian grid + conservative remap.
- This is a multi-week build (a new compressible-gel backend + myosin transport + the emergent
  instability), not the ~1 day the placeholder implied.

---

### Superseded placeholder (kept for the record)

The earlier sketch was a *kinematic* graded normal protrusion `v_n = β(s − s̄)·n̂` (front protrusion +
back retraction, volume-conserving) bolted onto the surface-tension motion. It would move the CoM, but it
is a phenomenological caricature: it imposes the protrusion profile by hand instead of deriving it from
the actin–myosin force balance + substrate drag, and it keeps the (wrong, incompressible) surface-tension
flow. Useful only as a quick "does the CoM move at all" probe; the 2017 model above is the real target.

# FV ↔ FEniCSx cross-validation

End-to-end numerics comparison of VCell's finite-volume solver against this repo's FEniCSx
backend, on a **common VCML** both solvers consume. This is the §"comparison baseline" milestone:
the same authored model is run by VCell's FV solver (ground truth) and imported + realized + solved
through `vcell_fenics`, then compared point-by-point on the FV solver's own grid.

## The models

Single 2D diffusing species `u` on the box `[-1, 1]²`, no-flux boundaries (mass conserved), with a
spatially non-uniform (off-centre Gaussian) initial condition:

| stem  | initial condition          | reaction            |
|-------|----------------------------|---------------------|
| `v1`  | Gaussian, width `a = 0.02` | pure diffusion      |
| `v1b` | Gaussian, width `a = 0.05` | pure diffusion      |
| `v2`  | Gaussian, width `a = 0.02` | first-order decay `k = 0.5` |

`D = 0.1`, end time `1.0`, 11 output times, FV mesh `128 × 128`.

## Two-stage flow (two environments)

The two solvers live in different environments (the dependency boundary: libvcell + fvsolver stay
out of the DOLFINx env), so the comparison is two stages:

1. **Author + run FV** — `author_and_run.py`, in the pyvcell `[native,solver]` env. Authors the
   biomodel, runs VCell's FV solver, and writes next to each model:
   - `<stem>.vcml` — the exact VCML the FV solver consumed;
   - `<stem>_math.yaml` / `<stem>_geom.yaml` — the lowered MathDescription / Geometry (what our
     bridge imports), extracted from the VCML via `VcmlReader`;
   - `<stem>_reference.npz` — the 4D FV field `u(t, c, y, x)` on the solver's grid, plus the
     free-space analytic reference. **Not committed** (large, ~2.4 MB each) — regenerate with
     `../pyvcell/.venv/bin/python cross_validation/author_and_run.py`.

2. **Import + solve + compare** — `compare_fenics_vs_fv.py`, in the vcell-fenics **dev** env.
   Imports the same lowered math + geometry, realizes the box, solves through our backend, samples
   `u_h` at the FV grid points, and reports relative-L2 agreement per output time plus mass.

```bash
../pyvcell/.venv/bin/python  cross_validation/author_and_run.py            # stage 1 (regenerates .npz)
.pixi/envs/dev/bin/python    cross_validation/compare_fenics_vs_fv.py      # stage 2 (needs the .npz)
```

## Result

The two independent solvers agree to **< 2.5 %** relative L2 at every output time (tightening as the
field smooths). Both reflect mass at the walls, so each diverges from the *free-space* analytic only
once the front reaches the boundary (relL2 → ~14 % by `t = 1`) — an expected physical difference, not
solver error. `v2`'s total mass decays `0.628 → 0.384 ≈ 0.628·e^(−0.5)` on both solvers, confirming
the decay reaction rides correctly on top of diffusion.

## Membrane jump-condition sign convention

`membrane_flux_sign.py` (pyvcell `[native,solver]` env) confirms the **sign** of the membrane
jump-condition → Neumann mapping against the FV solver, on a two-compartment cell (inner-disk
`cytosol` + `background` extracellular, `membrane` between them), in both directions:

| case   | reaction                          | VCell `in_flux(u)`   | FV cytosol mass |
|--------|-----------------------------------|----------------------|-----------------|
| efflux | mass-action `u → ∅` (`Kf·u`)      | negative (`∝ Kf·u`)  | decreases       |
| influx | general-kinetics `∅ → u` (`J=0.5`)| positive (constant)  | increases       |

The bridge maps `in_flux` **directly** (no sign flip) to `BCNeumann(u, membrane, in_flux)`, and the
backend's Neumann sign is independently verified (`d(mass)/dt = ∫_Γ h ds`). Both FV directions match
the sign of `in_flux`, so the convention is correct (locked in the dev env by
`tests/test_pyvcell_bridge.py::test_jump_condition_preserves_vcell_flux_sign`). A constant membrane
source needs **general kinetics** (the net rate set directly) — mass-action gives an empty reactant
product a rate of 0, so a reactant-less `∅ → u` is silently inert.

A full multi-compartment numerical FV-vs-ours *field* comparison would additionally need the backend
to apply a one-sided flux on an internal interface (solve one compartment's submesh with the membrane
as its boundary) — a later increment; this check confirms the sign at the import boundary.

## Time-dependent membrane flux (field comparison)

`membrane_timeflux_fv.py` (stage 1, pyvcell env) + `compare_membrane_timeflux.py` (stage 2, dev env)
cross-validate a **time-dependent** membrane flux — the case that exercises the backend's `sim.t` path
end-to-end (a `g(t)` Neumann re-evaluated each step). Same cell, with a general-kinetics membrane
influx whose net rate is an explicit function of time, `J(t) = 100·exp(−2t)`. The flux depends only on
`t`, so the cytosol PDE decouples from the extracellular and the membrane is the whole cytosol-disk
boundary: stage 2 imports the real lowered math, reduces it to the cytosol (the membrane survives as
the disk's external boundary carrying the imported `BCNeumann(u, membrane_dom, g(t))`), and solves on
a single-compartment disk.

Result: **0.9–2.1 % relL2** against the FV cytosol field at every output time, with mass tracking to
~1 %; the mass-rise rate slows over time exactly as `J` decays — the time signature. The VCell unit
factors (`KFlux`, `UnitFactor = KMOLE`, reconciling membrane molecules·µm⁻² ↔ volume µM) are carried
as parameters, so our applied Neumann matches FV in **magnitude** as well as sign. This needed the
backend to bind a `ParameterExpression` (those unit factors are `Area/Volume`, `pow(KMOLE,1)`, not
bare constants) — `test_backend_assemble.py::test_expression_parameter_binds_against_constants`.

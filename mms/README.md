# Manufactured-solution (MMS) verification suite

A **persistent** suite of Method-of-Manufactured-Solutions cases, each recorded as a **math description**
whose `source` term is the manufactured forcing that makes a chosen exact solution `u*(x, t)` the *true*
solution of the PDE. Each case runs through the solvers and its numerical result is compared to the
**known** `u*` — measuring the true `L2`/`L∞` error and the **convergence order**.

Why MMS (vs the conservation tests): a scheme can conserve every integral exactly while converging at the
wrong rate, or to the wrong answer. MMS checks the *solution itself* against a closed form, so it catches
**correctness** defects — wrong order, stale boundary data, term-consistency bugs — that conservation is
blind to. (The moving-bulk case below was found this way in the first hour of building the suite.)

## Three solvers, one manufactured model

Each case is solver-agnostic — a math description + an exact solution + metadata — so the **same** case is
checked against every applicable solver:

| solver | runner | env | applies to |
|---|---|---|---|
| vcell-fenics (BE + MOL/ALE) | `runner.py` | `.pixi/envs/dev` | static + moving |
| pyvcell **fvsolver** (fixed-grid FV) | `runner_fv.py` | `../pyvcell/.venv` | static (box) |
| pyvcell **mbsolver** (moving front) | `runner_mb.py` | `../pyvcell/.venv` | moving boundary |

```bash
.pixi/envs/dev/bin/python  mms/runner.py    mms/cases/<case>.yaml   # vcell-fenics (BE + MOL)
../pyvcell/.venv/bin/python mms/runner_fv.py mms/cases/<case>.yaml   # fvsolver  (fixed-grid FV)
../pyvcell/.venv/bin/python mms/runner_mb.py mms/cases/<case>.yaml   # mbsolver  (moving front)
```

`applicable_solvers` in each case declares which apply. The fv/mb runners author the case as VCML — the
manufactured forcing is a **general-kinetics** volume source `∅ → u` with net rate `J = f` (mass-action
*drops* a spatial/signed zeroth-order source; general kinetics sets the rate directly — verified). They run
the native solver and compare to the same recorded `u*`:
- **fvsolver** — fixed-grid, so its natural MMS is a **box** with zero-flux-compatible `u*` (e.g.
  `cos(x)cos(y)` on `[0,π]²`); reads the case's `fv:` block (`ic`, `source_j`, `diffusion`) + `fv_mesh_sizes`,
  and reports the h-order (verified: `box_static_diffusion` → order 1.98).
- **mbsolver** — the front deformation dictates the closed form, so the manufactured solution is the
  **radially dilating disk** (`u = u0·e^{−2k t}`, `R = R0·e^{k t}`); reads the `mb:` block and checks the
  per-frame mean + front radius against those `exact:` forms (verified: `mb_expansion_dilution` →
  0.9 %/1.5 %). A spatially *varying* `u*` under the moving front needs per-node sampling this build does
  not expose — a follow-up.

## Running

```bash
.pixi/envs/dev/bin/python mms/runner.py --all                 # every case, vcell-fenics
.pixi/envs/dev/bin/python mms/runner.py mms/cases/<case>.yaml # one case
```

## Case format (`cases/*.yaml`)

```yaml
name: ...
description: ...            # what it exercises; note if it exposes a defect
harness: bulk              # runner dispatch: bulk | membrane | coupled
dim: 2
geometry: {kind: disk, radius: 1.0, volume_subdomain: cyto, boundary: edge}
exact: {c: "2 + np.sin(x)*np.cos(y)"}   # u*(x,y,z,t) — numpy-evaluable, for the error norm
t_final: 1.0
dt: 0.01
resolutions_h: [0.1, 0.05, 0.025]
expected_order_h: 2.0
applicable_solvers: [fenics-be, fenics-mol, mbsolver]
params: {k: 0.25, D: 0.1}    # for reference / VCML authoring
math: |                      # the vcell-fenics MathDescription: the manufactured source lives in `source`,
  math_description: ...      # the IC is u*(·,0), the BC is u* on the boundary
```

**The load-bearing discipline — every moving case must be validated against its static reduction.** A
manufactured forcing is only trustworthy if the *static* version of the case (motion off) converges at the
theoretical order; if it doesn't, the forcing derivation (or the runner) is wrong, not the solver. Each
moving case therefore has, or is paired with, its static twin (see `static_bulk_diffusion.yaml`). A moving
case that fails while its static twin passes is a genuine solver finding.

## Manufacturing the forcing

For a lab-frame solution `u*` on a mesh moving (Lagrangian) at velocity `v`, the ALE conservation-form
residual is `∂_t u* + ∇·(u* v) − D∇²u* − reaction(u*)`, so the source is

```
f = ∂_t u* + ∇·(u* v) − D∇²u* − reaction(u*)      (= v·∇u* + u*∇·v + ∂_t u* − D∇²u* − reaction  )
```

A *static* case is the `v = 0` reduction, `f = ∂_t u* − D∇²u* − reaction`. On a moving boundary, a
position-dependent Dirichlet `u*` must be re-evaluated at the moved boundary each step (the runner calls
`set_time` to refresh it — see the finding below).

## Findings so far

**The moving-mesh solver is consistent — but first-order in time. Measure spatial order with dt ∝ h².**
An MMS sweep (8 contrived motion × physics cases) first appeared to show a defect: every case with
`∇·v ≠ 0` (expansion, contraction, anisotropic stretch) failed to hold a steady `u*` — `L∞` order ≈ 0 at
the fixed `dt` the runner used. **On independent verification this was a measurement artifact, not a solver
bug.** The backward-Euler + strided-motion moving scheme is **O(dt) in time** (expected), so at fixed `dt`
the temporal floor dominates and the `h`-order collapses. The tell: even a *constant* and a *linear* `u*`
(zero P1 interpolation error) show the same plateau, and the error halves *cleanly with dt* (ratio 2.00) —
a temporal floor, not an `h`-inconsistency. Refining **dt ∝ h²** recovers the true spatial order: the
expansion case then converges at **order 2.01 / 1.99**. The `∇·v = 0` cases (translation, shear) are the
same story. So the runner now refines `dt ∝ h²` on moving/time-dependent cases (`_dt_for`), and the moving
cases pass. *(Lesson baked into the suite: for a time-first-order scheme, an `h`-only MMS sweep measures the
temporal floor; always refine dt with h, or the sweep manufactures phantom defects. The adversarial verify
phase of the authoring workflow shared this blind spot — it is corrected here.)*

**Real gotcha that survived: the moving Dirichlet-BC refresh.** `step()` moves the mesh but does **not**
re-evaluate a position-dependent Dirichlet BC `g(x)` at the moved boundary unless `set_time` is called — so
a spatially-varying Dirichlet on a moving boundary silently uses stale (initial) positions (worth ~100× of
the error before the runner started refreshing it). The runner works around it by calling `set_time` each
step; a cleaner fix would refresh position-dependent BCs inside `step()` after the move. Tracked as a
follow-up. (This one is a genuine, if minor, solver gap — independent of the temporal floor above.)

## Roadmap

- Fill the matrix (solver × motion × physics) — the bulk sweep is done; extend to membrane, coupled,
  unknown-motion, and to more physics (advection, reaction, binding).
- Wire **box** geometry into the dev runner so the fvsolver box cases also run through vcell-fenics (true
  three-solver on one case); add a **spatially-varying** mbsolver case once per-node sampling is available.
- Promote the passing cases into the `pixi run check` gate (order-regression guard).

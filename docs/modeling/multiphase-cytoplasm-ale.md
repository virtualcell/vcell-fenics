# Multiphase cytoplasm on a moving membrane — an ALE design sketch

**Status:** design note, not implemented. Captures the architecture decision for the
eventual use case of a **two-phase cytoplasm** — a mechanically active cytoskeleton
plus an overdamped fluid — inside a moving, mechanically-coupled membrane. The goal of
this note is to choose a skeleton that keeps the **simplicity we already have** (one
conforming ALE mesh, the mixed-dimensional bulk↔surface coupling of `coupled.py`) while
admitting the **real fluid/solid physics**, and to identify which parts are shared
across modelling choices versus which force a fork.

It does **not** decide the biology. It decides the *frame* and the *software seams* so
that the two main modelling choices below are either both reachable, or cheaply
swappable.

## 1. The reframe that makes this tractable

The tension we kept hitting — "the membrane can slide tangentially relative to the bulk,
how do we couple that?" — dissolves once we separate two things we have so far
conflated:

- **Mesh velocity `w`** — how the *nodes* move. Geometric bookkeeping. Its **normal**
  component at the membrane is fixed by the membrane geometry (the compartments must stay
  conforming — fundamental); its **tangential** component is *free* (a quality /
  redistribution choice).
- **Phase velocities `vₐ`** — how the *material* of each phase moves. Physical. These can
  differ from `w` and from each other; the difference is carried by the ALE convective
  term `(vₐ − w)·∇(·)` — exactly the `relative_advection` slot already implemented
  (`backend/assemble.py`, `TermKind.ADVECTION`).

**Physical slip is not mesh slip.** A fluid sliding tangentially past the membrane is two
velocity *fields* differing tangentially — `v_fluid ≠ v_membrane` — not nodes sliding
relative to nodes. So the mesh can stay perfectly conforming (coincident membrane/bulk
nodes, the entity-map coupling intact), and every velocity difference lives in the
physics. This is the whole reason the coupled-mesh simplicity survives contact with
multiphase flow.

## 2. The eventual use case

Inside the membrane, two interpenetrating phases:

- **Cytoskeleton** (network `n`): a mechanically active material — viscous and/or
  elastic, with an **active contractile stress** — that is **mechanically coupled to the
  membrane**, at least tangentially (cortical attachment, cortical/retrograde flow drag).
  It is what drives and resists membrane shape change.
- **Overdamped fluid** (solvent `s`): the cytosol. Inertia-free (low Reynolds), so its
  momentum balance is quasi-static. It permeates / coexists with the network and may
  **slip** relative to the membrane.

The two phases exchange momentum through an **interphase drag** `ξ(v_s − v_n)`. The
membrane carries its own surface mechanics (tension, bending — the curvature forces
already built) and is **loaded by the cytoskeleton** at the boundary.

"Overdamped" everywhere means there are **no inertial `∂_t v` terms**: each time step is a
*quasi-static force balance* (an elliptic solve), exactly like the unknown-motion
force balance already in `backend/unknown_motion.py` — but now for two coupled vector
fields with a drag coupling, plus a pressure if the mixture is incompressible.

## 3. The two decision axes (and why they correlate)

Two independent-looking choices turn out to track each other:

**Axis A — two-phase formulation.**
- **A1. Two-fluid / mixture (active gel).** Two velocity fields `(v_n, v_s)`, volume
  fractions `θ_n + θ_s = 1`, a momentum balance for each, interphase drag. The network is
  treated as a (visco-active) *fluid*. Canonical for cortical flows and migration.
- **A2. Poroelastic / Brinkman.** An *elastic solid* network with a Darcy (or Brinkman)
  fluid flowing through it; Biot-style coupling through pressure. The network stores
  elastic energy.

**Axis B — cytoskeleton rheology / deformation.**
- **B1. Viscous active fluid — no memory.** Stress = viscous + active; turnover dominates,
  so the network *flows* and has no long-lived elastic memory. **No reference
  configuration.**
- **B2. (Visco)elastic solid — large deformation, memory.** Stress depends on deformation
  from a **reference configuration** (a deformation gradient `F`); the network *remembers*
  its rest shape.

The natural pairings are **(A1, B1)** — an *active viscous two-fluid* — and **(A2, B2)** —
a *poroelastic cytoplasm*. The note treats those as the two reference use cases:

- **UC-Viscous** = A1+B1: active gel + solvent, no reference configuration.
- **UC-Poroelastic** = A2+B2: elastic network + Darcy fluid, with a reference
  configuration.

## 4. The shared ALE skeleton (identical for both use cases)

The claim of this note is that the following skeleton is **the same** for both UCs, so it
can be built once:

1. **One conforming ALE mesh** of the cell interior, with the membrane as the boundary
   (or as a coincident-node submesh — §7). Reuses the current mesh-motion + coupling
   machinery.
2. **The mesh tracks a chosen "frame" phase.** Its velocity defines `w`. Normal motion at
   the membrane is fundamental; tangential is a free/quality choice. (Which phase the mesh
   follows is the one genuine fork — §6.)
3. **Every other phase is Eulerian relative to the mesh** and transported by
   `relative_advection = (vₐ − w)`. Already implemented.
4. **Overdamped ⇒ a quasi-static force-balance block each step.** A coupled solve for the
   phase velocities (and pressure), assembled with the **weak-form escape hatch**
   (`backend/weakform.py`) — extended from one vector field to a block of two plus a drag
   coupling. The existing multi-field block machinery (`coupled.py`,
   `ufl.MixedFunctionSpace` / `extract_blocks`) is the template.
5. **Interphase drag** `ξ(v_s − v_n)` — a symmetric off-diagonal coupling between the two
   velocity blocks. Mechanically trivial to add; it is what makes the solve genuinely
   two-phase (a block system, not two independent solves).
6. **Interface conditions on the phase velocities**, not on the mesh: normal-matching
   (`vₐ·n = w·n`, fundamental) and a tangential condition (no-slip / Navier-slip / drag —
   the modelling choice the membrane-vs-fluid coupling actually is). These need
   **normal-component velocity BCs**, which are the one piece of genuinely new
   machinery (the boundary normal is not axis-aligned ⇒ Nitsche or a rotated local
   frame).
7. **Membrane force balance loaded by the cortex.** The membrane's own surface mechanics
   (curvature/tension forces — built) gain a traction term from the network stress at the
   boundary. This is the mechanical coupling, an interface term.

Everything in §4 is phase-agnostic: it does not care whether the network is a fluid or an
elastic solid. That is the door we want to keep open.

## 5. Where the two use cases diverge

Only three things change between UC-Viscous and UC-Poroelastic, and all three are
**local to the constitutive law**, not the skeleton:

| Ingredient | UC-Viscous (A1/B1) | UC-Poroelastic (A2/B2) |
|---|---|---|
| Network stress `σ_n` | viscous `η(∇v+∇vᵀ)` + active `ζQ` | elastic `P(F)` (+ active), needs `F` |
| Reference configuration | **none** | **required** (rest state, `F = ∇φ`) |
| Fluid momentum | Stokes/Brinkman (a second viscous field) | Darcy (relative flux ∝ −∇p) |
| Natural mesh "frame" | membrane (or either phase) | the **network** (its material frame) |
| Incompressibility | mixture incompressible ⇒ a pressure | Biot pressure already present |

The crucial asymmetry: **UC-Poroelastic needs a reference configuration and a deformation
gradient; UC-Viscous does not.** A reference configuration means the mesh has to *be* (or
track) the network's material map from rest to current — i.e. the mesh follows the network
Lagrangianly, and `F` is read off the mesh deformation. That is the one place the "one
Eulerian mesh" story acquires solid-mechanics bookkeeping.

## 6. The one genuine fork — which phase the mesh follows

Because the mesh can follow only one velocity tangentially, and the others drift relative
to it via `relative_advection`, the real architectural decision is **which phase is the
mesh's frame**:

- **Mesh follows the membrane (lipids).** Membrane species are Lagrangian (carried, exact
  — what we have). Network *and* fluid are Eulerian relative to the mesh
  (`relative_advection` for both). Cortical/retrograde flow is then a relative-advection
  of the network. *Downside:* an elastic network's reference frame is now a moving-but-not-
  material frame — awkward for `F`. Best for **UC-Viscous**.
- **Mesh follows the network (cortex).** The network is Lagrangian ⇒ `F` is the mesh
  deformation from the reference mesh — exactly what elasticity wants. The membrane lipids
  and the fluid are Eulerian relative to the mesh (`relative_advection`, incl. the
  membrane species drifting under cortical flow). Best for **UC-Poroelastic**.

**This is the choice to keep abstract.** If "which phase the mesh follows" is a single
seam — the velocity handed to the mesh-motion solver — then switching frames is a
one-line change and both UCs reuse §4 wholesale.

## 7. Mesh topology — coupled submeshes vs. membrane-as-boundary

Orthogonal to §6, the membrane can be represented two ways (this predates multiphase; see
`approaches.md` A vs B):

- **Coupled submeshes (current, `coupled.py`):** membrane submesh + bulk, coincident
  nodes, entity-map coupling. Conservative and explicit. *Multiphase wrinkle:* if the
  membrane redistributes tangentially (BGN), the bulk-boundary nodes must redistribute
  *with it* to stay coincident — redistribution becomes a **shared interface** operation
  on both meshes. Doable, but a standing constraint.
- **Membrane-as-boundary (Approach A — single bulk mesh):** no submesh, no entity map, no
  coincidence constraint. Tangential boundary redistribution is just mesh motion. The
  multiphase fields are plain bulk fields; the membrane species are boundary **traces**
  (we already paid for the conservative trace-remap in `core/surface_remap_trace.py`).

For a multiphase *volume*, **membrane-as-boundary is the simpler home** — it removes the
coincidence constraint that fights tangential redistribution, at the cost of trace
coupling for the surface species. Worth deciding deliberately and early, because it is
expensive to change once physics is layered on.

## 8. Answers to the two open questions

**Is there one approach that keeps the door open for both formulations?** Yes — the §4
skeleton, with two seams kept abstract: (i) the **network constitutive stress** (a
swappable weak-form: viscous+active *or* elastic), and (ii) **which phase the mesh
follows** (§6). With those two behind interfaces, UC-Viscous and UC-Poroelastic are the
same code with a different stress plug-in and a different frame choice. Everything
expensive — the ALE mesh, the block force-balance, `relative_advection`, the drag, the
interface velocity BCs, the membrane coupling — is shared.

**Is there a solution that is much simpler if we commit?** Yes, and the simplification is
lopsided:

- **Commit to UC-Viscous (A1/B1): much simpler.** No reference configuration, no
  deformation gradient, no solid mechanics. Both phases are velocity fields; the whole
  model is *two overdamped Stokes/Brinkman fields + drag + a pressure*, on the ALE mesh,
  with `relative_advection` for transport — **almost entirely from pieces that already
  exist or are sketched** (weak-form escape hatch, `relative_advection`, block assembly,
  harmonic-extension mesh motion, BGN redistribution, remap-on-remesh). The new work is
  the interface velocity BCs (§4.6) and the saddle-point/pressure handling.
- **Commit to UC-Poroelastic (A2/B2): genuinely more.** Adds a reference configuration,
  hyperelastic stress `P(F)`, and the bookkeeping of a material frame — the one part that
  is *not* in the codebase in any form. But it is an **additive** extension of the
  viscous skeleton: swap the network stress, pin the mesh to the network frame, add the
  reference mesh.

So the low-risk path is: **build the viscous (UC-Viscous) skeleton first; treat
poroelasticity as a later constitutive swap rather than a parallel architecture.** That
choice keeps the door open (§4 is identical) while paying only for what the simpler model
needs now.

## 9. What is reused vs. new

| Piece | Status |
|---|---|
| One conforming ALE mesh + harmonic-extension motion | **have** (`discrete.py`, `coupled.py`) |
| Mixed-dimensional bulk↔surface coupling | **have** (`coupled.py`) |
| `relative_advection` — phase transport relative to the mesh | **have** (`assemble.py`) |
| Weak-form escape hatch — overdamped force balances | **have** (`weakform.py`) |
| Multi-field block solve (`MixedFunctionSpace` / `extract_blocks`) | **have** (`coupled.py`) |
| Tangential redistribution (membrane), remap-on-remesh | **have** (`bgn_curve*`, `core/` remaps) |
| Two coupled velocity blocks + interphase drag | **done** — `backend/multiphase.py` (`solve_two_phase_overdamped`), monolithic mixed-element block; verified (`test_backend_multiphase.py`) |
| **Interface velocity BCs** (normal-match / tangential slip), Nitsche or rotated frame | **first piece done** — `backend/slip.py` (`nitsche_normal_slip`, `solve_overdamped_slip`): perfect-slip `v·n = g`, free tangential, via Nitsche; **symmetric (L2-optimal) and non-symmetric penalty-free (no β to tune — robust for cut/weak-coercivity) variants** both verified (`test_backend_slip.py`). Stokes-traction / pressure variants pending |
| Incompressible-mixture **pressure** (saddle point, stable elements e.g. Taylor–Hood) | **single-phase done** — `backend/stokes.py` (`solve_incompressible_stokes`), Taylor–Hood P2/P1 + MUMPS, verified (`test_backend_stokes.py`); slip-with-Stokes-traction and the two-phase mixture pending |
| Membrane force balance loaded by cortex traction | **static coupling done** — `solve_incompressible_stokes_traction` (traction/Neumann BC); Laplace's law verified (`test_backend_stokes.py`). The *dynamic* moving-membrane loop is the remaining integration |
| **Exact mass conservation on a moving boundary** (the dynamic-FSI blocker) | **done** — `backend/stokes_hdiv.py` (`solve_incompressible_stokes_hdiv_slip`): the Taylor–Hood Nitsche slip leaks ~3% `div` on a *moving* boundary (the pressure-test in the Nitsche boundary term breaks `∫q∇·u=0`), even on H(div). Fix: an **H(div)** (BDM/DG) element — `∇·u=0` *pointwise* — with the slip BC `v·n=g` imposed **strongly** on the normal dofs (no Nitsche on the normal ⇒ no divergence pollution), tangential free. Interior-penalty DG for the (tangentially-discontinuous) viscous operator. Verified: `div` at round-off on the `cos2θ` bulging boundary at every `h`, and the fluid still slips (`test_backend_stokes_hdiv.py`). This is the "div-conforming flow element" targeted fix (`approaches.md`, discretization axis), not the full DG backend |
| **Dynamic FSI loop** (prescribed-motion foundation) | **done** — `backend/fsi.py` (`step_prescribed_fsi`): a moving membrane drives the conservative H(div) bulk (`v·n=w·n`), the ALE mesh follows (boundary by `dt·w`, interior harmonic). `∇·v` at round-off at *every* step of a deforming loop, and the enclosed volume conserved to O(dt) (vs ~3% for the leaky bulk; `test_backend_fsi.py`). **Consistency finding:** the prescribed motion must be volume-conserving on the *current* geometry (`∮w·n=0`) — a divergence-free `w` guarantees it; `cos2θ·n` (volume-conserving only on the circle) loses it once the boundary deforms and the incompressible solve becomes inconsistent. The force-balance closure (normal motion *solved*) sidesteps this — see the next row |
| **Dynamic FSI loop** (force-balance closure) | **done** — `backend/fsi.py` (`step_force_balance_fsi`) + `solve_incompressible_stokes_surface_tension` (`backend/stokes.py`): the membrane moves under its *own* surface tension `−γ∮_Γ∇_Γ·v ds` (curvature-free Laplace–Beltrami load, works on a piecewise-linear boundary) plus the bulk pressure — *no* prescribed motion. The pressure (the `∇·v=0` multiplier) enforces `∮v·n=0` itself, so the volume is conserved **automatically**; the consistency arrangement of the prescribed loop vanishes. On **Taylor–Hood** (continuous velocity), so the ALE mesh moves by the solved `v` directly with `∮v·n=0` exact (an H(div) velocity loses that when interpolated for mesh motion — measured 4% area drift vs Taylor–Hood's <1%). Verified (`test_backend_fsi.py`): a circle is a Laplace fixed point (`p=γ/R`, `v≈0`); a perturbed ellipse relaxes monotonically toward the minimal-perimeter circle at conserved area |
| Reference configuration + hyperelastic `P(F)` (poroelastic only) | new — deferred to the constitutive swap |

## 10. A staged, verifiable path (when we build it)

Each step is a known-answer check before the next is added:

1. **One overdamped fluid phase on a moving mesh + a normal-matching interface BC. ✓ done.**
   Proved the velocity-BC machinery and the mesh-vs-physical-velocity split. `backend/slip.py`
   — the perfect-slip Nitsche BC (`v·n = g`, free tangential, symmetric + non-symmetric
   variants) on a screened vector Laplacian; a manufactured field is recovered to round-off,
   the constraint tightens with the penalty, and a tangential forcing slips where no-slip
   kills it (`test_backend_slip.py`). The **integration** then composes the slip solve, the
   mesh motion, and `relative_advection`: each step solves `v` with `g = w·n`, transports a
   species by `relative_advection = v − w`, and moves the mesh — the velocity-solved
   rotating-mesh discriminator holds a lab-frame field static to the O(h) geometric floor
   (`test_backend_slip_moving.py`). The accuracy floor is the discrete-normal facet-leakage
   (a known limitation), confirmed to shrink under refinement.
2. **Add the second (network) phase + interphase drag** (block solve). ✓ done.
   `backend/multiphase.py` (`solve_two_phase_overdamped`): two overdamped velocity fields
   `(v_n, v_s)` with viscous + optional substrate friction + the symmetric interphase drag
   `ξ(v_a − v_b)`, each with its own normal-slip BC, assembled monolithically over a mixed
   element of two vector spaces. Verified (`test_backend_multiphase.py`): a manufactured
   two-field solution recovered to round-off (block + drag + slip assembly); increasing `ξ`
   **locks** the phases (slip `|v_n − v_s|` shrinks ~`1/ξ`); and `ξ = 0` decouples into the
   independent single-phase slip solves (the drag is the only coupling). `-ν∇²` still stands
   in for the viscous stress; the symmetric-gradient Stokes operator + pressure are step 3.
3. **Incompressible mixture pressure** (Taylor–Hood or stabilised). **3a ✓ done:** single
   incompressible Stokes — `backend/stokes.py` (`solve_incompressible_stokes`), the
   symmetric-gradient stress `2ν ε(u)` + a pressure Lagrange multiplier `∇·u = 0`, the first
   saddle-point system, on inf-sup-stable **Taylor–Hood** (P2/P1) elements with a pivoting
   (MUMPS) direct solve. Verified against a manufactured solution (div-free quadratic
   velocity + linear pressure recovered to round-off, `∇·u ≈ 0`; `test_backend_stokes.py`),
   with a strong Dirichlet velocity BC. **3b ✓ done:** the Nitsche normal-slip BC with
   the *Stokes* traction `n·σ·n = 2ν n·ε(u)·n − p` — the pressure (and the pressure test)
   enter the boundary terms — `solve_incompressible_stokes_slip`. Verified: a slip-compatible
   manufactured solution (constant `u` + linear `p`) recovered to round-off for both Nitsche
   variants; a well-posed (null-mode-orthogonal) slip flow is divergence-free to round-off;
   and the slip BC leaves a tangential boundary flow where no-slip kills it. *Caveat learnt:*
   a pure no-penetration BC on a rotationally-symmetric domain leaves **rigid rotation as a
   null mode** — a forcing aligned with it is inconsistent; a substrate-friction `screening`
   (present in the real overdamped model) removes it. **3c ✓ done:** the **two-phase
   incompressible mixture** — `solve_two_phase_stokes` (`backend/multiphase.py`): two
   velocity fields + one **mixture pressure** enforcing `∇·(v_n + v_s) = 0`, the interphase
   drag, and a per-phase Nitsche slip BC whose Stokes traction carries the *shared* pressure,
   over a `[P2, P2, P1]` Taylor–Hood element. Verified: a manufactured mixture (`v_n=[1,0]`,
   `v_s=[−1,0]` so the sum is divergence-free, linear pressure) recovered to round-off for
   both Nitsche variants, and `∇·(v_n + v_s)` zero to round-off on a generic flow. **Step 3
   complete.**
4. **Membrane mechanical coupling** — cortex traction loads the membrane force balance.
   **✓ done:** `solve_incompressible_stokes_traction` (`backend/stokes.py`) applies the
   membrane's surface-mechanics force on the enclosed fluid as a **traction (Neumann) BC**
   `σ·n = t` — the natural Stokes BC, so it enters only the RHS and also fixes the pressure
   level. Verified by **Laplace's law**: a tense membrane's curvature traction `−γ κ n`
   leaves the fluid at rest with the exact internal pressure `p = γ/R` (`test_backend_stokes.py`,
   to round-off across radii and tensions) — the membrane–bulk mechanical balance.
   With the bulk pressure now set by membrane tension, the staged momentum path is complete;
   the remaining work is the *dynamic* coupling (the moving-membrane ALE loop driving the
   two-phase bulk) and the constitutive/frame seams (§5–§6).
5. **(Later) Poroelastic swap** — reference configuration + `P(F)`, mesh pinned to the
   network frame. Verify: a poroelastic relaxation / Biot consolidation known solution.

## 11. Open decisions (to settle before step 1)

- **Two-fluid vs poroelastic as the *first* target.** This note recommends two-fluid
  (UC-Viscous) first, poroelastic as a constitutive extension.
- **Mesh topology** (§7): coupled submeshes vs membrane-as-boundary. Recommend deciding
  toward membrane-as-boundary for the multiphase volume, but it interacts with the
  surface-trace conservation work.
- **Which phase the mesh follows** (§6): membrane for viscous, network for elastic — keep
  it a single seam.
- **Incompressibility and stabilisation**: incompressible mixture ⇒ pressure ⇒ first
  saddle-point system in the codebase (element-stability and a Stokes-capable solver,
  beyond the current P1 + LU).
- **Active stress model**: how `ζQ` (orientation/activity) is represented — a prescribed
  field first, a solved order parameter later.

---

*Cross-references:* the mesh/physical-velocity split and the Eulerian volume term
(`relative_advection`) are in `declarative-formalism.md §3.6` and the "Lagrangian surface
vs Eulerian volume" note in `approaches.md`; the membrane mechanics (curvature/tension
force balance, weak-form escape hatch) are the `unknown_motion.py` / `weakform.py`
backend; Approach A (membrane-as-boundary) and the conservative trace remap are in
`approaches.md` and `conservative-surface-remap.md`.

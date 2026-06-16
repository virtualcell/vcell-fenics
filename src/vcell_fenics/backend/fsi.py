"""Prescribed-motion fluid–structure loop on a conservative (H(div)) bulk.

The dynamic integration of the multiphase path (`docs/modeling/multiphase-cytoplasm-ale.md`):
a moving membrane drives an incompressible bulk while the ALE mesh follows. Two steps:

- `step_prescribed_fsi` — the **foundation**: the membrane motion is *prescribed*, the bulk
  is the exactly mass-conserving H(div) Stokes (`backend/stokes_hdiv.py`), divergence-free
  as the domain deforms.
- `step_force_balance_fsi` — the **closure**: the membrane moves under its *own* surface
  tension and the bulk pressure (no prescribed motion). On Taylor–Hood (continuous velocity)
  so the ALE mesh can move by the solved `v` directly with `∮ v·n = 0` exact (an H(div)
  velocity loses that when interpolated for the mesh — see `step_force_balance_fsi`).
- `step_two_phase_fsi` — the closure for a **two-phase** (network + solvent) incompressible
  mixture with interphase drag: the membrane tension loads the mixture and the mesh follows a
  chosen *frame* phase (§6 of the design note). The mixture-average frame conserves volume via
  the shared mixture pressure, just as the single-phase closure does.
- `step_two_phase_fsi_with_species` — the same, **carrying a co-moving volume species** through
  the moving cell. The volume is Eulerian, so the species rides a physical phase while the mesh
  moves at the bookkeeping velocity; the transport is `relative_advection` plus the bulk
  dilution `c ∇·v_carrier` (the volume analogue of the mandatory surface `ρ ∇_Γ·v_Γ`).

Each step:

  1. solve the conservative H(div) fluid with `v·n = w·n` (the fluid follows the moving
     boundary normally, slips tangentially) — exactly divergence-free;
  2. advance the ALE mesh by `dt·w` — boundary nodes by the prescribed velocity, the
     interior filled by harmonic extension.

**The volume-conservation condition is an artefact of *prescribing* the motion, not a
constraint to babysit.** An incompressible bulk forces `∮ v·n = 0`, and the BC ties
`v·n = w·n`, so a *prescribed* `w` with non-zero net normal flux makes the two demands
contradict — the problem is genuinely **inconsistent** (no solution), and the solve blows
up. That is over-determination, not numerical fragility, and the cure is a `w` that is
actually consistent: a **divergence-free** `w` guarantees `∮ w·n = 0` on *any* shape
(`∮ w·n = ∫ ∇·w = 0`), whereas a `w` volume-conserving only on the initial shape (e.g.
`cos2θ·n` on a circle) loses that once the boundary deforms.

In the real model the condition **enforces itself**: in the force-balance closure the
normal motion is *solved*, and the pressure — the Lagrange multiplier for `∇·v = 0` — builds
up to whatever makes `∮ v·n = 0`, so the enclosed volume is conserved *automatically and
exactly*, with no source and no tuning. (Adding genuine water transport later generalises
the balance to `dV/dt = ∫ s` for a volume source `s`, or `dV/dt = −∮ J` for a transmembrane
flux `J` — conserved physics that *relaxes* the constraint when present, never slack that
hides a leak; with `s = 0` this reduces to the strict, exactly-conservative case here.)

Verified (`tests/test_backend_fsi.py`): the fluid is divergence-free to round-off at every
step of a deforming loop, and the enclosed volume is conserved (to the O(dt) forward-Euler
integration error, which refines with dt), where the Taylor–Hood Nitsche slip leaked ~3%.
"""

from __future__ import annotations

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from scipy.spatial import cKDTree

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.discrete import _HarmonicExtension
from vcell_fenics.backend.multiphase import solve_two_phase_stokes_surface_tension
from vcell_fenics.backend.stokes import solve_incompressible_stokes_surface_tension
from vcell_fenics.backend.stokes_hdiv import solve_incompressible_stokes_hdiv_slip


def _harmonic_displacement(mesh: Mesh, velocity: UflExpr, dt: float) -> fem.Function:
    """The ALE mesh displacement `dt·w`: `dt·velocity` on the boundary, harmonically extended
    into the interior. `w = displacement/dt` is the **mesh velocity** field — the geometric
    bookkeeping velocity, which the species transport needs as the frame to advect relative to.
    `velocity` is any UFL expression (a solved `Function` or e.g. `0.5·(v_n + v_s)`)."""
    gdim = mesh.geometry.dim
    lagrange = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    displacement = fem.Function(lagrange)
    displacement.interpolate(fem.Expression(dt * velocity, lagrange.element.interpolation_points))
    _HarmonicExtension(lagrange).fill(displacement)  # boundary kept, interior harmonic
    return displacement


def _apply_displacement(mesh: Mesh, displacement: fem.Function) -> None:
    """Move `mesh` in place by a P1 vector `displacement` field (permuting dof order → geometry
    node order, since they differ for a P1 vector field)."""
    gdim = mesh.geometry.dim
    dof_coords = displacement.function_space.tabulate_dof_coordinates()
    geom_from_dof = cKDTree(dof_coords).query(mesh.geometry.x)[1]
    mesh.geometry.x[:, :gdim] += displacement.x.array.reshape(-1, gdim)[geom_from_dof]


def _advance_ale_mesh(mesh: Mesh, velocity: UflExpr, dt: float) -> None:
    """Move `mesh` in place by `dt·velocity`: boundary nodes by the velocity directly, the
    interior by harmonic extension (the ALE fill)."""
    _apply_displacement(mesh, _harmonic_displacement(mesh, velocity, dt))


def step_prescribed_fsi(
    mesh: Mesh,
    *,
    boundary_velocity: fem.Function,
    dt: float,
    viscosity: float = 1.0,
    screening: float = 1.0,
) -> tuple[fem.Function, fem.Function]:
    """Advance one prescribed-motion FSI step, in place, returning the fluid `(v, p)`.

    Solves the conservative H(div) bulk for `v` (`v·n = boundary_velocity·n`, free
    tangential), then moves `mesh` by `dt·boundary_velocity` (boundary directly, interior by
    harmonic extension). `boundary_velocity` is a vector `Function`; because the motion is
    *prescribed* here, it must be consistent with incompressibility — `∮ w·n = 0` on the
    current geometry (use a divergence-free field); otherwise the bulk solve is
    over-determined. (When the motion is *solved* — the force-balance closure — the pressure
    enforces `∮ v·n = 0` itself and the volume is conserved automatically; see the module
    docstring.) Returns the exactly divergence-free `v` and its pressure `p`.
    """

    v, p = solve_incompressible_stokes_hdiv_slip(
        mesh, boundary_velocity=boundary_velocity, viscosity=viscosity, screening=screening
    )
    _advance_ale_mesh(mesh, boundary_velocity, dt)
    return v, p


def step_force_balance_fsi(
    mesh: Mesh,
    *,
    tension: float,
    dt: float,
    viscosity: float = 1.0,
    screening: float = 1.0,
) -> tuple[fem.Function, fem.Function]:
    """Advance one **force-balance** FSI step, in place — the membrane moves under its *own*
    surface tension and the bulk pressure, with no prescribed motion.

    Solves the surface-tension Stokes (`backend/stokes.py`) for the fluid `v`, then moves the
    membrane (and ALE mesh) by `dt·v`. The membrane is the material boundary, so it is carried
    by the fluid; the pressure (the `∇·v = 0` multiplier) enforces `∮ v·n = 0`, so the enclosed
    volume is **conserved automatically** — no prescribed-motion consistency to arrange. A
    circle is a fixed point (Laplace `p = γ/R`, `v ≈ 0`); a perturbed shape relaxes toward the
    minimal-perimeter circle at conserved area. Returns the fluid `(v, p)`.
    """

    v, p = solve_incompressible_stokes_surface_tension(mesh, tension=tension, viscosity=viscosity, screening=screening)
    _advance_ale_mesh(mesh, v, dt)
    return v, p


def step_two_phase_fsi(
    mesh: Mesh,
    *,
    tension: float,
    drag: float,
    dt: float,
    viscosity_n: float = 1.0,
    viscosity_s: float = 1.0,
    screening_n: float = 1.0,
    screening_s: float = 1.0,
    frame: str = "mixture",
) -> tuple[fem.Function, fem.Function, fem.Function]:
    """Advance one **two-phase** force-balance FSI step, in place — a tense membrane bounding a
    two-phase (network + solvent) incompressible mixture, with no prescribed motion.

    Solves the surface-tension two-phase Stokes mixture (`backend/multiphase.py`) for the phase
    velocities `(v_n, v_s)` and the mixture pressure `p`, then moves the ALE mesh by `dt` times
    the **frame** velocity. `drag` is the interphase friction `ξ(v_n − v_s)`; `frame` selects
    which velocity carries the mesh — the *one genuine fork* of the multiphase model
    (`docs/modeling/multiphase-cytoplasm-ale.md` §6):

    - `"mixture"` (default) — the volume-averaged `0.5·(v_n + v_s)`. The mixture pressure
      enforces `∮(v_n + v_s)·n = 0`, so `∮ v_mix·n = 0` and the enclosed **volume is conserved
      automatically**, exactly as in the single-phase closure.
    - `"network"` / `"solvent"` — follow `v_n` / `v_s` (the membrane attached to the cortex, or
      to the solvent). Volume is then conserved only insofar as that phase's normal flux
      vanishes; use when one phase is the material boundary.

    A circle is a fixed point (Laplace `p = γ/R`, both phases at rest, drag inactive); a
    perturbed shape relaxes toward the minimal-perimeter circle. Returns `(v_n, v_s, p)`.
    """

    v_n, v_s, p = solve_two_phase_stokes_surface_tension(
        mesh,
        tension=tension,
        drag=drag,
        viscosity_n=viscosity_n,
        viscosity_s=viscosity_s,
        screening_n=screening_n,
        screening_s=screening_s,
    )
    _advance_ale_mesh(mesh, _phase_velocity(frame, v_n, v_s), dt)
    return v_n, v_s, p


def _phase_velocity(phase: str, v_n: fem.Function, v_s: fem.Function) -> UflExpr:
    """Select a velocity by phase name: the volume-averaged `"mixture"`, the `"network"` `v_n`,
    or the `"solvent"` `v_s`. Raises on an unknown name."""
    velocity = {"mixture": 0.5 * (v_n + v_s), "network": v_n, "solvent": v_s}.get(phase)
    if velocity is None:
        raise ValueError(f"phase must be 'mixture', 'network', or 'solvent', not {phase!r}")
    return velocity


def step_two_phase_fsi_with_species(
    mesh: Mesh,
    species: fem.Function,
    *,
    tension: float,
    drag: float,
    dt: float,
    diffusivity: float,
    carrier: str = "mixture",
    frame: str = "mixture",
    viscosity_n: float = 1.0,
    viscosity_s: float = 1.0,
    screening_n: float = 1.0,
    screening_s: float = 1.0,
) -> tuple[fem.Function, fem.Function, fem.Function]:
    """Advance one two-phase FSI step **carrying a co-moving volume species** `c`, in place.

    Solves the surface-tension mixture (`(v_n, v_s, p)`), transports the species, then moves the
    ALE mesh by the `frame` velocity. The volume is **Eulerian** — it has no material points, so
    the species rides a *physical phase* (`carrier`) while the mesh moves at the bookkeeping
    velocity `w` (the harmonic extension of the `frame` velocity). The ALE transport is

        ∂c/∂t|_mesh + (v_carrier − w)·∇c + c ∇·v_carrier = D ∇²c

    — the **relative advection** `(v_carrier − w)·∇c` (`backend/assemble.py`'s term, here on the
    solved flow) plus the **bulk dilution** `c ∇·v_carrier`, the volume analogue of the
    mandatory surface dilution `ρ ∇_Γ·v_Γ`. With `carrier = "mixture"` the carrier is
    divergence-free so the dilution vanishes and `∫c` is conserved; with a single phase
    (`"network"`/`"solvent"`) the dilution is what keeps `∫c` conserved as that phase compresses.

    `species` is a scalar P1 `Function` on `mesh`, mutated in place (its current values are the
    backward-Euler previous step). `diffusivity` is `D`; `carrier`/`frame` are phase names
    (`"mixture"`/`"network"`/`"solvent"`). Returns `(v_n, v_s, p)`. A no-flux (natural) boundary
    is assumed; when the carrier matches the frame, `v_carrier·n = w·n` on the boundary so there
    is no transmembrane flux and `∫c` is conserved exactly.
    """

    v_n, v_s, p = solve_two_phase_stokes_surface_tension(
        mesh,
        tension=tension,
        drag=drag,
        viscosity_n=viscosity_n,
        viscosity_s=viscosity_s,
        screening_n=screening_n,
        screening_s=screening_s,
    )
    carrier_velocity = _phase_velocity(carrier, v_n, v_s)
    displacement = _harmonic_displacement(mesh, _phase_velocity(frame, v_n, v_s), dt)
    mesh_velocity = displacement / dt  # w = dt·w / dt

    space = species.function_space
    c, q = ufl.TrialFunction(space), ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=mesh)
    relative_advection = carrier_velocity - mesh_velocity
    a = (
        c / dt * q
        + diffusivity * ufl.inner(ufl.grad(c), ufl.grad(q))
        + ufl.dot(relative_advection, ufl.grad(c)) * q  # (v_carrier − w)·∇c
        + c * ufl.div(carrier_velocity) * q  # bulk dilution c ∇·v_carrier
    ) * dx
    rhs = species / dt * q * dx

    updated = fem.Function(space)
    LinearProblem(
        a,
        rhs,
        u=updated,
        petsc_options_prefix=f"vcellfenics_fsispecies_{id(updated):x}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    ).solve()
    species.x.array[:] = updated.x.array  # in place: the new step becomes next step's previous
    _apply_displacement(mesh, displacement)
    return v_n, v_s, p


def enclosed_volume(mesh: Mesh) -> float:
    """The volume (area in 2D) the mesh encloses — the conserved quantity of a
    volume-preserving FSI loop."""
    from mpi4py import MPI

    local = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))

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
from dolfinx.mesh import Mesh
from scipy.spatial import cKDTree

from vcell_fenics.backend.discrete import _HarmonicExtension
from vcell_fenics.backend.stokes import solve_incompressible_stokes_surface_tension
from vcell_fenics.backend.stokes_hdiv import solve_incompressible_stokes_hdiv_slip


def _advance_ale_mesh(mesh: Mesh, velocity: fem.Function, dt: float) -> None:
    """Move `mesh` in place by `dt·velocity`: boundary nodes by the velocity directly, the
    interior by harmonic extension (the ALE fill)."""
    gdim = mesh.geometry.dim
    lagrange = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    displacement = fem.Function(lagrange)
    displacement.interpolate(fem.Expression(dt * velocity, lagrange.element.interpolation_points))
    _HarmonicExtension(lagrange).fill(displacement)  # boundary kept, interior harmonic
    geom_from_dof = cKDTree(lagrange.tabulate_dof_coordinates()).query(mesh.geometry.x)[1]
    mesh.geometry.x[:, :gdim] += displacement.x.array.reshape(-1, gdim)[geom_from_dof]


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


def enclosed_volume(mesh: Mesh) -> float:
    """The volume (area in 2D) the mesh encloses — the conserved quantity of a
    volume-preserving FSI loop."""
    from mpi4py import MPI

    local = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))

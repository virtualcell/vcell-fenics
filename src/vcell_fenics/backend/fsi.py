"""Prescribed-motion fluid–structure loop on a conservative (H(div)) bulk.

The dynamic integration of the multiphase path (`docs/modeling/multiphase-cytoplasm-ale.md`):
a moving membrane drives an incompressible bulk through the slip BC while the ALE mesh
follows. This is the *foundation* step — the membrane motion is **prescribed** (the
force-balance closure, where it is solved, is the next increment) — and it is built on the
exactly mass-conserving H(div) Stokes (`backend/stokes_hdiv.py`) so the bulk stays
divergence-free as the domain deforms.

Each step:

  1. solve the conservative H(div) fluid with `v·n = w·n` (the fluid follows the moving
     boundary normally, slips tangentially) — exactly divergence-free;
  2. advance the ALE mesh by `dt·w` — boundary nodes by the prescribed velocity, the
     interior filled by harmonic extension.

**Consistency requirement (a real FSI constraint, not a numerical nicety):** the prescribed
boundary motion must be **volume-conserving on the *current* geometry**, `∮ w·n = 0`. An
incompressible bulk forces `∮ v·n = 0`, so a `w` with non-zero net normal flux has *no*
consistent fluid and the solve blows up. A **divergence-free** `w` guarantees `∮ w·n = 0`
on *any* shape (`∮ w·n = ∫ ∇·w = 0`); a `w` that is volume-conserving only on the initial
shape (e.g. `cos2θ·n` on a circle) loses that once the boundary deforms. The force-balance
closure sidesteps this — there the normal motion is *solved* and is automatically
consistent.

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
from vcell_fenics.backend.stokes_hdiv import solve_incompressible_stokes_hdiv_slip


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
    harmonic extension). `boundary_velocity` is a vector `Function`; **it must be
    volume-conserving on the current geometry** (`∮ w·n = 0` — use a divergence-free field).
    Returns the exactly divergence-free `v` and its pressure `p`.
    """

    v, p = solve_incompressible_stokes_hdiv_slip(
        mesh, boundary_velocity=boundary_velocity, viscosity=viscosity, screening=screening
    )

    gdim = mesh.geometry.dim
    lagrange = fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    displacement = fem.Function(lagrange)
    displacement.interpolate(fem.Expression(dt * boundary_velocity, lagrange.element.interpolation_points))
    _HarmonicExtension(lagrange).fill(displacement)  # boundary kept, interior harmonic
    geom_from_dof = cKDTree(lagrange.tabulate_dof_coordinates()).query(mesh.geometry.x)[1]
    mesh.geometry.x[:, :gdim] += displacement.x.array.reshape(-1, gdim)[geom_from_dof]
    return v, p


def enclosed_volume(mesh: Mesh) -> float:
    """The volume (area in 2D) the mesh encloses — the conserved quantity of a
    volume-preserving FSI loop."""
    from mpi4py import MPI

    local = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh)))
    return float(mesh.comm.allreduce(local, op=MPI.SUM))

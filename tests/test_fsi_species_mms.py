"""Method-of-manufactured-solutions convergence check for the FSI **co-moving volume species**
transport (`backend/fsi._transport_co_moving_species`, the operator inside
`step_two_phase_fsi_with_species`).

The existing FSI species tests (`test_backend_fsi.py`) are strong on **conservation**, **equilibria**,
and **discrimination** — but conservation is *blind to a wrong-but-conservative operator* (exactly how
the interface-flux over-count survived for the whole history of the coupled BE solver, since equilibration
never pinned an absolute flux). This test pins the transport to a known analytical solution and measures
its **spatial convergence order**, so a sign error / missing term / mis-scaled dilution in the real code
fails the build even though `∫c` might still be conserved.

Design (isolates the transport from the Stokes solve so the velocity is exactly known):
- **Manufactured solution** — steady lab-frame `u*(x) = 2 + sin(x)·cos(y)`.
- **Carrier velocity** `v = [0.3·x, 0.1·y]` (∇·v = 0.4) **≠ frame velocity** `w = [0.25·x, 0.25·y]`, so *both*
  the relative advection `(v − w)·∇c` **and** the bulk dilution `c ∇·v` (the volume analogue of the
  mandatory surface `ρ ∇_Γ·v_Γ`) are active — every transport term is exercised. Both fields are linear, so
  the ALE mesh's *harmonic* extension reproduces the frame velocity exactly and `w` is closed-form.
- The mesh-velocity `w` **cancels analytically** in the manufactured forcing (the transport is by the
  carrier): `f = ∇·(u*·v) − D∇²u* = v·∇u* + u*·∇·v − D∇²u*` — independent of the frame. Time-dependent
  Dirichlet `u*` at the moving boundary; `D = 0.1`; `dt ∝ h²` so the O(dt) strided-motion floor stays below
  the O(h²) spatial error.

On the straight-boundary structured box the P1 order is a clean O(h²) (measured ≈ 2.0). This is step 2 of
the FSI verification plan; the nested-disk (curved) variant and a decaying `exp(-t)·…` variant are natural
follow-ups, as is a fixed-domain Stokes-order check (step 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices
from mpi4py import MPI

from vcell_fenics.backend.fsi import _harmonic_displacement, _transport_co_moving_species

_D = 0.1
_T_FINAL = 1.0
_RESOLUTIONS_H = [0.1, 0.05, 0.025]


def _ustar(x: np.ndarray) -> np.ndarray:
    """The steady lab-frame manufactured solution u*(x) = 2 + sin(x)·cos(y)."""
    return np.asarray(2.0 + np.sin(x[0]) * np.cos(x[1]), dtype=float)


def _linf_error_at(h: float, *, dilution_in_source: bool = True) -> float:
    """Advance the FSI co-moving species to t_final on a box refined at h, return the nodal L∞ error vs u*.

    `dilution_in_source=False` drops the dilution's contribution `u*·∇·v = 0.8 + 0.4·sin(x)cos(y)` from the
    manufactured forcing — a **negative control**: since the real operator still carries `c ∇·v`, u* is then
    no longer the exact solution and the error must stop converging, proving the positive test pins that term."""
    nx = round(2.0 / h)
    mesh = create_rectangle(
        MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [nx, nx], CellType.triangle
    )
    space = fem.functionspace(mesh, ("Lagrange", 1))
    c = fem.Function(space)
    c.interpolate(_ustar)  # initial condition = u*

    mesh.topology.create_connectivity(1, 2)
    boundary_dofs = fem.locate_dofs_topological(space, 1, exterior_facet_indices(mesh.topology))
    boundary_value = fem.Function(space)
    bc = fem.dirichletbc(boundary_value, boundary_dofs)

    x = ufl.SpatialCoordinate(mesh)  # symbolic — re-evaluated on the moved mesh each assemble
    carrier = ufl.as_vector([0.3 * x[0], 0.1 * x[1]])  # ∇·v = 0.4, ≠ frame
    frame = ufl.as_vector([0.25 * x[0], 0.25 * x[1]])  # → mesh velocity w (linear ⇒ harmonic ext exact)
    source = (
        0.3 * x[0] * ufl.cos(x[0]) * ufl.cos(x[1])
        - 0.1 * x[1] * ufl.sin(x[0]) * ufl.sin(x[1])
        + 0.8
        + 0.6 * ufl.sin(x[0]) * ufl.cos(x[1])
    )  # f = ∇·(u*·v) − D∇²u*, with D = 0.1
    if not dilution_in_source:
        source = source - (0.8 + 0.4 * ufl.sin(x[0]) * ufl.cos(x[1]))  # drop u*·∇·v (negative control)

    dt = 0.5 * h * h  # first-order in dt (BE + strided motion) ⇒ refine dt ∝ h² to expose the O(h²) order
    for _ in range(round(_T_FINAL / dt)):
        boundary_value.interpolate(_ustar)  # refresh Dirichlet u* at the current (moved) boundary
        displacement = _harmonic_displacement(mesh, frame, dt)
        _transport_co_moving_species(
            mesh, c, carrier_velocity=carrier, displacement=displacement, dt=dt, diffusivity=_D, source=source, bcs=[bc]
        )

    coords = space.tabulate_dof_coordinates()  # final (moved) node positions, shape (npoints, 3)
    return float(np.abs(c.x.array - _ustar(coords.T)).max())


@pytest.mark.integration
def test_fsi_co_moving_species_transport_is_second_order() -> None:
    """The FSI co-moving species ALE transport holds its P1 spatial order (O(h²)) against a manufactured
    solution — a wrong-but-conservative operator (sign/scale/missing term) would collapse this order."""
    errors = [_linf_error_at(h) for h in _RESOLUTIONS_H]
    order = math.log(errors[0] / errors[-1]) / math.log(_RESOLUTIONS_H[0] / _RESOLUTIONS_H[-1])
    assert order > 1.6, f"FSI species transport order {order:.2f} (errors {errors}) — expected ≈ 2 (O(h²))"


@pytest.mark.integration
def test_fsi_species_mms_discriminates_the_dilution_term() -> None:
    """Negative control giving the O(h²) test teeth: drop the dilution's contribution from the manufactured
    forcing while the real operator keeps `c ∇·v`. u* is then not the exact solution, so the error must NOT
    converge — a mis-scaled/missing/sign-flipped dilution in the real code cannot masquerade as passing."""
    two = [0.1, 0.05]  # two coarse resolutions suffice to show non-convergence (cheap)
    errors = [_linf_error_at(h, dilution_in_source=False) for h in two]
    order = math.log(errors[0] / errors[-1]) / math.log(two[0] / two[-1])
    assert order < 0.5, f"dilution negative control converged (order {order:.2f}, errors {errors}) — no teeth"

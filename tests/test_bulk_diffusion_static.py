"""Bulk diffusion on a fixed disk with no-flux (Neumann) BC.

Validates:
- Constant initial condition stays constant (Neumann sanity).
- Total mass is conserved (no flux out, no reaction).
- A single Bessel eigenmode J_0(λr) — with λR equal to the first zero of
  J_1, the Neumann eigenvalue condition — decays at the analytical rate
  exp(-D λ² t). This is the bulk analogue of the cos(kθ) surface-decay
  test in test_surface_diffusion_static.py.
"""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from scipy.special import j0, jn_zeros

from vcell_fenics.approaches.static import BulkPDE, create_disk


def test_constant_initial_condition_stays_constant():
    disk = create_disk(radius=1.0, h=0.2)
    pde = BulkPDE(disk.mesh, D=0.5, dt=0.05)
    pde.set_initial(3.0)
    for _ in range(20):
        pde.step()
    assert np.allclose(pde.c.x.array, 3.0, atol=1e-10)


def test_mass_conserved_without_reaction():
    disk = create_disk(radius=1.0, h=0.1)
    pde = BulkPDE(disk.mesh, D=0.2, dt=0.01)

    def ic(x):
        return 1.0 + 0.3 * x[0]  # arbitrary smooth nonconstant IC

    pde.set_initial(ic)
    M0 = pde.total_mass()
    for _ in range(50):
        pde.step()
    M1 = pde.total_mass()
    assert abs(M1 - M0) / abs(M0) < 1e-10


def test_bessel_eigenmode_decays_at_analytical_rate():
    R, D, dt, n_steps = 1.0, 0.1, 0.01, 50
    T = dt * n_steps

    # Neumann eigenvalue: smallest λ s.t. J_0'(λR) = -J_1(λR) = 0.
    # I.e. λR is the first positive zero of J_1.
    alpha = jn_zeros(1, 1)[0]
    lam = alpha / R

    disk = create_disk(radius=R, h=0.05)
    pde = BulkPDE(disk.mesh, D=D, dt=dt)

    def eigenmode(x):
        r = np.sqrt(x[0] ** 2 + x[1] ** 2)
        return j0(lam * r)

    # IC = eigenmode (mass-neutral, since ∫_disk J_0(λr) dΩ = 0 at this λ).
    pde.set_initial(eigenmode)

    # Build a Function holding the eigenmode for projection onto.
    phi = fem.Function(pde.V)
    phi.interpolate(eigenmode)

    amp0 = pde.project_onto(phi)
    for _ in range(n_steps):
        pde.step()
    amp1 = pde.project_onto(phi)

    expected = np.exp(-D * lam**2 * T)
    observed = amp1 / amp0

    assert abs(observed - expected) / expected < 0.02, (
        f"expected decay {expected:.4f}, observed {observed:.4f}"
    )

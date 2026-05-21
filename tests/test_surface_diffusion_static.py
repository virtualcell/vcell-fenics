"""v0: pure surface diffusion on a static circular membrane (Approach B).

A closed 1D manifold has no boundary, so the surface Laplacian has no flux
out of the domain — total mass ∫_Γ ρ dΓ must be conserved exactly (up to
linear-solver tolerance) for ∂_t ρ = D Δ_Γ ρ.
"""

from __future__ import annotations

import numpy as np

from vcell_fenics.approaches.submesh import create_disk_with_membrane, SurfacePDE


def test_constant_rho_stays_constant():
    dm = create_disk_with_membrane(radius=1.0, h=0.2)
    pde = SurfacePDE(dm.submesh, D=0.5, dt=0.05)
    pde.set_initial(2.5)
    for _ in range(20):
        pde.step()
    assert np.allclose(pde.rho.x.array, 2.5, atol=1e-10)


def test_diffusion_matches_analytical_decay_and_conserves_mass():
    """A cos(kθ) mode on a circle of radius r decays as exp(-D k²/r² · t).

    This both verifies the surface-Laplacian assembly (eigenvalue is correct)
    and that mass is conserved on a closed manifold.
    """
    r, k, D, dt, n_steps = 1.0, 2, 0.1, 0.01, 50
    T = dt * n_steps
    expected_decay = np.exp(-D * (k**2) / (r**2) * T)

    dm = create_disk_with_membrane(radius=r, h=0.05)
    pde = SurfacePDE(dm.submesh, D=D, dt=dt)

    amp0 = 0.5

    def ic(x):
        theta = np.arctan2(x[1], x[0])
        return 1.0 + amp0 * np.cos(k * theta)

    pde.set_initial(ic)
    M0 = pde.total_mass()
    amp_initial = 0.5 * (pde.rho.x.array.max() - pde.rho.x.array.min())

    for _ in range(n_steps):
        pde.step()

    M1 = pde.total_mass()
    amp_final = 0.5 * (pde.rho.x.array.max() - pde.rho.x.array.min())
    observed_decay = amp_final / amp_initial

    assert abs(M1 - M0) / M0 < 1e-10, f"mass not conserved: M0={M0}, M1={M1}"
    # Backward Euler + piecewise-linear mesh adds a few % error; loosen tol.
    assert abs(observed_decay - expected_decay) / expected_decay < 0.02, (
        f"expected decay {expected_decay:.4f}, observed {observed_decay:.4f}"
    )

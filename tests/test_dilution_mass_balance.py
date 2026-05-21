"""v2: prescribed-motion + ρ ∇_Γ · v_Γ dilution.

This is the canonical sanity check for any moving-membrane surface-density
code. Per docs/modeling/approaches.md, omitting the dilution term is the
single most common bug in this problem class, so this test is structured
as a *discriminator*: it asserts both that the correct code conserves mass,
AND that without dilution the same configuration would fail.

Setup: uniform initial ρ_0 on a circle of radius r_0, with prescribed
radial expansion r(t) = r_0 + ṙ t. Surface stretch rate is ∇_Γ · v_Γ = ṙ/r.

Analytical solution (no diffusion):
    ∂_t ρ + ρ ṙ/r = 0  ⇒  ρ(t) = ρ_0 · r_0/r(t)
    M(t) = ρ(t) · 2π r(t) = 2π ρ_0 r_0 = M(0).
"""

from __future__ import annotations

from dolfinx import fem

from vcell_fenics.approaches.submesh import create_disk_with_membrane, SurfacePDE
from vcell_fenics.approaches.submesh.geometry import scale_radially


def _run_expansion(*, with_dilution: bool, h: float = 0.1, dt: float = 0.01):
    dm = create_disk_with_membrane(radius=1.0, h=h)
    r0, r_dot, T = 1.0, 1.0, 1.0
    n_steps = int(round(T / dt))

    div_v_fn = None
    if with_dilution:
        V = fem.functionspace(dm.submesh, ("Lagrange", 1))
        div_v_fn = fem.Function(V, name="div_v_gamma")

    pde = SurfacePDE(dm.submesh, D=0.0, dt=dt, div_v_gamma=div_v_fn)
    pde.set_initial(1.0)
    M0 = pde.total_mass()

    r = r0
    for i in range(n_steps):
        r_new = r0 + r_dot * dt * (i + 1)
        scale_radially(dm.submesh, r_new / r)
        if with_dilution:
            # Uniform radial expansion ⇒ ∇_Γ · v_Γ = ṙ / r everywhere on Γ.
            # Backward-Euler evaluates the dilution coefficient at t^{n+1}.
            div_v_fn.x.array[:] = r_dot / r_new
        r = r_new
        pde.step()

    return M0, pde.total_mass(), pde.surface_length()


def test_mass_conserved_with_dilution_under_uniform_stretch():
    M0, M, L = _run_expansion(with_dilution=True)
    # Membrane doubled in length; mass should stay put up to BE O(dt) error.
    assert abs(L / (2 * 3.141592653589793 * 2.0) - 1.0) < 1e-3, (
        f"sanity: expected L ≈ 2π·r(T)=4π, got {L}"
    )
    assert abs(M - M0) / M0 < 0.02, f"M/M0 - 1 = {M / M0 - 1.0:.4f}"


def test_omitting_dilution_corrupts_mass():
    """Negative control: without ρ ∇_Γ · v_Γ, mass grows ∝ stretch.

    Confirms the test above is actually exercising the dilution term, not
    merely passing because the problem is trivially mass-conserving.
    """
    M0, M, L = _run_expansion(with_dilution=False)
    # Without dilution, ρ is unchanged, so M = ρ_0 · L(T) = 2 · M0.
    assert abs(M / M0 - 2.0) < 1e-3

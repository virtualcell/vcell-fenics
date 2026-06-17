"""Inc-1 end-to-end: static surface diffusion driven through the formalism.

The bespoke equivalent is tests/test_surface_diffusion_static.py; here the same
physics is expressed as a `surface_pde_with_dilution` MathDescription on a static
membrane and run through the backend. A static membrane has no motion, so no
dilution term — just ∂_t ρ = D Δ_Γ ρ — which the structural check confirms. The
cos(kθ) eigenmode decays at the analytical rate and mass is conserved on the
closed manifold, exactly as the bespoke test asserts.
"""

from __future__ import annotations

import numpy as np

from vcell_fenics.backend import TermKind, assemble, make_disk_membrane_geometry
from vcell_fenics.formalism import load_yaml


def _surface_model(*, diffusion: str, ic: str) -> str:
    return f"""
math_description:
  geometry: disk_membrane
  subdomains:
    - name: membrane
      kind: surface
      motion: {{ kind: none }}
  variables:
    - {{ name: rho, subdomain: membrane }}
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      terms:
        diffusion: "{diffusion}"
      initial_condition: "{ic}"
"""


def test_static_surface_has_no_dilution_term() -> None:
    md = load_yaml(_surface_model(diffusion="0.1", ic="2.5"))
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.2)
    dp = assemble(md, geometry, dt=0.05)
    # Static membrane: diffusion only, no DILUTION (the canonical moving-membrane bug).
    assert dp.term_kinds() == {TermKind.TIME_DERIVATIVE, TermKind.DIFFUSION}


def test_constant_surface_density_stays_constant() -> None:
    md = load_yaml(_surface_model(diffusion="0.5", ic="2.5"))
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.2)
    dp = assemble(md, geometry, dt=0.05)
    assert np.allclose(dp.unknown.x.array, 2.5)
    for _ in range(20):
        dp.step()
    assert np.allclose(dp.unknown.x.array, 2.5, atol=1e-10)


def test_cos_mode_decays_at_analytical_rate_and_conserves_mass() -> None:
    # A cos(kθ) mode on a circle of radius r decays as exp(-D k²/r² · t); mass
    # is conserved on the closed manifold. Mirrors test_surface_diffusion_static.
    r, k, D, dt, n_steps = 1.0, 2, 0.1, 0.01, 50
    T = dt * n_steps
    expected_decay = np.exp(-D * (k**2) / (r**2) * T)

    md = load_yaml(_surface_model(diffusion=str(D), ic="1.0 + 0.5 * cos(2 * geom.azimuth)"))
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=r, h=0.05)
    dp = assemble(md, geometry, dt=dt)

    m0 = dp.total_mass()
    amp_initial = 0.5 * (dp.unknown.x.array.max() - dp.unknown.x.array.min())
    for _ in range(n_steps):
        dp.step()
    m1 = dp.total_mass()
    amp_final = 0.5 * (dp.unknown.x.array.max() - dp.unknown.x.array.min())

    assert abs(m1 - m0) / m0 < 1e-10, f"mass not conserved: {m0} -> {m1}"
    observed_decay = amp_final / amp_initial
    assert abs(observed_decay - expected_decay) / expected_decay < 0.02, (
        f"expected decay {expected_decay:.4f}, observed {observed_decay:.4f}"
    )

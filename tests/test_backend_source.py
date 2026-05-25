"""Inc-3a: single-variable source slot, verified against an analytical decay.

A `source` slot that is linear in the governed variable lands in the implicit
backward-Euler bilinear form (compiled with the variable bound to the trial
function). The cleanest check is a spatially-uniform linear decay: with a
constant IC and no-flux diffusion, ∂_t c = D Δc − k c reduces to ∂_t c = −k c,
so c stays uniform and decays as c0·exp(−k t). This pins both the source's
presence and its sign (a sign error would grow the field instead).
"""

from __future__ import annotations

import math

import numpy as np

from vcell_fenics.backend import TermKind, assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml

_DECAY = """
math_description:
  geometry: disk_2d
  subdomains:
    - name: cytoplasm
      kind: volume
  variables:
    - { name: c, subdomain: cytoplasm }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cytoplasm
      temporality: time_dependent
      terms:
        diffusion: "0.5"
        source: "-k * c"
      initial_condition: "2.0"
  parameters:
    - { name: k, value: 0.3 }
"""


def test_source_slot_adds_a_source_term() -> None:
    md = load_yaml(_DECAY)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)
    dp = assemble(md, geometry, dt=0.01)
    assert dp.term_kinds() == {TermKind.TIME_DERIVATIVE, TermKind.DIFFUSION, TermKind.SOURCE}


def test_linear_decay_matches_analytical() -> None:
    md = load_yaml(_DECAY)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)
    dt, n_steps, k, c0 = 0.01, 100, 0.3, 2.0
    dp = assemble(md, geometry, dt=dt)

    for _ in range(n_steps):
        dp.step()

    expected = c0 * math.exp(-k * dt * n_steps)  # spatially uniform exponential decay
    assert np.allclose(dp.unknown.x.array, expected, rtol=2e-2), (
        f"expected ≈ {expected:.4f}, got range [{dp.unknown.x.array.min():.4f}, {dp.unknown.x.array.max():.4f}]"
    )

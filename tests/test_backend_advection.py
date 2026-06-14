"""Verification of the `relative_advection` slot — the Eulerian volume term (§1.4 T1).

`relative_advection` is the species' drift *relative to the substrate/mesh*. It is what
makes a volume species **Eulerian**: where the dilution term `ρ ∇_Γ·v` is the Lagrangian
(co-moving) treatment, advection lets the medium move independently of the mesh.

1. **Static-mesh Eulerian fluid** — the formalism's documented Eulerian setup
   (`motion: none`, fluid velocity in `relative_advection`): a bump advects at exactly
   the prescribed velocity (its centre of mass moves at `U`).
2. **Eulerian medium on a moving mesh — the discriminator.** The volume has no material
   points, so a lab-frame field must stay put while the mesh sweeps through it. On a
   *rotating* (divergence-free, hence dilution-free) mesh, setting `relative_advection =
   −v_mesh` holds a lab-frame field static; the co-moving treatment (no advection) instead
   rotates the field with the mesh. This is the `(u − w)·∇c` term the moving-boundary
   volume needs, with `u = 0` (still medium).
3. **Structural** — the slot adds an `ADVECTION` term; omitting it does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend import DiscreteProblem, TermKind, assemble, make_disk_geometry
from vcell_fenics.formalism import load_yaml


def _bulk_model(*, motion: str, terms: str, ic: str) -> str:
    return f"""
math_description:
  geometry: g
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ {motion} }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {{ {terms} }}
      initial_condition: "{ic}"
"""


def test_static_mesh_advects_at_the_prescribed_velocity() -> None:
    # motion: none + fluid velocity in relative_advection = the Eulerian-fluid setup.
    # A Gaussian bump's centre of mass moves at exactly U.
    u = 0.4
    model = _bulk_model(
        motion="kind: none",
        terms=f'diffusion: "0.002", relative_advection: "[{u}, 0.0]"',
        ic="exp(-((x[0]+0.4)*(x[0]+0.4) + x[1]*x[1])/0.05)",
    )
    dp = assemble(load_yaml(model), make_disk_geometry("g", volume_subdomain="cyto", radius=1.5, h=0.04), dt=0.01)
    coords_x = dp.V.tabulate_dof_coordinates()[:, 0]

    def _com_x() -> float:
        c = dp.unknown.x.array
        return float((c * coords_x).sum() / c.sum())

    x0 = _com_x()
    for _ in range(50):
        dp.step()
    assert (_com_x() - x0) / (0.01 * 50) == pytest.approx(u, rel=2e-2)


def test_eulerian_field_stays_put_while_mesh_rotates_through_it() -> None:
    # The discriminator: a lab-frame field on an Eulerian (still) medium must not move
    # when the mesh rotates through it. With relative_advection = −v_mesh it stays put;
    # co-moving (no advection) carries the field around with the mesh.
    omega = 0.6
    rotation = f"{omega} * [-x[1], x[0]]"  # divergence-free ⇒ no dilution to confound the test

    def _max_error_vs_lab_frame(relative_advection: str | None) -> float:
        terms = 'diffusion: "0.0"'
        if relative_advection is not None:
            terms += f', relative_advection: "{relative_advection}"'
        model = _bulk_model(motion=f'kind: prescribed, velocity: "{rotation}"', terms=terms, ic="1.0 + 0.3*x[0]")
        dp = assemble(load_yaml(model), make_disk_geometry("g", volume_subdomain="cyto", radius=1.0, h=0.05), dt=0.01)
        for _ in range(40):
            dp.step()
        coords = dp.V.tabulate_dof_coordinates()[:, :2]
        lab_field = 1.0 + 0.3 * coords[:, 0]  # the static lab field, at each node's CURRENT position
        return float(np.abs(dp.unknown.x.array - lab_field).max())

    eulerian = _max_error_vs_lab_frame(f"{omega} * [x[1], -x[0]]")  # −v_mesh
    comoving = _max_error_vs_lab_frame(None)
    assert eulerian < 0.01  # the lab-frame field is held static as the mesh rotates
    assert comoving > 0.05  # the co-moving field rotates with the mesh — order of magnitude worse
    assert comoving > 10 * eulerian


def test_relative_advection_adds_an_advection_term() -> None:
    geom = make_disk_geometry("g", volume_subdomain="cyto", radius=1.0, h=0.3)
    advecting = _bulk_model(motion="kind: none", terms='diffusion: "0.1", relative_advection: "[1.0, 0.0]"', ic="1.0")
    plain = _bulk_model(motion="kind: none", terms='diffusion: "0.1"', ic="1.0")
    with_adv = assemble(load_yaml(advecting), geom, dt=0.01)
    without = assemble(load_yaml(plain), geom, dt=0.01)
    assert isinstance(with_adv, DiscreteProblem)
    assert TermKind.ADVECTION in with_adv.term_kinds()
    assert TermKind.ADVECTION not in without.term_kinds()

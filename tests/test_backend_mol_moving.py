"""Method-of-lines on a moving (ALE) subdomain — `integrate_discrete_problem_moving`.

PETSc `TS` integrates a *fixed* spatial operator; an ALE mesh moves it in time. The driver
mirrors the backward-Euler `step` (move the mesh discretely, then solve) but integrates each
inter-move stride with the adaptive-BDF `TS` instead of one backward-Euler step — first-order in
the mesh motion (frozen within a stride), high-order/adaptive in the reaction-diffusion. The
checks pin the three physical regimes against closed-form answers:

1. **Rigid translation + diffusion** — a constant velocity preserves area, so mass is conserved
   exactly and the centroid tracks `v·T`, while the interior gradient diffuses in the moving frame.
2. **Dilution under expansion** — `velocity = x` grows the disk (`∇·v = 2`); the uniform field must
   dilute as `e^{−2T}` (the mandatory `ρ∇·v` term) with mass conserved, and the split error shrinks
   as the stride count rises.
3. **Stiff reaction under translation** — a stiff decay is integrated to the analytic `e^{−kT}` with
   only a handful of mesh strides (the adaptive sub-stepping absorbs the stiffness), the centroid
   still tracking `v·T`. This is the regime that motivates MOL over backward Euler here.

Plus the API guards: the fixed-domain integrator rejects a moving subdomain and vice-versa.
"""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import (
    DiscreteProblem,
    assemble,
    integrate_discrete_problem,
    integrate_discrete_problem_moving,
    make_disk_geometry,
)
from vcell_fenics.formalism import load_yaml


def _model(velocity: str, *, diffusion: str = "0.0", source: str | None = None, ic: str = "1.0") -> str:
    terms = f'{{ diffusion: "{diffusion}"' + (f', source: "{source}"' if source else "") + " }"
    return f"""
math_description:
  geometry: disk
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: prescribed, velocity: "{velocity}" }} }}
  variables:
    - {{ name: c, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: {terms}
      initial_condition: "{ic}"
"""


def _problem(velocity: str, *, h: float = 0.13, **model_kwargs: str) -> DiscreteProblem:
    geom = make_disk_geometry("disk", volume_subdomain="cyto", radius=1.0, h=h)
    return assemble(load_yaml(_model(velocity, **model_kwargs)), geom, dt=0.01)


def _area(dp: DiscreteProblem) -> float:
    return float(dp.V.mesh.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * dp.dx)), op=MPI.SUM))


def _centroid_x(dp: DiscreteProblem) -> float:
    return float(dp.V.mesh.geometry.x[:, 0].mean())


# ---------------------------------------------------------------------------
# 1. rigid translation + diffusion: area & mass conserved, centroid = v·T
# ---------------------------------------------------------------------------


def test_translation_with_diffusion_conserves_mass_and_translates() -> None:
    dp = _problem("[0.5, 0.0]", diffusion="0.1", ic="1.0 + 0.3*geom.x[0]")
    x0, m0, a0 = _centroid_x(dp), dp.total_mass(), _area(dp)
    spread0 = float(dp.unknown.x.array.max() - dp.unknown.x.array.min())

    result = integrate_discrete_problem_moving(dp, t_final=1.0, motion_steps=20)

    assert _centroid_x(dp) - x0 == pytest.approx(0.5, rel=1e-9)  # centroid tracks v·T
    assert _area(dp) == pytest.approx(a0, rel=1e-9)  # translation preserves area (∇·v = 0)
    assert dp.total_mass() == pytest.approx(m0, rel=1e-9)  # ⇒ no spurious dilution
    assert float(dp.unknown.x.array.max() - dp.unknown.x.array.min()) < 0.7 * spread0  # gradient diffused
    assert result.time == pytest.approx(1.0)
    assert result.steps > 20  # the adaptive TS sub-stepped within the strides


# ---------------------------------------------------------------------------
# 2. dilution under expansion: c → e^{−2T}, mass conserved, split error → 0
# ---------------------------------------------------------------------------


def test_expansion_dilutes_the_field_as_volume_grows() -> None:
    # velocity = x ⇒ the disk expands (r → r·e^t, area → e^{2t}); a uniform field must dilute as
    # e^{−2t} via the ρ∇·v term, conserving total mass. This is the decisive dilution check.
    t_final = 0.5
    dp = _problem("geom.x", diffusion="0.0", ic="1.0")
    m0 = dp.total_mass()

    integrate_discrete_problem_moving(dp, t_final=t_final, motion_steps=40)

    assert dp.unknown.x.array.mean() == pytest.approx(np.exp(-2.0 * t_final), rel=2e-3)  # diluted e^{−2T}
    assert dp.total_mass() == pytest.approx(m0, rel=1e-2)  # mass conserved (first-order split)
    assert _area(dp) > 2.0 * (np.pi)  # the disk genuinely grew (area ≈ π·e^{2T})


def test_dilution_split_error_shrinks_with_more_strides() -> None:
    # The mesh is frozen within a stride, so mass conservation is first-order in the stride length:
    # refining motion_steps must reduce the mass drift. (The win of MOL is *within* the stride; the
    # mesh-motion split is the controllable first-order part.)
    def mass_drift(strides: int) -> float:
        dp = _problem("geom.x", diffusion="0.0", ic="1.0")
        m0 = dp.total_mass()
        integrate_discrete_problem_moving(dp, t_final=0.5, motion_steps=strides)
        return abs(dp.total_mass() - m0) / m0

    coarse, fine = mass_drift(10), mass_drift(40)
    assert fine < coarse  # refining the motion split tightens conservation
    assert coarse / fine > 2.5  # ~first-order: 4× the strides ⇒ markedly smaller drift


# ---------------------------------------------------------------------------
# 3. stiff reaction under translation — the regime that motivates MOL
# ---------------------------------------------------------------------------


def test_stiff_decay_under_translation_is_accurate_with_few_strides() -> None:
    # A stiff decay (k = 20: the field decays ~150× over the run, and an explicit method would need
    # dt < 2/k = 0.1 just for stability) while translating. The adaptive BDF absorbs the stiffness
    # inside each stride, so only 8 mesh strides reach the analytic e^{−kT} (to ~0.2 %) — a fixed
    # backward-Euler dt of t_final/8 = 0.03 would carry a visible first-order time error — and the
    # centroid still tracks v·T.
    k, v, t_final = 20.0, 0.4, 0.25
    dp = _problem("[0.4, 0.0]", diffusion="0.0", source=f"-{k}*c", ic="2.0")
    x0 = _centroid_x(dp)

    result = integrate_discrete_problem_moving(dp, t_final=t_final, motion_steps=8)

    assert dp.unknown.x.array.mean() == pytest.approx(2.0 * np.exp(-k * t_final), rel=1e-2)  # analytic decay
    assert _centroid_x(dp) - x0 == pytest.approx(v * t_final, rel=1e-9)  # centroid tracks v·T
    assert result.steps > 8  # the strides sub-stepped to resolve the stiffness


# ---------------------------------------------------------------------------
# 4. API guards — each integrator rejects the other's domain
# ---------------------------------------------------------------------------


def test_fixed_domain_integrator_rejects_a_moving_subdomain() -> None:
    dp = _problem("[0.5, 0.0]", diffusion="0.1")
    with pytest.raises(NotImplementedError, match="integrate_discrete_problem_moving"):
        integrate_discrete_problem(dp, t_final=0.1)


def test_moving_integrator_rejects_a_static_subdomain() -> None:
    geom = make_disk_geometry("disk", volume_subdomain="cyto", radius=1.0, h=0.2)
    static = """
math_description:
  geometry: disk
  subdomains:
    - { name: cyto, kind: volume }
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.1" }
      initial_condition: "1.0"
"""
    dp = assemble(load_yaml(static), geom, dt=0.01)
    with pytest.raises(NotImplementedError, match="moving subdomain"):
        integrate_discrete_problem_moving(dp, t_final=0.1, motion_steps=4)


def test_moving_integrator_requires_positive_motion_steps() -> None:
    dp = _problem("[0.5, 0.0]", diffusion="0.1")
    with pytest.raises(ValueError, match="motion_steps must be >= 1"):
        integrate_discrete_problem_moving(dp, t_final=0.1, motion_steps=0)

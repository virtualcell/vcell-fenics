"""Inc-2 end-to-end: prescribed membrane motion + auto-dilution.

Reproduces tests/test_dilution_mass_balance.py through the formalism. A membrane
under uniform radial expansion r(t) = r0 + ṙt carries a surface density ρ. The
mandatory dilution term ρ ∇_Γ·v_Γ (§1.4.2) is added automatically whenever the
subdomain moves, with ∇_Γ·v_Γ taken as div of the velocity expression; the
DiscreteProblem advances the mesh by dt·v each step.

Three checks:

1. **Structural** — a prescribed-motion model assembles with a DILUTION term and
   a motion velocity; a static one does not (covered in the surface-diffusion test).
2. **Positive (formalism-driven)** — with auto-dilution, total mass is conserved
   under expansion even as the membrane doubles in length.
3. **Negative control (IR-level)** — the same moving problem *without* the DILUTION
   term lets mass grow ∝ stretch (≈ doubles). This is the canonical moving-membrane
   bug; the formalism makes it unreachable (dilution is automatic), so the control
   is built directly on the IR to show the term is load-bearing.
"""

from __future__ import annotations

import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend import (
    BackwardEuler,
    DiscreteProblem,
    Term,
    TermKind,
    assemble,
    make_disk_membrane_geometry,
)
from vcell_fenics.formalism import load_yaml

_MOVING_MEMBRANE = """
math_description:
  geometry: disk_membrane
  subdomains:
    - name: membrane
      kind: surface
      motion:
        kind: prescribed
        velocity: "r_dot * x / r(x)"
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      initial_condition: "1.0"
  parameters:
    - { name: r_dot, value: 1.0 }
"""


def _measure(dp: DiscreteProblem) -> float:
    """The total measure (length, here) of the membrane's current configuration."""
    local = fem.assemble_scalar(fem.form(1.0 * dp.dx))
    return float(dp.V.mesh.comm.allreduce(local, op=MPI.SUM))


# ---------------------------------------------------------------------------
# 1. Structural — motion implies a DILUTION term.
# ---------------------------------------------------------------------------


def test_prescribed_motion_adds_dilution_term() -> None:
    md = load_yaml(_MOVING_MEMBRANE)
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.2)
    dp = assemble(md, geometry, dt=0.01)
    assert dp.term_kinds() == {TermKind.TIME_DERIVATIVE, TermKind.DILUTION}
    assert dp.motion_velocity is not None


# ---------------------------------------------------------------------------
# 2. Positive — auto-dilution conserves mass under expansion.
# ---------------------------------------------------------------------------


def test_expansion_conserves_mass_through_formalism() -> None:
    md = load_yaml(_MOVING_MEMBRANE)
    geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=1.0, h=0.1)
    dp = assemble(md, geometry, dt=0.01)

    length0 = _measure(dp)
    mass0 = dp.total_mass()
    for _ in range(100):  # ṙ = 1, T = 1 ⇒ r: 1 → 2
        dp.step()

    # The membrane doubled in length (the motion really happened)...
    assert _measure(dp) / length0 == pytest.approx(2.0, abs=2e-2)
    # ...and mass is conserved to backward-Euler O(dt) error thanks to dilution.
    assert abs(dp.total_mass() - mass0) / mass0 < 0.02


# ---------------------------------------------------------------------------
# 3. Negative control — omitting dilution lets mass grow with stretch.
# ---------------------------------------------------------------------------


def _expansion_problem(*, with_dilution: bool, h: float = 0.1, dt: float = 0.01) -> DiscreteProblem:
    mesh = make_disk_membrane_geometry("g", surface_subdomain="m", radius=1.0, h=h).mesh_of("m")
    V = fem.functionspace(mesh, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    x = ufl.SpatialCoordinate(mesh)
    velocity = 1.0 * x / ufl.sqrt(ufl.dot(x, x))  # ṙ = 1, radial unit field

    terms = [Term(TermKind.TIME_DERIVATIVE)]
    if with_dilution:
        terms.append(Term(TermKind.DILUTION, ufl.div(velocity) * trial * test))

    dp = DiscreteProblem(
        variable_name="rho",
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name="rho"),
        previous=fem.Function(V, name="rho_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(dt)),  # type: ignore[operator]
        terms=tuple(terms),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=velocity,
    )
    dp.set_initial(1.0)
    return dp


def test_omitting_dilution_doubles_mass() -> None:
    # Without ρ ∇_Γ·v_Γ, ρ is unchanged while the membrane doubles, so
    # M = ρ0 · L(T) ≈ 2 · M0 — the bug the dilution term prevents.
    dp = _expansion_problem(with_dilution=False)
    mass0 = dp.total_mass()
    for _ in range(100):
        dp.step()
    assert abs(dp.total_mass() / mass0 - 2.0) < 0.05

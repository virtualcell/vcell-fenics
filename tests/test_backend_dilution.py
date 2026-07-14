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

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend import (
    BackwardEuler,
    DiscreteProblem,
    MeshQualityError,
    Term,
    TermKind,
    assemble,
    make_disk_geometry,
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
        velocity: "r_dot * geom.x / geom.radius"
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


# ---------------------------------------------------------------------------
# 4. Mesh-quality guard — a tangling motion fails loudly.
# ---------------------------------------------------------------------------


def test_degenerate_motion_fails_loudly() -> None:
    # A strong inward contraction (unit inward velocity, dt = 0.5) drives the
    # membrane from r = 1 to r = 0 in two steps — every cell collapses to zero
    # length. The backend moves nodes but never remeshes, so the quality guard
    # must raise rather than silently solve on a collapsed mesh. (Uniform
    # *dilation*, by contrast, preserves cell-size ratios and never trips it —
    # see test_expansion_conserves_mass_through_formalism.)
    mesh = make_disk_membrane_geometry("g", surface_subdomain="m", radius=1.0, h=0.2).mesh_of("m")
    V = fem.functionspace(mesh, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    x = ufl.SpatialCoordinate(mesh)
    contraction = -x / ufl.sqrt(ufl.dot(x, x))  # inward unit velocity → r shrinks to 0

    dp = DiscreteProblem(
        variable_name="rho",
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name="rho"),
        previous=fem.Function(V, name="rho_old"),
        dt=fem.Constant(mesh, PETSc.ScalarType(0.5)),  # type: ignore[operator]
        terms=(Term(TermKind.TIME_DERIVATIVE), Term(TermKind.DILUTION, ufl.div(contraction) * trial * test)),
        scheme=BackwardEuler(),
        bcs=[],
        motion_velocity=contraction,
    )
    dp.set_initial(1.0)

    with pytest.raises(MeshQualityError):
        for _ in range(5):
            dp.step()


# ---------------------------------------------------------------------------
# 5. Translation — a non-normal prescribed velocity transports rigidly.
# ---------------------------------------------------------------------------

_TRANSLATING_MEMBRANE = """
math_description:
  geometry: g
  subdomains:
    - { name: membrane, kind: surface, motion: { kind: prescribed, velocity: "[0.5, 0.0]" } }
  variables:
    - { name: rho, subdomain: membrane }
  equations:
    - template: surface_pde_with_dilution
      variable: rho
      subdomain: membrane
      temporality: time_dependent
      initial_condition: "1.0 + 0.5*cos(2*geom.azimuth)"
"""


def test_translation_transports_rigidly_with_no_spurious_dilution() -> None:
    # A constant velocity [0.5, 0] is *tangential*, not normal — the worry was whether
    # the moving-surface machinery silently assumes normal motion. It does not: a rigid
    # translation has ∇_Γ·v = 0, so the dilution term contributes nothing, the membrane
    # length is unchanged, ∫_Γ ρ ds is conserved, and each (material) node carries its ρ
    # value. This is the membrane checked against its static co-moving-frame solution.
    geometry = make_disk_membrane_geometry("g", surface_subdomain="membrane", radius=1.0, h=0.05)
    dp = assemble(load_yaml(_TRANSLATING_MEMBRANE), geometry, dt=0.01)
    mesh = dp.V.mesh

    center0 = mesh.geometry.x[:, :2].mean(axis=0)
    rho0 = dp.unknown.x.array.copy()
    mass0, length0 = dp.total_mass(), _measure(dp)
    for _ in range(50):
        dp.step()

    center1 = mesh.geometry.x[:, :2].mean(axis=0)
    assert center1[0] - center0[0] == pytest.approx(0.5 * 0.01 * 50)  # translated by v·t
    assert abs(center1[1] - center0[1]) < 1e-12  # no drift off-axis
    assert _measure(dp) == pytest.approx(length0, rel=1e-12)  # rigid: length unchanged
    assert dp.total_mass() == pytest.approx(mass0, rel=1e-12)  # no spurious dilution
    assert np.abs(dp.unknown.x.array - rho0).max() < 1e-10  # each material node keeps its ρ


# ---------------------------------------------------------------------------
# 6. Bulk — the conservative ALE time term conserves mass exactly, for any motion.
# ---------------------------------------------------------------------------

_DISTORTING_BULK = """
math_description:
  geometry: disk_2d
  subdomains:
    - name: cyto
      kind: volume
      motion:
        kind: prescribed
        velocity: "(1.0 + 0.7 * cos(2 * geom.azimuth)) * geom.x / geom.radius"
  variables:
    - { name: c, subdomain: cyto }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cyto
      temporality: time_dependent
      terms: { diffusion: "0.0" }
      initial_condition: "1.0"
"""


def _bulk_mass_drift(dt: float, *, t_final: float = 0.6) -> float:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.10)
    dp = assemble(load_yaml(_DISTORTING_BULK), geometry, dt=dt)
    mass0 = dp.total_mass()
    for _ in range(round(t_final / dt)):
        dp.step()
    return abs(dp.total_mass() - mass0) / mass0


def test_nonaffine_bulk_motion_conserves_mass_to_roundoff() -> None:
    # The conservative ALE time term (BackwardEuler) rescales the carried previous field per cell by the
    # actual swept-volume ratio |Kⁿ|/|Kⁿ⁺¹|, so dilution lives in the changing measure and mass is
    # conserved to solver precision for *any* bulk motion — including this non-affine one
    # ((1 + 0.7 cos 2θ) radial, where the interior moves by the harmonic extension and ∇·v_mesh varies
    # cell to cell). Because the ratio is the true volume change of the harmonically-moved cells,
    # conservation is exact and dt-independent — no O(dt) geometric-conservation-law drift, a strictly
    # stronger guarantee than the earlier advective (uⁿ⁺¹−uⁿ)·w + ρ∇·v form's small-but-nonzero drift.
    coarse = _bulk_mass_drift(0.02)
    fine = _bulk_mass_drift(0.01)

    assert coarse < 1e-11  # machine precision — not a dt-convergent drift
    assert fine < 1e-11
    assert abs(fine - coarse) < 1e-11  # dt-independent: refining the step does not change the (zero) drift

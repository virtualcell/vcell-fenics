"""Verification of the method-of-lines RDA integrator (backend/reaction_diffusion).

`integrate_reaction_diffusion` turns the FEM-discretised reaction–diffusion–advection system
into an implicit ODE `M ċ = G(c)` and integrates it with PETSc `TS` (adaptive BDF) — the
method-of-lines strategy of VCell's finite-volume solver (which uses SUNDIALS/CVODE), here with
a GMRES + ILU inner solve.

1.  **Diffusion eigenmode + the implicit-stiffness win** — `cos(πx)` decays at the analytical
    rate `exp(−Dπ²t)`, in far fewer steps than an explicit `dt < h²/2D` stability bound forces.
2.  **Conservation + equilibrium** — a conservative `A ⇌ B` holds `∫(A+B)` to round-off and
    relaxes to detailed balance `A/B = k_off/k_on`.
3.  **Nonlinear reaction** — a bilinear `A + B ⇌ C` reaches `C/(A·B) = k_f/k_r` (the inner
    Newton handles the nonlinearity).
4.  **Advection** — a bump advects at the prescribed speed.
5.  **Integrator choice** — Crank–Nicolson integrates a non-stiff decay to tight accuracy.
6.  **Preconditioner choice** — GMRES + ILU and a direct LU give the same answer.
"""

from __future__ import annotations

import math

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from vcell_fenics.backend import integrate_reaction_diffusion


def _unit_square(n: int, components: int):  # type: ignore[no-untyped-def]
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    return mesh, fem.functionspace(mesh, ("Lagrange", 1, (components,)))


def _integral(expr, mesh) -> float:  # type: ignore[no-untyped-def]
    return float(fem.assemble_scalar(fem.form(expr * ufl.dx(domain=mesh))).real)


def test_diffusion_eigenmode_decays_at_the_analytical_rate() -> None:
    # cos(πx) is a Neumann eigenmode of −∇² on the unit square (eigenvalue π²), so it decays as
    # exp(−Dπ²t). The implicit integrator reaches t in far fewer steps than an explicit scheme,
    # whose diffusion stability limit dt < h²/2D would force ~hundreds of steps.
    n, diffusivity, t_final = 64, 0.5, 0.1
    mesh, space = _unit_square(n, 1)
    c = fem.Function(space)
    c.interpolate(lambda x: np.array([np.cos(math.pi * x[0])]))
    result = integrate_reaction_diffusion(mesh, c, diffusivities=[diffusivity], t_final=t_final)

    exact = fem.Function(space)
    decay = math.exp(-diffusivity * math.pi**2 * result.time)
    exact.interpolate(lambda x: np.array([decay * np.cos(math.pi * x[0])]))
    l2_error = float(_integral((c[0] - exact[0]) ** 2, mesh) ** 0.5)
    assert l2_error < 5e-3  # spatial-(P1)-limited accuracy at the analytical decay rate

    explicit_stability_steps = t_final / ((1.0 / n) ** 2 / (2.0 * diffusivity))
    assert result.steps < explicit_stability_steps / 3  # the implicit / adaptive-BDF win


def test_conservative_reaction_conserves_total_and_reaches_equilibrium() -> None:
    # A ⇌ B (k_on·A − k_off·B): conservative, so ∫(A+B) is invariant; the steady state is the
    # detailed-balance ratio A/B = k_off/k_on.
    k_on, k_off = 2.0, 0.5
    mesh, space = _unit_square(12, 2)
    c = fem.Function(space)
    values = c.x.array.reshape(-1, 2)
    values[:, 0], values[:, 1] = 1.0, 0.3
    total0 = _integral(c[0], mesh) + _integral(c[1], mesh)

    def reaction(u):  # type: ignore[no-untyped-def]
        r = k_on * u[0] - k_off * u[1]
        return ufl.as_vector([-r, r])

    integrate_reaction_diffusion(mesh, c, diffusivities=[0.1, 0.1], t_final=3.0, reaction=reaction)
    total = _integral(c[0], mesh) + _integral(c[1], mesh)
    assert abs(total - total0) / total0 < 1e-10  # ∫(A+B) conserved to round-off
    assert abs(_integral(c[0], mesh) / _integral(c[1], mesh) - k_off / k_on) < 1e-2  # detailed balance


def test_nonlinear_reaction_reaches_bilinear_equilibrium() -> None:
    # A + B ⇌ C with rate k_f·A·B − k_r·C — bilinear (nonlinear), handled by the TS inner Newton.
    # Reaches detailed balance C/(A·B) = k_f/k_r.
    k_forward, k_reverse = 2.0, 0.5
    mesh, space = _unit_square(10, 3)
    c = fem.Function(space)
    values = c.x.array.reshape(-1, 3)
    values[:, 0], values[:, 1], values[:, 2] = 1.0, 1.0, 0.0

    def reaction(u):  # type: ignore[no-untyped-def]
        r = k_forward * u[0] * u[1] - k_reverse * u[2]
        return ufl.as_vector([-r, -r, r])

    integrate_reaction_diffusion(mesh, c, diffusivities=[0.0, 0.0, 0.0], t_final=20.0, reaction=reaction)
    a, b, conc_c = (float(values[:, k].mean()) for k in range(3))
    assert abs(conc_c / (a * b) - k_forward / k_reverse) < 1e-2  # nonlinear detailed balance


def test_advection_transports_at_the_prescribed_speed() -> None:
    # A Gaussian bump advected by a uniform velocity (u, 0) has its centre of mass move at speed
    # u over a short time (before it reaches the no-flux boundary).
    mesh, space = _unit_square(60, 1)
    c = fem.Function(space)
    c.interpolate(lambda x: np.array([np.exp(-80.0 * ((x[0] - 0.3) ** 2 + (x[1] - 0.5) ** 2))]))
    coordinate = ufl.SpatialCoordinate(mesh)

    def centre_of_mass() -> float:
        return _integral(c[0] * coordinate[0], mesh) / _integral(c[0], mesh)

    speed = 0.5
    start = centre_of_mass()
    result = integrate_reaction_diffusion(
        mesh, c, diffusivities=[0.0], t_final=0.4, advection=fem.Constant(mesh, (speed, 0.0))
    )
    measured = (centre_of_mass() - start) / result.time
    assert abs(measured - speed) < 0.05  # bump advects at the prescribed speed


def test_crank_nicolson_integrates_a_decay_to_tight_accuracy() -> None:
    # The integrator is correct; on a non-stiff problem Crank–Nicolson reaches near-exact
    # accuracy (the BDF default trades some accuracy for L-stability, which matters when stiff).
    k, t_final, c0 = 2.0, 1.0, 3.0
    mesh, space = _unit_square(8, 1)
    c = fem.Function(space)
    c.x.array[:] = c0
    result = integrate_reaction_diffusion(
        mesh, c, diffusivities=[0.0], t_final=t_final, reaction=lambda u: ufl.as_vector([-k * u[0]]), ts_type="cn"
    )
    assert abs(float(c.x.array.mean()) - c0 * math.exp(-k * result.time)) < 1e-3


def test_ilu_and_lu_inner_solvers_agree() -> None:
    # The inner linear solver is configurable; GMRES + ILU (the scalable default, as in VCell's
    # SPGMR + ILU) and a direct LU give the same answer — ILU is a preconditioner, not a model
    # change. (ILU is what keeps the per-step cost low as the system grows.)
    def run_eigenmode(ksp_type: str, pc_type: str) -> fem.Function:
        mesh, space = _unit_square(40, 1)
        c = fem.Function(space)
        c.interpolate(lambda x: np.array([np.cos(math.pi * x[0])]))
        integrate_reaction_diffusion(mesh, c, diffusivities=[0.5], t_final=0.1, ksp_type=ksp_type, pc_type=pc_type)
        return c

    ilu = run_eigenmode("gmres", "ilu")
    lu = run_eigenmode("preonly", "lu")
    assert float(np.abs(ilu.x.array - lu.x.array).max()) < 1e-9  # preconditioner ⇒ same solution


def test_validates_diffusivity_count() -> None:
    mesh, space = _unit_square(4, 2)
    c = fem.Function(space)
    with pytest.raises(ValueError, match="diffusivities has length"):
        integrate_reaction_diffusion(mesh, c, diffusivities=[0.1], t_final=0.1)

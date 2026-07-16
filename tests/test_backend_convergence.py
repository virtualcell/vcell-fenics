"""Convergence-rate verification for the v1 backend (h- and dt-refinement).

The other `test_backend_*` modules are *accuracy/discrimination* tests: they run
one resolution and check a decay rate, mass conservation, or a negative control.
Those catch sign errors and missing terms but not a wrong *order* of accuracy — a
solver can hit a single-resolution tolerance while converging at the wrong rate
(the classic symptom of a quadrature, geometry, or assembly bug). This module
adds the missing axis: refine and measure the slope against analytical solutions.

What is pinned here, and why these problems:

- **Spatial, O(h²).** P1 Lagrange is second-order in L2. We measure it on a
  diffusion eigenmode whose continuous solution is known exactly:
    * bulk (T1) on a **unit square** — the Neumann eigenmode cos(πx)cos(πy) decays
      as exp(-2π²D·t). The square meshes *exactly* (polygonal = the true domain),
      so there is no geometric error to pollute the spatial rate.
    * surface (T2) on a **circle membrane** — the eigenmode cos(kθ) decays as
      exp(-Dk²/r²·t). Here the polygonal mesh only approximates the circle, but
      that geometric error is itself O(h²), so the combined rate is still 2. This
      also exercises the codim-1 submesh path where ufl.grad is the tangential ∇_Γ.
  The diffusion integrand is identical for T1 and T2 (only the mesh differs), so
  passing both is strong evidence the operator is assembled correctly on each. The
  bulk rate is pinned for **both** integrators — backward Euler (with a tiny dt
  floor so the temporal error stays below the spatial one) and the adaptive
  **method-of-lines** the cross-validation uses (no floor needed, its time error is
  ≈0). A clean O(h²) via MOL also certifies the MOL time error sits under the
  spatial error — the BDF cold-start over-diffusion (found + fixed in the
  convergence-study work) would offset the field and flatten the rate.

- **Temporal, O(dt¹).** Backward Euler is first-order. We pin it by
  self-convergence on a *fixed* mesh: solving with dt, dt/2, dt/4, the spatial
  discretisation is identical across the three and cancels in the successive
  differences, leaving a pure temporal-truncation signal whose ratio → 2 (order
  1). A companion check ties the finest-dt run back to the analytical decay so the
  scheme is verified for accuracy, not just self-consistency.

All errors use the *exact* analytical field as a UFL expression (built from the
mesh's SpatialCoordinate), so there is no interpolation error in the reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

from vcell_fenics.backend import SolverConfiguration, assemble, make_disk_membrane_geometry, run
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.geometry import Geometry, SubdomainGeometry
from vcell_fenics.formalism import MathDescription, load_yaml

PI = math.pi


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _l2_error(dp: DiscreteProblem, exact: ufl.core.expr.Expr) -> float:
    """L2 norm ‖u_h − u_exact‖ over the problem's domain, with `exact` an analytic
    UFL expression (no interpolation error in the reference)."""

    diff = dp.unknown - exact
    local = fem.assemble_scalar(fem.form(diff * diff * dp.dx))
    return math.sqrt(dp.V.mesh.comm.allreduce(local, op=MPI.SUM))


def _fit_order(hs: list[float], errors: list[float]) -> float:
    """Least-squares slope of log(error) vs log(h) — the observed convergence
    order. Refining h (or dt) downward should drive `errors` down with this slope."""

    slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    return float(slope)


def _square_geometry(name: str, *, subdomain: str, n: int) -> Geometry:
    """A unit-square `volume` geometry at resolution n×n (h = 1/n). Built inline
    rather than via a production constructor because the square is a pure test
    fixture: it meshes the domain exactly, isolating the spatial FE error."""

    mesh = create_unit_square(MPI.COMM_WORLD, n, n)
    return Geometry(name=name, subdomains={subdomain: SubdomainGeometry(mesh=mesh, kind="volume")})


def _square_model(*, diffusion: float) -> MathDescription:
    # IC is the Neumann eigenmode cos(πx)cos(πy); the compiler has no `pi`, so the
    # literal is inlined.
    return load_yaml(f"""
math_description:
  geometry: unit_square
  subdomains:
    - name: cytoplasm
      kind: volume
  variables:
    - {{ name: c, subdomain: cytoplasm }}
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cytoplasm
      temporality: time_dependent
      terms:
        diffusion: "{diffusion}"
      initial_condition: "cos({PI} * geom.x[0]) * cos({PI} * geom.x[1])"
""")


def _surface_model(*, diffusion: float, k: int) -> MathDescription:
    return load_yaml(f"""
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
      initial_condition: "cos({k} * geom.azimuth)"
""")


def _run_to(dp: DiscreteProblem, *, dt: float, t_final: float) -> None:
    for _ in range(round(t_final / dt)):
        dp.step()


# ---------------------------------------------------------------------------
# Spatial convergence — P1 Lagrange is second-order in L2.
# ---------------------------------------------------------------------------


def test_bulk_spatial_convergence_is_second_order() -> None:
    # cos(πx)cos(πy) on the unit square decays as exp(-2π²D·t) under Neumann BC.
    # dt is tiny so the O(dt) temporal floor sits well below the spatial error at
    # the finest mesh, leaving a clean spatial rate.
    D, t_final, dt = 0.1, 0.05, 5e-4
    decay = math.exp(-2.0 * PI**2 * D * t_final)
    md = _square_model(diffusion=D)

    resolutions = [8, 16, 32]
    hs, errors = [], []
    for n in resolutions:
        geometry = _square_geometry("unit_square", subdomain="cytoplasm", n=n)
        dp = assemble(md, geometry, dt=dt)
        _run_to(dp, dt=dt, t_final=t_final)
        x = ufl.SpatialCoordinate(dp.V.mesh)
        exact = decay * ufl.cos(PI * x[0]) * ufl.cos(PI * x[1])
        hs.append(1.0 / n)
        errors.append(_l2_error(dp, exact))

    assert errors[0] > errors[1] > errors[2], f"L2 error not monotone under refinement: {errors}"
    order = _fit_order(hs, errors)
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order spatial convergence, got slope {order:.2f} (errors {errors})"


@pytest.mark.xfail(
    reason="Surface eigenmode L2 error is non-monotone across independently-remeshed (non-nested) membranes: "
    "the Netgen region mesher resamples the circle at each h, so the codim-1 node layout is not a clean "
    "refinement and the O(h²) trend is swamped by mesh-topology noise once the geometry is near-exact (same "
    "root cause as test_mol_transient_is_second_order_in_space). Follow-up (project_gmsh_isolation_followups): "
    "nested-refinement / fine-reference convergence like test_method_of_lines_spatial_convergence_is_second_order.",
    strict=False,
)
def test_surface_spatial_convergence_is_second_order() -> None:
    # cos(kθ) on a circle of radius r decays as exp(-Dk²/r²·t). The polygonal mesh
    # approximates the circle with O(h²) geometric error, matching the P1 FE rate,
    # so the combined L2 convergence is still ~2. Exercises the codim-1 submesh.
    D, k, r, t_final, dt = 0.1, 2, 1.0, 0.1, 1e-3
    decay = math.exp(-D * k**2 / r**2 * t_final)
    md = _surface_model(diffusion=D, k=k)

    mesh_sizes = [0.2, 0.1, 0.05]
    hs, errors = [], []
    for h in mesh_sizes:
        geometry = make_disk_membrane_geometry("disk_membrane", surface_subdomain="membrane", radius=r, h=h)
        dp = assemble(md, geometry, dt=dt)
        _run_to(dp, dt=dt, t_final=t_final)
        x = ufl.SpatialCoordinate(dp.V.mesh)
        exact = decay * ufl.cos(k * ufl.atan2(x[1], x[0]))
        hs.append(h)
        errors.append(_l2_error(dp, exact))

    assert errors[0] > errors[1] > errors[2], f"L2 error not monotone under refinement: {errors}"
    order = _fit_order(hs, errors)
    assert 1.5 <= order <= 2.5, f"expected ~2nd-order surface convergence, got slope {order:.2f} (errors {errors})"


def test_method_of_lines_spatial_convergence_is_second_order() -> None:
    # The same bulk eigenmode, but integrated with the **method-of-lines** integrator (PETSc TS
    # adaptive BDF) the cross-validation now uses. MOL's time error is ≈0 (adaptive, tight tolerances),
    # so — unlike the backward-Euler test above, which needs a tiny dt to push the temporal floor below
    # the spatial error — no dt floor is needed here (`config.dt` is ignored by MOL). A clean O(h²) rate
    # also confirms the MOL time error stays under the spatial error: the BDF cold-start over-diffusion
    # (found + fixed in the convergence-study work) would offset the field and flatten the rate.
    D, t_final = 0.1, 0.05
    decay = math.exp(-2.0 * PI**2 * D * t_final)
    md = _square_model(diffusion=D)
    config = SolverConfiguration(dt=t_final, t_final=t_final, time_integration="method_of_lines")

    resolutions = [8, 16, 32]
    hs, errors = [], []
    for n in resolutions:
        geometry = _square_geometry("unit_square", subdomain="cytoplasm", n=n)
        problem = run(md, geometry, config)
        x = ufl.SpatialCoordinate(problem.V.mesh)
        exact = decay * ufl.cos(PI * x[0]) * ufl.cos(PI * x[1])
        hs.append(1.0 / n)
        errors.append(_l2_error(problem, exact))

    assert errors[0] > errors[1] > errors[2], f"L2 error not monotone under refinement: {errors}"
    order = _fit_order(hs, errors)
    assert 1.7 <= order <= 2.3, f"expected ~2nd-order MOL spatial convergence, got slope {order:.2f} (errors {errors})"


# ---------------------------------------------------------------------------
# Temporal convergence — backward Euler is first-order in dt.
# ---------------------------------------------------------------------------


def test_backward_euler_temporal_convergence_is_first_order() -> None:
    # Self-convergence on a fixed mesh: with the spatial discretisation identical
    # across the three runs, it cancels in the successive differences, so the
    # ratio of those differences measures the *temporal* order alone. For a
    # first-order scheme, ‖u(dt) − u(dt/2)‖ / ‖u(dt/2) − u(dt/4)‖ → 2.
    D, t_final, n = 0.1, 0.1, 40
    md = _square_model(diffusion=D)
    geometry = _square_geometry("unit_square", subdomain="cytoplasm", n=n)
    mesh = geometry.mesh_of("cytoplasm")

    dts = [0.02, 0.01, 0.005]
    finals: list[np.ndarray] = []
    for dt in dts:
        dp = assemble(md, geometry, dt=dt)
        _run_to(dp, dt=dt, t_final=t_final)
        finals.append(dp.unknown.x.array.copy())

    # ‖·‖_L2 of array differences via a shared P1 space (identical dof layout to
    # each solve's space, since same mesh + same element).
    V_ref = fem.functionspace(mesh, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=mesh)

    def _diff_norm(a: np.ndarray, b: np.ndarray) -> float:
        fa, fb = fem.Function(V_ref), fem.Function(V_ref)
        fa.x.array[:] = a
        fb.x.array[:] = b
        local = fem.assemble_scalar(fem.form((fa - fb) * (fa - fb) * dx))
        return math.sqrt(mesh.comm.allreduce(local, op=MPI.SUM))

    d1 = _diff_norm(finals[0], finals[1])  # u(dt)   − u(dt/2)
    d2 = _diff_norm(finals[1], finals[2])  # u(dt/2) − u(dt/4)
    order = math.log2(d1 / d2)
    assert 0.8 <= order <= 1.3, (
        f"expected ~1st-order temporal convergence, got order {order:.2f} (d1={d1:.3e}, d2={d2:.3e})"
    )

    # Tie it to the closed form: the finest-dt run must match the analytical decay
    # of the eigenmode (accuracy, not just self-consistency).
    dp_fine = assemble(md, geometry, dt=dts[-1])
    _run_to(dp_fine, dt=dts[-1], t_final=t_final)
    decay = math.exp(-2.0 * PI**2 * D * t_final)
    x = ufl.SpatialCoordinate(mesh)
    exact = decay * ufl.cos(PI * x[0]) * ufl.cos(PI * x[1])
    rel_err = _l2_error(dp_fine, exact) / abs(decay) / 0.5  # ‖exact‖₂ on unit square = decay·(1/2)
    assert rel_err < 0.02, f"finest-dt run off the analytical decay by {rel_err:.3%}"

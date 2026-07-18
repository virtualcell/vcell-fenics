"""Method-of-manufactured-solutions convergence check for the incompressible **Stokes** solver
(`backend/stokes.solve_incompressible_stokes`, the Taylor–Hood P2/P1 fluid solve underneath the FSI
force-balance closure).

This is **step 1** of the FSI verification plan (the fixed-domain fluid-order check), complementing the
co-moving species transport MMS (`test_fsi_species_mms.py`, step 2). The FSI fluid solve is a coupled
momentum balance with an incompressibility constraint — *not* scalar reaction-advection-diffusion — so it has
no place in the math-description formalism (see ADR 009 non-goals) and is verified here by pytest, on a
**fixed** domain to isolate the spatial discretisation from any mesh motion.

Stokes is linear, so the MMS is exact: pick a divergence-free `u*` and a pressure `p*`, let UFL differentiate
them to get the consistent body force `f = −∇·(2ν ε(u*)) + ∇p*` for the solver's own operator, impose `u*` as
the Dirichlet velocity, and measure the convergence of `(u_h, p_h)`. Taylor–Hood P2/P1 has the textbook
optimal rates **‖u−u_h‖_L2 = O(h³)** (P2 velocity) and **‖p−p_h‖_L2 = O(h²)** (P1 pressure); a solver defect
(wrong viscosity scaling, broken pressure coupling, a mis-assembled strain form) collapses them. Pressure is
compared mean-subtracted (it is defined only up to a constant).
"""

from __future__ import annotations

import math

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import create_unit_square
from mpi4py import MPI

from vcell_fenics.backend.stokes import solve_incompressible_stokes

_NU = 1.0
_PI = float(np.pi)
_RESOLUTIONS_NX = [8, 16, 32]


def _stokes_l2_errors(nx: int) -> tuple[float, float]:
    """Solve the manufactured Stokes problem on an nx×nx unit square; return (velocity, pressure) L2 errors."""
    mesh = create_unit_square(MPI.COMM_WORLD, nx, nx)
    x = ufl.SpatialCoordinate(mesh)
    # Divergence-free manufactured velocity (steady Taylor–Green) + a curved pressure (not exact in P1).
    u_star = ufl.as_vector(
        [ufl.sin(_PI * x[0]) * ufl.cos(_PI * x[1]), -ufl.cos(_PI * x[0]) * ufl.sin(_PI * x[1])]
    )
    p_star = ufl.sin(_PI * x[0]) * ufl.sin(_PI * x[1])
    forcing = -ufl.div(2.0 * _NU * ufl.sym(ufl.grad(u_star))) + ufl.grad(p_star)

    u_h, p_h = solve_incompressible_stokes(mesh, forcing=forcing, velocity=u_star, viscosity=_NU)

    dx = ufl.dx(domain=mesh)

    def integral(form: ufl.Form) -> float:
        return float(fem.assemble_scalar(fem.form(form)).real)

    area = integral(1.0 * dx)
    e_u = math.sqrt(integral(ufl.inner(u_h - u_star, u_h - u_star) * dx))
    p_mean = integral(p_h * dx) / area
    ps_mean = integral(p_star * dx) / area
    e_p = math.sqrt(integral(((p_h - p_mean) - (p_star - ps_mean)) ** 2 * dx))
    return e_u, e_p


def _order(errors: list[float]) -> float:
    return math.log(errors[0] / errors[-1]) / math.log(_RESOLUTIONS_NX[-1] / _RESOLUTIONS_NX[0])


def test_incompressible_stokes_hits_taylor_hood_optimal_orders() -> None:
    """The P2/P1 Taylor–Hood Stokes solve converges at its optimal rates against a manufactured solution:
    velocity O(h³), pressure O(h²). A solver defect collapses these below the asserted thresholds."""
    errors = [_stokes_l2_errors(nx) for nx in _RESOLUTIONS_NX]
    velocity_order = _order([e[0] for e in errors])
    pressure_order = _order([e[1] for e in errors])
    assert velocity_order > 2.5, f"velocity L2 order {velocity_order:.2f} (errors {[e[0] for e in errors]}) < 3"
    assert pressure_order > 1.5, f"pressure L2 order {pressure_order:.2f} (errors {[e[1] for e in errors]}) < 2"

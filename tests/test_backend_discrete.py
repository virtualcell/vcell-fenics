"""Tests for the DiscreteProblem IR, scheme, and lowering (ADR 004).

The IR is exercised on a hand-built static bulk-diffusion problem — the same
physics as approaches/static/bulk_pde.py, but assembled through the IR. Three
verification layers, per ADR 004:

1. **Structural** (no solve) — the term tags and BC set are what we expect.
2. **Operator invariants** — properties the assembled matrices must have,
   independent of any solution: the stiffness matrix annihilates constants
   (K·1 ≈ 0, the Neumann Laplacian), and the backward-Euler operator maps a
   constant to the mass term (A·1 = M·1).
3. **End-to-end** — a constant IC stays constant and total mass is conserved.

The hand-built problem here is what the T1 assembler will construct from a
MathDescription in a later commit; building it directly keeps this module
independent of the expression compiler and geometry adapter.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem
from petsc4py import PETSc

from vcell_fenics.approaches.static import create_disk
from vcell_fenics.backend import BackwardEuler, DiscreteProblem, Term, TermKind


def _bulk_diffusion_problem(
    *, radius: float = 1.0, h: float = 0.2, D: float = 0.5, dt: float = 0.05
) -> DiscreteProblem:
    mesh = create_disk(radius=radius, h=h).mesh
    V = fem.functionspace(mesh, ("Lagrange", 1))
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    # petsc4py stubs type PETSc.ScalarType as a non-callable dtype; it is a
    # callable scalar alias at runtime.
    Dc = fem.Constant(mesh, PETSc.ScalarType(D))  # type: ignore[operator]
    dtc = fem.Constant(mesh, PETSc.ScalarType(dt))  # type: ignore[operator]

    terms = (
        Term(TermKind.TIME_DERIVATIVE),
        Term(TermKind.DIFFUSION, Dc * ufl.dot(ufl.grad(trial), ufl.grad(test))),
    )
    return DiscreteProblem(
        variable_name="c",
        V=V,
        trial=trial,
        test=test,
        dx=dx,
        unknown=fem.Function(V, name="c"),
        previous=fem.Function(V, name="c_old"),
        dt=dtc,
        terms=terms,
        scheme=BackwardEuler(),
        bcs=[],
    )


def _dense(form: Any) -> npt.NDArray[Any]:
    matrix = fem.assemble_matrix(fem.form(form))
    return cast(npt.NDArray[Any], matrix.to_dense())


# ---------------------------------------------------------------------------
# 1. Structural — no solve.
# ---------------------------------------------------------------------------


def test_term_kinds_are_time_derivative_and_diffusion() -> None:
    dp = _bulk_diffusion_problem()
    assert dp.term_kinds() == {TermKind.TIME_DERIVATIVE, TermKind.DIFFUSION}


def test_no_dirichlet_constraints() -> None:
    # Natural-Neumann is the default; static bulk diffusion creates no BC object.
    assert _bulk_diffusion_problem().bcs == []


# ---------------------------------------------------------------------------
# 2. Operator invariants — assembled matrices, no solution needed.
# ---------------------------------------------------------------------------


def test_stiffness_annihilates_constants() -> None:
    dp = _bulk_diffusion_problem()
    K = _dense(dp.integrand_of(TermKind.DIFFUSION) * dp.dx)
    ones = np.ones(K.shape[1])
    assert np.allclose(K @ ones, 0.0, atol=1e-10)


def test_backward_euler_operator_maps_constant_to_mass() -> None:
    # A = M + dt·K, and K·1 ≈ 0, so A·1 = M·1.
    dp = _bulk_diffusion_problem()
    A = _dense(dp.bilinear_form)
    M = _dense(dp.trial * dp.test * dp.dx)
    ones = np.ones(A.shape[1])
    assert np.allclose(A @ ones, M @ ones, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. End-to-end — solve through the IR.
# ---------------------------------------------------------------------------


def test_constant_initial_condition_stays_constant() -> None:
    dp = _bulk_diffusion_problem()
    dp.set_initial(3.0)
    for _ in range(20):
        dp.step()
    assert np.allclose(dp.unknown.x.array, 3.0, atol=1e-10)


def test_total_mass_is_conserved() -> None:
    dp = _bulk_diffusion_problem(h=0.1, D=0.2, dt=0.01)

    def ic(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return cast(npt.NDArray[Any], 1.0 + 0.3 * x[0])

    dp.set_initial(ic)
    m0 = dp.total_mass()
    for _ in range(50):
        dp.step()
    assert abs(dp.total_mass() - m0) / abs(m0) < 1e-10

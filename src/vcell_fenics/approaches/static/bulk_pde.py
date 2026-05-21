"""Bulk reaction-diffusion on a fixed mesh: ∂_t c = D Δc (+ R).

Backward Euler. No-flux (Neumann) BC is the natural BC for the weak form,
so it is enforced implicitly — no `bcs` argument needed for closed domains.
Reaction is left for a follow-up increment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy.typing as npt
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

# Scalar field initializer in dolfinx style: (x: shape (gdim, n_points))
# -> shape (n_points,). Used by fem.Function.interpolate.
ScalarField = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


class BulkPDE:
    def __init__(
        self,
        mesh: dmesh.Mesh,
        D: float,
        dt: float,
        degree: int = 1,
    ) -> None:
        self.mesh = mesh
        self.V = fem.functionspace(mesh, ("Lagrange", degree))
        self.c = fem.Function(self.V, name="c")
        self.c_old = fem.Function(self.V, name="c_old")

        # petsc4py stubs mark PETSc.ScalarType as a non-callable numpy.dtype;
        # at runtime it's a callable scalar type alias.
        self.D = fem.Constant(mesh, PETSc.ScalarType(D))  # type: ignore[operator]
        self.dt = fem.Constant(mesh, PETSc.ScalarType(dt))  # type: ignore[operator]

        u = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=mesh)

        a = u * w * dx + self.dt * self.D * ufl.dot(ufl.grad(u), ufl.grad(w)) * dx
        L = self.c_old * w * dx

        self._problem = LinearProblem(
            a,
            L,
            u=self.c,
            petsc_options_prefix="vcellfenics_bulk_pde_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def set_initial(self, value: float | ScalarField) -> None:
        if callable(value):
            self.c.interpolate(value)
        else:
            self.c.x.array[:] = float(value)
        self.c_old.x.array[:] = self.c.x.array

    def step(self) -> None:
        self._problem.solve()
        self.c_old.x.array[:] = self.c.x.array

    def total_mass(self) -> float:
        dx = ufl.Measure("dx", domain=self.mesh)
        form = fem.form(self.c * dx)
        local = fem.assemble_scalar(form)
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def domain_volume(self) -> float:
        dx = ufl.Measure("dx", domain=self.mesh)
        form = fem.form(1.0 * dx)
        local = fem.assemble_scalar(form)
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def project_onto(self, mode: fem.Function) -> float:
        """Return ⟨c, φ⟩ / ⟨φ, φ⟩ for the L²(Ω) inner product.

        Useful for measuring the amplitude of a single eigenmode in c.
        """
        dx = ufl.Measure("dx", domain=self.mesh)
        num = self.mesh.comm.allreduce(fem.assemble_scalar(fem.form(self.c * mode * dx)), op=MPI.SUM)
        den = self.mesh.comm.allreduce(fem.assemble_scalar(fem.form(mode * mode * dx)), op=MPI.SUM)
        return float(num / den)

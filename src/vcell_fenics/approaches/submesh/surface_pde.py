"""Surface PDE on a codim-1 submesh: diffusion + optional stretch dilution.

Solves

    ∂ₜρ + ρ (∇_Γ · v_Γ) = D Δ_Γ ρ

on the submesh with backward Euler. The dilution term ρ ∇_Γ · v_Γ is
mandatory whenever the membrane moves (see docs/modeling/approaches.md);
this implementation takes ∇_Γ · v_Γ as a user-supplied scalar Function so
the caller is forced to think about it rather than silently default it to
zero on a moving mesh.
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


class SurfacePDE:
    def __init__(
        self,
        submesh: dmesh.Mesh,
        D: float,
        dt: float,
        div_v_gamma: fem.Function | None = None,
        degree: int = 1,
    ) -> None:
        self.submesh = submesh
        self.V = fem.functionspace(submesh, ("Lagrange", degree))
        self.rho = fem.Function(self.V, name="rho")
        self.rho_old = fem.Function(self.V, name="rho_old")

        # petsc4py stubs mark PETSc.ScalarType as a non-callable numpy.dtype;
        # at runtime it's a callable scalar type alias.
        self.D = fem.Constant(submesh, PETSc.ScalarType(D))  # type: ignore[operator]
        self.dt = fem.Constant(submesh, PETSc.ScalarType(dt))  # type: ignore[operator]
        self.div_v = div_v_gamma  # None ⇒ static membrane (no dilution)

        u = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=submesh)

        # ∇ on a codim-1 submesh embedded in R^d evaluates the intrinsic
        # (tangential) gradient — i.e. ∇_Γ — so no projector is needed.
        a = u * w * dx + self.dt * self.D * ufl.dot(ufl.grad(u), ufl.grad(w)) * dx
        if self.div_v is not None:
            a = a + self.dt * self.div_v * u * w * dx
        L = self.rho_old * w * dx

        self._problem = LinearProblem(
            a,
            L,
            u=self.rho,
            petsc_options_prefix="vcellfenics_surface_pde_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def set_initial(self, value: float | ScalarField) -> None:
        if callable(value):
            self.rho.interpolate(value)
        else:
            self.rho.x.array[:] = float(value)
        self.rho_old.x.array[:] = self.rho.x.array

    def step(self) -> None:
        self._problem.solve()
        self.rho_old.x.array[:] = self.rho.x.array

    def total_mass(self) -> float:
        dx = ufl.Measure("dx", domain=self.submesh)
        form = fem.form(self.rho * dx)
        local = fem.assemble_scalar(form)
        return float(self.submesh.comm.allreduce(local, op=MPI.SUM))

    def surface_length(self) -> float:
        dx = ufl.Measure("dx", domain=self.submesh)
        form = fem.form(1.0 * dx)
        local = fem.assemble_scalar(form)
        return float(self.submesh.comm.allreduce(local, op=MPI.SUM))

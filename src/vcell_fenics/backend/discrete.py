"""The DiscreteProblem intermediate representation and its lowering.

Per ADR 004 (docs/decisions/004-discreteproblem-ir.md), a `DiscreteProblem` is
the FEniCSx-backend-internal representation of one coupled solve. It records the
spatial weak-form residual as a small enum of **tagged terms** — each carrying a
UFL integrand — plus a **time scheme** (`BackwardEuler` for v1). Lowering applies
the scheme to the term list to compose the bilinear / linear forms and build a
DOLFINx `LinearProblem`.

The point of the IR is that the FEniCSx target assumptions (backward Euler,
natural-Neumann, the mass term, which spatial terms are present) are explicit and
inspectable — a test can assert "this problem has a DILUTION term" without solving,
and operator invariants (`K·1 ≈ 0`, `A·1 = M·1`) can be checked on the assembled
matrices independently of any solution.

This module owns no formalism types: the assembler (a separate module) builds the
UFL integrands from a MathDescription and hands them here as `Term`s. v1 supports
time-dependent problems only; steady-state lowering raises until a later increment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh
from mpi4py import MPI
from scipy.spatial import cKDTree

from vcell_fenics.backend._typing import UflExpr


class TermKind(Enum):
    """The kinds of term a v1 template (T1 / T2) can contribute. A closed set,
    extended deliberately as new templates land (ADR 004)."""

    TIME_DERIVATIVE = "time_derivative"
    DIFFUSION = "diffusion"
    ADVECTION = "advection"
    DILUTION = "dilution"
    SOURCE = "source"


@dataclass(frozen=True)
class Term:
    """One tagged term of the spatial residual.

    `integrand` is a UFL expression — the term's contribution *before* the time
    scheme applies (so a diffusion term carries `D ∇u·∇w`, not `dt·D ∇u·∇w`). The
    `TIME_DERIVATIVE` marker carries no integrand; the scheme materialises the
    mass term from the problem's trial/test functions during lowering.
    """

    kind: TermKind
    integrand: UflExpr | None = None


@dataclass(frozen=True)
class BackwardEuler:
    """First-order implicit (backward-Euler) time scheme.

    Composes, for a time-dependent problem, the implicit step

        ∫ u·w dx + dt·a_spatial(u, w) = ∫ u_old·w dx  (+ dt·source, when present)

    multiplying the spatial terms through by `dt` and keeping the mass term's
    coefficient at 1 — the convention the bespoke prototypes use, which avoids
    dividing by `dt`.
    """

    name: str = "backward_euler"

    def compose(self, problem: DiscreteProblem) -> tuple[ufl.Form, ufl.Form]:
        kinds = problem.term_kinds()
        if TermKind.TIME_DERIVATIVE not in kinds:
            raise NotImplementedError("steady-state lowering is not in the v1 backend yet")

        u, w, dx, dt = problem.trial, problem.test, problem.dx, problem.dt
        bilinear = u * w  # mass term, coefficient 1
        for term in problem.terms:
            if term.kind in (TermKind.TIME_DERIVATIVE, TermKind.SOURCE):
                continue  # mass handled above; SOURCE goes to the RHS (a later increment)
            if term.integrand is not None:
                bilinear = bilinear + dt * term.integrand
        a = bilinear * dx
        linear = problem.previous * w * dx  # ∫ u_old·w dx; source RHS contributions are a later increment
        return a, linear


class _MeshMotion:
    """Advances mesh nodes by dt·velocity each step (prescribed motion, §1.10).

    The displacement is interpolated into a P1 vector field; its dof order does
    not match the mesh's geometry-node order (especially on a submesh), so a
    one-time topological permutation maps interpolated values onto geometry rows.
    The permutation is invariant under motion (it is purely topological), so it
    is computed once from the initial coordinates and reused.
    """

    def __init__(self, mesh: Mesh, velocity: UflExpr, dt: fem.Constant) -> None:
        self._mesh = mesh
        self._gdim = mesh.geometry.dim
        space = fem.functionspace(mesh, ("Lagrange", 1, (self._gdim,)))
        self._displacement = fem.Function(space)
        self._expression = fem.Expression(dt * velocity, space.element.interpolation_points)
        self._geom_from_dof = cKDTree(space.tabulate_dof_coordinates()).query(mesh.geometry.x)[1]

    def advance(self) -> None:
        self._displacement.interpolate(self._expression)
        increment = self._displacement.x.array.reshape((-1, self._gdim))[self._geom_from_dof]
        self._mesh.geometry.x[:, : self._gdim] += increment


@dataclass(eq=False)
class DiscreteProblem:
    """A single coupled solve: a function space, the tagged residual terms, a
    time scheme, and the DOLFINx solver state. Lowering happens at construction.

    `unknown` is the solution Function (also the LinearProblem's output);
    `previous` holds the prior time step. `trial` / `test` / `dx` are shared with
    the terms' integrands, so the scheme reuses the same objects.
    """

    variable_name: str
    V: fem.FunctionSpace
    trial: UflExpr
    test: UflExpr
    dx: ufl.Measure
    unknown: fem.Function
    previous: fem.Function
    dt: fem.Constant
    terms: tuple[Term, ...]
    scheme: BackwardEuler
    bcs: list[fem.DirichletBC]
    # Prescribed substrate velocity (a UFL vector field). When set, each step
    # advances the mesh by dt·velocity before solving — the moving-subdomain
    # protocol of §1.10. None means a static subdomain.
    motion_velocity: UflExpr | None = None

    def __post_init__(self) -> None:
        self._a, self._L = self.scheme.compose(self)
        self._problem = LinearProblem(
            self._a,
            self._L,
            u=self.unknown,
            bcs=self.bcs,
            petsc_options_prefix=f"vcellfenics_dp_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        self._motion = (
            _MeshMotion(self.V.mesh, self.motion_velocity, self.dt) if self.motion_velocity is not None else None
        )

    # -- inspection (structural verification, no solve) ----------------------

    def term_kinds(self) -> set[TermKind]:
        return {term.kind for term in self.terms}

    def integrand_of(self, kind: TermKind) -> UflExpr:
        """The UFL integrand of the (unique) term of `kind`. For invariant tests
        that assemble a single term's matrix (e.g. the stiffness `K`)."""

        return next(term.integrand for term in self.terms if term.kind is kind)

    @property
    def bilinear_form(self) -> ufl.Form:
        return self._a

    @property
    def linear_form(self) -> ufl.Form:
        return self._L

    # -- solve / state -------------------------------------------------------

    def set_initial(self, value: float | Callable[[Any], Any]) -> None:
        if callable(value):
            self.unknown.interpolate(value)
        else:
            self.unknown.x.array[:] = float(value)
        self.previous.x.array[:] = self.unknown.x.array

    def interpolate_initial(self, ufl_expr: UflExpr) -> None:
        """Set the initial state from a compiled UFL expression. Handles
        spatially-varying ICs (a constant interpolates to a constant field), so
        it is the assembler's path for applying a MathDescription's IC."""

        expr = fem.Expression(ufl_expr, self.V.element.interpolation_points)
        self.unknown.interpolate(expr)
        self.previous.x.array[:] = self.unknown.x.array

    def step(self) -> None:
        # For a moving subdomain, advance the mesh first; field values are
        # carried (material frame) and the mass + DILUTION terms re-assemble on
        # the new configuration, so the dilution `div(velocity)` reflects it.
        if self._motion is not None:
            self._motion.advance()
        self._problem.solve()
        self.previous.x.array[:] = self.unknown.x.array

    def total_mass(self) -> float:
        local = fem.assemble_scalar(fem.form(self.unknown * self.dx))
        return float(self.V.mesh.comm.allreduce(local, op=MPI.SUM))

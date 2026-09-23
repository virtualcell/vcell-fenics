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

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, exterior_facet_indices
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from ufl.algorithms.check_arities import ArityMismatch, check_form_arity

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.diagnostics import NonlinearTermError, nonlinear_backward_euler_message


class TermKind(Enum):
    """The kinds of term a v1 template (T1 / T2) can contribute. A closed set,
    extended deliberately as new templates land (ADR 004)."""

    TIME_DERIVATIVE = "time_derivative"
    DIFFUSION = "diffusion"
    ADVECTION = "advection"
    DILUTION = "dilution"
    SOURCE = "source"
    # Boundary contributions (integrated over a labelled boundary measure, not dx).
    NEUMANN = "neumann"
    ROBIN = "robin"


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
class BoundaryTerm:
    """A weak-form contribution integrated over a labelled boundary `measure` (a
    restricted `ds`) — for non-zero Neumann and Robin BCs.

    `integrand` is the *signed* residual contribution: the assembler bakes in the
    sign and coefficients, so lowering just adds `dt · integrand · measure` to the
    residual form. The implicit-in-`u` part of a Robin term lands in the bilinear
    form via the same `ufl.lhs`/`rhs` split the `SOURCE` term relies on. Dirichlet
    BCs are *strong* (they live in `DiscreteProblem.bcs`), not here; the implicit
    zero-Neumann default is simply the absence of any boundary term.
    """

    kind: TermKind
    integrand: UflExpr
    measure: ufl.Measure


@dataclass(frozen=True)
class BackwardEuler:
    """First-order implicit (backward-Euler) time scheme.

    Builds the per-step residual (multiplied through by `dt`, so the mass term
    keeps coefficient 1 — the convention the bespoke prototypes use, avoiding
    division by `dt`):

        F = (uⁿ⁺¹ − uⁿ)·w  +  dt·[ diffusion + dilution + advection ]  −  dt·source·w

    and splits it with `ufl.lhs` / `ufl.rhs`: terms in the trial function form the
    bilinear `a`, the rest form the linear `L`. Letting UFL do the split is what
    lets a `source` linear in the unknown(s) — including cross-variable coupling
    on a mixed space — land in `a` automatically, while a constant forcing lands
    in `L`, with no per-term bookkeeping here.
    """

    name: str = "backward_euler"

    def compose(self, problem: DiscreteProblem) -> tuple[ufl.Form, ufl.Form]:
        if TermKind.TIME_DERIVATIVE not in problem.term_kinds():
            raise NotImplementedError("steady-state lowering is not in the v1 backend yet")

        trial, test, dt = problem.trial, problem.test, problem.dt
        # On a **moving mesh that carries a dilution term** (bulk *or* membrane) the mass term is the
        # *conservative ALE* time term: the previous field is rescaled per cell by `|Kⁿ|/|Kⁿ⁺¹|`
        # (`motion.volume_ratio()` — cell volumes for a bulk, facet lengths/areas for a membrane) so its
        # mass on the moved mesh equals its integral on the pre-move mesh exactly. Dilution then lives
        # entirely in the changing measure — the explicit `ρ ∇·v` DILUTION term is dropped below — and mass
        # is conserved to solver precision, eliminating the O(dt) geometric-conservation-law drift the
        # same-mesh (uⁿ⁺¹−uⁿ)·w + ρ∇·v split incurs. The trigger is the DILUTION term itself (the "this
        # field dilutes" signal): a moving problem *without* one is left advective (mass follows the
        # measure — the classic bug; see test_omitting_dilution_doubles_mass), and a rigid motion gives
        # `volume_ratio ≡ 1`, so either way this reduces to the plain form.
        motion = problem._motion
        conservative = motion is not None and TermKind.DILUTION in problem.term_kinds()
        if conservative:
            assert motion is not None  # implied by `conservative`; narrows for the type checker
            previous = motion.volume_ratio() * problem.previous
        else:
            previous = problem.previous
        # mass: (uⁿ⁺¹ − r·uⁿ)·w. `inner` so a mixed/vector space (coupled species)
        # sums its components; for a scalar space it is just the product.
        form = ufl.inner(trial - previous, test) * problem.dx
        for term in problem.terms:
            if term.kind is TermKind.TIME_DERIVATIVE or term.integrand is None:
                continue
            if conservative and term.kind is TermKind.DILUTION:
                continue  # subsumed into the conservative time term's changing measure
            if term.kind is TermKind.SOURCE:
                form = form - dt * term.integrand * problem.dx  # +s on the PDE's RHS ⇒ −s in the residual
            else:
                form = form + dt * term.integrand * problem.dx  # diffusion, dilution, advection
        # Boundary contributions integrate over their own (restricted) measure; the
        # assembler has baked the residual sign into each integrand.
        for boundary in problem.boundary_terms:
            form = form + dt * boundary.integrand * boundary.measure
        return ufl.lhs(form), ufl.rhs(form)


# A motion is "degenerate" once the spread of cell sizes (max/min cell volume)
# grows this many times beyond the initial mesh's. Scale-invariant: uniform
# dilation or contraction preserves the ratio and never trips it; only
# non-uniform distortion (cells collapsing / tangling) does. A future
# SolverConfiguration `ale.remesh_when_quality_below` (§3.4) would supply this.
_MAX_CELL_RATIO_GROWTH = 100.0


class MeshQualityError(RuntimeError):
    """Prescribed motion degraded the mesh beyond a usable state — a cell
    collapsed, inverted, or the mesh distorted badly. The backend moves nodes
    but never remeshes, so rather than silently solving on a broken mesh it
    fails here. Remeshing / field transfer is future work (§3.3)."""


class _HarmonicExtension:
    """Extends a boundary displacement into a bulk mesh's interior by a vector
    Laplace solve — the standard ALE mesh-motion fill (Approach A).

    Given a displacement field whose *boundary* values are the prescribed dt·v, it
    solves ∇²d = 0 with those values fixed on ∂Ω (Dirichlet) and overwrites the
    field with the harmonic result. So interior nodes follow the moving boundary
    *smoothly* rather than being dragged by the raw velocity formula — which for a
    codim-0 mesh is both the physically right ALE choice (interior motion is a mesh
    bookkeeping device, not a material velocity) and avoids interior singularities
    of the boundary velocity expression (e.g. `x / r(x)` at the centre).

    Bound to the mesh; the bilinear form re-assembles each solve, so a moving mesh
    is handled automatically.
    """

    def __init__(self, space: fem.FunctionSpace) -> None:
        mesh = space.mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facets = exterior_facet_indices(mesh.topology)
        boundary_dofs = fem.locate_dofs_topological(space, tdim - 1, facets)
        self._boundary = fem.Function(space)  # the prescribed boundary displacement (BC source)
        self._solution = fem.Function(space)  # the harmonic result
        bc = fem.dirichletbc(self._boundary, boundary_dofs)
        u, w = ufl.TrialFunction(space), ufl.TestFunction(space)
        a = ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        zero = fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType))
        L = ufl.inner(zero, w) * ufl.dx
        self._problem = LinearProblem(
            a,
            L,
            u=self._solution,
            bcs=[bc],
            petsc_options_prefix=f"vcellfenics_ale_{id(self):x}_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def fill(self, displacement: fem.Function) -> None:
        """Replace `displacement`'s interior with the harmonic extension of its
        boundary values (the boundary values themselves are preserved). Interior
        values of the input are ignored — only the boundary DOFs feed the BC."""

        self._boundary.x.array[:] = displacement.x.array
        self._problem.solve()
        displacement.x.array[:] = self._solution.x.array


class _MeshMotion:
    """Advances mesh nodes by dt·velocity each step (prescribed motion, §1.10).

    The displacement is interpolated into a P1 vector field; its dof order does
    not match the mesh's geometry-node order (especially on a submesh), so a
    one-time topological permutation maps interpolated values onto geometry rows.
    The permutation is invariant under motion (it is purely topological), so it
    is computed once from the initial coordinates and reused.

    On a codim-0 (bulk) mesh the prescribed velocity only defines the *boundary*
    motion; the interior is filled by harmonic extension (`_HarmonicExtension`) so
    interior quality is preserved. On a codim-1 mesh (a membrane) every node is on
    the boundary, so the interpolated dt·v moves them all directly — the original
    behaviour, unchanged.

    After each move the mesh quality is checked (`MeshQualityError` on failure):
    a node-displacement scheme with no remeshing can only follow motions that
    keep the elements valid, so a tangling motion must fail loudly.
    """

    def __init__(self, mesh: Mesh, velocity: UflExpr, dt: fem.Constant) -> None:
        self._mesh = mesh
        self._dt = dt
        self._gdim = mesh.geometry.dim
        space = fem.functionspace(mesh, ("Lagrange", 1, (self._gdim,)))
        self._displacement = fem.Function(space)
        self._expression = fem.Expression(dt * velocity, space.element.interpolation_points)
        self._geom_from_dof = cKDTree(space.tabulate_dof_coordinates()).query(mesh.geometry.x)[1]
        # A bulk mesh (codim 0) has interior nodes to fill harmonically; a membrane
        # (codim 1) is all boundary, so dt·v moves every node directly.
        self._extension = _HarmonicExtension(space) if mesh.topology.dim == self._gdim else None
        dg0 = fem.functionspace(mesh, ("DG", 0))
        # Per-cell volume form (DG0 test function integrates to each cell's
        # volume); re-assembling after a move reports the deformed cell sizes.
        self._cell_volume_form = fem.form(ufl.TestFunction(dg0) * ufl.dx)
        # Per-cell |Kⁿ|/|Kⁿ⁺¹| (old/new cell volume), refreshed each `advance`. The conservative BE mass
        # term multiplies the carried previous field by this, so its mass integrated on the moved mesh
        # equals its integral on the pre-move mesh exactly — dilution then lives in the changing measure
        # (no explicit ρ∇·v term) and mass is conserved to solver precision, without the O(dt) GCL drift
        # the same-mesh time-term + `ρ∇·v` split incurs. 1.0 before the first move and for a rigid motion.
        self._volume_ratio = fem.Function(dg0)
        self._volume_ratio.x.array[:] = 1.0
        # Per-cell *GCL-consistent* dilution rate `ln(|Kⁿ⁺¹|/|Kⁿ|)/dt` (DG0), refreshed each `advance`.
        # The strided ALE-MOL (`TS`) path applies dilution *continuously* over a stride while the mesh
        # jumps *discretely* at its start; using this effective rate (not the instantaneous `∇·v`) makes
        # the continuous decay `exp(−d_eff·dt) = |Kⁿ|/|Kⁿ⁺¹|` exactly cancel the discrete swept-volume
        # jump, so mass is conserved (no O(dt) GCL drift). `→ ∇·v` as dt→0, so it is consistent. 0 before
        # the first move. (The backward-Euler path drops the dilution term entirely — see BackwardEuler.)
        self._effective_dilution = fem.Function(dg0)
        self._reference_ratio = self._cell_volume_ratio()

    def advance(self) -> None:
        self._displacement.interpolate(self._expression)
        if self._extension is not None:
            self._extension.fill(self._displacement)
        volumes_before = self._cell_volumes()
        increment = self._displacement.x.array.reshape((-1, self._gdim))[self._geom_from_dof]
        self._mesh.geometry.x[:, : self._gdim] += increment
        volumes_after = self._cell_volumes()
        v_min, v_max = float(volumes_after.min()), float(volumes_after.max())
        if not (math.isfinite(v_min) and math.isfinite(v_max)) or v_min <= 0.0:
            raise MeshQualityError(
                f"prescribed motion produced a non-positive or non-finite cell volume (min={v_min:.3g}); "
                f"an element has collapsed or inverted."
            )
        # |Kⁿ|/|Kⁿ⁺¹| per cell: the P1 local mass matrix scales linearly with cell volume, so this ratio
        # rescales the carried previous field's mass on the moved mesh back to its pre-move value exactly.
        self._volume_ratio.x.array[:] = volumes_before / volumes_after
        # GCL-consistent continuous dilution rate for the strided TS path (see __init__): the log of the
        # *actual* per-cell swept-volume ratio over the stride, so exp(−d_eff·dt) undoes the mesh jump.
        self._effective_dilution.x.array[:] = np.log(volumes_after / volumes_before) / float(self._dt.value)
        ratio = v_max / v_min
        if ratio > _MAX_CELL_RATIO_GROWTH * self._reference_ratio:
            raise MeshQualityError(
                f"prescribed motion degraded the mesh: cell-volume max/min ratio {ratio:.3g} exceeds "
                f"{_MAX_CELL_RATIO_GROWTH:g}x the initial {self._reference_ratio:.3g}. The backend moves nodes "
                f"but does not remesh; reduce the step, the motion magnitude, or use a better-behaved velocity."
            )

    def velocity(self) -> UflExpr:
        """The mesh (frame) velocity of the step just taken — the moved node displacement over dt: the
        prescribed motion on the boundary, its harmonic extension inside a bulk. Zero before the first
        move. An Eulerian (lab-frame) species is transported relative to it (the ``advection`` slot)."""

        return self._displacement / self._dt

    def current_growth(self) -> float:
        """The cell-size (max/min volume) ratio relative to the fresh reference
        mesh: 1.0 when undistorted, rising as motion deforms the mesh. The ALE
        driver polls this to decide when to remesh, below the hard limit `advance`
        enforces."""
        return self._cell_volume_ratio() / self._reference_ratio

    def volume_ratio(self) -> fem.Function:
        """Per-cell `|Kⁿ|/|Kⁿ⁺¹|` (old/new cell volume), refreshed each `advance`. The conservative
        backward-Euler mass term (`BackwardEuler.compose`) multiplies the carried previous field by this
        so its mass on the moved mesh equals its pre-move mass exactly, making dilution implicit in the
        changing measure. 1.0 before the first move and for a rigid (`∇·v = 0`) motion."""
        return self._volume_ratio

    def effective_dilution_rate(self) -> fem.Function:
        """Per-cell `ln(|Kⁿ⁺¹|/|Kⁿ|)/dt` (DG0), refreshed each `advance` — the GCL-consistent dilution
        rate the **strided ALE-MOL (`TS`) path** uses so its continuous over-a-stride decay exactly
        cancels the discrete mesh jump (conservation to solver precision). `→ ∇·v` as dt→0. The DILUTION
        term (built in `assemble`) reads this; the backward-Euler path drops that term and conserves via
        `volume_ratio` instead. 0 before the first move (dilution is only assembled after a move)."""
        return self._effective_dilution

    def _cell_volumes(self) -> Any:
        """Per-cell volumes (DG0) on the current configuration — |K| for each cell."""
        return fem.assemble_vector(self._cell_volume_form).array

    def _cell_volume_ratio(self) -> float:
        volumes = self._cell_volumes()
        v_min, v_max = float(volumes.min()), float(volumes.max())
        if not (math.isfinite(v_min) and math.isfinite(v_max)) or v_min <= 0.0:
            raise MeshQualityError(
                f"prescribed motion produced a non-positive or non-finite cell volume (min={v_min:.3g}); "
                f"an element has collapsed or inverted."
            )
        return v_max / v_min


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
    # Non-zero Neumann / Robin contributions (Dirichlet BCs are strong, in `bcs`).
    boundary_terms: tuple[BoundaryTerm, ...] = ()
    # Prescribed substrate velocity (a UFL vector field). When set, each step
    # advances the mesh by dt·velocity before solving — the moving-subdomain
    # protocol of §1.10. None means a static subdomain.
    motion_velocity: UflExpr | None = None
    # The mesh-motion object. The assembler builds it *before* the dilution term so that term can read
    # the actual mesh (substrate) velocity (`substrate_velocity`, the harmonic extension for a bulk),
    # not the raw prescribed velocity. When omitted, it is created here from `motion_velocity` (the raw
    # velocity — correct for a membrane or affine bulk; a direct-construction convenience).
    motion: _MeshMotion | None = None
    # Whether `step()` moves the mesh (calls `motion.advance()`). Normally True. Set False when an
    # *external* driver already advanced this same `motion` before calling `step()` — e.g. the solved-
    # velocity `unknown_motion` path moves the membrane itself, then steps the receptor: the receptor
    # then only *reads* the motion's `volume_ratio` for the conservative time term, without moving again.
    owns_mesh_motion: bool = True
    # The bound time Constant `t` (compile context). A driver advances it via
    # `set_time` so a time-dependent expression — e.g. a Dirichlet g(t) — tracks it.
    time: fem.Constant | None = None
    # (value, expression) pairs for each Dirichlet BC: re-interpolating the value from
    # its expression re-evaluates g against the (advanced) time Constant.
    dirichlet_refreshers: tuple[tuple[fem.Function, fem.Expression], ...] = ()

    def __post_init__(self) -> None:
        # The backward-Euler `ufl.lhs`/`rhs` split is composed LAZILY (`_compose_backward_euler`), not
        # here. A model with a nonlinear source cannot be represented by that split — for a *polynomial*
        # nonlinearity (e.g. `c*c`) the split silently mis-buckets it (caught later by the arity check),
        # and for a *rational* one (a trial in a denominator, e.g. `c**2/(1+c**2)`) `ufl.lhs` raises
        # outright. Deferring the split means such a model can still be assembled and handed to the
        # method-of-lines integrator (`backend/reaction_diffusion.py`, which builds its own nonlinear
        # residual) — only a backward-Euler solve (or `bilinear_form`/`linear_form`) triggers the split.
        self._a: ufl.Form | None = None
        self._L: ufl.Form | None = None
        self._problem: LinearProblem | None = None
        if self.motion is not None:
            self._motion: _MeshMotion | None = self.motion
        elif self.motion_velocity is not None:
            self._motion = _MeshMotion(self.V.mesh, self.motion_velocity, self.dt)
        else:
            self._motion = None

    def _compose_backward_euler(self) -> tuple[ufl.Form, ufl.Form]:
        # The BE bilinear/linear forms, composed once on first use. Backward Euler can only assemble a
        # residual affine in the unknown; checking arity here, in pure UFL *before* `fem.form`/FFCx, gives
        # the modeller a named fix instead of a deep traceback (and never form-compiles a bad form, which
        # would poison the JIT cache). A rational nonlinearity instead makes `ufl.lhs` raise a `ValueError`
        # ("Argument in denominator"); both routes are reported as the same `NonlinearTermError`.
        a, ell = self._a, self._L
        if a is None or ell is None:
            try:
                a, ell = self.scheme.compose(self)
                check_form_arity(a, a.arguments())
            except (ArityMismatch, ValueError) as nonlinear:
                raise NonlinearTermError(nonlinear_backward_euler_message()) from nonlinear
            self._a, self._L = a, ell
        return a, ell

    def _backward_euler_problem(self) -> LinearProblem:
        if self._problem is None:
            a, ell = self._compose_backward_euler()
            self._problem = LinearProblem(
                a,
                ell,
                u=self.unknown,
                bcs=self.bcs,
                petsc_options_prefix=f"vcellfenics_dp_{id(self):x}_",
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            )
        return self._problem

    # -- inspection (structural verification, no solve) ----------------------

    def term_kinds(self) -> set[TermKind]:
        return {term.kind for term in self.terms}

    def boundary_kinds(self) -> set[TermKind]:
        """The kinds of boundary term present (NEUMANN / ROBIN). Empty when every
        boundary is the implicit zero-Neumann default or a strong Dirichlet."""

        return {boundary.kind for boundary in self.boundary_terms}

    def mesh_quality_growth(self) -> float:
        """How far the mesh has distorted since this problem was built, as a
        cell-size ratio growth factor (1.0 for a fresh or static mesh, larger as
        prescribed motion deforms it). The ALE driver (`backend/ale.py`) remeshes
        when this crosses a configured limit, kept well below the hard
        `MeshQualityError` threshold `_MeshMotion.advance` enforces."""

        return 1.0 if self._motion is None else self._motion.current_growth()

    def integrand_of(self, kind: TermKind) -> UflExpr:
        """The UFL integrand of the (unique) term of `kind`. For invariant tests
        that assemble a single term's matrix (e.g. the stiffness `K`)."""

        return next(term.integrand for term in self.terms if term.kind is kind)

    @property
    def bilinear_form(self) -> ufl.Form:
        return self._compose_backward_euler()[0]

    @property
    def linear_form(self) -> ufl.Form:
        return self._compose_backward_euler()[1]

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
        # For a moving subdomain, advance the mesh first; field values are carried (material frame) and
        # the conservative mass term re-assembles on the new configuration via `motion.volume_ratio`.
        # When `owns_mesh_motion` is False an external driver already advanced this motion (and refreshed
        # its swept measures) — we only read them, so skip the move to avoid double-advancing.
        if self._motion is not None and self.owns_mesh_motion:
            self._motion.advance()
        self._backward_euler_problem().solve()
        self.previous.x.array[:] = self.unknown.x.array

    def advance_mesh(self) -> None:
        """Move the mesh by `dt·velocity` (prescribed motion), carrying the field values, **without**
        solving — the mesh-move half of `step` on its own. The method-of-lines ALE driver
        (`integrate_discrete_problem_moving`) calls this once per stride and then `TS`-integrates the
        reaction–diffusion on the now-fixed configuration, so the move magnitude per stride is set via
        `dt`. Raises on a static subdomain (no prescribed motion)."""

        if self._motion is None:
            raise RuntimeError("advance_mesh requires a prescribed motion velocity; this subdomain is static")
        self._motion.advance()

    def set_time(self, t: float) -> None:
        """Advance the bound time `t` and refresh any time-dependent boundary values.

        Sets the compile context's time Constant and re-interpolates each Dirichlet value
        Function, so a `g(t)` reflects the new time. With no time-dependent expressions it is a
        cheap no-op refresh. The driver calls this each step (the backward-Euler loop) or each
        `TS` callback (method-of-lines), so the same model works under either integrator."""

        if self.time is not None:
            self.time.value = t
        for value, expression in self.dirichlet_refreshers:
            value.interpolate(expression)

    def total_mass(self) -> float:
        local = fem.assemble_scalar(fem.form(self.unknown * self.dx))
        return float(self.V.mesh.comm.allreduce(local, op=MPI.SUM))

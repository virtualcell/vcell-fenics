"""A diffusion that loses its sign stops the run, with the cause named (ADR 021 decision 4).

A diffusion coefficient that reads a species can turn negative, as ``D (1 − m/m_max,d)`` does above ``m_max,d``.
Diffusion backwards in time is ill-posed: a run that crossed it would fail inside the integrator with a generic
message, or carry growing mesh-scale noise. Only a coefficient that reads an unknown (`formalism.linearity`) can
change sign during a run, so only those are checked:

- a scalar coefficient, bulk or surface, must not be negative at the vertices of its domain's cells;
- a tensor coefficient's symmetric part must not have a negative eigenvalue there;
- the minimum is collective over the ranks.

A coefficient that reaches zero is degenerate, not backwards, and is let through, as is a value below zero by
round-off (1e-12 of its largest magnitude).

The check reads the model's own expression, compiled as the forms compile it, so it needs no knowledge of
``m_max,d``. It is exact for a coefficient affine in a P1 field (its extremes are at the vertices); for another
coefficient a negative value between vertices can be missed (*inferred*).

Each method-of-lines integrator installs it as a TS monitor, on every path, strides included, so it runs after
every accepted step; and its failure path runs it once more on the last iterate (`explain`). A monitor's exception
leaves ``ts.solve`` as a ``PETSc.Error`` with it as the context; `raised_by` finds it there.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import basix
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl.algorithms.analysis import extract_coefficients, extract_constants

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.diagnostics import SolveError

# Relative to the coefficient's largest magnitude: a value this far below zero is round-off of a zero.
_ROUND_OFF = 1.0e-12


class DiffusionSignError(SolveError):
    """A species' diffusion coefficient became negative during a run (ADR 021 decision 4)."""


@dataclass(frozen=True)
class DiffusionCoefficient:
    """The diffusion of `species`, `coefficient` compiled on `mesh` from the current values of the unknowns, over
    `cells` (local indices of `mesh`'s cells; all of them when None)."""

    species: str
    coefficient: UflExpr
    mesh: dmesh.Mesh
    cells: npt.NDArray[np.int32] | None = None


class DiffusionSignCheck:
    """The check of a set of coefficients. `refresh(t, x)` makes the Functions (and the time) the coefficients read
    hold the accepted state `x` at `t`; the monitor calls it before each check."""

    def __init__(
        self,
        coefficients: Sequence[DiffusionCoefficient],
        refresh: Callable[[float, PETSc.Vec], None] | None = None,
    ) -> None:
        self._refresh = refresh
        self._entries: list[tuple[str, fem.Expression, dmesh.Mesh, npt.NDArray[np.int32], MPI.Comm]] = []
        self._syncs: list[Callable[[], None]] = []
        for c in coefficients:
            on_mesh = _on_mesh(c.coefficient, c.mesh)
            if on_mesh is None:
                warnings.warn(
                    f"the diffusion of {c.species!r} reads a field on another mesh; its sign is not checked "
                    f"(ADR 021 decision 4)",
                    stacklevel=2,
                )
                continue
            coefficient, sync = on_mesh
            self._syncs.append(sync)
            cell_type = basix.CellType[c.mesh.topology.cell_name()]
            vertices = basix.geometry(cell_type)  # the reference cell's vertices
            if c.cells is None:
                cell_map = c.mesh.topology.index_map(c.mesh.topology.dim)
                assert cell_map is not None
                cells = np.arange(cell_map.size_local, dtype=np.int32)
            else:
                cells = np.asarray(c.cells, dtype=np.int32)
            self._entries.append((c.species, fem.Expression(coefficient, vertices), c.mesh, cells, c.mesh.comm))

    def __bool__(self) -> bool:
        return bool(self._entries)

    def least(self) -> list[tuple[str, float]]:
        """Each species' least value of its coefficient (a tensor's least eigenvalue of its symmetric part), over
        every rank. Collective."""
        return [(species, low) for species, low, _ in self._extremes()]

    def _extremes(self) -> list[tuple[str, float, float]]:
        """Each species' least value and largest magnitude, over every rank. Collective."""

        for sync in self._syncs:
            sync()
        out: list[tuple[str, float, float]] = []
        for species, expression, mesh, cells, comm in self._entries:
            local, size = np.inf, 0.0
            if len(cells):
                values = np.real(expression.eval(mesh, cells))  # (cells, vertices, *shape)
                shape = expression.value_shape
                if len(shape) == 2:
                    tensors = values.reshape(-1, *shape)
                    symmetric = 0.5 * (tensors + np.swapaxes(tensors, 1, 2))
                    eigenvalues = np.linalg.eigvalsh(symmetric)
                    local, size = float(np.min(eigenvalues)), float(np.max(np.abs(eigenvalues)))
                else:
                    local, size = float(np.min(values)), float(np.max(np.abs(values)))
            out.append((species, float(comm.allreduce(local, op=MPI.MIN)), float(comm.allreduce(size, op=MPI.MAX))))
        return out

    def failure(self, time: float) -> str | None:
        """The message naming the first species whose coefficient is negative, or None. Collective."""

        for species, value, size in self._extremes():
            if value < -_ROUND_OFF * size:  # a zero that round-off leaves just below is degenerate, not negative
                return (
                    f"the diffusion of {species!r} became negative at t ≈ {time:.4g} (least value {value:.4g}): "
                    f"diffusion backwards in time is ill-posed, so the run stops. Its coefficient reads the "
                    f"species; check that it stays positive over the range the species reach (e.g. a crowding "
                    f"cut-off D·(1 − m/m_max) above m_max)."
                )
        return None

    def check(self, time: float) -> None:
        message = self.failure(time)
        if message is not None:
            raise DiffusionSignError(message)

    def monitor(self, _ts: PETSc.TS, _step: int, time: float, x: PETSc.Vec) -> None:
        """A TS monitor: after each accepted step (and at the start), check the accepted state."""

        if self._refresh is not None:
            self._refresh(time, x)
        self.check(time)

    def attach(self, ts: PETSc.TS) -> None:
        if self._entries:
            ts.setMonitor(self.monitor)

    def explain(self, ts: PETSc.TS) -> str | None:
        """After a failed solve: the message if the last iterate (left in the Functions by the last residual
        evaluation) has a negative coefficient, else None. Collective."""

        return self.failure(float(ts.getTime())) if self._entries else None


def _on_mesh(expression: UflExpr, mesh: dmesh.Mesh) -> tuple[UflExpr, Callable[[], None]] | None:
    """`expression` with everything it reads on `mesh`, so it can be evaluated there, and the function that brings
    the copies up to date. A species on a submesh reads parameters and the time as Constants on the parent mesh,
    and a region variable as a one-value (Real) Function there: each becomes a Constant on `mesh`, refreshed from
    its original before each check. None if it reads any other field on another mesh."""

    domain = mesh.ufl_domain()
    mirrors: dict[object, fem.Constant] = {}
    copies: list[Callable[[], None]] = []
    for constant in extract_constants(expression):
        if not isinstance(constant, fem.Constant) or constant.ufl_domain() == domain:
            continue
        mirror = fem.Constant(mesh, np.array(constant.value, dtype=PETSc.ScalarType))
        mirrors[constant] = mirror

        def copy_constant(source: fem.Constant = constant, target: fem.Constant = mirror) -> None:
            target.value = source.value

        copies.append(copy_constant)
    for coefficient in extract_coefficients(expression):
        if not isinstance(coefficient, fem.Function) or coefficient.function_space.mesh.ufl_domain() == domain:
            continue
        index_map = coefficient.function_space.dofmap.index_map
        if index_map.size_global * coefficient.function_space.dofmap.index_map_bs != 1:
            return None  # a field on another mesh: not ours to evaluate here
        real = fem.Constant(mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]  # petsc4py stubs ScalarType as a dtype
        mirrors[coefficient] = real

        def copy_real(source: fem.Function = coefficient, target: fem.Constant = real) -> None:
            owned = source.x.array[: source.function_space.dofmap.index_map.size_local]
            target.value = source.function_space.mesh.comm.allreduce(complex(np.sum(owned)).real, op=MPI.SUM)

        copies.append(copy_real)

    def sync() -> None:
        for copy in copies:
            copy()

    return (ufl.replace(expression, mirrors) if mirrors else expression), sync


def raised_by(error: BaseException) -> DiffusionSignError | None:
    """The `DiffusionSignError` a monitor raised inside ``ts.solve``, found in `error`'s chain."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, DiffusionSignError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


__all__ = ["DiffusionCoefficient", "DiffusionSignCheck", "DiffusionSignError", "raised_by"]

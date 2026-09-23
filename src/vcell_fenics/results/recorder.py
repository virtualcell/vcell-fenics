"""Interpolate each domain's solution into P1, reduce its statistics, and write one bundle row per time.

The successor of the CLI's XDMF ``_Recorder``: the same P1 interpolation (so ``--fe-degree 2`` and
several species blocked in one unknown still produce one scalar P1 field per species) and the same
MPI-reduced statistics — ``total = ∫u``, ``mean = total / |domain|``, ``min``/``max`` over owned dofs —
but for any number of domains at once, written through a :class:`~vcell_fenics.results.writer.BundleWriter`.

A domain registers its channels with a default *source* per channel (the live solver state); a capture
may pass other sources instead — the method-of-lines integrators hand over interpolated snapshots at
each output time rather than the live unknown.

A **moving** domain (``moving=True``, ALE) has its measure re-assembled at every capture, since the
domain changes size as it moves, and the writer records its point coordinates per row. After a remesh,
:meth:`BundleRecorder.start_segment` moves a domain's output onto the new mesh and opens a new bundle
segment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.mesh import Mesh
from mpi4py import MPI

from vcell_fenics.results.schema import DomainKind
from vcell_fenics.results.writer import BundleWriter, Stats


@dataclass
class _DomainChannels:
    name: str
    names: list[str]
    sources: list[Any]
    fields: list[fem.Function]
    integrals: list[fem.Form]
    measure: float
    space: fem.FunctionSpace
    measure_form: fem.Form
    moving: bool = False


class BundleRecorder:
    """Owns the P1 output functions and statistics forms for every domain of a run."""

    def __init__(self, writer: BundleWriter, *, comm: MPI.Comm) -> None:
        self._writer = writer
        self._comm = comm
        self._domains: list[_DomainChannels] = []
        self._records: list[dict[str, Any]] = []
        self._on_row: list[Callable[[float, int], None]] = []

    def add_domain(
        self, name: str, kind: DomainKind, mesh: Mesh, channels: Sequence[tuple[str, Any]], *, moving: bool = False
    ) -> None:
        """Register a domain and its ``(variable name, default source)`` channels (collective). ``moving``:
        an ALE domain whose mesh moves in place (see the module docstring)."""

        space = fem.functionspace(mesh, ("Lagrange", 1))
        self._writer.add_domain(name, kind, space, moving=moving)
        for variable, _ in channels:
            self._writer.add_variable(name, variable)
        self._domains.append(self._channels(name, space, [v for v, _ in channels], [s for _, s in channels], moving))

    def _channels(
        self, name: str, space: fem.FunctionSpace, names: list[str], sources: list[Any], moving: bool
    ) -> _DomainChannels:
        mesh = space.mesh
        fields = [fem.Function(space, name=variable) for variable in names]
        dx = ufl.Measure("dx", domain=mesh)
        measure_form = fem.form(1.0 * dx)
        return _DomainChannels(
            name=name,
            names=names,
            sources=sources,
            fields=fields,
            integrals=[fem.form(u * dx) for u in fields],
            measure=self._reduce(float(fem.assemble_scalar(measure_form).real), MPI.SUM),
            space=space,
            measure_form=measure_form,
            moving=moving,
        )

    def start_segment(self, remeshed: Mapping[str, tuple[Mesh, Sequence[Any]]]) -> int:
        """After a remesh (collective): move each named domain's output onto its new mesh, with new default
        sources, and start a new bundle segment. Returns the segment index."""

        for index, domain in enumerate(self._domains):
            if domain.name in remeshed:
                mesh, sources = remeshed[domain.name]
                space = fem.functionspace(mesh, ("Lagrange", 1))
                self._domains[index] = self._channels(domain.name, space, domain.names, list(sources), domain.moving)
        return self._writer.new_segment({domain.name: domain.space for domain in self._domains})

    def on_row(self, callback: Callable[[float, int], None]) -> None:
        """Call ``callback(t, row)`` after each row is written (e.g. to report a VCell DATA event)."""

        self._on_row.append(callback)

    def open(self) -> None:
        self._writer.open()

    def capture(
        self, t: float, sources: Mapping[str, Sequence[Any]] | None = None, *, progress: float | None = None
    ) -> dict[str, Any]:
        """Interpolate, reduce and write the state at time ``t`` (collective). ``sources`` maps a domain
        name to replacement sources for its channels (in registration order). Returns the row's
        statistics as ``{variable: {domain, total, mean, min, max}}``."""

        owned: dict[tuple[str, str], np.typing.NDArray[np.float64]] = {}
        stats: dict[tuple[str, str], Stats] = {}
        summary: dict[str, Any] = {}
        for domain in self._domains:
            if domain.moving:  # the domain's size changes as its mesh moves
                domain.measure = self._reduce(float(fem.assemble_scalar(domain.measure_form).real), MPI.SUM)
            live = sources.get(domain.name, domain.sources) if sources is not None else domain.sources
            if len(live) != len(domain.fields):
                raise ValueError(f"domain {domain.name!r} expects {len(domain.fields)} sources, got {len(live)}")
            for variable, source, out, integral in zip(
                domain.names, live, domain.fields, domain.integrals, strict=True
            ):
                out.interpolate(source)
                n_owned = out.function_space.dofmap.index_map.size_local
                values = out.x.array[:n_owned]
                total = self._reduce(float(fem.assemble_scalar(integral).real), MPI.SUM)
                mean = total / domain.measure if domain.measure > 0.0 else float("nan")
                low = self._reduce(float(np.min(values)) if values.size else np.inf, MPI.MIN)
                high = self._reduce(float(np.max(values)) if values.size else -np.inf, MPI.MAX)
                owned[(domain.name, variable)] = values
                stats[(domain.name, variable)] = (mean, total, low, high)
                summary[variable] = {"domain": domain.name, "total": total, "mean": mean, "min": low, "max": high}
        row = self._writer.write(t, owned, stats, progress=progress)
        self._records.append({"t": t, "species": summary})
        for callback in self._on_row:
            callback(t, row)
        return summary

    @property
    def records(self) -> list[dict[str, Any]]:
        """Every captured row's statistics, in order (for ``summary.json``)."""

        return self._records

    @property
    def domains(self) -> list[str]:
        """The registered domain names, in order."""

        return [d.name for d in self._domains]

    def measure(self, domain: str) -> float:
        return next(d.measure for d in self._domains if d.name == domain)

    def _reduce(self, value: float, op: MPI.Op) -> float:
        return float(self._comm.allreduce(value, op=op))

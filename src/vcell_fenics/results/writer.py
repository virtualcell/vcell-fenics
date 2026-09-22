"""Write a results bundle (ADR 010): one VTU per domain, zarr-v2 ``(T, N)`` fields, an atomic manifest.

Lifecycle (every call is collective; rank 0 does the file I/O)::

    writer = BundleWriter(path, comm=comm, planned_times=times, source=..., solver=...)
    writer.add_domain("cyt", "volume", V_cyt); writer.add_variable("cyt", "C")
    writer.open()                                  # VTUs + preallocated arrays + manifest(status=running)
    writer.write(t, {("cyt", "C"): owned}, {("cyt", "C"): (mean, total, min, max)})   # per output time
    writer.finalize("completed")                   # or "failed", message=...

A row is written before its time is appended to the manifest's ``times``, and the manifest is replaced
atomically (temp file + ``os.replace``), so a reader polling mid-run always sees complete rows only.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from numpy.typing import NDArray

from vcell_fenics.results.gather import P1Layout
from vcell_fenics.results.schema import (
    SCHEMA_VERSION,
    STATS_COLUMNS,
    DomainInfo,
    DomainKind,
    Manifest,
    Segment,
    SolverInfo,
    SourceInfo,
    Status,
    VariableInfo,
    manifest_to_attrs,
)
from vcell_fenics.results.vtu import write_vtu

Stats = tuple[float, float, float, float]  # mean, total, min, max (STATS_COLUMNS)


@dataclass
class _Domain:
    kind: DomainKind
    layout: P1Layout
    variables: list[str]


class BundleWriter:
    def __init__(
        self,
        path: Path,
        *,
        comm: MPI.Comm,
        planned_times: Sequence[float],
        source: SourceInfo,
        solver: SolverInfo,
        write_fields: bool = True,
    ) -> None:
        if not planned_times:
            raise ValueError("a bundle needs at least one planned output time")
        self.path = Path(path)
        self._comm = comm
        self._planned = tuple(float(t) for t in planned_times)
        self._source = source
        self._solver = solver
        self._write_fields = write_fields
        self._domains: dict[str, _Domain] = {}
        self._arrays: dict[tuple[str, str], Any] = {}
        self._stats: dict[tuple[str, str], Any] = {}
        self._manifest: Manifest | None = None

    # -- set-up -------------------------------------------------------------------------------------

    def add_domain(self, name: str, kind: DomainKind, space: fem.FunctionSpace) -> P1Layout:
        """Register a domain by its VCell name and its scalar P1 output space (collective)."""

        if name in self._domains:
            raise ValueError(f"domain {name!r} added twice")
        _check_name(name)
        layout = P1Layout(space)
        self._domains[name] = _Domain(kind=kind, layout=layout, variables=[])
        return layout

    def add_variable(self, domain: str, name: str) -> None:
        _check_name(name)
        if name in self._domains[domain].variables:
            raise ValueError(f"variable {name!r} added twice to domain {domain!r}")
        self._domains[domain].variables.append(name)

    def open(self) -> None:
        """Create the bundle: remove any previous one at ``path``, write each domain's VTU, preallocate
        the arrays, and publish the manifest with ``status: running`` and no times (collective)."""

        if self._comm.rank == 0:
            import numcodecs
            import zarr

            if self.path.exists():
                shutil.rmtree(self.path)
            group = zarr.open_group(str(self.path), mode="w", zarr_format=2)
            n_rows = len(self._planned)
            for name, domain in self._domains.items():
                (self.path / "mesh").mkdir(exist_ok=True)
                points, cells = domain.layout.points(), domain.layout.cells()
                assert points is not None and cells is not None
                write_vtu(self.path / "mesh" / f"{name}.vtu", points, cells, domain.layout.vtk_type)
                for variable in domain.variables:
                    key = (name, variable)
                    if self._write_fields:
                        self._arrays[key] = _create(
                            group, f"{name}/{variable}", (n_rows, domain.layout.n_points), ("time", "point"), numcodecs
                        )
                    self._stats[key] = _create(
                        group,
                        f"stats/{name}/{variable}",
                        (n_rows, len(STATS_COLUMNS)),
                        ("time", "statistic"),
                        numcodecs,
                    )
            self._manifest = self._initial_manifest()
            self._publish()
        self._comm.barrier()

    # -- per output time ----------------------------------------------------------------------------

    def write(
        self,
        t: float,
        fields: Mapping[tuple[str, str], NDArray[np.float64]],
        stats: Mapping[tuple[str, str], Stats],
        *,
        progress: float | None = None,
    ) -> int:
        """Write one output row: each ``(domain, variable)``'s owned P1 values and its statistics
        (already MPI-reduced). Collective. Returns the row index."""

        expected = {(d, v) for d, domain in self._domains.items() for v in domain.variables}
        if set(stats) != expected or (self._write_fields and set(fields) != expected):
            raise ValueError(f"write() needs every registered (domain, variable) exactly once: {sorted(expected)}")
        gathered = {key: self._domains[key[0]].layout.gather(values) for key, values in fields.items()}
        if self._comm.rank != 0:
            return -1
        manifest = self._require_manifest()
        row = len(manifest.times)
        if row >= self._arrays_rows():
            self._grow(row + 1)
        for key, values in gathered.items():
            if self._write_fields:
                assert values is not None
                self._arrays[key][row, :] = values
        for key, values_stats in stats.items():
            self._stats[key][row, :] = np.asarray(values_stats, dtype=np.float64)
        # Data first, then the manifest that makes the row visible.
        segment = replace(manifest.segments[0], count=row + 1)
        self._manifest = replace(
            manifest,
            times=(*manifest.times, float(t)),
            segments=(segment, *manifest.segments[1:]),
            progress=manifest.progress if progress is None else float(progress),
            updated=_now(),
        )
        self._publish()
        return row

    def write_provenance(self, name: str, text: str) -> None:
        """A convenience copy (the resolved math/geometry, a summary) under ``provenance/`` — rank 0."""

        if self._comm.rank == 0:
            target = self.path / "provenance" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def finalize(self, status: Status, message: str | None = None) -> None:
        """Publish the terminal status (collective; rank 0 writes). Safe to call if never opened."""

        if self._comm.rank == 0 and self._manifest is not None:
            self._manifest = replace(
                self._manifest,
                status=status,
                message=message,
                progress=1.0 if status == "completed" else self._manifest.progress,
                updated=_now(),
            )
            self._publish()
        self._comm.barrier()

    @property
    def manifest(self) -> Manifest | None:
        """The current manifest on rank 0 (``None`` elsewhere or before :meth:`open`)."""

        return self._manifest

    # -- internals ----------------------------------------------------------------------------------

    def _initial_manifest(self) -> Manifest:
        domains = {
            name: DomainInfo(
                kind=domain.kind,
                dim=domain.layout.tdim,
                gdim=domain.layout.gdim,
                mesh=f"mesh/{name}.vtu",
                n_points=domain.layout.n_points,
                n_cells=domain.layout.n_cells,
                cell_type=domain.layout.vtk_type,
            )
            for name, domain in self._domains.items()
        }
        variables = tuple(
            VariableInfo(name=v, domain=d, path=f"{d}/{v}", stats=f"stats/{d}/{v}")
            for d, domain in self._domains.items()
            for v in domain.variables
        )
        return Manifest(
            schema=SCHEMA_VERSION,
            status="running",
            times=(),
            planned_times=self._planned,
            segments=(Segment(index=0, t0=self._planned[0], count=0),),
            domains=domains,
            variables=variables,
            solver=self._solver,
            source=self._source,
            updated=_now(),
        )

    def _require_manifest(self) -> Manifest:
        if self._manifest is None:
            raise RuntimeError("BundleWriter.open() must be called before write()")
        return self._manifest

    def _arrays_rows(self) -> int:
        first = next(iter(self._stats.values()), None)
        return int(first.shape[0]) if first is not None else 0

    def _grow(self, rows: int) -> None:
        """More rows than planned (should not happen for a VCell schedule): extend every array."""

        for arrays in (self._arrays, self._stats):
            for array in arrays.values():
                array.resize((rows, *array.shape[1:]))

    def _publish(self) -> None:
        """Replace the root ``.zattrs`` atomically with the current manifest."""

        manifest = self._require_manifest()
        document = json.dumps(manifest_to_attrs(manifest), indent=1, sort_keys=True)
        handle, temp = tempfile.mkstemp(dir=self.path, prefix=".zattrs.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(document)
            os.replace(temp, self.path / ".zattrs")
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise


def _create(group: Any, path: str, shape: tuple[int, int], dims: tuple[str, str], numcodecs: Any) -> Any:
    array = group.create_array(
        path,
        shape=shape,
        chunks=(1, shape[1]),  # one chunk per output row: a row lands in a single file
        dtype="<f8",
        compressors=numcodecs.Zlib(level=1),
        fill_value=np.nan,
        order="C",
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)  # the xarray convention, so xarray can open the bundle
    return array


def _check_name(name: str) -> None:
    if not name or "/" in name or name.startswith(".") or name in ("mesh", "stats", "provenance"):
        raise ValueError(f"{name!r} cannot name a bundle domain or variable")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

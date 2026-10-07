"""Write a results bundle (ADR 010): one VTU per domain, zarr-v2 ``(T, N)`` fields, an atomic manifest.

Lifecycle (every call is collective; rank 0 does the file I/O)::

    writer = BundleWriter(path, comm=comm, planned_times=times, source=..., solver=...)
    writer.add_domain("cyt", "volume", V_cyt); writer.add_variable("cyt", "C")
    writer.open()                                  # VTUs + preallocated arrays + manifest(status=running)
    writer.write(t, {("cyt", "C"): owned}, {("cyt", "C"): (mean, total, min, max)})   # per output time
    writer.finalize("completed")                   # or "failed", message=...

A row is written before its time is appended to the manifest's ``times``, and the manifest is replaced
atomically (temp file + ``os.replace``), so a reader polling mid-run always sees complete rows only.

**Moving meshes** (ADR 010 §2–3). A domain added with ``moving=True`` is an ALE domain: its geometry moves
in place under a fixed topology, and each row also records the domain's point coordinates in
``<prefix><domain>/_coords`` ``(T_seg, N, 3)``. Such a bundle is ``profile: segmented`` with segments of
``motion: ale``. A remesh changes the topology: :meth:`BundleWriter.new_segment` starts a segment under
``seg000N/`` with its own meshes and arrays, and arrays are indexed by the row *within* their segment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import warnings
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
    ADJACENT_DIR,
    SCHEMA_VERSION,
    STATS_COLUMNS,
    Adjacency,
    DomainInfo,
    DomainKind,
    Manifest,
    Motion,
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
    moving: bool = False
    sides: tuple[str, ...] | None = None  # a membrane's adjacent compartments


COORDS = "_coords"  # a moving domain's per-row point coordinates, (T_seg, N, 3)


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
        self._coords: dict[str, Any] = {}
        self._manifest: Manifest | None = None
        self._segment_row0 = 0  # the global row at which the current segment starts

    # -- set-up -------------------------------------------------------------------------------------

    def add_domain(
        self,
        name: str,
        kind: DomainKind,
        space: fem.FunctionSpace,
        *,
        moving: bool = False,
        sides: tuple[str, ...] | None = None,
    ) -> P1Layout:
        """Register a domain by its VCell name and its scalar P1 output space (collective). ``moving``: the
        mesh moves in place (ALE) and each row records its point coordinates. ``sides``: a membrane's two
        adjacent compartments — each segment then records, for every one that is a domain of the bundle, the
        map from the membrane's points to that compartment's (ADR 010 §3, "Membrane adjacency"). The
        membrane and its compartments must be submeshes of one parent mesh."""

        if name in self._domains:
            raise ValueError(f"domain {name!r} added twice")
        _check_name(name)
        if sides is not None and kind != "membrane":
            raise ValueError(f"only a membrane has sides; {name!r} is a {kind}")
        layout = P1Layout(space)
        self._domains[name] = _Domain(kind=kind, layout=layout, variables=[], moving=moving, sides=sides)
        return layout

    @property
    def moving(self) -> bool:
        return any(domain.moving for domain in self._domains.values())

    def add_variable(self, domain: str, name: str) -> None:
        _check_name(name)
        if name in self._domains[domain].variables:
            raise ValueError(f"variable {name!r} added twice to domain {domain!r}")
        self._domains[domain].variables.append(name)

    def open(self) -> None:
        """Create the bundle: write each domain's VTU, preallocate the arrays, and publish the manifest
        with ``status: running`` and no times (collective).

        The bundle appears **atomically**: it is built in a hidden staging directory beside ``path`` and
        renamed into place only once its manifest is written, so a reader polling ``path`` sees either
        no bundle or a complete one — never zarr's placeholder ``.zattrs`` without a manifest. A previous
        bundle at ``path`` is replaced."""

        if self._comm.rank == 0:
            import numcodecs
            import zarr

            staging = self.path.parent / f".{self.path.name}.staging-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            group = zarr.open_group(str(staging), mode="w", zarr_format=2)
            array_paths = self._create_segment_arrays(group, staging, "", len(self._planned), numcodecs)
            self._manifest = self._initial_manifest()
            self._publish(staging)
            if self.path.exists():
                shutil.rmtree(self.path)
            os.rename(staging, self.path)
            self._bind_arrays(zarr.open_group(str(self.path), mode="r+", zarr_format=2), array_paths)
        self._comm.barrier()

    def new_segment(self, spaces: Mapping[str, fem.FunctionSpace]) -> int:
        """Start a new segment after a remesh (collective): every registered domain's new scalar P1 output
        space, so each gets a new VTU and new arrays under the segment's prefix. Rows written after this
        land in the new segment. Returns the segment index."""

        if set(spaces) != set(self._domains):
            raise ValueError(f"new_segment() needs a space for every domain: {sorted(self._domains)}")
        for name, space in spaces.items():
            self._domains[name].layout = P1Layout(space)
        index = -1
        if self._comm.rank == 0:
            import numcodecs
            import zarr

            manifest = self._require_manifest()
            index = len(manifest.segments)
            prefix = f"seg{index:04d}/"
            rows = max(1, len(self._planned) - len(manifest.times))
            group = zarr.open_group(str(self.path), mode="r+", zarr_format=2)
            array_paths = self._create_segment_arrays(group, self.path / prefix, prefix, rows, numcodecs)
            self._arrays.clear()
            self._stats.clear()
            self._coords.clear()
            self._bind_arrays(zarr.open_group(str(self.path), mode="r+", zarr_format=2), array_paths)
            motion: Motion = "ale" if self.moving else "none"
            # t0 is the segment's first written time; until then, the next planned one (JSON has no NaN)
            done = len(manifest.times)
            t0 = self._planned[done] if done < len(self._planned) else (manifest.times[-1] if done else 0.0)
            segment = Segment(index=index, t0=float(t0), count=0, motion=motion, prefix=prefix)
            self._manifest = replace(manifest, segments=(*manifest.segments, segment), updated=_now())
            self._segment_row0 = len(manifest.times)
            self._publish()
        index = int(self._comm.bcast(index, root=0))
        self._segment_row0 = int(self._comm.bcast(self._segment_row0, root=0))
        return index

    def _create_segment_arrays(
        self, group: Any, root: Path, prefix: str, n_rows: int, numcodecs: Any
    ) -> dict[tuple[str, str], tuple[str | None, str]]:
        """Write each domain's VTU under ``root/mesh`` and preallocate the segment's arrays (rank 0)."""

        array_paths: dict[tuple[str, str], tuple[str | None, str]] = {}
        for name, domain in self._domains.items():
            (root / "mesh").mkdir(parents=True, exist_ok=True)
            points, cells = domain.layout.points(), domain.layout.cells()
            assert points is not None and cells is not None
            write_vtu(root / "mesh" / f"{name}.vtu", points, cells, domain.layout.vtk_type)
            n_points = domain.layout.n_points
            if domain.moving:
                _create(group, f"{prefix}{name}/{COORDS}", (n_rows, n_points, 3), ("time", "point", "xyz"), numcodecs)
                array_paths[(name, COORDS)] = (f"{prefix}{name}/{COORDS}", "")
            for variable in domain.variables:
                field_path = f"{prefix}{name}/{variable}" if self._write_fields else None
                stats_path = f"{prefix}stats/{name}/{variable}"
                if field_path is not None:
                    _create(group, field_path, (n_rows, n_points), ("time", "point"), numcodecs)
                _create(group, stats_path, (n_rows, len(STATS_COLUMNS)), ("time", "statistic"), numcodecs)
                array_paths[(name, variable)] = (field_path, stats_path)
            for compartment in self._adjacent(name):
                _write_map(group, f"{prefix}{_map_path(name, compartment)}", self._map(name, compartment), numcodecs)
        return array_paths

    def _adjacent(self, membrane: str) -> list[str]:
        """The membrane's adjacent compartments that are domains of the bundle."""

        sides = self._domains[membrane].sides or ()
        return [c for c in sides if c in self._domains and self._domains[c].kind == "volume"]

    def _map(self, membrane: str, compartment: str) -> NDArray[np.int32]:
        """Each membrane point's index among the compartment's points — the point with the same parent-mesh
        input index, so the same vertex — or -1 (rank 0)."""

        keys = self._domains[membrane].layout.keys()
        targets = self._domains[compartment].layout.keys()
        assert keys is not None and targets is not None
        where = np.searchsorted(targets, keys)
        found = where < targets.size
        found[found] = targets[where[found]] == keys[found]
        mapped = np.where(found, where, -1).astype(np.int32)
        if not found.all():
            warnings.warn(
                f"{int((~found).sum())} of {keys.size} points of membrane {membrane!r} are not points of "
                f"{compartment!r}; they map to -1",
                stacklevel=2,
            )
        return mapped

    def _bind_arrays(self, group: Any, array_paths: Mapping[tuple[str, str], tuple[str | None, str]]) -> None:
        for (domain, variable), (field_path, stats_path) in array_paths.items():
            if variable == COORDS:
                assert field_path is not None
                self._coords[domain] = group[field_path]
                continue
            if field_path is not None:
                self._arrays[(domain, variable)] = group[field_path]
            self._stats[(domain, variable)] = group[stats_path]

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
        coords = {name: domain.layout.gather_coords() for name, domain in self._domains.items() if domain.moving}
        if self._comm.rank != 0:
            return -1
        manifest = self._require_manifest()
        row = len(manifest.times)
        local = row - self._segment_row0  # arrays are indexed within their segment
        if local >= self._arrays_rows():
            self._grow(local + 1)
        for key, values in gathered.items():
            if self._write_fields:
                assert values is not None
                self._arrays[key][local, :] = values
        for key, values_stats in stats.items():
            self._stats[key][local, :] = np.asarray(values_stats, dtype=np.float64)
        for name, points in coords.items():
            assert points is not None
            self._coords[name][local, :, :] = points
        # Data first, then the manifest that makes the row visible.
        current = manifest.segments[-1]
        segment = replace(current, count=local + 1, t0=float(t) if local == 0 else current.t0)
        self._manifest = replace(
            manifest,
            times=(*manifest.times, float(t)),
            segments=(*manifest.segments[:-1], segment),
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
                adjacent=self._adjacency(name),
            )
            for name, domain in self._domains.items()
        }
        variables = tuple(
            VariableInfo(name=v, domain=d, path=f"{d}/{v}", stats=f"stats/{d}/{v}")
            for d, domain in self._domains.items()
            for v in domain.variables
        )
        moving = self.moving
        return Manifest(
            schema=SCHEMA_VERSION,
            status="running",
            times=(),
            planned_times=self._planned,
            segments=(Segment(index=0, t0=self._planned[0], count=0, motion="ale" if moving else "none"),),
            profile="segmented" if moving else "fixed",
            domains=domains,
            variables=variables,
            solver=self._solver,
            source=self._source,
            updated=_now(),
        )

    def _adjacency(self, name: str) -> Adjacency | None:
        sides = self._domains[name].sides
        if sides is None:
            return None
        return Adjacency(compartments=sides, maps={c: _map_path(name, c) for c in self._adjacent(name)})

    def _require_manifest(self) -> Manifest:
        if self._manifest is None:
            raise RuntimeError("BundleWriter.open() must be called before write()")
        return self._manifest

    def _arrays_rows(self) -> int:
        first = next(iter(self._stats.values()), None)
        return int(first.shape[0]) if first is not None else 0

    def _grow(self, rows: int) -> None:
        """More rows than planned (should not happen for a VCell schedule): extend every array."""

        for arrays in (self._arrays, self._stats, self._coords):
            for array in arrays.values():
                array.resize((rows, *array.shape[1:]))

    def _publish(self, root: Path | None = None) -> None:
        """Replace the root ``.zattrs`` atomically with the current manifest."""

        manifest = self._require_manifest()
        directory = self.path if root is None else root
        # Not sorted: `domains` keeps registration order, which readers take as the default (compartments first).
        document = json.dumps(manifest_to_attrs(manifest), indent=1)
        handle, temp = tempfile.mkstemp(dir=directory, prefix=".zattrs.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(document)
            os.replace(temp, directory / ".zattrs")
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise


def _map_path(membrane: str, compartment: str) -> str:
    return f"{membrane}/{ADJACENT_DIR}/{compartment}"


def _write_map(group: Any, path: str, values: NDArray[np.int32], numcodecs: Any) -> None:
    """A membrane's point map onto one side: int32 ``(n_points,)`` in one chunk, written at once."""

    array = group.create_array(
        path,
        shape=values.shape,
        chunks=values.shape if values.size else (1,),
        dtype="<i4",
        compressors=numcodecs.Zlib(level=1),
        fill_value=-1,
        order="C",
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["point"]
    if values.size:
        array[:] = values


def _create(group: Any, path: str, shape: tuple[int, ...], dims: tuple[str, ...], numcodecs: Any) -> None:
    array = group.create_array(
        path,
        shape=shape,
        chunks=(1, *shape[1:]),  # one chunk per output row: a row lands in a single file
        dtype="<f8",
        compressors=numcodecs.Zlib(level=1),
        fill_value=np.nan,
        order="C",
    )
    array.attrs["_ARRAY_DIMENSIONS"] = list(dims)  # the xarray convention, so xarray can open the bundle


_SEGMENT_DIR = re.compile(r"^seg\d{4}$")


def _check_name(name: str) -> None:
    if (
        not name
        or "/" in name
        or name.startswith(".")
        or name in ("mesh", "stats", "provenance", COORDS, ADJACENT_DIR)
        or _SEGMENT_DIR.match(name)
    ):
        raise ValueError(f"{name!r} cannot name a bundle domain or variable")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

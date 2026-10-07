"""Read a results bundle (ADR 010) — the reference reader, used by tests, export and CI smoke checks.

Follows the reader rules: the manifest's ``times`` (per segment, ``count``) says how many rows exist —
arrays are preallocated and unwritten rows read as NaN — and a bundle with a newer ``schema`` is
refused. Safe to use while the run is still writing (it re-reads the manifest on :meth:`Bundle.refresh`).

    python -m vcell_fenics.results BUNDLE [--require-status completed] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vcell_fenics.results.schema import MANIFEST_KEY, Manifest, Segment, manifest_from_attrs
from vcell_fenics.results.vtu import VtuGrid, read_vtu_strict


class Bundle:
    def __init__(self, path: Path, manifest: Manifest) -> None:
        self.path = path
        self.manifest = manifest
        self._group: Any = None

    @classmethod
    def open(cls, path: str | Path) -> Bundle:
        root = Path(path)
        attrs_file = root / ".zattrs"
        if not attrs_file.is_file():
            raise FileNotFoundError(f"{root} is not a results bundle (no .zattrs)")
        return cls(root, manifest_from_attrs(json.loads(attrs_file.read_text())))

    def refresh(self) -> Bundle:
        """Re-read the manifest (a running solver appends times as rows land)."""

        self.manifest = Bundle.open(self.path).manifest
        return self

    @property
    def times(self) -> tuple[float, ...]:
        return self.manifest.times

    @property
    def status(self) -> str:
        return self.manifest.status

    def segment_of(self, row: int) -> tuple[Segment, int]:
        """The segment holding output row ``row``, and the row's index within it (a segment's arrays are
        indexed by that local row)."""

        self._check_row(row)
        start = 0
        for segment in self.manifest.segments:
            if row < start + segment.count:
                return segment, row - start
            start += segment.count
        # the manifest lists a time before its segment's count catches up: it is the last segment's
        last = self.manifest.segments[-1]
        return last, row - (start - last.count)

    def mesh(self, domain: str, row: int | None = None) -> VtuGrid:
        """The domain's mesh — of the segment holding ``row`` (default: the first segment). A moving
        domain's VTU holds its segment's *initial* points; see :meth:`coords` for a row's points."""

        prefix = self.segment_of(row)[0].prefix if row is not None else self.manifest.segments[0].prefix
        return read_vtu_strict(self.path / (prefix + self.manifest.domains[domain].mesh))

    def coords(self, domain: str, row: int) -> NDArray[np.float64]:
        """The domain's point coordinates at output row ``row``, (n_points, 3): the recorded positions of a
        moving (ALE) segment, else the segment mesh's points."""

        segment, local = self.segment_of(row)
        if segment.motion == "ale":
            path = f"{segment.prefix}{domain}/_coords"
            if (self.path / path).is_dir():
                coords: NDArray[np.float64] = np.asarray(self._array(path)[local, :, :])
                return coords
        points = self.mesh(domain, row).points
        if points.shape[1] < 3:
            points = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
        return points

    def adjacent(self, membrane: str, compartment: str, row: int | None = None) -> NDArray[np.int32] | None:
        """The membrane's point map onto an adjacent ``compartment``, for the segment holding ``row``
        (default: the first): point ``i`` of the membrane is point ``map[i]`` of the compartment's mesh, or
        -1. ``None`` when the bundle records no such map (an older writer, or the compartment is not a domain
        of the bundle)."""

        adjacency = self.manifest.domains[membrane].adjacent
        if adjacency is None or compartment not in adjacency.maps:
            return None
        prefix = self.segment_of(row)[0].prefix if row is not None else self.manifest.segments[0].prefix
        values: NDArray[np.int32] = np.asarray(self._array(prefix + adjacency.maps[compartment])[:], dtype=np.int32)
        return values

    def field(self, domain: str, variable: str, row: int) -> NDArray[np.float64]:
        """The P1 values of ``variable`` at output row ``row`` (the point order of that row's mesh)."""

        segment, local = self.segment_of(row)
        values: NDArray[np.float64] = np.asarray(
            self._array(segment.prefix + self.manifest.variable(domain, variable).path)[local, :]
        )
        return values

    def series(self, domain: str, variable: str) -> NDArray[np.float64]:
        """All written rows of ``variable``: shape (len(times), n_points). A bundle whose mesh changes
        between segments has no single point set — read it row by row with :meth:`field`."""

        if len(self.manifest.segments) > 1:
            raise ValueError("a remeshed bundle has a different mesh per segment; read rows with field()")
        segment = self.manifest.segments[0]
        values: NDArray[np.float64] = np.asarray(
            self._array(segment.prefix + self.manifest.variable(domain, variable).path)[: len(self.times), :]
        )
        return values

    def stats(self, domain: str, variable: str) -> NDArray[np.float64]:
        """All written rows of the statistics: shape (len(times), len(stats_columns)), across segments."""

        path = self.manifest.variable(domain, variable).stats
        parts = []
        remaining = len(self.times)
        for segment in self.manifest.segments:
            count = min(segment.count, remaining)
            if count > 0:
                parts.append(np.asarray(self._array(segment.prefix + path)[:count, :]))
            remaining -= count
        if not parts:
            return np.empty((0, len(self.manifest.stats_columns)))
        values: NDArray[np.float64] = np.concatenate(parts)
        return values

    def _check_row(self, row: int) -> None:
        if not 0 <= row < len(self.times):
            raise IndexError(f"row {row} not written (bundle has {len(self.times)} rows)")

    def _array(self, path: str) -> Any:
        if self._group is None:
            import zarr

            self._group = zarr.open_group(str(self.path), mode="r", zarr_format=2)
        return self._group[path]


def _summary(bundle: Bundle) -> dict[str, Any]:
    manifest = bundle.manifest
    out: dict[str, Any] = {
        "status": manifest.status,
        "message": manifest.message,
        "rows": len(manifest.times),
        "planned": len(manifest.planned_times),
        "times": list(manifest.times),
        "domains": {
            name: {"kind": d.kind, "points": d.n_points, "cells": d.n_cells} for name, d in manifest.domains.items()
        },
        "variables": {},
    }
    for variable in manifest.variables:
        stats = bundle.stats(variable.domain, variable.name)
        last = stats[-1] if stats.shape[0] else [math.nan] * len(manifest.stats_columns)
        out["variables"][f"{variable.domain}/{variable.name}"] = dict(
            zip(manifest.stats_columns, (float(v) for v in last), strict=True)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vcell_fenics.results", description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--require-status", choices=("running", "completed", "failed"))
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    try:
        bundle = Bundle.open(args.bundle)
        summary = _summary(bundle)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{args.bundle}: {summary['status']}, {summary['rows']}/{summary['planned']} rows")
        for name, domain in summary["domains"].items():
            print(f"  {name} ({domain['kind']}): {domain['points']} points, {domain['cells']} cells")
        for name, last in summary["variables"].items():
            print(
                f"  {name} @ t={summary['times'][-1] if summary['times'] else 'n/a'}: "
                + ", ".join(f"{k}={v:.6g}" for k, v in last.items())
            )
    if args.require_status and bundle.status != args.require_status:
        print(f"error: status {bundle.status!r}, required {args.require_status!r}", file=sys.stderr)
        return 1
    return 0


__all__ = ["MANIFEST_KEY", "Bundle", "main"]

if __name__ == "__main__":
    sys.exit(main())

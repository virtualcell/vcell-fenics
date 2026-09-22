"""Read a results bundle (ADR 010) — the reference reader, used by tests, export and CI smoke checks.

Follows the reader rules: the manifest's ``times`` (per segment, ``count``) says how many rows exist —
arrays are preallocated and unwritten rows read as NaN — and a bundle with a newer ``schema`` is
refused. Safe to use while the run is still writing (it re-reads the manifest on :meth:`Bundle.refresh`).

    python -m vcell_fenics.results.reader BUNDLE [--require-status completed] [--json]
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

from vcell_fenics.results.schema import MANIFEST_KEY, Manifest, manifest_from_attrs
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

    def mesh(self, domain: str) -> VtuGrid:
        return read_vtu_strict(self.path / self.manifest.domains[domain].mesh)

    def field(self, domain: str, variable: str, row: int) -> NDArray[np.float64]:
        """The P1 values of ``variable`` at output row ``row`` (VTU point order)."""

        self._check_row(row)
        values: NDArray[np.float64] = np.asarray(self._array(self.manifest.variable(domain, variable).path)[row, :])
        return values

    def series(self, domain: str, variable: str) -> NDArray[np.float64]:
        """All written rows of ``variable``: shape (len(times), n_points)."""

        values: NDArray[np.float64] = np.asarray(
            self._array(self.manifest.variable(domain, variable).path)[: len(self.times), :]
        )
        return values

    def stats(self, domain: str, variable: str) -> NDArray[np.float64]:
        """All written rows of the statistics: shape (len(times), len(stats_columns))."""

        values: NDArray[np.float64] = np.asarray(
            self._array(self.manifest.variable(domain, variable).stats)[: len(self.times), :]
        )
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
    parser = argparse.ArgumentParser(prog="python -m vcell_fenics.results.reader", description=__doc__)
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

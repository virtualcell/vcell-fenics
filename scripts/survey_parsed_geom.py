#!/usr/bin/env python
"""Survey the parsed VCell geometries for import-layer coverage (geometry analogue of
`survey_parsed_math.py`).

Runs in the vcell-fenics **dev env**. For each `vcml_biomodels/parsed/*_geom.yaml` it
reconstructs a pyvcell `Geometry`, imports it to a `GeometryDescription`, validates, and
records structure (dim, subvolume types, #surfaces) and any validation findings. Writes a
printed summary + `geom_survey.csv`.

Note: the parsed geom YAMLs had the raw image voxel blob (`compressed_content`) stripped at
parse time, so an `image` is patched with an empty blob purely so the pydantic model
reconstructs — the importer carries image metadata only, never the blob.

Usage: `.pixi/envs/dev/bin/python scripts/survey_parsed_geom.py [--limit N]`
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import pyvcell.vcml.models_geometry as g
import yaml

from vcell_fenics.formalism import validate_geometry
from vcell_fenics.pyvcell_bridge import import_geometry

_ROOT = Path(__file__).resolve().parent.parent
_PARSED = _ROOT / "vcml_biomodels" / "parsed"


def _reconstruct(data: dict[str, Any]) -> Any:
    image = data.get("image")
    if isinstance(image, dict) and "compressed_content" not in image:
        image["compressed_content"] = ""  # blob stripped at parse time; importer never uses it
    return g.Geometry.model_validate(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey parsed VCell geometries")
    parser.add_argument("--parsed", type=Path, default=_PARSED)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    files = sorted(args.parsed.glob("*_geom.yaml"))
    if args.limit is not None:
        files = files[: args.limit]
    total = len(files)
    print(f"Surveying {total} geometries in {args.parsed}\n")

    buckets: Counter[str] = Counter()
    dims: Counter[int] = Counter()
    subvol_types: Counter[str] = Counter()
    n_surfaces: Counter[int] = Counter()
    validate_reasons: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for f in files:
        try:
            data = yaml.safe_load(f.read_text())
            vcml_geom = _reconstruct(data)
        except Exception as exc:
            buckets["reconstruct_fail"] += 1
            rows.append({"file": f.name, "bucket": "reconstruct_fail", "detail": f"{type(exc).__name__}: {exc}"[:120]})
            continue

        try:
            gd = import_geometry(vcml_geom)
        except Exception as exc:
            buckets["import_error"] += 1
            rows.append({"file": f.name, "bucket": "import_error", "detail": f"{type(exc).__name__}: {exc}"[:120]})
            continue

        errors = [d for d in validate_geometry(gd) if d.severity == "error"]
        bucket = "ok" if not errors else "validate_fail"
        buckets[bucket] += 1
        dims[gd.dim] += 1
        n_surfaces[len(gd.surfaces)] += 1
        for sv in gd.subvolumes:
            subvol_types[sv.type] += 1
        if errors:
            validate_reasons[errors[0].message[:90]] += 1
        rows.append(
            {
                "file": f.name,
                "bucket": bucket,
                "dim": gd.dim,
                "subvolumes": len(gd.subvolumes),
                "surfaces": len(gd.surfaces),
                "detail": errors[0].message[:120] if errors else "",
            }
        )

    print("=== import buckets ===")
    for name, n in buckets.most_common():
        print(f"  {n:5d}  {name}  ({100 * n / total:.1f}%)")
    print("\n=== dim ===")
    for d, n in sorted(dims.items()):
        print(f"  {n:5d}  dim {d}")
    print("\n=== subvolume types ===")
    for name, n in subvol_types.most_common():
        print(f"  {n:5d}  {name}")
    print("\n=== #surfaces per geometry ===")
    for k, n in sorted(n_surfaces.items()):
        print(f"  {n:5d}  {k} surface(s)")
    if validate_reasons:
        print("\n=== top validate_fail reasons ===")
        for msg, n in validate_reasons.most_common(12):
            print(f"  {n:5d}  {msg}")

    csv_path = args.parsed / "geom_survey.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file", "bucket", "dim", "subvolumes", "surfaces", "detail"], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

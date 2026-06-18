#!/usr/bin/env python
"""Parse downloaded VCML biomodels into per-application math + geometry YAML.

Run with the FULL pyvcell env (needs lxml + the VcmlReader; the vcell-fenics dev env has
pyvcell --no-deps):

    ../pyvcell/.venv/bin/python scripts/parse_biomodels_to_yaml.py
    # or: cd ../pyvcell && uv run python ../vcell-fenics/scripts/parse_biomodels_to_yaml.py

For each `vcml_biomodels/biomodel_<id>.vcml`, it loads the Biomodel and, for each
Application, writes (into `vcml_biomodels/parsed/` by default):

    biomodel_<id>_<app>_math.yaml   - the generated MathDescription (if the app has one)
    biomodel_<id>_<app>_geom.yaml   - the Application's Geometry

`<app>` is a filesystem-safe slug of the application name (Applications carry no id/key in
the pyvcell model). Both YAMLs are trimmed (`exclude_none`, `exclude_defaults`). The
geometry's raw segmented-image blob (`image.compressed_content`) is omitted — it is a large
base64 payload, not useful as readable YAML, and recoverable from the .vcml if ever needed;
the image metadata (name, size, pixel classes) and the analytic/CSG subvolumes are kept.

Failures and skips are recorded in `<out>/parse_failures.csv` with columns
`biomodel_id, application, stage, detail`, where stage is one of: `load` (the whole file
failed to parse), `no_applications` (biomodel had none), `no_math` (an app had no generated
math), `math` / `geometry` (a per-app dump failed).

Options: `--vcml-dir DIR`, `--out DIR`, `--limit N`.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pyvcell.vcml.vcml_reader import VcmlReader

_ROOT = Path(__file__).resolve().parent.parent
_VCML_DIR = _ROOT / "vcml_biomodels"


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("_")
    return cleaned[:80] or "app"


def _err(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:300] if text else type(exc).__name__


def _dump(model: Any, path: Path, *, exclude: Any = None) -> None:
    # mode="json" coerces StrEnum / tuples to YAML-safe primitives (the pyvcell models use
    # enums like SubVolumeType / MathVariableType that yaml.safe_dump cannot represent).
    data = model.model_dump(mode="json", exclude_none=True, exclude_defaults=True, exclude=exclude)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse VCML biomodels into per-app math/geom YAML")
    parser.add_argument("--vcml-dir", type=Path, default=_VCML_DIR, help="dir of biomodel_<id>.vcml files")
    parser.add_argument("--out", type=Path, default=_VCML_DIR / "parsed", help="output dir")
    parser.add_argument("--limit", type=int, default=None, help="process at most N vcml files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.vcml_dir.glob("biomodel_*.vcml"))
    if args.limit is not None:
        files = files[: args.limit]
    total = len(files)
    print(f"{total} vcml files -> {args.out}")

    failures: list[tuple[str, str, str, str]] = []
    loaded = apps = math_yaml = geom_yaml = 0
    for i, vcml_file in enumerate(files, 1):
        bid = vcml_file.stem[len("biomodel_") :]
        try:
            biomodel = VcmlReader.biomodel_from_file(vcml_file)
        except Exception as exc:  # malformed / unsupported VCML
            failures.append((bid, "", "load", _err(exc)))
            print(f"[{i}/{total}] LOAD FAIL {bid}: {exc!r}", file=sys.stderr)
            continue
        loaded += 1

        if not biomodel.applications:
            failures.append((bid, "", "no_applications", "biomodel has no applications (physiology only)"))

        seen: dict[str, int] = {}
        for app in biomodel.applications:
            apps += 1
            slug = _slug(app.name)
            if slug in seen:
                seen[slug] += 1
                slug = f"{slug}_{seen[slug]}"
            else:
                seen[slug] = 0
            stem = f"biomodel_{bid}_{slug}"

            try:
                _dump(app.geometry, args.out / f"{stem}_geom.yaml", exclude={"image": {"compressed_content"}})
                geom_yaml += 1
            except Exception as exc:
                failures.append((bid, app.name, "geometry", _err(exc)))

            if app.math_description is None:
                failures.append((bid, app.name, "no_math", "application has no generated math description"))
                continue
            try:
                _dump(app.math_description, args.out / f"{stem}_math.yaml")
                math_yaml += 1
            except Exception as exc:
                failures.append((bid, app.name, "math", _err(exc)))

        if i % 100 == 0 or i == total:
            print(f"[{i}/{total}] loaded={loaded} apps={apps} math={math_yaml} geom={geom_yaml} issues={len(failures)}")

    csv_path = args.out / "parse_failures.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["biomodel_id", "application", "stage", "detail"])
        writer.writerows(failures)

    print(
        f"\nDone. files={total} loaded={loaded} apps={apps} "
        f"math_yaml={math_yaml} geom_yaml={geom_yaml} issues={len(failures)}"
    )
    print(f"Issues CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

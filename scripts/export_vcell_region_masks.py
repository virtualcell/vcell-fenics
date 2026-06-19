#!/usr/bin/env python
"""Stage 1 of the VCell ↔ vcell-fenics geometry cross-check — run in the **pyvcell full env**
(`../pyvcell/.venv`; needs pyvcell >= 0.3.3 with the `native` + `solver` extras), NOT the vcell-fenics
dev env.

For each 2D-analytic parsed corpus geometry, reconstruct the pyvcell `Geometry` and call
`geometry.to_cartesian_mesh(...)` (pyvcell 0.3.3) — VCell's own rasterization of the analytic
geometry into a region image — then save the per-voxel centres and their VCell subvolume
(`domain_name`) labels to an `.npz`. The dev-env comparison (`check_vcell_membership.py`) reads these
and checks vcell-fenics classifies the same points into the same subvolumes VCell does.

The voxel centres come straight from `CartesianMesh.coordinates`, and the region map is aligned to
them, so no mesh-convention is re-derived downstream. Usage:

  ../pyvcell/.venv/bin/python scripts/export_vcell_region_masks.py [--limit N] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pyvcell.vcml.models_geometry as g
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_PARSED = _ROOT / "vcml_biomodels" / "parsed"


def _reconstruct(data: dict[str, Any]) -> Any:
    image = data.get("image")
    if isinstance(image, dict) and "compressed_content" not in image:
        image["compressed_content"] = ""
    return g.Geometry.model_validate(data)


def _export(geom_yaml: Path, out_dir: Path) -> str:
    data = yaml.safe_load(geom_yaml.read_text())
    geo = _reconstruct(data)
    if int(geo.dim) != 2 or not any(s.subvolume_type.value == "analytic" for s in geo.subvolumes):
        return "skip_not_2d_analytic"

    mesh = geo.to_cartesian_mesh()  # default resolution; VCell rasterizes the analytic geometry
    nx, ny, nz = mesh.size
    coords = np.asarray(mesh.coordinates).reshape(-1, 3)  # (N,3), C-order over (nx,ny,nz)
    # VCell flattens the region map x-fastest (m = i + nx*j + nx*ny*k); reshape (order='F') to
    # (nx,ny,nz) so it indexes [i,j,k] like coordinates, then flatten C-order to align with coords.
    region = np.asarray(mesh.volume_region_map).reshape((nx, ny, nz), order="F").reshape(-1)
    domain_of = {vid: dom for vid, _sub, _vol, dom in mesh.volume_regions}
    domains = np.array([domain_of[int(r)] for r in region])

    name = geom_yaml.stem
    np.savez(out_dir / f"{name}.npz", coords=coords, domains=domains, source=name, size=np.array(mesh.size))
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, default=_PARSED)
    parser.add_argument("--out", type=Path, default=_PARSED.parent / "region_masks")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for geom_yaml in sorted(args.parsed.glob("*_geom.yaml")):
        if saved >= args.limit:
            break
        try:
            status = _export(geom_yaml, args.out)
        except Exception as exc:  # noqa: BLE001 — math-gen / solver / non-spatial failures
            print(f"  error {geom_yaml.name}: {type(exc).__name__}: {exc}"[:120])
            continue
        if status == "ok":
            saved += 1
            print(f"  saved {geom_yaml.stem}.npz")

    print(f"\nsaved {saved} masks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

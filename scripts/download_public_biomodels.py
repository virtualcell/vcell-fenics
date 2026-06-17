#!/usr/bin/env python
"""Download VCell BioModels as raw VCML (`.vcml`) — a model pool for the import layer.

One-off data-collection utility. It opens a browser for **interactive OAuth2 login**
(use your VCell credentials), lists the BioModel summaries the server returns for you
(public + shared + owned), and saves each model's VCML to
`<vcell-fenics>/vcml_biomodels/<name>__<id>.vcml`. These serve as a pool to draw
MathDescriptions (and later geometry) from when extending `vcell_fenics.pyvcell_bridge`.

**Run it with a FULL pyvcell install.** pyvcell's remote client needs requests /
oauth2 / libvcell, which the vcell-fenics dev env deliberately omits (we depend only on
pyvcell's pure-Pydantic data model — see `pyvcell_bridge`). The simplest way is pyvcell's
own environment:

    cd ../pyvcell && uv run python ../vcell-fenics/scripts/download_public_biomodels.py

The output directory is resolved relative to this script (the vcell-fenics root), so the
working directory does not matter. Already-downloaded files are skipped, so re-running
resumes. Per-model failures are caught and reported at the end; they do not stop the run.

Options: `--out DIR`, `--limit N` (download at most N, for a quick test), `--server URL`,
`--timeout SECONDS`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyvcell.vcml as vc
from pyvcell._internal.api.vcell_client.api.bio_model_resource_api import BioModelResourceApi

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "vcml_biomodels"


def _slug(name: str) -> str:
    """A filesystem-safe slug; the BioModel id (appended separately) guarantees uniqueness."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("_")
    return cleaned[:120] or "model"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VCell BioModels as .vcml")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output dir (default: <repo>/vcml_biomodels)")
    parser.add_argument("--limit", type=int, default=None, help="download at most N models (for testing)")
    parser.add_argument("--server", default="https://vcell.cam.uchc.edu", help="VCell server URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-model request timeout (s)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.out}")

    print("Opening browser for VCell OAuth2 login ...")
    session = vc.connect(api_base_url=args.server, login=True)
    api = BioModelResourceApi(session._api_client)

    print("Listing BioModels ...")
    summaries = session.list_biomodels()
    print(f"Server returned {len(summaries)} BioModel summaries.")
    if args.limit is not None:
        summaries = summaries[: args.limit]

    downloaded = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    total = len(summaries)
    for i, summary in enumerate(summaries, 1):
        model_id = summary.get("id")
        name = summary.get("name") or "model"
        if not model_id:
            continue
        dest = args.out / f"{_slug(name)}__{model_id}.vcml"
        if dest.exists():
            skipped += 1
            continue
        try:
            vcml = api.get_bio_model_vcml(model_id, _request_timeout=args.timeout)
            dest.write_text(vcml)
            downloaded += 1
            print(f"[{i}/{total}] {dest.name}")
        except Exception as exc:  # keep going; report at the end
            failed += 1
            failures.append((f"{name} ({model_id})", str(exc).splitlines()[0][:200]))
            print(f"[{i}/{total}] FAILED {name} ({model_id}): {exc!r}", file=sys.stderr)

    print(f"\nDone. downloaded={downloaded} skipped={skipped} failed={failed} -> {args.out}")
    if failures:
        print("Failures:")
        for who, why in failures:
            print(f"  - {who}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

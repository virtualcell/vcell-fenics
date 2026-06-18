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
import sys
from pathlib import Path

import pyvcell.vcml as vc
import requests

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "vcml_biomodels"

# The VCML-download endpoint is `/api/v1/bioModel/{id}/vcml_download`, and it @Produces
# `text/xml` ONLY — `application/xml` 500s with NotAcceptableException, which is why the
# generated client (and an earlier guess) failed. We call the REST API directly (reusing
# the OAuth token, like list_biomodels) with the confirmed combo first, keeping the others
# as fallbacks in case the API shifts again; the first combo returning real VCML is pinned.
_VCML_PATH_TEMPLATES = (
    "/api/v1/bioModel/{id}/vcml_download",
    "/api/v1/bioModel/{id}/biomodel.vcml",
    "/api/v1/bioModel/{id}/vcml",
)
_ACCEPT_HEADERS = ("text/xml", "*/*", "application/xml")


def _looks_like_vcml(text: str) -> bool:
    head = text.lstrip()[:400].lower()
    return "<vcml" in head or "<biomodel" in head or ("<?xml" in head and "vcml" in head)


def _snippet(text: str, n: int = 200) -> str:
    """A one-line body snippet — strip CR/LF (VCML uses \\r\\n, which clobbers terminal output)."""
    return text.replace("\r", " ").replace("\n", " ").strip()[:n]


def _fetch_vcml(
    host: str, token: str | None, model_id: str, timeout: float, pinned: dict[str, str]
) -> tuple[str | None, str]:
    """Fetch one model's VCML, probing path/Accept variants until one returns XML. Returns
    (vcml_or_None, diagnostic). Once a (path, accept) combo works it is pinned for reuse."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    combos = (
        [(pinned["path"], pinned["accept"])]
        if pinned
        else [(p, a) for p in _VCML_PATH_TEMPLATES for a in _ACCEPT_HEADERS]
    )
    last = ""
    for path_template, accept in combos:
        url = host.rstrip("/") + path_template.format(id=model_id)
        try:
            resp = requests.get(url, headers={**headers, "Accept": accept}, timeout=timeout)
        except requests.RequestException as exc:
            last = f"{path_template} [{accept}] -> request error: {exc}"
            continue
        if resp.status_code == 200 and _looks_like_vcml(resp.text):
            pinned["path"], pinned["accept"] = path_template, accept
            return resp.text, ""
        last = f"{path_template} [{accept}] -> HTTP {resp.status_code}: {_snippet(resp.text)}"
    return None, last


def _diagnose(host: str, token: str | None, model_ids: list[str], timeout: float) -> None:
    """Probe every path/Accept variant for a few models and print status + content-type +
    body snippet — distinguishes a wrong/changed endpoint (all variants error for every
    model) from individually-broken models (some models 200, some 500)."""

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for model_id in model_ids:
        print(f"\n--- model {model_id} ---")
        for path_template in _VCML_PATH_TEMPLATES:
            for accept in _ACCEPT_HEADERS:
                url = host.rstrip("/") + path_template.format(id=model_id)
                try:
                    resp = requests.get(url, headers={**headers, "Accept": accept}, timeout=timeout)
                except requests.RequestException as exc:
                    print(f"  {path_template} [{accept}] -> request error: {exc}")
                    continue
                ok = "OK " if (resp.status_code == 200 and _looks_like_vcml(resp.text)) else "   "
                ct = resp.headers.get("content-type", "?")
                print(f"  {ok}{path_template} [{accept}] -> {resp.status_code} ct={ct}  {_snippet(resp.text)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VCell BioModels as .vcml")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output dir (default: <repo>/vcml_biomodels)")
    parser.add_argument("--limit", type=int, default=None, help="download at most N models (for testing)")
    parser.add_argument("--server", default="https://vcell.cam.uchc.edu", help="VCell server URL")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-model request timeout (s)")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="probe the VCML endpoint variants for the first few models and exit (no downloads)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {args.out}")

    print("Opening browser for VCell OAuth2 login ...")
    session = vc.connect(api_base_url=args.server, login=True)
    config = session._api_client.configuration
    host = config.host
    token = config.access_token

    print("Listing BioModels ...")
    summaries = session.list_biomodels()
    print(f"Server returned {len(summaries)} BioModel summaries.")

    if args.diagnose:
        ids = [s["id"] for s in summaries if s.get("id")][:3]
        _diagnose(host, token, ids, args.timeout)
        return 0

    if args.limit is not None:
        summaries = summaries[: args.limit]

    pinned: dict[str, str] = {}
    downloaded = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    total = len(summaries)
    for i, summary in enumerate(summaries, 1):
        model_id = summary.get("id")
        name = summary.get("name") or "model"
        if not model_id:
            continue
        dest = args.out / f"biomodel_{model_id}.vcml"
        if dest.exists():
            skipped += 1
            continue
        vcml, diagnostic = _fetch_vcml(host, token, model_id, args.timeout, pinned)
        if vcml is not None:
            dest.write_text(vcml)
            downloaded += 1
            print(f"[{i}/{total}] {dest.name}  (via {pinned['path']} [{pinned['accept']}])")
        else:
            failed += 1
            failures.append((f"{name} ({model_id})", diagnostic))
            print(f"[{i}/{total}] FAILED {name} ({model_id}): {diagnostic}", file=sys.stderr)

    print(f"\nDone. downloaded={downloaded} skipped={skipped} failed={failed} -> {args.out}")
    if failures:
        print("Failures (last variant tried per model):")
        for who, why in failures:
            print(f"  - {who}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

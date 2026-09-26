"""Regenerate the math of saved VCell BioModels, as VCell does before it runs a simulation.

A saved BioModel carries the MathDescription generated when it was last saved. Very old documents (and
VCell itself, then) treated a volume variable as defined in every volume compartment and a membrane
variable in every membrane, with no ``Domain``; the solvers never see that math, because VCell regenerates
it from the biology when it builds a SimulationTask. So a coverage survey that reads the saved math
directly over-counts "legacy math" refusals. This script writes a regenerated copy of each BioModel
(libvcell's ``vcml_to_vcml``) for the survey to read instead.

Runs in pyvcell's full environment (libvcell is native, kept out of the DOLFINx env)::

    ../pyvcell/.venv/bin/python scripts/regenerate_vcml.py [--sample 600] [--out vcml_biomodels/regenerated]

then ``survey_fenics_coverage.py run --vcml-dir vcml_biomodels/regenerated --work <another work dir>``
(with the same ``census.csv``: regeneration changes the math, not the simulations' settings).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import libvcell  # type: ignore[import-not-found]  # native: only in pyvcell's own env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import survey_fenics_coverage as survey  # a sibling script, for its census and seeded sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--vcml-dir", type=Path, default=survey._VCML_DIR)
    parser.add_argument("--work", type=Path, default=survey._WORK, help="the survey work dir holding census.csv")
    parser.add_argument("--out", type=Path, default=survey._VCML_DIR / "regenerated")
    parser.add_argument("--sample", type=int, default=None, help="only the files of the survey's seeded sample")
    args = parser.parse_args()

    if args.sample is not None:
        files = sorted({file for file, _ in survey._sampled(args.work, args.sample)})
    else:
        with (args.work / "census.csv").open() as source:
            files = sorted({row["file"] for row in csv.DictReader(source)})
    args.out.mkdir(parents=True, exist_ok=True)
    failures = args.out / "failures.csv"
    with failures.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(["file", "message"])
        for k, name in enumerate(files, 1):
            target = args.out / name
            if target.exists():
                continue
            ok, message = libvcell.vcml_to_vcml((args.vcml_dir / name).read_text(), target)
            if not ok:
                writer.writerow([name, message.replace("\n", " ")[:500]])
            if k % 50 == 0 or k == len(files):
                print(f"  {k}/{len(files)}", flush=True)
    print(f"regenerated -> {args.out} (failures: {failures})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

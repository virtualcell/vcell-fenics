"""Stage 2 (dev env): vcell-fenics vs mbsolver on a RacRho front that reads the normal (ADR 013).

Runs the linear RacRho variant written by stage 1 (``racrho_front_fv.py``) through the importer and the
moving-boundary runner: the front ``v = (a − 1)|a − 1| n`` becomes the membrane's prescribed motion with
``geom.normal`` and ``trace(a)``, and the cytoplasm's material stays at rest. Compares the front's area,
centroid and x-extent, and the area-weighted mean of ``a``, with mbsolver's at each output time, and
checks that ``a + b`` (which the reaction conserves) keeps its total.

Result (2026-11-09, h = 0.08, dt = 0.005, 43 remeshes to t = 5; mbsolver on its 96 × 39 grid): area within
3.0 % (ours larger throughout), centroid x 8 % ahead at t = 0.5 and 2–4 % ahead after, trailing edge within
0.03; the leading edge runs ahead of mbsolver's, by 0.28 (7 %) at t = 5; area-weighted mean ``a`` within
3.8 % (ours lower); ``∫(a + b)`` constant to 6 digits. The systematic lead of our leading edge is not
explained yet (no refinement study in either solver). At dt = 0.01 a step tangles the mesh near
t = 4.4: a front moving only normally gathers the retracting stretch's boundary nodes (no tangential
redistribution), so the run remeshes ever more often.

    .pixi/envs/dev/bin/python cross_validation/racrho_front.py [--h 0.08] [--dt 0.005] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lxml import etree
from pyvcell.vcml.vcml_reader import VcmlReader

from vcell_fenics.cli import ModelInput, _import_vcell
from vcell_fenics.pyvcell_bridge.simtask import _front_velocities, _tag
from vcell_fenics.results import Bundle
from vcell_fenics.runner import RunOptions, run_model, uniform_output_times

_HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=0.08)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--out", type=Path, default=_HERE / "racrho_front_out")
    args = parser.parse_args()
    source = _HERE / "racrho_front_linear.vcml"
    reference = np.load(_HERE / "racrho_front_fv.npz")

    biomodel = VcmlReader.biomodel_from_file(str(source))
    app = biomodel.applications[0]
    root = etree.parse(str(source)).getroot()
    spec = next(e for e in root.iter() if _tag(e) == "SimulationSpec" and e.get("Name") == app.name)
    (front,) = _front_velocities(next(e for e in spec if _tag(e) == "MathDescription"))
    geometry, math = _import_vcell(app.geometry, app.math_description, front_velocity=front)
    membrane = next(s for s in math.subdomains if s.kind == "surface")
    print(f"front on {membrane.name}: {membrane.motion}")
    model = ModelInput(geometry=geometry, math=math, source="vcml", provenance={"kind": "vcml", "file": str(source)})
    options = RunOptions(
        h=args.h,
        dt=args.dt,
        t_final=5.0,
        output_times=uniform_output_times(5.0, 0.5),
        fe_degree=1,
        time_integration="backward_euler",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        run_model(model, options, args.out, prefix="run")
    except Exception as error:  # compare the rows written before a failure
        print(f"run failed: {error}")
    bundle = Bundle.open(args.out / "run.fenics")

    for row, t in enumerate(bundle.times):
        xy = bundle.coords("cyt", row)[:, :2]
        cells = bundle.mesh("cyt", row).cells
        p = xy[cells]
        w = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) / 2
        area = w.sum()
        cx = float((w * p[:, :, 0].mean(axis=1)).sum() / area)
        a, b = bundle.field("cyt", "a", row), bundle.field("cyt", "b", row)
        a_mean = float((w * a[cells].mean(axis=1)).sum() / area)
        total = float((w * (a[cells] + b[cells]).mean(axis=1)).sum())
        k = int(np.argmin(np.abs(reference["times"] - t)))
        print(
            f"t = {t:3.1f}: area {area:.4f} / mbsolver {reference['areas'][k]:.4f}; "
            f"centroid x {cx:.4f} / {reference['centroids'][k][0]:.4f}; "
            f"x-extent [{xy[:, 0].min():.3f}, {xy[:, 0].max():.3f}] / [{reference['xmin'][k]:.3f}, {reference['xmax'][k]:.3f}]; "
            f"mean a {a_mean:.4f} / {reference['a_mean'][k]:.4f}; ∫(a+b) {total:.5f}"
        )


if __name__ == "__main__":
    main()

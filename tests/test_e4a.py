"""vcell-core's E4a against its own criteria, through the CLI (ADR 021 decision 7; stage 5a).

E4a is Nickaeen et al. 2017's Fig 4A on E2's fixed circle: myosin with the crowding cut-offs, the diffusion
`D (1 − m/m_max,d)` and the drift `(max(0, 1 − m/m_max,u) − 1) v`, carried by the force balance's velocity with the
membrane held. The document and its criteria are read from vcell-core's sibling checkout
(`conformance/math/myosin-on-moving-cell.jsonl`), not copied; vcell-core proposes the criteria and this test
asserts them, at every output:
- total myosin conserved to 1e-8 relative;
- `m ≤ m_max,d`, and `min m ≥ −0.01 · mean m` (a P1 undershoot allowed);
- by `t = 10`, the share of myosin in the left-edge band (`r ≥ 0.9`, `|θ − π| ≤ 45°`) at least 3 times its share at
  `t = 0`.

Measured 2026-11-19 at h = 0.1, Δt = 0.05 (200 strides, 2,607 steps, 22 s): the ratio 6.07 at t = 10 (3.70 at
t = 2), the peak 13.3, `min m / mean` 0.048, `∫m` kept to 7e-15. The Δt and h dependence is in ADR 021 decision 7.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vcell_fenics.cli import main
from vcell_fenics.results import Bundle

_E4 = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "math" / "myosin-on-moving-cell.jsonl"

_GEOMETRY = """geometry_description:
  name: disk_in_medium
  dim: 2
  extent: [5.0, 4.0, 1.0]
  origin: [-2.0, -2.0, 0.0]
  subvolumes:
    - { name: cell, type: analytic, expression: "geom.x[0]**2 + geom.x[1]**2 < 1.0" }
    - { name: ec, type: analytic, expression: "1.0" }
  surfaces:
    - { name: cell_ec_membrane, inside: cell, outside: ec }
"""


def _record(case: str) -> dict[str, Any]:
    records = [json.loads(line) for line in _E4.read_text().splitlines() if line.strip()]
    found: dict[str, Any] = next(r for r in records if r.get("case") == case)
    return found


def _band_share(bundle: Bundle, row: int, band: dict[str, float]) -> float:
    """The share of `∫m` over the cells whose centroid lies in the band: exact for P1 on each triangle."""

    points = bundle.coords("cell", row)[:, :2]
    cells = np.asarray(bundle.mesh("cell", row).cells).reshape(-1, 3)
    m = bundle.field("cell", "m", row)
    a, b, c = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    area = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    integral = area * m[cells].mean(axis=1)
    centre = (a + b + c) / 3.0
    r = np.hypot(centre[:, 0], centre[:, 1])
    theta = np.arctan2(centre[:, 1], centre[:, 0]) % (2.0 * np.pi)
    inside = (r >= band["r_min"]) & (
        np.abs(theta - np.radians(band["theta_centre_deg"])) <= np.radians(band["theta_half_width_deg"])
    )
    return float(integral[inside].sum() / integral.sum())


@pytest.mark.integration
@pytest.mark.skipif(not _E4.exists(), reason="vcell-core's E4 document needs the sibling checkout")
def test_vcell_cores_e4a_meets_its_criteria_at_every_output(tmp_path: Path) -> None:
    record = _record("e4a")
    criteria = record["criteria"]
    every = criteria["all"]
    (tmp_path / "geom.yaml").write_text(_GEOMETRY)
    (tmp_path / "math.json").write_text(json.dumps(record["fenics"]))
    argv = ["--math", str(tmp_path / "math.json"), "--geometry", str(tmp_path / "geom.yaml")]
    t_final = criteria["t"]
    out = tmp_path / "out"
    assert (
        main([*argv, "--h", "0.1", "--dt", "0.05", "--t-final", repr(t_final), "--output-dt", "1.0", "--out", str(out)])
        == 0
    )
    bundle = Bundle.open(out / "results.fenics")
    assert bundle.times[-1] == pytest.approx(t_final)
    stats = bundle.stats("cell", "m")  # mean, total, min, max per row
    column = {name: k for k, name in enumerate(bundle.manifest.stats_columns)}
    total0 = stats[0, column["total"]]
    for row in range(len(bundle.times)):
        mean, total = stats[row, column["mean"]], stats[row, column["total"]]
        assert abs(total / total0 - 1.0) <= every["myosin_conserved_relative"], (row, total)
        assert stats[row, column["max"]] <= every["m_max"], row
        assert stats[row, column["min"]] / mean >= every["m_min_over_mean"], row
    share0 = _band_share(bundle, 0, criteria["band"])
    ratio = _band_share(bundle, len(bundle.times) - 1, criteria["band"]) / share0
    assert ratio >= criteria["band_share_ratio_min"], ratio

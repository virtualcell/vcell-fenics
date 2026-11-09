"""The implicit-function evaluator takes VCell's trigonometric functions: a VCell analytic geometry may use them.
The copied moving_bleb application (BioModel 294718710) draws its cell as a 35-tooth wavy disk in polar form,
``r² − (1 + 0.4 sin(35·atan2(y − 5, x − 5)))² < 2.5²``, which the realizer refused before."""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend.implicit_fields import eval_field
from vcell_fenics.formalism.parser import parse


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("atan2(geom.x[1], geom.x[0])", np.arctan2),
        ("tan(geom.x[0]) + atan(geom.x[1])", lambda y, x: np.tan(x) + np.arctan(y)),
        ("asin(0.5 * geom.x[0]) + acos(0.5 * geom.x[1])", lambda y, x: np.arcsin(0.5 * x) + np.arccos(0.5 * y)),
    ],
)
def test_trigonometric_functions_evaluate_like_numpy(text: str, expected: object) -> None:
    x = np.array([0.3, -0.7, 1.2, -1.5])
    y = np.array([0.9, -0.2, -1.1, 0.4])
    assert np.allclose(eval_field(parse(text), (x, y)), expected(y, x))  # type: ignore[operator]


def test_a_polar_wavy_disk_classifies_its_teeth() -> None:
    teeth = "(1.0 + 0.4 * sin(35.0 * atan2(geom.x[1] - 5.0, geom.x[0] - 5.0)))"
    shape = parse(f"(geom.x[0] - 5.0)**2 + (geom.x[1] - 5.0)**2 - {teeth}**2 - 2.5**2")
    angle = np.pi / 70.0  # a tooth's crest: sin(35·θ) = 1, so the boundary sits at r² = 2.5² + 1.4²
    r_crest = np.sqrt(2.5**2 + 1.4**2)
    inside = eval_field(
        shape, (np.array([5.0 + 0.99 * r_crest * np.cos(angle)]), np.array([5.0 + 0.99 * r_crest * np.sin(angle)]))
    )
    outside = eval_field(
        shape, (np.array([5.0 + 1.01 * r_crest * np.cos(angle)]), np.array([5.0 + 1.01 * r_crest * np.sin(angle)]))
    )
    assert inside[0] < 0.0 < outside[0]

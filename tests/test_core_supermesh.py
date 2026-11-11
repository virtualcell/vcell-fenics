"""The libsupermesh transfer (`core.supermesh`, ADR 016 decisions 3 and 9): the L2 projection between two unrelated
simplex meshes of the same square or cube keeps the total to round-off, reproduces a linear field, agrees with the
pure-Python 2D kernel (`core.bulk_remap`), and its lumped variant keeps the total and non-negativity."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from vcell_fenics.core.bulk_remap import supermesh_project_2d
from vcell_fenics.core.supermesh import mass_matrix, mixed_mass_matrix, project


def _mesh(d: int, n: int, seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """A Delaunay mesh of the unit square or cube: its corners, points on its boundary and inside, at random."""
    rng = np.random.default_rng(seed)
    corners = np.array(np.meshgrid(*[[0.0, 1.0]] * d, indexing="ij")).reshape(d, -1).T
    boundary = rng.uniform(0.0, 1.0, (n // 2, d))
    face = rng.integers(0, d, n // 2)
    boundary[np.arange(n // 2), face] = rng.integers(0, 2, n // 2).astype(np.float64)
    points = np.vstack([corners, boundary, rng.uniform(0.0, 1.0, (n, d))])
    return points, Delaunay(points).simplices.astype(np.int64)


def _total(x: NDArray[np.float64], cells: NDArray[np.int64], c: NDArray[np.float64]) -> float:
    return float(np.ones(len(x)) @ (mass_matrix(x, cells) @ c))


@pytest.mark.parametrize("d", [2, 3])
def test_the_projection_keeps_the_total_and_reproduces_a_linear_field(d: int) -> None:
    old_x, old_cells = _mesh(d, 400, seed=1)
    new_x, new_cells = _mesh(d, 300, seed=2)
    B, covered = mixed_mass_matrix(old_x, old_cells, new_x, new_cells)
    assert covered == pytest.approx(1.0, abs=1e-12)  # the pieces tile the square / cube
    rough = np.random.default_rng(3).uniform(0.0, 1.0, len(old_x))
    assert _total(new_x, new_cells, project(old_x, old_cells, rough, new_x, new_cells, mixed=B)) == pytest.approx(
        _total(old_x, old_cells, rough), rel=1e-12
    )
    slope = np.arange(1.0, d + 1.0)
    linear = project(old_x, old_cells, 1.0 + old_x @ slope, new_x, new_cells, mixed=B)
    assert np.abs(linear - (1.0 + new_x @ slope)).max() < 1e-10


def test_it_agrees_with_the_python_kernel_in_2d() -> None:
    old_x, old_cells = _mesh(2, 200, seed=4)
    new_x, new_cells = _mesh(2, 150, seed=5)
    c_old = np.random.default_rng(6).uniform(0.0, 1.0, len(old_x))
    reference = supermesh_project_2d(old_x, old_cells, c_old, new_x, new_cells)
    assert np.abs(project(old_x, old_cells, c_old, new_x, new_cells) - reference).max() < 1e-10


@pytest.mark.parametrize("d", [2, 3])
def test_the_lumped_variant_keeps_the_total_and_stays_non_negative(d: int) -> None:
    old_x, old_cells = _mesh(d, 300, seed=7)
    new_x, new_cells = _mesh(d, 250, seed=8)
    step = np.where(old_x[:, 0] < 0.5, 1.0, 0.0)  # a sharp front, where the projection would undershoot
    lumped = project(old_x, old_cells, step, new_x, new_cells, lumped=True)
    assert lumped.min() >= 0.0
    assert _total(new_x, new_cells, lumped) == pytest.approx(_total(old_x, old_cells, step), rel=1e-12)


def test_meshes_that_do_not_overlap_give_an_empty_matrix() -> None:
    old_x, old_cells = _mesh(2, 50, seed=9)
    B, covered = mixed_mass_matrix(old_x, old_cells, old_x + 5.0, old_cells)
    assert B.nnz == 0
    assert covered == 0.0


def test_meshes_of_different_dimension_are_refused() -> None:
    x2, c2 = _mesh(2, 20, seed=10)
    x3, c3 = _mesh(3, 20, seed=11)
    with pytest.raises(ValueError, match="differ in dimension"):
        mixed_mass_matrix(x2, c2, x3, c3)

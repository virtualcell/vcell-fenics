"""`CellOrientation`: a moving bulk mesh's validity against the mesh as built (vcell-fenics ADR 014 decision 6).
An inverted cell must be caught although an assembled ``1·dx`` (|det J|) still reports a positive volume."""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from vcell_fenics.backend.discrete import CellOrientation, MeshQualityError, signed_cell_volumes


def _square() -> dmesh.Mesh:
    return dmesh.create_unit_square(MPI.COMM_SELF, 2, 2)


def test_an_inverted_cell_keeps_its_assembled_volume_but_not_its_orientation() -> None:
    mesh = _square()
    guard = CellOrientation(mesh)
    assert guard.check("fresh") == pytest.approx(1.0)
    x = mesh.geometry.x
    corner = int(np.argmin(np.linalg.norm(x[:, :2] - [1.0, 1.0], axis=1)))
    x[corner, :2] = [0.2, 0.2]  # pushed through its neighbours
    dg0 = fem.functionspace(mesh, ("DG", 0))
    assembled = fem.assemble_vector(fem.form(ufl.TestFunction(dg0) * ufl.dx)).array
    assert assembled.min() > 0.0  # |det J|: blind to the inversion
    assert (signed_cell_volumes(mesh) * np.sign(signed_cell_volumes(_square())) <= 0.0).any()
    with pytest.raises(MeshQualityError, match="inverted or collapsed"):
        guard.check("pushed corner")


def test_a_uniform_scaling_is_no_distortion_and_a_squeeze_is() -> None:
    mesh = _square()
    guard = CellOrientation(mesh)
    mesh.geometry.x[:, :2] *= 3.0
    assert guard.check("scaled") == pytest.approx(1.0)
    x = mesh.geometry.x
    centre = int(np.argmin(np.linalg.norm(x[:, :2] - [1.5, 1.5], axis=1)))
    x[centre, :2] = [2.9, 2.9]  # still valid, but its cells are stretched and squashed
    assert guard.check("squeezed") > 2.0

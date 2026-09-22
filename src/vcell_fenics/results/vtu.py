"""VTK XML UnstructuredGrid I/O for bundle meshes, pinned to what VCell's ``VtuGridParser`` reads.

``VtuGridParser`` (``vcell-client/.../viz/VtuGridParser.java``) is deliberately not a general VTU
reader: one ``Piece``, little-endian, each binary ``DataArray`` an inline base64 block of a byte-count
header (UInt32) followed by the raw values, *no compression* (it does not check the ``compressor``
attribute, so compressed data would be misread silently). :func:`write_vtu` produces exactly that with
VTK's own writer; :func:`read_vtu_strict` is a Python mirror of the parser that also refuses anything
else, so a test that round-trips through it proves VCell can read the file (ADR 010 §6(d)).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray

VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_TETRA = 10

_NUMPY_TYPE = {
    "Int8": "<i1",
    "UInt8": "<u1",
    "Int16": "<i2",
    "UInt16": "<u2",
    "Int32": "<i4",
    "UInt32": "<u4",
    "Int64": "<i8",
    "UInt64": "<u8",
    "Float32": "<f4",
    "Float64": "<f8",
}


@dataclass(frozen=True)
class VtuGrid:
    points: NDArray[np.float64]  # (N, 3)
    cells: NDArray[np.int64]  # (M, k) — every bundle mesh is single-cell-type simplices
    cell_types: NDArray[np.int64]  # (M,)


def write_vtu(path: Path, points: NDArray[np.float64], cells: NDArray[np.int64], vtk_type: int) -> None:
    """Write a single-cell-type unstructured grid. ``points`` may be (N, 1–3); it is padded to 3D."""

    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    xyz = np.zeros((points.shape[0], 3), dtype=np.float64)
    xyz[:, : points.shape[1]] = points
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(xyz, deep=True))
    n_cells, per_cell = cells.shape
    offsets = np.arange(0, (n_cells + 1) * per_cell, per_cell, dtype=np.int64)
    cell_array = vtkCellArray()
    cell_array.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(cells, dtype=np.int64).ravel(), deep=True),
    )
    grid = vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(vtk_type, cell_array)

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToBinary()  # inline base64 — not appended
    writer.SetCompressorTypeToNone()
    writer.SetHeaderTypeToUInt32()
    writer.SetByteOrderToLittleEndian()
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {path}")


def read_vtu_strict(path: Path) -> VtuGrid:
    """Read a VTU the way ``VtuGridParser.parse`` does, refusing what it cannot read."""

    root = ElementTree.fromstring(path.read_bytes())
    if root.tag != "VTKFile" or root.get("type") != "UnstructuredGrid":
        raise ValueError(f"{path}: not a VTKFile/UnstructuredGrid document")
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise ValueError(f"{path}: byte order {root.get('byte_order')!r} is not LittleEndian")
    if root.get("compressor"):
        raise ValueError(f"{path}: compressed data ({root.get('compressor')}) would be misread by VtuGridParser")
    header = 8 if root.get("header_type", "UInt32") == "UInt64" else 4
    pieces = root.findall(".//Piece")
    if len(pieces) != 1:
        raise ValueError(f"{path}: expected exactly one Piece, found {len(pieces)}")
    points_el, cells_el = pieces[0].find("Points"), pieces[0].find("Cells")
    first = points_el.find("DataArray") if points_el is not None else None
    if first is None or cells_el is None:
        raise ValueError(f"{path}: Piece is missing Points/DataArray or Cells")

    def read(array: ElementTree.Element) -> NDArray[np.float64]:
        fmt = array.get("format", "ascii")
        text = "".join(array.itertext())
        if fmt == "ascii":
            return np.array(text.split(), dtype=np.float64)
        if fmt != "binary":
            raise ValueError(f"{path}: DataArray format {fmt!r} (appended data is not read by VtuGridParser)")
        blob = base64.b64decode(text.split()[0])
        count = int.from_bytes(blob[:header], "little")
        values = np.frombuffer(blob[header : header + count], dtype=_NUMPY_TYPE[array.get("type", "")])
        return values.astype(np.float64)

    points = read(first).reshape(-1, 3)
    arrays = {array.get("Name"): read(array) for array in cells_el.iter("DataArray")}
    try:
        connectivity, offsets, types = arrays["connectivity"], arrays["offsets"], arrays["types"]
    except KeyError as missing:
        raise ValueError(f"{path}: Cells is missing {missing}") from None
    sizes = np.diff(np.concatenate([[0.0], offsets])).astype(np.int64)
    if sizes.size and not np.all(sizes == sizes[0]):
        raise ValueError(f"{path}: mixed cell sizes are not a bundle mesh")
    per_cell = int(sizes[0]) if sizes.size else 0
    cells = connectivity.astype(np.int64).reshape(-1, per_cell) if per_cell else np.empty((0, 0), np.int64)
    return VtuGrid(points=points, cells=cells, cell_types=types.astype(np.int64))

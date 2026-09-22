"""Spike: validate the ADR 010 results-bundle stack before building on it (integration step 2).

Checks, each printed as PASS/FAIL:

  (a) zarr-python 3 writes a *format v2* array with a zlib compressor, '<f8', C order, NaN fill.
  (b) A stdlib decoder (json + zlib + numpy) round-trips that array — the "tiny Java/JS reader" claim.
  (c) pyvcell's zarr 2.18 (../pyvcell/.venv) reads the same array.
  (d) A VTU written with vtkXMLUnstructuredGridWriter (Binary, no compressor, UInt32 header, LE) parses
      with a faithful Python port of VCell's VtuGridParser, for triangle, tetra and line cells.
  (e) PETSc TS BDF + TSInterpolate at output times, driven from a monitor, matches the exact solution to
      within the integrator tolerance and does not perturb the step sequence.
  (f) P1 point keys via mesh.geometry.input_global_indices give the same key-sorted points and cells in
      serial and under MPI, for a volume mesh and a boundary-facet submesh.

Run:
    pixi run -e dev python scripts/spike_results_bundle.py [--out DIR]
    pixi run -e dev mpiexec -n 2 python scripts/spike_results_bundle.py --mpi-only --out DIR
The serial run writes the reference for (f); the MPI run compares against it.
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from mpi4py import MPI

RESULTS: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    if MPI.COMM_WORLD.rank == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}", flush=True)


# ---------------------------------------------------------------------------------------------
# (a)-(c) zarr format v2
# ---------------------------------------------------------------------------------------------


def check_zarr(root: Path) -> None:
    import numcodecs
    import zarr

    store = root / "bundle.fenics"
    group = zarr.open_group(str(store), mode="w", zarr_format=2)
    data = np.arange(15, dtype="<f8").reshape(3, 5) * 0.5
    arr = group.create_array(
        "cyt/C",
        shape=(4, 5),  # one row more than written: preallocated planned_times
        chunks=(1, 5),
        dtype="<f8",
        compressors=numcodecs.Zlib(level=1),
        fill_value=np.nan,
        order="C",
    )
    arr[:3] = data
    meta = json.loads((store / "cyt" / "C" / ".zarray").read_text())
    ok_a = (
        meta.get("zarr_format") == 2
        and meta.get("compressor", {}).get("id") == "zlib"
        and meta.get("dtype") == "<f8"
        and meta.get("order") == "C"
        and meta.get("fill_value") == "NaN"
        and meta.get("chunks") == [1, 5]
    )
    report(
        "(a) zarr v2 metadata",
        ok_a,
        json.dumps(
            {
                k: meta.get(k)
                for k in ("compressor", "dtype", "order", "fill_value", "chunks", "filters", "dimension_separator")
            }
        ),
    )

    # (b) stdlib decode: chunk key "k.0" for row k (dimension_separator ".")
    sep = meta.get("dimension_separator", ".")
    rows = []
    for k in range(3):
        raw = zlib.decompress((store / "cyt" / "C" / f"{k}{sep}0").read_bytes())
        rows.append(np.frombuffer(raw, dtype="<f8"))
    decoded = np.vstack(rows)
    missing_row_absent = not (store / "cyt" / "C" / f"3{sep}0").exists()
    report(
        "(b) stdlib zlib decode round-trips",
        bool(np.array_equal(decoded, data)) and missing_row_absent,
        "unwritten row has no chunk file (reader must honour manifest.times)",
    )

    # (c) pyvcell's zarr 2.x
    pyvcell_py = Path(__file__).resolve().parents[2] / "pyvcell" / ".venv" / "bin" / "python"
    if not pyvcell_py.exists():
        report("(c) pyvcell zarr 2.x reads bundle", False, f"{pyvcell_py} not found")
        return
    code = (
        "import sys, zarr, numpy as np; g = zarr.open_group(sys.argv[1], mode='r'); a = g['cyt/C'];"
        "print(zarr.__version__, a.shape, float(a[2, 4]), bool(np.isnan(a[3, 0])))"
    )
    out = subprocess.run([str(pyvcell_py), "-c", code, str(store)], capture_output=True, text=True)
    ok_c = out.returncode == 0 and out.stdout.split()[-2:] == ["7.0", "True"]
    report("(c) pyvcell zarr 2.x reads bundle", ok_c, (out.stdout or out.stderr).strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------
# (d) VTU vs VtuGridParser
# ---------------------------------------------------------------------------------------------


def write_vtu(path: Path, points: np.ndarray, cells: np.ndarray, vtk_type: int) -> None:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

    pts3 = np.zeros((points.shape[0], 3))
    pts3[:, : points.shape[1]] = points
    vpts = vtkPoints()
    vpts.SetData(numpy_to_vtk(pts3, deep=True))
    n, k = cells.shape
    offsets = np.arange(0, (n + 1) * k, k, dtype=np.int64)
    conn = cells.astype(np.int64).ravel()
    ca = vtkCellArray()
    ca.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(conn, deep=True))
    grid = vtkUnstructuredGrid()
    grid.SetPoints(vpts)
    grid.SetCells(vtk_type, ca)
    w = vtkXMLUnstructuredGridWriter()
    w.SetFileName(str(path))
    w.SetInputData(grid)
    w.SetDataModeToBinary()
    w.SetCompressorTypeToNone()
    w.SetHeaderTypeToUInt32()
    w.SetByteOrderToLittleEndian()
    if w.Write() != 1:
        raise RuntimeError(f"vtk failed to write {path}")


_WIDTH = {
    "Int8": 1,
    "UInt8": 1,
    "Int16": 2,
    "UInt16": 2,
    "Float32": 4,
    "Int32": 4,
    "UInt32": 4,
    "Float64": 8,
    "Int64": 8,
    "UInt64": 8,
}
_NP = {
    "Int8": "<i1",
    "UInt8": "<u1",
    "Int16": "<i2",
    "UInt16": "<u2",
    "Float32": "<f4",
    "Int32": "<i4",
    "UInt32": "<u4",
    "Float64": "<f8",
    "Int64": "<i8",
    "UInt64": "<u8",
}


def parse_like_vtugridparser(raw: bytes) -> dict[str, Any]:
    """Python port of VtuGridParser.parse (../vcell/vcell-client/.../viz/VtuGridParser.java)."""
    root = ElementTree.fromstring(raw)
    if root.tag != "VTKFile" or root.get("type") != "UnstructuredGrid":
        raise ValueError("not a VTKFile/UnstructuredGrid document")
    if root.get("byte_order", "LittleEndian") != "LittleEndian":
        raise ValueError("unsupported byte order")
    header = 8 if root.get("header_type", "UInt32") == "UInt64" else 4
    if root.get("compressor"):
        raise ValueError(f"compressed VTU ({root.get('compressor')}) would be misread by VtuGridParser")
    pieces = root.findall(".//Piece")
    if len(pieces) != 1:
        raise ValueError(f"expected exactly one Piece, found {len(pieces)}")
    piece = pieces[0]

    def read(da: ElementTree.Element) -> np.ndarray:
        fmt = da.get("format", "ascii")
        typ = da.get("type", "")
        text = "".join(da.itertext())
        if fmt == "ascii":
            return np.array(text.split(), dtype=float)
        if fmt != "binary":
            raise ValueError(f"unsupported DataArray format {fmt!r}")
        blob = base64.b64decode(text.split()[0])
        count = int.from_bytes(blob[:header], "little")
        return np.frombuffer(blob[header : header + count], dtype=_NP[typ]).astype(float)

    points_el, cells_el = piece.find("Points"), piece.find("Cells")
    first = points_el.find("DataArray") if points_el is not None else None
    if first is None or cells_el is None:
        raise ValueError("Piece is missing Points/DataArray or Cells")
    points = read(first)
    arrays = {da.get("Name"): read(da) for da in cells_el.iter("DataArray")}
    return {
        "points": points.reshape(-1, 3),
        "connectivity": arrays["connectivity"],
        "offsets": arrays["offsets"],
        "types": arrays["types"],
    }


def check_vtu(root: Path) -> None:
    cases = {
        "triangle": (np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]]), np.array([[0, 1, 2], [1, 3, 2]]), 5),
        "tetra": (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]), np.array([[0, 1, 2, 3]]), 10),
        "line": (np.array([[0, 0], [1, 0], [1, 1.0]]), np.array([[0, 1], [1, 2]]), 3),
    }
    for name, (pts, cells, vtype) in cases.items():
        path = root / f"{name}.vtu"
        write_vtu(path, pts, cells, vtype)
        head = path.read_bytes()[:400].decode(errors="replace")
        parsed = parse_like_vtugridparser(path.read_bytes())
        k = cells.shape[1]
        ok = (
            np.allclose(parsed["points"][:, : pts.shape[1]], pts)
            and np.array_equal(parsed["connectivity"].astype(int), cells.ravel())
            and np.array_equal(parsed["offsets"].astype(int), np.arange(k, k * (len(cells) + 1), k))
            and np.all(parsed["types"] == vtype)
            and 'header_type="UInt32"' in head
            and "compressor" not in head
        )
        report(f"(d) VTU {name} parses like VtuGridParser", bool(ok), f"{path.stat().st_size} bytes")


# ---------------------------------------------------------------------------------------------
# (e) TS BDF + interpolate at output times
# ---------------------------------------------------------------------------------------------


def check_ts_interpolate() -> None:
    from petsc4py import PETSc

    k = np.array([1.0, 3.0])  # y' = -k y, y(0) = 1
    rtol = atol = 1e-8
    outputs = np.linspace(0.0, 2.0, 11)

    def run(with_monitor: bool, *, stride: bool = False) -> tuple[list[tuple[float, np.ndarray]], int]:
        ts = PETSc.TS().create(comm=PETSc.COMM_SELF)
        ts.setType(PETSc.TS.Type.BDF)
        ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
        y = PETSc.Vec().createSeq(2)
        y.set(1.0)
        f = y.duplicate()

        def ifunction(ts_: Any, t: float, u: Any, udot: Any, F: Any) -> None:
            F.setArray(udot.getArray(readonly=True) + k * u.getArray(readonly=True))

        def ijacobian(ts_: Any, t: float, u: Any, udot: Any, shift: float, J: Any, P: Any) -> None:
            P.zeroEntries()
            for i in range(2):
                P.setValue(i, i, shift + k[i])
            P.assemble()

        J = PETSc.Mat().createAIJ((2, 2), nnz=1, comm=PETSc.COMM_SELF)
        J.setUp()
        ts.setIFunction(ifunction, f)
        ts.setIJacobian(ijacobian, J)
        ts.setTime(0.0)
        ts.setMaxTime(float(outputs[-1]))
        ts.setTimeStep(1e-3)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
        ts.setTolerances(rtol=rtol, atol=atol)
        recorded: list[tuple[float, np.ndarray]] = [(0.0, y.getArray().copy())]
        work = y.duplicate()
        pending = list(outputs[1:])

        def monitor(ts_: Any, step: int, t: float, u: Any) -> None:
            while pending and pending[0] <= t + 1e-12 * max(1.0, abs(t)):
                tk = pending.pop(0)
                ts_.interpolate(tk, work)
                recorded.append((tk, work.getArray().copy()))

        if with_monitor:
            ts.setMonitor(monitor)
        ts.setFromOptions()
        if stride:  # the fallback: stop exactly on every output time (MATCHSTEP per interval)
            for tk in outputs[1:]:
                ts.setMaxTime(float(tk))
                ts.solve(y)
                recorded.append((float(tk), y.getArray().copy()))
        else:
            ts.solve(y)
        return recorded, ts.getStepNumber()

    def max_rel_err(rec: list[tuple[float, np.ndarray]]) -> float:
        return max(float(np.max(np.abs(v - np.exp(-k * t)) / np.exp(-k * t))) for t, v in rec)

    rec, steps_with = run(True)
    _, steps_without = run(False)
    rec_stride, steps_stride = run(False, stride=True)
    # The global BDF error accumulates past rtol (local control), so the yardstick is the stride
    # fallback — the accuracy we would get by stopping on every output time — not rtol itself.
    err, err_stride = max_rel_err(rec), max_rel_err(rec_stride)
    ok = len(rec) == len(outputs) and err <= 1.5 * err_stride and steps_with == steps_without
    report(
        "(e) TS BDF interpolate at output times",
        ok,
        f"{len(rec)} outputs, max rel err {err:.2e} vs stride {err_stride:.2e}; "
        f"steps {steps_with} (unperturbed {steps_without}, stride {steps_stride})",
    )


# ---------------------------------------------------------------------------------------------
# (f) MPI-invariant P1 point keys
# ---------------------------------------------------------------------------------------------


def keyed_layout(V: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Owned P1 dofs keyed by input_global_indices, gathered to rank 0: (keys, points, cells-as-keys)."""
    mesh = V.mesh
    comm = mesh.comm
    tdim = mesh.topology.dim
    imap = V.dofmap.index_map
    n_owned = imap.size_local
    n_all = n_owned + imap.num_ghosts
    igi = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    dof_list = np.asarray(V.dofmap.list)
    geo_list = np.asarray(mesh.geometry.dofmap)
    key = np.full(n_all, -1, dtype=np.int64)
    key[dof_list.ravel()] = igi[geo_list.ravel()]
    xyz = V.tabulate_dof_coordinates()
    n_cells = mesh.topology.index_map(tdim).size_local
    cell_keys = key[dof_list[:n_cells]]
    g_keys = comm.gather(key[:n_owned], root=0)
    g_xyz = comm.gather(xyz[:n_owned], root=0)
    g_cells = comm.gather(cell_keys, root=0)
    if comm.rank != 0:
        return None
    keys = np.concatenate(g_keys)
    pts = np.concatenate(g_xyz)
    order = np.argsort(keys, kind="stable")
    cells = np.concatenate(g_cells)
    return keys[order], pts[order], cells


def canonical_cells(keys_sorted: np.ndarray, cells_as_keys: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(keys_sorted, cells_as_keys)
    s = np.sort(pos, axis=1)
    return s[np.lexsort(s.T[::-1])]


def build_layouts() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    from dolfinx import fem, mesh

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, 12, 9, mesh.CellType.triangle)
    out: dict[str, Any] = {"volume": keyed_layout(fem.functionspace(msh, ("Lagrange", 1)))}
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    sub = mesh.create_submesh(msh, tdim - 1, facets)[0]
    out["membrane"] = keyed_layout(fem.functionspace(sub, ("Lagrange", 1)))
    return out


def check_mpi_keys(ref_path: Path, *, write_reference: bool) -> None:
    comm = MPI.COMM_WORLD
    layouts = build_layouts()
    if comm.rank != 0:
        return
    if write_reference:
        arrays: dict[str, Any] = {
            f"{d}_{n}": a
            for d, lay in layouts.items()
            if lay
            for n, a in zip(("keys", "pts", "cells"), lay, strict=True)
        }
        np.savez(ref_path, **arrays)
        for d, lay in layouts.items():
            assert lay is not None
            unique = len(np.unique(lay[0])) == len(lay[0])
            report(f"(f) serial {d} keys unique", unique, f"{len(lay[0])} points")
        return
    ref = np.load(ref_path)
    for d, lay in layouts.items():
        assert lay is not None
        keys, pts, cells = lay
        same_keys = np.array_equal(keys, ref[f"{d}_keys"])
        same_pts = np.allclose(pts, ref[f"{d}_pts"])
        same_cells = np.array_equal(canonical_cells(keys, cells), canonical_cells(ref[f"{d}_keys"], ref[f"{d}_cells"]))
        report(
            f"(f) {d}: n={comm.size} matches serial",
            same_keys and same_pts and same_cells,
            f"keys {same_keys}, points {same_pts}, cells {same_cells}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="scratch directory (default: a temp dir)")
    parser.add_argument("--mpi-only", action="store_true", help="only run (f), comparing against the serial reference")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    root = args.out or Path(tempfile.mkdtemp(prefix="spike_bundle_"))
    if comm.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    ref = root / "mpi_keys_serial.npz"
    if args.mpi_only:
        check_mpi_keys(ref, write_reference=False)
    else:
        if comm.size != 1:
            raise SystemExit("run the full spike serially; use --mpi-only under mpiexec")
        for sub in ("zarr", "vtu"):
            shutil.rmtree(root / sub, ignore_errors=True)
            (root / sub).mkdir()
        check_zarr(root / "zarr")
        check_vtu(root / "vtu")
        check_ts_interpolate()
        check_mpi_keys(ref, write_reference=True)
        print(f"scratch: {root}")
    failed = [name for name, ok, _ in RESULTS if not ok]
    return 1 if (comm.rank == 0 and failed) else 0


if __name__ == "__main__":
    sys.exit(main())

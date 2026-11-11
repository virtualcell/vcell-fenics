"""Conservative P1 transfer between two simplex meshes of the same region, by libsupermesh (ADR 016 decisions 3, 9).

The L2 projection of a P1 field from an old mesh onto a new one, ``M_new c_new = B c_old``, where ``M_new`` is the
new mesh's mass matrix and ``B[j, i] = ∫ φ_new_j φ_old_i`` is the *mixed* mass matrix over the supermesh, the common
refinement of the two meshes (Farrell–Maddison 2011). When both meshes cover the same region, ``Σ_j B[j, i] =
∫ φ_old_i``, so the total ``∫ c`` is kept to the solve's round-off with no rescale, and a linear field is
reproduced exactly. 2D (triangles) and 3D (tetrahedra).

libsupermesh (Firedrake's library, LGPL-2.1) supplies the two hard parts: an R-tree search for the pairs of cells
that may overlap, and the exact intersection of two triangles or two tetrahedra as a set of simplices. A small C
kernel here assembles ``B`` on each piece exactly: both P1 functions are linear on a piece, and on a simplex ``S``
in ``d`` dimensions ``∫ f g = |S| / ((d+1)(d+2)) (Σ_k f_k g_k + Σ_k f_k Σ_k g_k)`` over its vertices. Pairs whose
intersection is empty store nothing.

The kernel is compiled with cffi at first use, as DOLFINx compiles its forms, and cached under
``$XDG_CACHE_HOME/vcell-fenics`` (``~/.cache`` by default), keyed by a hash of its source and libsupermesh's
location. The container sets ``XDG_CACHE_HOME`` to a world-writable directory and warms it in its build.

`core.bulk_remap.supermesh_project_2d` is the pure-Python reference this is tested against (2D only, all-pairs).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

_CDEF = """
long vf_mixed_mass(int d, long nn_old, long ne_old, const double *x_old, const long *en_old,
                   long nn_new, long ne_new, const double *x_new, const long *en_new, double rel_tol,
                   long **rows, long **cols, double **vals, double *covered);
void vf_free(void *p);
"""

_SOURCE = r"""
#include <stdlib.h>
#include <math.h>
#include "libsupermesh-c.h"

/* barycentric coordinates `lam` (d+1) of point `p` in the simplex `v` ((d+1) x d, row-major) */
static void bary(int d, const double *v, const double *p, double *lam) {
  double A[9], r[3], s[3];
  for (int i = 0; i < d; i++) { r[i] = p[i] - v[i]; for (int j = 0; j < d; j++) A[i*d+j] = v[(j+1)*d+i] - v[i]; }
  if (d == 2) {
    double det = A[0]*A[3] - A[1]*A[2];
    s[0] = ( A[3]*r[0] - A[1]*r[1]) / det; s[1] = (-A[2]*r[0] + A[0]*r[1]) / det;
  } else {
    double det = A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);
    double inv[9] = { A[4]*A[8]-A[5]*A[7], A[2]*A[7]-A[1]*A[8], A[1]*A[5]-A[2]*A[4],
                      A[5]*A[6]-A[3]*A[8], A[0]*A[8]-A[2]*A[6], A[2]*A[3]-A[0]*A[5],
                      A[3]*A[7]-A[4]*A[6], A[1]*A[6]-A[0]*A[7], A[0]*A[4]-A[1]*A[3] };
    for (int i = 0; i < 3; i++) s[i] = (inv[i*3]*r[0] + inv[i*3+1]*r[1] + inv[i*3+2]*r[2]) / det;
  }
  double sum = 0; for (int i = 0; i < d; i++) { lam[i+1] = s[i]; sum += s[i]; } lam[0] = 1.0 - sum;
}

static int grow(long need, long *cap, long **rows, long **cols, double **vals) {
  if (need <= *cap) return 0;
  long c = *cap ? *cap : 1024; while (c < need) c *= 2;
  long *r = realloc(*rows, sizeof(long) * c); if (!r) return 1; *rows = r;
  long *q = realloc(*cols, sizeof(long) * c); if (!q) return 1; *cols = q;
  double *w = realloc(*vals, sizeof(double) * c); if (!w) return 1; *vals = w;
  *cap = c; return 0;
}

/* B as COO (rows: new nodes, cols: old nodes), allocated here and released with vf_free. Pieces below
   rel_tol of their new cell's measure are dropped; `covered` is the summed measure of the kept pieces.
   Returns the number of entries, or -1 when out of memory. */
long vf_mixed_mass(int d, long nn_old, long ne_old, const double *x_old, const long *en_old,
                   long nn_new, long ne_new, const double *x_new, const long *en_new, double rel_tol,
                   long **rows, long **cols, double **vals, double *covered) {
  int loc = d + 1, dd = d;
  *rows = NULL; *cols = NULL; *vals = NULL; *covered = 0.0;
  libsupermesh_tree_intersection_finder_set_input(&nn_new, &dd, &ne_new, &loc, &nn_old, &dd, &ne_old, &loc,
      (double *)x_new, (long *)en_new, (double *)x_old, (long *)en_old);
  long npairs; libsupermesh_tree_intersection_finder_query_output(&npairs);
  long *indices = malloc(sizeof(long) * (npairs > 0 ? npairs : 1)), *indptr = malloc(sizeof(long) * (ne_new + 1));
  if (!indices || !indptr) { free(indices); free(indptr); return -1; }
  libsupermesh_tree_intersection_finder_get_output(&ne_new, &npairs, indices, indptr);

  double vn[12], vo[12], out[3 * 4 * 81 * 4];  /* ample: tet ∩ tet gives at most 3^4 tets, tri ∩ tri 22 tris */
  long k = 0, cap = 0;
  for (long e = 0; e < ne_new; e++) {
    for (int a = 0; a < loc; a++) for (int c = 0; c < d; c++) vn[a*d+c] = x_new[en_new[e*loc+a]*d + c];
    double vol_new;
    if (d == 2) libsupermesh_triangle_area(vn, &vol_new); else libsupermesh_tetrahedron_volume(vn, &vol_new);
    vol_new = fabs(vol_new);
    for (long q = indptr[e]; q < indptr[e+1]; q++) {
      long o = indices[q];
      for (int a = 0; a < loc; a++) for (int c = 0; c < d; c++) vo[a*d+c] = x_old[en_old[o*loc+a]*d + c];
      int npieces = 0;
      if (d == 2) libsupermesh_intersect_tris_real(vn, vo, out, &npieces);
      else libsupermesh_intersect_tets_real(vn, vo, out, &npieces);
      double Bloc[16] = {0}; int any = 0;
      for (int p = 0; p < npieces; p++) {
        double *piece = out + p * loc * d, vol;
        if (d == 2) libsupermesh_triangle_area(piece, &vol); else libsupermesh_tetrahedron_volume(piece, &vol);
        vol = fabs(vol);
        if (vol < rel_tol * vol_new) continue;
        *covered += vol; any = 1;
        double ln[4][4], lo[4][4];
        for (int m = 0; m < loc; m++) { bary(d, vn, piece + m*d, ln[m]); bary(d, vo, piece + m*d, lo[m]); }
        double w = vol / ((d + 1.0) * (d + 2.0));
        for (int j = 0; j < loc; j++) {
          double sj = 0; for (int m = 0; m < loc; m++) sj += ln[m][j];
          for (int i = 0; i < loc; i++) {
            double si = 0, dot = 0;
            for (int m = 0; m < loc; m++) { si += lo[m][i]; dot += ln[m][j] * lo[m][i]; }
            Bloc[j*loc+i] += w * (dot + sj * si);
          }
        }
      }
      if (!any) continue;
      if (grow(k + loc * loc, &cap, rows, cols, vals)) { free(indices); free(indptr); return -1; }
      for (int j = 0; j < loc; j++) for (int i = 0; i < loc; i++) {
        (*rows)[k] = en_new[e*loc+j]; (*cols)[k] = en_old[o*loc+i]; (*vals)[k] = Bloc[j*loc+i]; k++;
      }
    }
  }
  free(indices); free(indptr);
  return k;
}

void vf_free(void *p) { free(p); }
"""


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root) / "vcell-fenics"


@cache
def _kernel() -> ModuleType:
    """The compiled kernel, built once per source and libsupermesh install (the module docstring)."""
    import cffi
    import libsupermesh

    include, library = libsupermesh.get_include(), libsupermesh.get_library()
    flags = ["-O2", "-Wno-unused-function"]  # -Wno-unused-function: the header's own unused helpers
    key = hashlib.sha256((_CDEF + _SOURCE + library + " ".join(flags)).encode()).hexdigest()[:16]
    name = f"_vcell_fenics_supermesh_{key}"
    target = _cache_dir() / "supermesh"
    built = sorted(target.glob(f"{name}.*.so")) + sorted(target.glob(f"{name}.so"))
    if not built:
        target.mkdir(parents=True, exist_ok=True)
        ffi = cffi.FFI()
        ffi.cdef(_CDEF)
        libdir = os.path.dirname(library)
        ffi.set_source(
            name,
            _SOURCE,
            include_dirs=[include],
            library_dirs=[libdir],
            libraries=["supermesh"],
            extra_link_args=[f"-Wl,-rpath,{libdir}"],
            extra_compile_args=flags,
        )
        # Compile in a private directory, then move the result into place: concurrent processes (test workers,
        # MPI ranks) may build at the same time, and a rename is atomic.
        with tempfile.TemporaryDirectory(dir=target) as tmp:
            so = Path(ffi.compile(tmpdir=tmp))
            os.replace(so, target / so.name)
        built = [target / so.name]
    spec = importlib.util.spec_from_file_location(name, built[0])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrays(x: Floats, cells: Ints) -> tuple[int, Floats, Ints]:
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    d = cells.shape[1] - 1
    if d not in (2, 3):
        raise ValueError(f"a supermesh needs triangles or tetrahedra, not cells of {cells.shape[1]} vertices")
    return d, np.ascontiguousarray(np.asarray(x, dtype=np.float64)[:, :d]), cells


def mixed_mass_matrix(
    old_x: Floats, old_cells: Ints, new_x: Floats, new_cells: Ints, *, rel_tol: float = 1.0e-14
) -> tuple[csr_matrix, float]:
    """``B[j, i] = ∫ φ_new_j φ_old_i`` over the supermesh (shape: new nodes x old nodes), and the summed measure of
    the pieces kept: the overlap of the two meshes, up to pieces below ``rel_tol`` of their new cell."""
    d, xo, eo = _arrays(old_x, old_cells)
    d_new, xn, en = _arrays(new_x, new_cells)
    if d_new != d:
        raise ValueError(f"the meshes differ in dimension ({d} and {d_new})")
    lib = _kernel()
    ffi, kernel = lib.ffi, lib.lib
    rows, cols, vals = ffi.new("long **"), ffi.new("long **"), ffi.new("double **")
    covered = ffi.new("double *")
    n = kernel.vf_mixed_mass(
        d,
        len(xo),
        len(eo),
        ffi.from_buffer("double[]", xo),
        ffi.from_buffer("long[]", eo),
        len(xn),
        len(en),
        ffi.from_buffer("double[]", xn),
        ffi.from_buffer("long[]", en),
        rel_tol,
        rows,
        cols,
        vals,
        covered,
    )
    try:
        if n < 0:
            raise MemoryError("the supermesh kernel ran out of memory")
        r = np.frombuffer(ffi.buffer(rows[0], 8 * n), dtype=np.int64).copy() if n else np.zeros(0, np.int64)
        c = np.frombuffer(ffi.buffer(cols[0], 8 * n), dtype=np.int64).copy() if n else np.zeros(0, np.int64)
        v = np.frombuffer(ffi.buffer(vals[0], 8 * n), dtype=np.float64).copy() if n else np.zeros(0)
    finally:
        for pointer in (rows[0], cols[0], vals[0]):
            kernel.vf_free(pointer)
    matrix = coo_matrix((v, (r, c)), shape=(len(xn), len(xo))).tocsr()
    return matrix, float(covered[0])


def mass_matrix(x: Floats, cells: Ints) -> csr_matrix:
    """The P1 mass matrix of a simplex mesh (exact)."""
    d, xs, en = _arrays(x, cells)
    corners = xs[en]
    measure = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / (2.0 if d == 2 else 6.0)
    loc = d + 1
    local = (np.ones((loc, loc)) + np.eye(loc)) / ((d + 1) * (d + 2))
    rows = np.repeat(en, loc, axis=1).ravel()
    cols = np.tile(en, (1, loc)).ravel()
    values = (measure[:, None, None] * local).ravel()
    return coo_matrix((values, (rows, cols)), shape=(len(xs), len(xs))).tocsr()


def project(
    old_x: Floats,
    old_cells: Ints,
    c_old: Floats,
    new_x: Floats,
    new_cells: Ints,
    *,
    lumped: bool = False,
    mixed: csr_matrix | None = None,
) -> Floats:
    """``c_old`` (P1, on the old mesh's nodes) transferred to the new mesh, keeping its total.

    - ``lumped=False``: the L2 projection, ``M_new c_new = B c_old``; second order, reproduces linear fields, may
      undershoot near a steep front.
    - ``lumped=True``: ``c_new = (B c_old) / m``, with ``m`` the new mesh's lumped node masses (``M_new``'s row
      sums); first order, non-negative where ``c_old`` is (``B`` is entrywise non-negative), total kept too.

    ``mixed`` reuses a ``B`` from `mixed_mass_matrix` for the same two meshes (one per compartment, shared by its
    species)."""
    if mixed is None:
        mixed, _ = mixed_mass_matrix(old_x, old_cells, new_x, new_cells)
    rhs = mixed @ np.asarray(c_old, dtype=np.float64)
    m = mass_matrix(new_x, new_cells)
    if lumped:
        lumped_mass = np.asarray(m.sum(axis=1)).ravel()
        result: Floats = rhs / lumped_mass
        return result
    # a mass matrix is well conditioned at any mesh size: CG with a diagonal preconditioner, solved tightly, since
    # the total is kept only to the residual
    diagonal = m.diagonal()
    preconditioner: Any = csr_matrix((1.0 / diagonal, (np.arange(len(diagonal)), np.arange(len(diagonal)))))
    c_new, info = cg(m, rhs, x0=rhs / np.asarray(m.sum(axis=1)).ravel(), rtol=1.0e-14, atol=0.0, M=preconditioner)
    if info != 0:
        raise RuntimeError(f"the supermesh projection's mass solve did not converge (CG info {info})")
    out: Floats = np.asarray(c_new, dtype=np.float64)
    return out

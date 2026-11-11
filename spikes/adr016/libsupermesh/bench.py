"""Spike: libsupermesh-based conservative P1 projection vs vcell-fenics' Python kernel (2D and 3D)."""
import ctypes, sys, time, numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spsolve
lib = ctypes.CDLL(sys.argv[1])
P = np.ctypeslib.ndpointer
lib.lsm_pairs.restype = ctypes.c_long
lib.lsm_pairs.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, P(np.float64), P(np.int64), ctypes.c_long, ctypes.c_long,
                          P(np.float64), P(np.int64), ctypes.POINTER(ctypes.POINTER(ctypes.c_long)), ctypes.POINTER(ctypes.POINTER(ctypes.c_long))]
lib.lsm_mixed_mass.restype = ctypes.c_double
lib.lsm_mixed_mass.argtypes = [ctypes.c_int, P(np.float64), P(np.int64), P(np.float64), P(np.int64), ctypes.c_long,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, P(np.int64), P(np.int64), P(np.float64)]

def arrays(m):
    d = m.topology.dim; x = np.ascontiguousarray(m.geometry.x[:, :d]); en = m.geometry.dofmap.astype(np.int64)
    return d, x, np.ascontiguousarray(en)

def perturb(m, amp, seed):
    d = m.topology.dim; x = m.geometry.x; rng = np.random.default_rng(seed)
    interior = np.all((x[:, :d] > 1e-12) & (x[:, :d] < 1 - 1e-12), axis=1)
    x[interior, :d] += amp * rng.uniform(-1, 1, (interior.sum(), d))

def mass(d, x, en):
    loc = d + 1; v = x[en]                                     # (ne, loc, d)
    J = v[:, 1:, :] - v[:, :1, :]; vol = np.abs(np.linalg.det(J)) / (1 if d == 1 else (2 if d == 2 else 6))
    local = (np.ones((loc, loc)) + np.eye(loc)) / ((d + 1) * (d + 2))
    r = np.repeat(en, loc, axis=1).ravel(); c = np.tile(en, (1, loc)).ravel()
    return coo_matrix(((vol[:, None, None] * local).ravel(), (r, c)), shape=(len(x), len(x))).tocsr(), vol

def project(old, new, c_old, rel_tol=1e-14):
    d, xo, eo = old; _, xn, en = new
    t0 = time.perf_counter()
    ind = ctypes.POINTER(ctypes.c_long)(); ptr = ctypes.POINTER(ctypes.c_long)()
    n = lib.lsm_pairs(d, len(xo), len(eo), xo, eo, len(xn), len(en), xn, en, ctypes.byref(ind), ctypes.byref(ptr))
    t1 = time.perf_counter()
    nnz = n * (d + 1) ** 2; rows = np.empty(nnz, np.int64); cols = np.empty(nnz, np.int64); vals = np.empty(nnz)
    covered = lib.lsm_mixed_mass(d, xo, eo, xn, en, len(en), ctypes.cast(ind, ctypes.c_void_p), ctypes.cast(ptr, ctypes.c_void_p),
                                 rel_tol, rows, cols, vals)
    lib.lsm_free(ind); lib.lsm_free(ptr)
    B = coo_matrix((vals, (rows, cols)), shape=(len(xn), len(xo))).tocsr()
    t2 = time.perf_counter()
    M, vol = mass(d, xn, en)
    c_new, info = cg(M, B @ c_old, rtol=1e-14, atol=0, M=None)
    t3 = time.perf_counter()
    return c_new, dict(pairs=n, finder=t1 - t0, assemble=t2 - t1, solve=t3 - t2, covered=covered, area=vol.sum(), cg=info)

def total(m_arrays, c):
    d, x, en = m_arrays; M, _ = mass(d, x, en); return float(np.ones(len(x)) @ (M @ c))

def case(name, old_m, new_m):
    old, new = arrays(old_m), arrays(new_m); d = old[0]
    rng = np.random.default_rng(1)
    for label, c_old in (("linear", 1 + old[1] @ np.arange(1, d + 1)), ("rough", rng.uniform(0, 1, len(old[1])))):
        c_new, info = project(old, new, c_old)
        T0, T1 = total(old, c_old), total(new, c_new)
        lin = np.max(np.abs(c_new - (1 + new[1] @ np.arange(1, d + 1)))) if label == "linear" else float("nan")
        print(f"{name} {label}: old {len(old[2])} cells, new {len(new[2])}; pairs {info['pairs']}; finder {info['finder']:.2f} s, "
              f"assemble {info['assemble']:.2f} s, solve {info['solve']:.2f} s; covered/area-1 {info['covered']/info['area']-1:.1e}; "
              f"total rel change {abs(T1-T0)/abs(T0):.1e}; linear max err {lin:.1e}", flush=True)
    return old, new

comm = MPI.COMM_SELF
which = sys.argv[2]
if which == "check2d":
    from vcell_fenics.core.bulk_remap import supermesh_project_2d
    a = dmesh.create_unit_square(comm, 40, 40); b = dmesh.create_unit_square(comm, 37, 37); perturb(a, 0.006, 2); perturb(b, 0.006, 3)
    old, new = case("2D small", a, b)
    c_old = np.random.default_rng(5).uniform(0, 1, len(old[1]))
    t0 = time.perf_counter(); c_py = supermesh_project_2d(old[1], old[2], c_old, new[1], new[2]); tp = time.perf_counter() - t0
    c_c, info = project(old, new, c_old)
    print(f"vs Python kernel: max |diff| {np.max(np.abs(c_py - c_c)):.1e} (values ~1); Python {tp:.2f} s, libsupermesh {info['finder']+info['assemble']+info['solve']:.3f} s")
elif which == "2d":
    n = int(sys.argv[3]); a = dmesh.create_unit_square(comm, n, n); b = dmesh.create_unit_square(comm, n - 7, n - 7)
    perturb(a, 0.15 / n, 2); perturb(b, 0.15 / n, 3); case("2D", a, b)
elif which == "3d":
    n = int(sys.argv[3]); a = dmesh.create_unit_cube(comm, n, n, n); b = dmesh.create_unit_cube(comm, n - 2, n - 2, n - 2)
    perturb(a, 0.15 / n, 2); perturb(b, 0.15 / n, 3); case("3D", a, b)

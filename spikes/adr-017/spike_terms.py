"""Spike: FEM shape-derivative forms vs vcell-core's vertex vectors (contract C), P1, fan-triangulated faces."""
import json, sys
import numpy as np
import ufl, basix.ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_vector

def build(points, faces):
    pts = [list(p) for p in points]; cells = []; ctag = []
    for ci, f in enumerate(faces):
        c = np.mean([points[i] for i in f], axis=0); k = len(pts); pts.append(list(c))
        for a, b in zip(f, f[1:] + f[:1]):
            cells.append([a, b, k]); ctag.append(ci)
    el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    m = dmesh.create_mesh(MPI.COMM_SELF, np.array(cells, dtype=np.int64), el, np.array(pts))
    # dolfinx reorders: map original node -> geometry index
    orig = m.geometry.input_global_indices
    tags = np.array(ctag, dtype=np.int32)[np.argsort(m.topology.original_cell_index)] if False else np.array(ctag, dtype=np.int32)[m.topology.original_cell_index]
    ct = dmesh.meshtags(m, 2, np.arange(len(cells), dtype=np.int32), tags)
    return m, ct, orig

def run(case):
    pts, faces, d = case["points"], case["faces"], case["depth"]
    m, ct, orig = build(pts, faces)
    V = fem.functionspace(m, ("Lagrange", 1, (2,)))
    phi = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=m, subdomain_data=ct)
    # indicator per face (DG0) to localise facet integrals to one face's boundary
    Q = fem.functionspace(m, ("DG", 0)); out = {}
    n = ufl.FacetNormal(m)
    def surf_div(side=None):
        g = ufl.grad(phi) if side is None else ufl.grad(phi)(side)
        nn = n if side is None else n(side)
        return ufl.tr(g) - ufl.dot(nn, ufl.dot(g, nn))
    def chi(ci):
        f = fem.Function(Q); f.x.array[:] = (ct.values == ci).astype(float)[np.argsort(ct.indices)]; return f
    nf = len(faces)
    chis = [chi(i) for i in range(nf)]
    # boundary of face ci = exterior facets with chi=1 plus interior facets where the jump of chi is nonzero
    def perim_form(ci, integrand_ext, integrand_int):
        c = chis[ci]
        return integrand_ext * c * ufl.ds + abs(c("+") - c("-")) * integrand_int * ufl.dS
    def vec(form):
        if form.empty(): return np.zeros((len(pts), 2))
        b = assemble_vector(fem.form(form)); b = b.array.reshape(-1, 2)
        # dof -> geometry node: P1 on P1 geometry; map dofs to vertices via coordinates
        X = V.tabulate_dof_coordinates()[:, :2]
        R = np.zeros((len(pts), 2))
        for i, p in enumerate(pts):
            j = np.argmin(np.linalg.norm(X - np.array(p), axis=1)); R[i] = b[j]
        return R
    def area(ci): return fem.assemble_scalar(fem.form(chis[ci] * ufl.dx))
    def perim(ci):
        c = chis[ci]; return fem.assemble_scalar(fem.form(c * ufl.ds + abs(c("+") - c("-")) * ufl.dS))
    T = case["terms"]; E = {}
    tot = {"area": [0, 0], "surface": [0, 0], "cortical": [0, 0], "interfacial": [0, 0]}
    def add(key, e, R):
        tot[key][0] += e; tot[key][1] = tot[key][1] + (-R)
    for ci_s, p in T["area_elasticity"].items():
        ci = int(ci_s); A = area(ci); Vc = A * d; K, V0 = p["K"], p["V0"]
        e = 0.5 * K * (Vc - V0) ** 2 / d
        c = chis[ci]
        R = (K * (Vc - V0)) * vec((c * ufl.dot(phi, n) * ufl.ds + (c("+") - c("-")) * ufl.dot(phi("+"), n("+")) * ufl.dS))
        add("area", e, R)
    for ci_s, p in T["surface_elasticity"].items():
        ci = int(ci_s); S = perim(ci) * d; KS, S0 = p["K_S"], p["S0"]
        e = 0.5 * KS * (S - S0) ** 2 / d
        add("surface", e, (KS * (S - S0)) * vec(perim_form(ci, surf_div(), surf_div("+"))))
    for ci_s, p in T["cortical_tension"].items():
        ci = int(ci_s); s = p["sigma"]
        add("cortical", s * perim(ci), s * vec(perim_form(ci, surf_div(), surf_div("+"))))
    for key, p in T["interfacial_tension"].items():
        a, b = map(int, key.split(",")); L = p["Lambda"]
        contact = chis[a]("+") * chis[b]("-") + chis[a]("-") * chis[b]("+")
        e = L * fem.assemble_scalar(fem.form(contact * ufl.dS))
        add("interfacial", e, L * vec(contact * surf_div("+") * ufl.dS))
    exp = case["expected"]; worst = 0.0
    for k, (e, F) in tot.items():
        if k not in exp: continue
        de = abs(e - exp[k]["energy"]); dF = np.max(np.abs(np.array(F) - np.array(exp[k]["forces"])))
        worst = max(worst, de, dF); print(f"  {k:12s} dE={de:.1e} dF={dF:.1e}")
    if "friction" in exp:
        z = sum(T["friction"][str(i)]["zeta"] * area(i) for i in range(nf))
        print(f"  friction     total mobility FEM {z:.6g} vs sum(zeta a_v) {sum(exp['friction']['mobility']):.6g}")
    return worst

worst = 0.0
for line in open(sys.argv[1]):
    c = json.loads(line); print(c["case"]); worst = max(worst, run(c))
print("worst", worst)

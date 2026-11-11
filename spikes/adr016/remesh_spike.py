"""Spike (ADR 016): a moving partition run with whole-partition remeshing. Usage: remesh_spike.py SIMTASK [h]"""
import sys, time, math, numpy as np, ufl
from pathlib import Path
from dolfinx import fem
from scipy.spatial import cKDTree
from vcell_fenics.cli import load_simtask, build_parser, resolve_options
from vcell_fenics.runner import _stride_ends
from vcell_fenics.backend.multi_compartment import realize_multi_compartment, integrate_multi_compartment
from vcell_fenics.backend.partition_remesh import remesh_partition, transfer_bulk, transfer_membrane

path = Path(sys.argv[1]); model = load_simtask(path)
args = build_parser().parse_args(["--simtask", str(path)] + (["--h", sys.argv[2]] if len(sys.argv) > 2 else []))
opts = resolve_options(model, args)
membrane = next(s.name for s in model.math.subdomains if getattr(s, "motion", None) is not None and s.kind == "surface")
variables = {eq.variable: eq.subdomain for eq in model.math.equations}
wanted = {sub for sub in variables.values() if sub not in {sc.name for sc in model.geometry.surfaces}}
inside = next(s.inside for s in model.geometry.surfaces if s.name == membrane); wanted.add(inside)
g = realize_multi_compartment(model.geometry, h=opts.h, compartments=wanted)
ends_all = _stride_ends(opts)
print(f"h {opts.h:g}, {len(ends_all)} strides to t = {opts.t_final:g}, parent cells {g.parent_mesh.topology.index_map(2).size_local}", flush=True)

def totals(fields, geometry):
    return {n: float(np.real(fem.assemble_scalar(fem.form(f * ufl.dx)))) for n, f in fields.items()}

def front_gap(geometry):
    """Smallest distance between points of the moving membrane more than h apart along it."""
    m = geometry.membranes[membrane].mesh; x = m.geometry.x[:, :2]
    from vcell_fenics.core.surface_remap_mesh import ordered_membrane_loop
    V = fem.functionspace(m, ("Lagrange", 1)); q, _ = ordered_membrane_loop(V)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]; L = s[-1] + np.linalg.norm(q[-1] - q[0])
    best = np.inf
    for i, j in cKDTree(q).query_pairs(3 * opts.h):
        da = abs(s[i] - s[j]); da = min(da, L - da)
        if da > opts.h: best = min(best, float(np.linalg.norm(q[i] - q[j])))
    return best

t, pending, initial, remeshes, log = 0.0, None, None, 0, []
first = None
while True:
    ends = [e for e in ends_all if e > t + 1e-12]
    t0 = time.perf_counter()
    r = integrate_multi_compartment(model.math, g, t_final=opts.t_final, rtol=opts.rtol, atol=opts.atol, motion_membrane=membrane,
                                    stride_ends=ends, t_start=t, initial=initial, pending_move=pending, remesh_on_tangle=True,
                                    output_times=opts.output_times, on_output=lambda tt, f: print(f'    t = {tt:.2f} ({time.strftime("%H:%M:%S")})', flush=True))
    tot = totals(r.fields, g)
    if first is None:
        first = tot  # totals at the end of the first segment, for reference
    print(f"segment to t = {r.final_time:.4f} ({time.perf_counter()-t0:.0f} s, {r.steps} steps); totals " + ", ".join(f"{k} {v:.12g}" for k, v in tot.items())
          + (f"; front gap {front_gap(g):.4f}" if r.needs_remesh else ""), flush=True)
    if not r.needs_remesh:
        break
    if remeshes >= 3 and r.steps == 0:
        print('a fresh mesh tangled on its first move: stopping (ADR 016 decision 2)'); break
    if remeshes >= 40:
        print("too many remeshes; stopping"); break
    t, pending = r.final_time, r.pending_move
    t0 = time.perf_counter()
    g_new = remesh_partition(g, model.geometry, h=opts.h, compartments=wanted)
    t_mesh = time.perf_counter() - t0
    old = r.fields; kinds = {}
    t0 = time.perf_counter()
    cache = {}
    def make_initial(old=old, g_old=g, g_new=g_new):
        def initial(name, V):
            sub = variables[name]
            if sub in g_new.membranes:
                return transfer_membrane(old[name], V)
            u, kind = transfer_bulk(old[name], V); kinds[name] = kind; return u
        return initial
    initial = make_initial()
    # transfer now (to time it and check totals), and hand the results over
    new_fields = {}
    for name in old:
        sub = variables[name]; V = fem.functionspace(g_new.mesh_of(sub), ("Lagrange", 1))
        new_fields[name] = initial(name, V)
    t_tr = time.perf_counter() - t0
    new_tot = totals(new_fields, g_new)
    worst = max(abs(new_tot[k] - tot[k]) / max(abs(tot[k]), 1e-300) for k in tot)
    remeshes += 1
    print(f"  remesh {remeshes} at t = {t:.4f}{' (pending half-move)' if pending else ''}: {g_new.parent_mesh.topology.index_map(2).size_local} cells, "
          f"mesh {t_mesh:.1f} s, transfer {t_tr:.1f} s ({kinds}), worst total change {worst:.2e}", flush=True)
    stash = new_fields
    initial = (lambda stash: (lambda name, V: _assign(stash[name], V)))(stash)
    def _assign(src, V):
        dst = fem.Function(V); 
        if dst.x.array.size != src.x.array.size: raise RuntimeError("space mismatch")
        # same mesh, possibly different dof layout: map by coordinates
        xs = src.function_space.tabulate_dof_coordinates(); xd = V.tabulate_dof_coordinates()
        idx = cKDTree(xs).query(xd)[1]; dst.x.array[:] = src.x.array[idx]; return dst
    g = g_new
print(f"done: {remeshes} remeshes, reached t = {r.final_time:.4f}", flush=True)

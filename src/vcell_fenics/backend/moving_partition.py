"""Prescribed motion of one membrane of a multi-compartment partition (ADR 014 decision 5).

The multi-compartment solver keeps every compartment and membrane on one parent mesh, with a submesh and an
`EntityMap` for each. This moves them together, one method-of-lines stride at a time:

1. **The front.** The membrane's prescribed velocity `v` is evaluated on the membrane mesh at the move's end:
   `geom.normal` is the membrane's normal out of its inside compartment (the area-weighted facet normal of the
   inside compartment's submesh, `BoundaryNormal`), `trace(X)` reads the bulk species `X` at the membrane's
   nodes, and `sim.t` is the move's end time. Its displacement over a move of length `h` is `h·v`.
2. **The slide** (2D, decision 4). The membrane's nodes are slid along the moved membrane, along the periodic
   spline through them, back to the fractions of its length they had when the mesh was built; only the
   front's normal motion is physical, and the membrane's species are transported relative to the slide.
3. **The parent.** A harmonic extension of that displacement fills the parent's interior, with the box and
   every other membrane held fixed (a membrane without a velocity is static, as in VCell). Every mesh — the
   parent, each compartment's and each membrane's submesh — then takes the same displacement at its coincident
   nodes, so the topological `EntityMap`s stay valid.
4. **The checks.** The front must not come into contact with itself, the box or another membrane
   (`FrontContact`, 2D: ADR 016 decision 5), and the moved parent must keep every cell's orientation
   (`CellOrientation`): either stops the run.

The solver carries each species' nodal moments through every move (ADR 015) and reads `parent_velocity` (the
mesh velocity `w` in the bulk, the transport relative to the mesh `−u (c − w)·∇q`), `membrane_velocity` (the
material velocity `v` of the membrane's species) and the membrane's own `membrane_mesh_velocity` (the slid
nodes' velocity).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend.boundary_normal import BoundaryNormal
from vcell_fenics.backend.compiler import CompileContext, compile_expression, region_size_symbol
from vcell_fenics.backend.discrete import (
    _MAX_CELL_RATIO_GROWTH,
    CellOrientation,
    MeshQualityError,
    MeshTangled,
    _arc_fractions,
    _slid_to_fractions,
    signed_cell_volumes,
)
from vcell_fenics.backend.geometry import prescribed_velocity
from vcell_fenics.backend.interface_coupled import _param_symbols
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import MathDescription

# a node of one mesh and its twin on another coincide to round-off (the submeshes copy the parent's geometry)
_COINCIDENT = 1.0e-9


class FrontContact(MeshQualityError):
    """The moving membrane has come into contact with itself, the box or another membrane (ADR 016 decision 5).
    A remesh keeps every region's topology, so it cannot carry a run past contact: the run stops."""


class FrontCannotAdvance(MeshQualityError):
    """A freshly remeshed partition tangled on its very first move (ADR 016 decision 6): the cause is the front's
    own geometry (a valley its normal motion folds into a cusp, or two parts closing), not the mesh, so another
    remesh would not help and the run stops."""


# a cell that has shrunk below this fraction of its area when its mesh was built triggers a remesh (ADR 016
# decision 2): the single-mesh path's 4x limit, read per cell
REMESH_SHRINK = 0.25


def _point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distance from each `points[k]` to the segment `a[k]`–`b[k]`."""
    ab = b - a
    length2 = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", points - a, ab) / np.where(length2 > 0.0, length2, 1.0), 0.0, 1.0)
    closest = a + t[:, None] * ab
    distances: np.ndarray = np.linalg.norm(points - closest, axis=1)
    return distances


def _nearest_segment_gap(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    cap: float,
    exclude: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> float:
    """The smallest distance from any of `points` to any segment `a[j]`–`b[j]`, or `cap` if none is nearer.
    `exclude(i, j)` (vectorized over candidate pairs) drops pairs. Between non-crossing segments the smallest
    distance is attained at an endpoint, so checking every point of one set against the other set's segments,
    both ways, gives the polylines' gap."""
    if len(points) == 0 or len(a) == 0:
        return cap
    midpoints = 0.5 * (a + b)
    half = 0.5 * float(np.max(np.linalg.norm(b - a, axis=1)))
    tree = cKDTree(midpoints)
    near = tree.query_ball_point(points, cap + half)
    i = np.repeat(np.arange(len(points)), [len(n) for n in near])
    if i.size == 0:
        return cap
    j = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    if exclude is not None:
        keep = ~exclude(i, j)
        i, j = i[keep], j[keep]
        if i.size == 0:
            return cap
    return min(cap, float(_point_segment_distances(points[i], a[j], b[j]).min()))


class _ContactCheck:
    """ADR 016 decision 5, in 2D: after each move, the moving membrane's gap to itself, to the box and to every
    other membrane. Two parts of the front count as separate only if they lie more than `h` apart along it. The
    run stops when a gap falls below `delta = min(h/4, g0/2)`, with `g0` the smallest such gap as built, so a
    geometry built with a narrow gap stops only close to true contact. Gaps are measured exactly between the
    polylines (every vertex against every segment, both ways), and only up to `h/2`, which bounds `delta`."""

    def __init__(
        self,
        front: np.ndarray,
        others: Mapping[str, tuple[np.ndarray, np.ndarray]],
        box: tuple[np.ndarray, np.ndarray],
        h: float,
    ) -> None:
        self._others = others
        self._box = box
        self._h = h
        self._cap = 0.5 * h
        g0, _ = self.gap(front)
        self.initial_gap = g0
        self.threshold = min(0.25 * h, 0.5 * g0)

    def gap(self, front: np.ndarray) -> tuple[float, str]:
        """The front's smallest gap (at most `h/2`) and what it is to."""
        cap, h = self._cap, self._h
        a, b = front, np.roll(front, -1, axis=0)
        lengths = np.linalg.norm(b - a, axis=1)
        s = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        perimeter = float(lengths.sum())

        def along(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            d = np.abs(x - y)
            result: np.ndarray = np.minimum(d, perimeter - d)
            return result

        def same_part(i: np.ndarray, j: np.ndarray) -> np.ndarray:
            ends = np.minimum(along(s[i], s[j]), along(s[i], s[(j + 1) % len(s)]))
            result: np.ndarray = ends <= h
            return result

        gaps = [(_nearest_segment_gap(a, a, b, cap, same_part), "itself")]
        lo, hi = self._box
        gaps.append((min(cap, float(np.min(np.minimum(front - lo, hi - front)))), "the box"))
        for name, (oa, ob) in self._others.items():
            g = min(_nearest_segment_gap(a, oa, ob, cap), _nearest_segment_gap(oa, a, b, cap))
            gaps.append((g, f"membrane {name!r}"))
        return min(gaps, key=lambda item: item[0])

    def check(self, front: np.ndarray, where: str, membrane: str) -> None:
        gap, what = self.gap(front)
        if gap < self.threshold:
            raise FrontContact(
                f"{where}: the moving membrane {membrane!r} has come within {gap:.3g} of {what} (the contact "
                f"threshold is {self.threshold:.3g} = min(h/4, half its narrowest gap as built, "
                f"{self.initial_gap:.3g}), at h = {self._h:g}). A remesh cannot carry a run past contact (vcell-fenics "
                f"ADR 016 decision 5); the prescribed motion brings the membrane into contact here."
            )


def _matching(points: np.ndarray, onto: np.ndarray, what: str) -> np.ndarray:
    """For each of `points`, the index of its coincident twin among `onto`."""
    distance, index = cKDTree(onto).query(points)
    if distance.size and float(distance.max()) > _COINCIDENT:
        raise RuntimeError(f"{what}: {int((distance > _COINCIDENT).sum())} nodes have no coincident twin")
    return np.asarray(index)


class MovingPartition:
    """Moves a `MultiCompartmentGeometry` under the prescribed velocity of its membrane `membrane` (the module
    docstring). `traces` maps each bulk species the velocity may read to its P1 field on its compartment's
    submesh (the solver's per-species views, refreshed before each move)."""

    def __init__(
        self,
        md: MathDescription,
        geometry: object,
        membrane: str,
        *,
        traces: Mapping[str, fem.Function],
    ) -> None:
        from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry

        assert isinstance(geometry, MultiCompartmentGeometry)
        self._geometry = geometry
        sub = next(s for s in md.subdomains if s.name == membrane)
        velocity = prescribed_velocity(sub)
        if velocity is None:
            raise ValueError(f"membrane {membrane!r} has no prescribed, non-zero velocity")
        part = geometry.membranes[membrane]
        if part.inside not in geometry.compartments:
            raise NotImplementedError(
                f"the moving membrane {membrane!r} needs its inside compartment {part.inside!r} realized (its "
                "normal is that compartment's outward one)"
            )
        parent = geometry.parent_mesh
        self._gdim = gdim = parent.geometry.dim
        tdim = parent.topology.dim
        parent.topology.create_connectivity(tdim - 1, tdim)
        mem_mesh = part.mesh

        # --- the front: v on the membrane, reading the normal, the traces and the time ---------------------
        self.time = fem.Constant(mem_mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]
        V_m = fem.functionspace(mem_mesh, ("Lagrange", 1, (gdim,)))
        S_m = fem.functionspace(mem_mesh, ("Lagrange", 1))
        mem_xyz = V_m.tabulate_dof_coordinates()[:, :gdim]
        self._traces: list[tuple[fem.Function, fem.Function, np.ndarray]] = []
        trace_symbols: dict[str, fem.Function] = {}
        for name, source in traces.items():
            on_membrane = fem.Function(S_m, name=name)
            source_xyz = source.function_space.tabulate_dof_coordinates()[:, :gdim]
            self._traces.append((source, on_membrane, _matching(mem_xyz, source_xyz, f"trace of {name!r}")))
            trace_symbols[name] = on_membrane
        inside = geometry.compartments[part.inside].mesh
        inside.topology.create_connectivity(tdim - 1, tdim)
        exterior = dmesh.exterior_facet_indices(inside.topology)
        midpoints = dmesh.compute_midpoints(inside, tdim - 1, exterior)[:, :gdim]
        membrane_midpoints = dmesh.compute_midpoints(
            mem_mesh, tdim - 1, np.arange(mem_mesh.topology.index_map(tdim - 1).size_local, dtype=np.int32)
        )[:, :gdim]
        on_front = exterior[cKDTree(membrane_midpoints).query(midpoints)[0] < _COINCIDENT]
        self._inside_normal = BoundaryNormal(inside, on_front)
        self.normal = fem.Function(V_m, name="normal")
        inside_xyz = self._inside_normal.function.function_space.tabulate_dof_coordinates()[:, :gdim]
        self._normal_from_inside = _matching(mem_xyz, inside_xyz, "the membrane's normal")
        # the parameters, `region_size(...)` among them: its Constants (`size_constants`) are the solver's to
        # re-measure after each move, so the front reads the sizes of the mesh it moves from
        params = _param_symbols(md, mem_mesh, geometry.sizes, self.time)
        self.size_constants = [(name, params[region_size_symbol(name)]) for name in geometry.sizes]
        symbols = {
            **params,
            "geom.x": ufl.SpatialCoordinate(mem_mesh),
            "sim.t": self.time,
            "__normal__": self.normal,
            **trace_symbols,
        }
        self.membrane_velocity = fem.Function(V_m, name="v")
        self._velocity = fem.Expression(
            compile_expression(parse(velocity), CompileContext(mem_mesh, symbols)), V_m.element.interpolation_points
        )
        self._membrane_displacement = fem.Function(V_m)
        self.membrane_mesh_velocity = fem.Function(V_m, name="w_membrane")

        # --- the slide: the membrane's nodes back to their fractions of its length (2D) --------------------
        self._loop: np.ndarray | None = None
        if gdim == 2 and parent.comm.size == 1:
            from vcell_fenics.core.surface_remap_mesh import ordered_membrane_loop

            _, order = ordered_membrane_loop(S_m)
            scalar_xyz = S_m.tabulate_dof_coordinates()[:, :gdim]
            self._loop = _matching(scalar_xyz[order], mem_xyz, "the membrane's loop")
            self._fractions = _arc_fractions(mem_xyz[self._loop])

        # --- the parent: a harmonic extension, the moving membrane's displacement on its facets, 0 elsewhere -
        V_p = fem.functionspace(parent, ("Lagrange", 1, (gdim,)))
        parent_xyz = V_p.tabulate_dof_coordinates()[:, :gdim]
        front_facets = geometry.facet_tags.find(part.facet_tag)
        fixed_facets = np.setdiff1d(
            np.concatenate(
                [dmesh.exterior_facet_indices(parent.topology)]
                + [geometry.facet_tags.find(m.facet_tag) for name, m in geometry.membranes.items() if name != membrane]
            ),
            front_facets,
        ).astype(np.int32)
        front_dofs = fem.locate_dofs_topological(V_p, tdim - 1, front_facets)
        fixed_dofs = np.setdiff1d(fem.locate_dofs_topological(V_p, tdim - 1, fixed_facets), front_dofs)
        self._front_dofs = front_dofs
        self._front_from_membrane = _matching(parent_xyz[front_dofs], mem_xyz, "the front's parent nodes")
        self._front_value = fem.Function(V_p)
        self._displacement = fem.Function(V_p)
        u, q = ufl.TrialFunction(V_p), ufl.TestFunction(V_p)
        zero = fem.Constant(parent, np.zeros(gdim, dtype=PETSc.ScalarType))
        self._extension = LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(q)) * ufl.dx,
            ufl.inner(zero, q) * ufl.dx,
            u=self._displacement,
            bcs=[
                fem.dirichletbc(self._front_value, front_dofs),
                fem.dirichletbc(fem.Function(V_p), fixed_dofs.astype(np.int32)),
            ],
            petsc_options_prefix=f"vcellfenics_partition_motion_{id(self):x}_",
            # CG with algebraic multigrid: a direct solve grows as (cells)^2.5 in 3D (46 s per move at 136k cells,
            # against 0.8 s here, ADR 015). Each move's moments are carried exactly whatever mesh this yields, so
            # the extension need not be solved tightly.
            petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1.0e-10},
        )
        self.parent_velocity = fem.Function(V_p, name="w")

        # every mesh moves by the parent's displacement at its coincident nodes (topological, so computed once)
        meshes = [
            parent,
            *(c.mesh for c in geometry.compartments.values()),
            *(m.mesh for m in geometry.membranes.values()),
        ]
        self._moves = [(m, _matching(m.geometry.x[:, :gdim], parent_xyz, "a submesh")) for m in meshes]
        # The contact stop (ADR 016 decision 5), in 2D on one process, where the front is an ordered loop.
        self._membrane = membrane
        self._contact: _ContactCheck | None = None
        if self._loop is not None:
            front = mem_xyz[self._loop]
            h_mesh = geometry.h
            if h_mesh is None:  # a geometry built by hand: the front's own spacing stands in for h
                h_mesh = float(np.median(np.linalg.norm(np.roll(front, -1, axis=0) - front, axis=1)))
            others = {}
            for name, m in geometry.membranes.items():
                if name == membrane:
                    continue
                x, segments = m.mesh.geometry.x[:, :gdim], np.asarray(m.mesh.geometry.dofmap)
                others[name] = (x[segments[:, 0]].copy(), x[segments[:, 1]].copy())
            box = (parent_xyz.min(axis=0), parent_xyz.max(axis=0))
            self._contact = _ContactCheck(front, others, box, h_mesh)
        # The moved parent's validity after every move: a move that tangles it, or distorts it past the hard limit,
        # is undone (`advance`), for a driver to remesh (ADR 016 decision 2) or stop.
        self._orientation = CellOrientation(parent)
        self._growth = 1.0
        self._built_volume = np.abs(signed_cell_volumes(parent))

    def degraded(self) -> bool:
        """Whether any cell has shrunk below `REMESH_SHRINK` of its area when this mesh was built (the original
        mesh, or the remesh that made it): the proactive remesh trigger (ADR 016 decision 2). Per cell, not the
        global size ratio, which a geometry's own small cells would hold back until a notch cell nearly inverts."""
        current = np.abs(signed_cell_volumes(self._geometry.parent_mesh))
        ratio = current / np.where(self._built_volume > 0.0, self._built_volume, 1.0)
        local = float(ratio.min()) if ratio.size else 1.0
        from mpi4py import MPI

        return bool(self._geometry.parent_mesh.comm.allreduce(local, op=MPI.MIN) < REMESH_SHRINK)

    def advance(self, t_end: float, h: float) -> None:
        """Move every mesh by `h` times the front's velocity read at `t_end` (the module docstring): a whole stride,
        or half of one (ADR 015).

        A move that would invert a cell, or distort the mesh past the hard limit, is undone, every mesh restored
        to its stored coordinates (not `x + d - d`, which is not bitwise `x`), and raises `MeshTangled`: a driver
        that can remesh does so and redoes the move (ADR 016 decision 2). Contact (`FrontContact`) is a stop."""
        saved = [mesh.geometry.x.copy() for mesh, _ in self._moves]
        try:
            self._advance(t_end, h)
        except FrontContact:
            raise
        except MeshQualityError as error:
            for (mesh, _), x in zip(self._moves, saved, strict=True):
                mesh.geometry.x[:] = x
            raise MeshTangled(str(error)) from error

    def _advance(self, t_end: float, h: float) -> None:
        gdim = self._gdim
        self.time.value = t_end
        for source, on_membrane, index in self._traces:
            on_membrane.x.array[:] = source.x.array[index]
        self._inside_normal.refresh()
        normal = self._inside_normal.function.x.array.reshape((-1, gdim))
        self.normal.x.array[:] = normal[self._normal_from_inside].ravel()
        self.membrane_velocity.interpolate(self._velocity)

        here = self.membrane_velocity.function_space.tabulate_dof_coordinates()[:, :gdim]
        displacement = h * self.membrane_velocity.x.array.reshape((-1, gdim))
        if self._loop is not None:
            moved = here[self._loop] + displacement[self._loop]
            displacement[self._loop] = _slid_to_fractions(moved, self._fractions) - here[self._loop]
        self._membrane_displacement.x.array[:] = displacement.ravel()
        self.membrane_mesh_velocity.x.array[:] = displacement.ravel() / h

        front = self._front_value.x.array.reshape((-1, gdim))
        front[:] = 0.0
        front[self._front_dofs] = displacement[self._front_from_membrane]
        self._extension.solve()
        parent_displacement = self._displacement.x.array.reshape((-1, gdim))
        self.parent_velocity.x.array[:] = parent_displacement.ravel() / h

        for mesh, index in self._moves:
            mesh.geometry.x[:, :gdim] += parent_displacement[index]
        where = f"the moving partition at t = {t_end:g}"
        if self._contact is not None and self._loop is not None:  # before the orientation: contact is the cause
            moved_front = self.membrane_velocity.function_space.tabulate_dof_coordinates()[:, :gdim][self._loop]
            self._contact.check(moved_front, where, self._membrane)
        self._growth = self._orientation.check(where)
        if self._growth > _MAX_CELL_RATIO_GROWTH:
            raise MeshQualityError(
                f"{where}: the motion has distorted the mesh (cell-size ratio grown {self._growth:.3g}x since it "
                f"was built), past the hard limit of {_MAX_CELL_RATIO_GROWTH:g}x."
            )

    def mesh_quality_growth(self) -> float:
        """The parent mesh's cell-size ratio growth since it was built (1.0 undistorted), as of the last stride:
        what a remesh trigger would read."""
        return self._growth

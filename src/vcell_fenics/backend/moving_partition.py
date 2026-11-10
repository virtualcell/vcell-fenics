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
4. **The check.** The moved parent must keep every cell's orientation (`CellOrientation`): a tangled mesh
   stops the run.

The solver carries each species' nodal moments through every move (ADR 015) and reads `parent_velocity` (the
mesh velocity `w` in the bulk, the transport relative to the mesh `−u (c − w)·∇q`), `membrane_velocity` (the
material velocity `v` of the membrane's species) and the membrane's own `membrane_mesh_velocity` (the slid
nodes' velocity).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend.boundary_normal import BoundaryNormal
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.discrete import (
    _MAX_CELL_RATIO_GROWTH,
    CellOrientation,
    MeshQualityError,
    _arc_fractions,
    _slid_to_fractions,
)
from vcell_fenics.backend.geometry import prescribed_velocity
from vcell_fenics.backend.interface_coupled import _param_symbols
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import MathDescription

# a node of one mesh and its twin on another coincide to round-off (the submeshes copy the parent's geometry)
_COINCIDENT = 1.0e-9


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
        symbols = {
            **_param_symbols(md, mem_mesh, None, self.time),
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
        # The moved parent's validity after every stride (ADR 014 decision 6: no remeshing yet, so a mesh the
        # motion has tangled or badly distorted stops the run rather than being solved on).
        self._orientation = CellOrientation(parent)
        self._growth = 1.0

    def advance(self, t_end: float, h: float) -> None:
        """Move every mesh by `h` times the front's velocity read at `t_end` (the module docstring): a whole stride,
        or half of one (ADR 015)."""
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
        self._growth = self._orientation.check(where)
        if self._growth > _MAX_CELL_RATIO_GROWTH:
            raise MeshQualityError(
                f"{where}: the motion has distorted the mesh (cell-size ratio grown {self._growth:.3g}x since it "
                f"was built), and a moving partition cannot be remeshed yet (vcell-fenics ADR 014 decision 6); "
                f"use a coarser mesh, a smaller motion, or a shorter run."
            )

    def mesh_quality_growth(self) -> float:
        """The parent mesh's cell-size ratio growth since it was built (1.0 undistorted), as of the last stride:
        what a remesh trigger would read."""
        return self._growth

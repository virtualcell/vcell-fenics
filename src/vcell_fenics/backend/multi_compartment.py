"""Any number of compartments and membranes: the multi-compartment solver.

A VCell cell is rarely just a cytosol in extracellular space. A nucleus, an ER, organelles, or two cells side
by side give three or more compartments with a membrane between each adjacent pair; about a fifth of the saved
spatial BioModels are like that. This module solves them on one body-fitted partition of the box.

**Geometry** (:func:`realize_multi_compartment` → :class:`MultiCompartmentGeometry`). The realized partition
(any analytic, box-touching or image geometry) is kept as the *parent* mesh with its cell tags (one per
subvolume) and facet tags: each membrane (SurfaceClass) keeps its tag, and each box face is split by
compartment, so a compartment's species see only their own share of a face. Every compartment the math
needs gets a submesh and an `EntityMap` to the parent; every membrane with facets gets a codim-1 submesh.
A subvolume that carries nothing stays in the parent mesh with no unknowns: its membranes are plain walls
(or carry the single-sided flux the import writes for them).

**Unknowns.** One scalar P1 block per species on its own compartment or membrane mesh, and one Real per
region variable (§1.4.2 T5), in declaration order. All integrals are on the parent: a compartment's cells
``dx(tag)``, a membrane's interior facets ``dS(tag)``, a compartment's share of a box face ``ds(tag)``, and a
membrane species' own mass and surface diffusion on its membrane mesh.

**Coupling.** A membrane couples only its own two sides. On ``dS`` a compartment's function is read through a
**side-masked trace** ``f('+')·χ('+') + f('-')·χ('-')`` (χ the exact 0/1 indicator of the compartment): the
same value as ``f('+') + f('-')`` for a coefficient, and — unlike it — the right one for a test or trial
function, which DOLFINx 0.10 aliases to both restrictions (memory `project_interface_flux_overcount`). Each
jump-condition flux goes into its own species' block on its own side; a membrane species' reaction reads the
traces of both sides.

**Time.** Method of lines (PETSc TS adaptive BDF). The Jacobian is assembled exactly, except that DOLFINx
assembles a bulk-row × membrane-species column on ``dS`` as zero; so with membrane species the implicit
Jacobian is applied matrix-free (finite differences of the residual, as `integrate_membrane_coupled`) and the
assembled one preconditions it.

Box faces: a Dirichlet value (VCell's Value boundary) is a weak penalty on the compartment's share of that
face, a Neumann value (Flux) a boundary flux; a face the compartment does not touch has nothing to act on
(VCell's per-face boilerplate for an interior compartment) and is dropped.

Fixed geometry only; lab-frame ``advection`` and the interface value-equality constraint are refused.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import scifem
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem import petsc
from dolfinx.mesh import EntityMap, Mesh, MeshTags
from mpi4py import MPI
from petsc4py import PETSc
from ufl.algorithms import expand_derivatives

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression
from vcell_fenics.backend.equations import as_field_equation
from vcell_fenics.backend.interface_coupled import (
    _RESERVOIR_PENALTY,
    _accumulate_ghosts,
    _interpolate_component_ics,
    _interpolate_ic,
    _MatrixFreeShiftedJacobian,
    _param_symbols,
    _refuse_lab_frame_advection,
    connected_region_count,
)
from vcell_fenics.backend.linear_solvers import set_preconditioner
from vcell_fenics.backend.output_times import OutputMonitor
from vcell_fenics.backend.realize import (
    _FACE_NAMES_2D,
    _FACE_NAMES_3D,
    _FACE_TAG_BASE,
    RealizationError,
    _realize_2d_partition,
    _realize_3d_partition,
)
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
    BCInterfaceValueEquality,
    BCNeumann,
    MathDescription,
    TemplateEquation,
)
from vcell_fenics.formalism.validator import validate_or_raise

# a compartment's share of a box face: its own facet tag (distinct from the membranes' and the faces')
_BOX_TAG_BASE = 10_000


def _box_tag(region_tag: int, face_index: int) -> int:
    return _BOX_TAG_BASE + 100 * region_tag + face_index


@dataclass(frozen=True)
class CompartmentPart:
    """A volume compartment: its submesh and entity map, its cell tag on the parent, and the facet tag of
    its share of each box face it touches."""

    name: str
    mesh: Mesh
    entity_map: EntityMap
    region_tag: int
    faces: dict[str, int]


@dataclass(frozen=True)
class MembranePart:
    """A membrane between two subvolumes: its codim-1 submesh and entity map, and its facet tag."""

    name: str
    inside: str
    outside: str
    mesh: Mesh
    entity_map: EntityMap
    facet_tag: int


@dataclass
class MultiCompartmentGeometry:
    """A realized partition of the box into any number of compartments and membranes (the module docstring).
    ``sizes`` holds every subvolume's volume (area in 2D) and every membrane's area (length), for
    `region_size(...)` — including subvolumes without a submesh."""

    name: str
    parent_mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags
    compartments: dict[str, CompartmentPart]
    membranes: dict[str, MembranePart]
    region_tags: dict[str, int]
    sizes: dict[str, float] = field(default_factory=dict)

    def mesh_of(self, subdomain: str) -> Mesh:
        if subdomain in self.compartments:
            return self.compartments[subdomain].mesh
        if subdomain in self.membranes:
            return self.membranes[subdomain].mesh
        raise KeyError(f"{subdomain!r} is not a realized compartment or membrane of {self.name!r}")

    def entity_maps(self) -> list[EntityMap]:
        return [c.entity_map for c in self.compartments.values()] + [m.entity_map for m in self.membranes.values()]


def realize_multi_compartment(
    description: GeometryDescription,
    *,
    h: float,
    compartments: Collection[str] | None = None,
    resolution: int | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> MultiCompartmentGeometry:
    """Realize ``description`` (2D or 3D, two or more subvolumes) for the multi-compartment solver. Submeshes
    are made for the subvolumes in ``compartments`` (default: all) and for every membrane with facets."""

    if len(description.subvolumes) < 2:
        raise RealizationError(f"geometry {description.name!r} has one subvolume: there are no compartments to couple")
    face_names: tuple[str, ...]
    if description.dim == 3:
        parent, tagging = _realize_3d_partition(description, h=h, resolution=resolution, comm=comm)
        face_names = _FACE_NAMES_3D
    elif description.dim == 2:
        parent, tagging = _realize_2d_partition(description, h=h, resolution=resolution, comm=comm)
        face_names = _FACE_NAMES_2D
    else:
        raise NotImplementedError(f"multi-compartment realization of a {description.dim}D geometry is not implemented")

    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = parent.topology.connectivity(tdim - 1, tdim)
    cell_map = parent.topology.index_map(tdim)
    region_of_cell = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
    region_of_cell[tagging.cell_tags.indices] = tagging.cell_tags.values
    offsets = np.asarray(facet_to_cell.offsets)
    links = np.asarray(facet_to_cell.array)

    # Facet tags: each membrane keeps its tag; each box face is split by the compartment beside it.
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for tag in tagging.surface_tags.values():
        facets = tagging.facet_tags.find(tag)
        indices.append(facets)
        values.append(np.full(facets.size, tag, dtype=np.int32))
    for i in range(len(face_names)):
        facets = tagging.facet_tags.find(_FACE_TAG_BASE + i)
        regions = region_of_cell[links[offsets[facets]]]
        indices.append(facets)
        values.append((_BOX_TAG_BASE + 100 * regions + i).astype(np.int32))
    idx = np.concatenate(indices).astype(np.int32) if indices else np.empty(0, dtype=np.int32)
    val = np.concatenate(values).astype(np.int32) if values else np.empty(0, dtype=np.int32)
    order = np.argsort(idx)
    facet_tags = dmesh.meshtags(parent, tdim - 1, idx[order], val[order])

    def global_count(entities: np.ndarray) -> int:
        return int(comm.allreduce(int(entities.size), op=MPI.SUM))

    wanted = set(compartments) if compartments is not None else {sv.name for sv in description.subvolumes}
    parts: dict[str, CompartmentPart] = {}
    for sv in description.subvolumes:
        tag = tagging.region_tags[sv.name]
        cells = tagging.cell_tags.find(tag)
        if sv.name not in wanted or not global_count(cells):  # collective: every rank decides alike
            continue
        sub, emap, *_ = dmesh.create_submesh(parent, tdim, cells)
        faces = {
            face: _box_tag(tag, i)
            for i, face in enumerate(face_names)
            if global_count(facet_tags.find(_box_tag(tag, i)))
        }
        parts[sv.name] = CompartmentPart(name=sv.name, mesh=sub, entity_map=emap, region_tag=tag, faces=faces)

    membranes: dict[str, MembranePart] = {}
    for sc in description.surfaces:
        tag = tagging.surface_tags[sc.name]
        facets = facet_tags.find(tag)
        if not global_count(facets):
            continue  # the declared membrane has no realized interface (its subvolumes don't meet)
        sub, emap, *_ = dmesh.create_submesh(parent, tdim - 1, facets)
        membranes[sc.name] = MembranePart(
            name=sc.name, inside=sc.inside, outside=sc.outside, mesh=sub, entity_map=emap, facet_tag=tag
        )

    geometry = MultiCompartmentGeometry(
        name=description.name,
        parent_mesh=parent,
        cell_tags=tagging.cell_tags,
        facet_tags=facet_tags,
        compartments=parts,
        membranes=membranes,
        region_tags=dict(tagging.region_tags),
    )
    geometry.sizes = _measures(geometry, tagging.surface_tags)
    return geometry


def _measures(geometry: MultiCompartmentGeometry, surface_tags: dict[str, int]) -> dict[str, float]:
    parent = geometry.parent_mesh
    one = fem.Constant(parent, PETSc.ScalarType(1.0))  # type: ignore[operator]
    cells = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)
    facets = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags)

    def measure(form: UflExpr) -> float:
        return float(parent.comm.allreduce(float(np.real(fem.assemble_scalar(fem.form(form)))), op=MPI.SUM))

    sizes = {name: measure(one * cells(tag)) for name, tag in geometry.region_tags.items()}
    sizes |= {name: measure(one("+") * facets(tag)) for name, tag in surface_tags.items()}
    return sizes


def _indicator(parent: Mesh, cell_tags: MeshTags, region_tag: int) -> fem.Function:
    chi = fem.Function(fem.functionspace(parent, ("DG", 0)))
    chi.x.array[:] = 0.0
    chi.x.array[cell_tags.indices[cell_tags.values == region_tag]] = 1.0
    chi.x.scatter_forward()
    return chi


@dataclass
class MultiCompartmentResult:
    """The final state: every species and region variable by name (a Real for a region variable), and the
    subdomain each lives on."""

    fields: dict[str, fem.Function]
    subdomain_of: dict[str, str]
    steps: int
    final_time: float


def integrate_multi_compartment(
    md: MathDescription,
    geometry: MultiCompartmentGeometry,
    *,
    t_final: float,
    dt_initial: float | None = None,
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
    ksp_type: str = "gmres",
    pc_type: str = "ilu",
    output_times: Sequence[float] = (),
    on_output: Callable[[float, dict[str, fem.Function]], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> MultiCompartmentResult:
    """Integrate ``md`` on ``geometry`` to ``t_final`` (the module docstring). ``on_output(t, fields)`` receives
    every species and region variable by name (snapshots valid for the call) at each of ``output_times``,
    recorded from the TS monitor so the step sequence is unchanged; ``on_progress(t)`` follows every step."""

    validate_or_raise(md)
    template_eqs = [eq for eq in md.equations if isinstance(eq, TemplateEquation)]
    kind_of = {sd.name: sd.kind for sd in md.subdomains}
    region_eqs = [eq for eq in template_eqs if eq.template == "region_ode"]
    field_eqs = [as_field_equation(eq, kind_of[eq.subdomain]) for eq in template_eqs if eq.template != "region_ode"]
    for eq in field_eqs:
        _refuse_lab_frame_advection(eq)
        if eq.subdomain not in geometry.compartments and eq.subdomain not in geometry.membranes:
            raise NotImplementedError(
                f"species {eq.variable!r} lives on {eq.subdomain!r}, which geometry {geometry.name!r} does not realize "
                f"(compartments {sorted(geometry.compartments)}, membranes {sorted(geometry.membranes)})"
            )
    for eq in region_eqs:
        if eq.subdomain not in geometry.compartments and eq.subdomain not in geometry.membranes:
            raise NotImplementedError(
                f"region variable {eq.variable!r} lives on {eq.subdomain!r}, which is not realized"
            )
        pieces = connected_region_count(geometry.mesh_of(eq.subdomain))
        if pieces > 1:
            raise NotImplementedError(
                f"region variable {eq.variable!r} lives on {eq.subdomain!r}, realized as {pieces} disconnected "
                "regions; one value per region (per-region instancing) is not supported yet (#196)"
            )
    for bc in md.boundary_conditions:
        if isinstance(bc, BCInterfaceValueEquality):
            raise NotImplementedError("the interface value-equality constraint (u = k·u_adjacent) is not supported")

    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    cells = ufl.Measure("dx", domain=parent, subdomain_data=geometry.cell_tags)
    facets = ufl.Measure("dS", domain=parent, subdomain_data=geometry.facet_tags, metadata={"quadrature_degree": 4})
    box = ufl.Measure("ds", domain=parent, subdomain_data=geometry.facet_tags)

    # sim.t on the parent and on each membrane mesh (a Constant belongs to one mesh), set to each stage time
    time = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    params = _param_symbols(md, parent, geometry.sizes, time)
    ctx = CompileContext(parent, {"geom.x": ufl.SpatialCoordinate(parent), **params})
    membrane_times = {
        name: fem.Constant(m.mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]
        for name, m in geometry.membranes.items()
    }
    surface_ctx = {
        name: CompileContext(
            m.mesh,
            {
                "geom.x": ufl.SpatialCoordinate(m.mesh),
                **_param_symbols(md, m.mesh, geometry.sizes, membrane_times[name]),
            },
        )
        for name, m in geometry.membranes.items()
    }

    # --- the blocks: P1 blocks for the species, then one Real per region variable. With membrane species the
    # Newton is matrix-free and assembles the residual once per Krylov iteration, a cost that grows with the
    # number of blocks: so one block per subdomain there, a component per species. Otherwise the Jacobian is
    # assembled exactly, and one scalar block per species keeps it sparse (a subdomain's species that never
    # meet have no block between them, where a vector block would couple every pair at every node).
    matrix_free = any(eq.subdomain in geometry.membranes for eq in field_eqs)
    members = {
        sub: [eq for eq in field_eqs if eq.subdomain == sub] for sub in dict.fromkeys(e.subdomain for e in field_eqs)
    }
    groups = list(members.values()) if matrix_free else [[eq] for eq in field_eqs]

    def field_space(group: list[TemplateEquation]) -> fem.FunctionSpace:
        n = len(group)
        element = ("Lagrange", 1, (n,)) if n > 1 else ("Lagrange", 1)
        return fem.functionspace(geometry.mesh_of(group[0].subdomain), element)

    spaces = [field_space(group) for group in groups]
    real = scifem.create_real_functionspace(parent)
    spaces += [real] * len(region_eqs)
    n_fields = len(groups)
    n_blocks = len(spaces)
    all_eqs = [*field_eqs, *region_eqs]
    names = [eq.variable for eq in all_eqs]
    states = [fem.Function(V) for V in spaces]
    rates = [fem.Function(V) for V in spaces]
    tests = [ufl.TestFunction(V) for V in spaces]
    trials = [ufl.TrialFunction(V) for V in spaces]
    sizes = [V.dofmap.index_map.size_local * V.dofmap.index_map_bs for V in spaces]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    subdomain_of = {eq.variable: eq.subdomain for eq in all_eqs}
    block_sub = [*(group[0].subdomain for group in groups), *(eq.subdomain for eq in region_eqs)]
    where: dict[str, tuple[int, int | None]] = {}  # a variable's block, and its component (None: the block is it)
    for blk, group in enumerate(groups):
        for j, eq in enumerate(group):
            where[eq.variable] = (blk, j if len(group) > 1 else None)
    for r, eq in enumerate(region_eqs):
        where[eq.variable] = (n_fields + r, None)

    def part(functions: Sequence[UflExpr], name: str) -> UflExpr:
        blk, j = where[name]
        return functions[blk] if j is None else functions[blk][j]

    region_value = {eq.variable: states[n_fields + r] for r, eq in enumerate(region_eqs)}
    region_on_facet = {name: u("+") for name, u in region_value.items()}

    chi = {
        name: _indicator(parent, geometry.cell_tags, part_.region_tag) for name, part_ in geometry.compartments.items()
    }

    def trace(compartment: str, f: UflExpr) -> UflExpr:
        return f("+") * chi[compartment]("+") + f("-") * chi[compartment]("-")

    def bulk_ctx(compartment: str) -> CompileContext:
        """A compartment's own species (the states) and every region variable, beside the parameters."""
        own_species = {eq.variable: part(states, eq.variable) for eq in members.get(compartment, [])}
        return CompileContext(parent, {**ctx.symbols, **region_value, **own_species})

    def membrane_ctx(membrane: str) -> CompileContext:
        """On a membrane's dS: both sides' species by their traces, its own species, every region variable."""
        m = geometry.membranes[membrane]
        symbols: dict[str, UflExpr] = {**ctx.symbols, **region_on_facet}
        for side in (m.inside, m.outside):
            if side in geometry.compartments:
                symbols |= {eq.variable: trace(side, part(states, eq.variable)) for eq in members.get(side, [])}
        symbols |= {eq.variable: part(states, eq.variable)("+") for eq in members.get(membrane, [])}
        return CompileContext(parent, symbols)

    # A block form integrates over ONE domain. So the residual is two blocked forms, assembled and summed:
    # `residual` on the parent (every bulk and region term, and every term on a membrane's dS) and `own` on a
    # membrane block's own mesh (its species' mass, surface diffusion and drift). A block with no terms in one
    # of them gets a structural zero there, so the two blocked vectors line up.
    zero = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    residual: list[UflExpr | None] = [None] * n_blocks
    own: dict[int, UflExpr] = {}

    def add(blk: int, form: UflExpr) -> None:
        current = residual[blk]
        residual[blk] = form if current is None else current + form

    for eq in field_eqs:
        blk, _ = where[eq.variable]
        w, u, rate = part(tests, eq.variable), part(states, eq.variable), part(rates, eq.variable)
        if eq.subdomain in geometry.compartments:
            dx = cells(geometry.compartments[eq.subdomain].region_tag)
            local = ctx
        else:
            dx = ufl.Measure("dx", domain=geometry.membranes[eq.subdomain].mesh)
            local = surface_ctx[eq.subdomain]
        form = rate * w * dx
        if "diffusion" in eq.terms:
            d = compile_expression(parse(eq.terms["diffusion"]), local)
            form += d * ufl.dot(ufl.grad(u), ufl.grad(w)) * dx
        if "relative_advection" in eq.terms:  # drift relative to the (fixed) mesh
            drift = compile_expression(parse(eq.terms["relative_advection"]), local)
            form += ufl.dot(drift, ufl.grad(u)) * w * dx
        if eq.subdomain in geometry.compartments:
            if "source" in eq.terms:
                form += -compile_expression(parse(eq.terms["source"]), bulk_ctx(eq.subdomain)) * w * dx
            add(blk, form)
            continue
        own[blk] = own[blk] + form if blk in own else form
        dS = facets(geometry.membranes[eq.subdomain].facet_tag)
        add(blk, zero * w("+") * dS)
        if "source" in eq.terms:  # a membrane reaction reads both sides' traces: on the parent's dS
            add(blk, -compile_expression(parse(eq.terms["source"]), membrane_ctx(eq.subdomain)) * w("+") * dS)

    for r, eq in enumerate(region_eqs):
        blk = n_fields + r
        w = tests[blk]
        if eq.subdomain in geometry.compartments:
            measure = cells(geometry.compartments[eq.subdomain].region_tag)
            here, rate_here, rate_ctx = w, rates[blk], bulk_ctx(eq.subdomain)
        else:
            measure = facets(geometry.membranes[eq.subdomain].facet_tag)
            here, rate_here, rate_ctx = w("+"), rates[blk]("+"), membrane_ctx(eq.subdomain)
        form = rate_here * here * measure
        for slot in ("uniform_rate", "region_rate"):
            if slot in eq.terms:
                form += -compile_expression(parse(eq.terms[slot]), rate_ctx) * here * measure
        add(blk, form)

    # --- boundary conditions: membrane fluxes into their own side, box-face values and fluxes ------------
    face_names = _FACE_NAMES_3D if tdim == 3 else _FACE_NAMES_2D
    for bc in md.boundary_conditions:
        if not isinstance(bc, (BCInterfaceFlux, BCNeumann, BCDirichlet)) or bc.variable not in where:
            continue
        blk, _ = where[bc.variable]
        home = subdomain_of[bc.variable]
        w = part(tests, bc.variable)
        if bc.boundary in geometry.membranes and not isinstance(bc, BCDirichlet):
            m = geometry.membranes[bc.boundary]
            dS = facets(m.facet_tag)
            flux = compile_expression(parse(bc.expression), membrane_ctx(m.name))  # D∇u·n = flux INTO u's side
            if blk >= n_fields or home == m.name:  # a region variable's (or a membrane quantity's) own balance
                add(blk, -flux * w("+") * dS)
            elif home in (m.inside, m.outside):
                add(blk, -flux * trace(home, w) * dS)
            else:
                raise NotImplementedError(
                    f"flux BC on {bc.variable!r} at membrane {m.name!r}: {home!r} is not beside that membrane "
                    f"({m.inside!r} | {m.outside!r})"
                )
        elif bc.boundary in face_names and home in geometry.compartments and blk < n_fields:
            compartment = geometry.compartments[home]
            if bc.boundary not in compartment.faces:
                continue  # VCell's per-face boilerplate for a compartment that does not reach this face
            ds = box(compartment.faces[bc.boundary])
            value = compile_expression(parse(bc.expression), bulk_ctx(home))
            if isinstance(bc, BCDirichlet):  # a held value, weakly: β ∮ (u − g) w
                add(blk, _RESERVOIR_PENALTY * (part(states, bc.variable) - value) * w * ds)
            else:
                add(blk, -value * w * ds)
        elif bc.boundary in face_names and home in geometry.membranes:
            continue  # a membrane's edge on the box: VCell's per-face values for membrane species, no-flux here
        else:
            raise NotImplementedError(
                f"{type(bc).__name__} on {bc.variable!r} at {bc.boundary!r}: the multi-compartment solver applies "
                f"membrane fluxes ({sorted(geometry.membranes)}) and box-face values/fluxes ({list(face_names)})"
            )
    parent_forms = [form for form in residual if form is not None]
    assert len(parent_forms) == n_blocks  # every block has its mass (or a structural zero) on the parent

    # --- the Jacobian (exact but for DOLFINx's zero bulk × membrane-species block on dS) -----------------
    shift = fem.Constant(parent, PETSc.ScalarType(0.0))  # type: ignore[operator]
    shift_surface = {
        name: fem.Constant(m.mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]
        for name, m in geometry.membranes.items()
    }

    def home_cells(blk: int) -> ufl.Measure:
        """Where a bulk block (or a region variable) lives on the parent: its compartment's cells, not the
        whole parent — its test function is defined only there."""
        sub = block_sub[blk]
        if sub in geometry.compartments:
            return cast(ufl.Measure, cells(geometry.compartments[sub].region_tag))
        return cast(ufl.Measure, ufl.dx(domain=parent))  # a Real on a membrane: one value, any cell

    def structural_diagonal(blk: int, on_parent: bool) -> UflExpr:
        """A zero diagonal block (DOLFINx places a row and a column by their blocks): on the parent a membrane
        block's is on its dS, elsewhere any block's on its home cells."""
        if on_parent and blk < n_fields and block_sub[blk] in geometry.membranes:
            dS = facets(geometry.membranes[block_sub[blk]].facet_tag)
            return zero * ufl.inner(trials[blk]("+"), tests[blk]("+")) * dS
        return zero * ufl.inner(trials[blk], tests[blk]) * home_cells(blk)

    def blocks_of(forms: Sequence[UflExpr], on_parent: bool) -> list[list[UflExpr | None]]:
        rows: list[list[UflExpr | None]] = []
        for i in range(n_blocks):
            row: list[UflExpr | None] = []
            for j in range(n_blocks):
                on_membrane = j < n_fields and block_sub[j] in geometry.membranes
                sigma = shift_surface[block_sub[j]] if on_membrane else shift
                # each part expanded on its own (ufl.derivative is lazy: an unrelated block is empty only once
                # expanded), or its zero integrals reach DOLFINx, compiled against the wrong mesh
                parts = [expand_derivatives(ufl.derivative(forms[i], v)) for v in (rates[j], states[j])]
                d_rate, d_state = (None if p.empty() else p for p in parts)
                if d_rate is not None and d_state is not None:
                    row.append(sigma * d_rate + d_state)
                elif d_rate is not None:
                    row.append(sigma * d_rate)
                else:
                    row.append(d_state)
            if row[i] is None:  # every row and column needs a block (a decoupled species' column is empty)
                row[i] = structural_diagonal(i, on_parent)
            rows.append(row)
        return rows

    emaps = geometry.entity_maps()
    residual_form = fem.form(parent_forms, entity_maps=emaps)
    jacobian_form = fem.form(cast(Any, blocks_of(parent_forms, True)), entity_maps=emaps)  # None: a zero block
    own_residual_form = own_jacobian_form = None
    if own:
        own_forms = [own.get(k, _zero_like(zero, tests[k]) * home_cells(k)) for k in range(n_blocks)]
        own_residual_form = fem.form(own_forms, entity_maps=emaps)
        own_jacobian_form = fem.form(cast(Any, blocks_of(own_forms, False)), entity_maps=emaps)

    # The matrices are made once (their sparsity is fixed) and assembled into in place: making one is a
    # sparsity build, which cost as much as the assembly itself when done per Jacobian evaluation.
    parent_matrix: PETSc.Mat = petsc.create_matrix(jacobian_form, kind="mpi")
    own_matrix: PETSc.Mat | None = petsc.create_matrix(own_jacobian_form, kind="mpi") if own_jacobian_form else None

    def assemble_jacobian(into: PETSc.Mat) -> None:
        """The Jacobian into ``into``, whose sparsity holds both groups'."""
        parent_matrix.zeroEntries()
        petsc.assemble_matrix(parent_matrix, jacobian_form)  # type: ignore[arg-type]  # singledispatch: the Mat overload
        parent_matrix.assemble()
        into.zeroEntries()
        into.axpy(1.0, parent_matrix, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)  # type: ignore[arg-type]
        if own_matrix is not None and own_jacobian_form is not None:
            own_matrix.zeroEntries()
            petsc.assemble_matrix(own_matrix, own_jacobian_form)  # type: ignore[arg-type]  # singledispatch: the Mat overload
            own_matrix.assemble()
            into.axpy(1.0, own_matrix, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)  # type: ignore[arg-type]
        into.assemble()

    for blk, group in enumerate(groups):
        ic_ctx = ctx if block_sub[blk] in geometry.compartments else surface_ctx[block_sub[blk]]
        if len(group) > 1:
            _interpolate_component_ics(states[blk], group, ic_ctx)
        else:
            _interpolate_ic(states[blk], group[0], ic_ctx)
    for r, eq in enumerate(region_eqs):
        states[n_fields + r].x.array[:] = _region_initial(eq, geometry, ctx, cells, facets)

    def scatter(x: PETSc.Vec, targets: Sequence[fem.Function]) -> None:
        for k, target in enumerate(targets):
            target.x.array[: sizes[k]] = x.array_r[offsets[k] : offsets[k + 1]]
            target.x.scatter_forward()  # ghosts too: a partition-boundary cell reads them

    def set_time(t: float) -> None:
        time.value = t
        for constant in membrane_times.values():
            constant.value = t

    def assemble_residual(state: PETSc.Vec, rate: PETSc.Vec, out: PETSc.Vec) -> None:
        scatter(state, states)
        scatter(rate, rates)
        b = _accumulate_ghosts(petsc.assemble_vector(residual_form, kind="mpi"))
        if own_residual_form is not None:
            own_b = _accumulate_ghosts(petsc.assemble_vector(own_residual_form, kind="mpi"))
            b.axpy(1.0, own_b)
            own_b.destroy()
        b.copy(out)
        b.destroy()

    def evaluate_residual(_ts: PETSc.TS, t: float, x: PETSc.Vec, x_dot: PETSc.Vec, result: PETSc.Vec) -> None:
        set_time(t)
        assemble_residual(x, x_dot, result)

    state_vec = petsc.create_vector(spaces, kind="mpi")
    for k, state in enumerate(states):
        state_vec.array[offsets[k] : offsets[k + 1]] = state.x.array[: sizes[k]]
    shift.value = 1.0
    for constant in shift_surface.values():
        constant.value = 1.0
    # the preconditioner holds the union of both groups' sparsity (off-diagonal blocks too); each matrix is
    # assembled once first, so its sparsity is the full one (DOLFINx inserts every element entry)
    petsc.assemble_matrix(parent_matrix, jacobian_form)  # type: ignore[arg-type]  # singledispatch: the Mat overload
    parent_matrix.assemble()
    precond = parent_matrix.duplicate(copy=True)
    if own_matrix is not None:
        petsc.assemble_matrix(own_matrix, own_jacobian_form)  # type: ignore[arg-type]  # singledispatch: the Mat overload
        own_matrix.assemble()
        precond.axpy(1.0, own_matrix, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)  # type: ignore[arg-type]
    assemble_jacobian(precond)

    mf: _MatrixFreeShiftedJacobian | None = None
    operator = precond
    if matrix_free:
        mf = _MatrixFreeShiftedJacobian(assemble_residual, state_vec)
        vec_sizes = state_vec.getSizes()
        operator = PETSc.Mat().createPython((vec_sizes, vec_sizes), comm=parent.comm)  # type: ignore[arg-type]
        operator.setPythonContext(mf)
        operator.setUp()

    def evaluate_jacobian(
        _ts: PETSc.TS, t: float, x: PETSc.Vec, x_dot: PETSc.Vec, sigma: float, _mat: PETSc.Mat, pre: PETSc.Mat
    ) -> None:
        set_time(t)
        if mf is not None:
            mf.set_base(x, x_dot, sigma)
        scatter(x, states)
        shift.value = sigma
        for constant in shift_surface.values():
            constant.value = sigma
        assemble_jacobian(pre)

    ts = PETSc.TS().create(parent.comm)
    ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)  # type: ignore[arg-type]
    ts.setType("bdf")
    ts.setIFunction(evaluate_residual, state_vec.duplicate())
    ts.setIJacobian(evaluate_jacobian, operator, precond)
    ts.setTimeStep(dt_initial if dt_initial is not None else t_final / 1.0e4)
    ts.setMaxTime(t_final)
    ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)  # type: ignore[arg-type]
    ts.setTolerances(atol, rtol)
    ts.setMaxSNESFailures(-1)
    snes = ts.getSNES()
    snes.setUseEW(False)
    snes.getKSP().setType(ksp_type)
    set_preconditioner(snes.getKSP(), pc_type)  # ILU → block-Jacobi/ILU(0) under MPI
    ts.setFromOptions()

    # Each variable by name: a scalar P1 view of its block's component (a copy), or its Real.
    views = _Views(names, where, spaces, n_fields)
    monitor: OutputMonitor | None = None
    work: PETSc.Vec | None = None
    if on_output is not None or on_progress is not None:
        snaps = [fem.Function(V) for V in spaces]
        work = state_vec.duplicate()

        def emit(t: float, x: PETSc.Vec) -> None:
            scatter(x, snaps)
            if on_output is not None:
                on_output(t, views.of(snaps))

        monitor = OutputMonitor(
            output_times if on_output is not None else (),
            t_start=0.0,
            t_final=t_final,
            work=work,
            emit=emit,
            progress=on_progress,
        )
        ts.setMonitor(monitor)

    ts.solve(state_vec)
    steps, final_time = ts.getStepNumber(), float(ts.getTime())
    if monitor is not None:
        monitor.finish(ts, final_time, state_vec)
    scatter(state_vec, states)
    for obj in (ts, state_vec, precond, parent_matrix, own_matrix, work):
        if obj is not None:
            obj.destroy()
    if operator is not precond:
        operator.destroy()
    final = {name: _copy(f) for name, f in views.of(states).items()}
    return MultiCompartmentResult(final, subdomain_of, steps, final_time)


def _zero_like(zero: fem.Constant, test: UflExpr) -> UflExpr:
    """``0 · test`` summed over its components: a structural zero for a block of any shape."""
    return zero * (sum(test[i] for i in range(test.ufl_shape[0])) if test.ufl_shape else test)


class _Views:
    """Each variable by name as its own Function: a scalar P1 copy of its block's component (the output
    and the result speak per species), or a region variable's Real itself."""

    def __init__(
        self,
        names: Sequence[str],
        where: dict[str, tuple[int, int | None]],
        spaces: Sequence[fem.FunctionSpace],
        n_fields: int,
    ) -> None:
        self._where = where
        self._scalar: dict[str, tuple[fem.Function, np.ndarray | None]] = {}
        collapsed: dict[int, list[tuple[fem.FunctionSpace, np.ndarray]]] = {}
        for name in names:
            blk, j = where[name]
            if blk >= n_fields:
                continue
            if j is None:
                self._scalar[name] = (fem.Function(spaces[blk], name=name), None)
                continue
            if blk not in collapsed:
                bs = spaces[blk].dofmap.index_map_bs
                collapsed[blk] = [spaces[blk].sub(k).collapse() for k in range(bs)]
            sub_space, dofs = collapsed[blk][j]
            self._scalar[name] = (fem.Function(sub_space, name=name), np.asarray(dofs))

    def of(self, blocks: Sequence[fem.Function]) -> dict[str, fem.Function]:
        out: dict[str, fem.Function] = {}
        for name, (blk, _) in self._where.items():
            if name not in self._scalar:
                out[name] = blocks[blk]  # a region variable: its Real
                continue
            view, dofs = self._scalar[name]
            view.x.array[:] = blocks[blk].x.array if dofs is None else blocks[blk].x.array[dofs]
            out[name] = view
        return out


def _copy(f: fem.Function) -> fem.Function:
    g = fem.Function(f.function_space, name=f.name)
    g.x.array[:] = f.x.array
    return g


def _region_initial(
    eq: TemplateEquation,
    geometry: MultiCompartmentGeometry,
    ctx: CompileContext,
    cells: ufl.Measure,
    facets: ufl.Measure,
) -> float:
    """A region variable's initial value: its initial condition averaged over its region."""

    assert eq.initial_condition is not None  # the validator requires one (time-dependent)
    parent = geometry.parent_mesh
    ic = compile_expression(parse(eq.initial_condition), ctx)
    one = fem.Constant(parent, PETSc.ScalarType(1.0))  # type: ignore[operator]
    if eq.subdomain in geometry.compartments:
        measure = cells(geometry.compartments[eq.subdomain].region_tag)
    else:
        measure = facets(geometry.membranes[eq.subdomain].facet_tag)
        ic, one = ic("+"), one("+")
    total = parent.comm.allreduce(fem.assemble_scalar(fem.form(ic * measure)), op=MPI.SUM)
    size = parent.comm.allreduce(fem.assemble_scalar(fem.form(one * measure)), op=MPI.SUM)
    return float(np.real(total) / np.real(size))


def species_mass(result: MultiCompartmentResult, name: str) -> float:
    """∫ of a species over its compartment or membrane mesh (a region variable's value, not its amount)."""

    f = result.fields[name]
    mesh = f.function_space.mesh
    return float(
        mesh.comm.allreduce(float(np.real(fem.assemble_scalar(fem.form(f * ufl.dx(domain=mesh))))), op=MPI.SUM)
    )

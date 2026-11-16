"""The cell force balance (T9) from a MathDescription, solved on a fixed geometry (ADR 018 staging step 3a).

This is the assembler between the formalism and the term builders of `force_balance.py`:

- **The members' mesh.** The realized partition (`realize_multi_compartment`) is restricted to the members'
  cells. Their region tags are kept as the member tags; each realized membrane keeps its facet tag, and each box
  face a member touches gets one tag (its share of the face, merged over the members).
- **The coefficients.** Each slot of the `volumes:` and `surfaces:` maps, each parameter, `geom.x`, `sim.t`
  and `region_size(...)` are compiled on that mesh; `geom.normal` is the facet normal, for a `traction`. A slot
  that reads a species or a region variable is refused: chemistry on the cells is staging step 5.
- **The velocity conditions.** Each `dirichlet` BC on the velocity becomes a `VelocityCondition` on its box
  face or membrane.
- **The T9 rules that need the geometry** (ADR 018 decisions 2 and 4, ADR 019 decisions 4 and 5), before any
  solve: each surface in `surfaces:` bounds a member; every membrane bounding a member moves with `v` or is a
  growing front, and one moving with `v` bounds a member; a growing front is a free membrane meeting no other
  membrane or the box; a membrane with a `dirichlet` or a traction is a free one (a member against a
  non-member); a member touching the box is held there (`MemberOnFreeBox`); each connected group of members
  has a drag or a `dirichlet` condition (`UnconstrainedRigidMotion`).
- **The checks at every solve** (ADR 018 decisions 3, 4 and 6): the viscosity `> 0`, the bulk viscosity and
  the drag `≥ 0` at each member's quadrature points and vertices; a given tension `> 0` at its facets'
  midpoints (`NonPositiveTension`); a drag positive somewhere in each group no `dirichlet` holds; a finite
  velocity (`SolveFailed`); and a warning where `η_b/η` is large enough to lock a low-order element.
- **`geom.normal`** is read only in a `traction`, where it points out of the surface's `inside`.

The solve is quasi-static: `solve()` gives the velocity of the current configuration. Moving the cells with it
is stage 4, and the runner refuses T9 until then (ADR 018, "Not decided here").
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import basix
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.mesh import Mesh, MeshTags
from mpi4py import MPI
from numpy.typing import NDArray
from petsc4py import PETSc
from scipy.spatial import cKDTree

from vcell_fenics.backend._typing import UflExpr
from vcell_fenics.backend.compiler import CompileContext, compile_expression, region_size_symbol
from vcell_fenics.backend.discrete import DiscreteProblem
from vcell_fenics.backend.force_balance import (
    MemberLaw,
    MembersMesh,
    SurfaceLaw,
    VelocityCondition,
    assemble_force_balance,
    velocity_dirichlet,
)
from vcell_fenics.backend.multi_compartment import MultiCompartmentGeometry, realize_multi_compartment
from vcell_fenics.backend.realize import _FACE_NAMES_2D, _FACE_NAMES_3D
from vcell_fenics.formalism.expr import (
    BinaryOp,
    Expr,
    FunctionCall,
    IndexAccess,
    Name,
    Number,
    TensorLiteral,
    UnaryOp,
    VectorLiteral,
)
from vcell_fenics.formalism.geometry_schema import GeometryDescription
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    MathDescription,
    MotionPrescribedVelocity,
    MotionUnknown,
    ParameterConstant,
    ParameterExpression,
    TemplateEquation,
)
from vcell_fenics.formalism.templates import CELL_FORCE_BALANCE
from vcell_fenics.formalism.validator import validate_or_raise

# The members' mesh keeps the parent's region tags (members) and membrane tags (surfaces); a box face gets
# this base plus its index, merged over the members that touch it.
_FACE_TAG = 3_000_000
# Above this ratio of bulk to shear viscosity, warn of locking (ADR 018 decision 6): a low-order velocity
# stiffens as the interior nears incompressibility. Measured on the disk's exact solution (P1, symmetric
# operator, `tests/test_backend_cell_force_balance.py`, 2026-11-15), relative L² error:
#     η_b/η    h = 0.1   h = 0.05   observed order
#        1     9.0e-4    2.5e-4     1.8
#       10     7.9e-4    1.5e-4     2.4
#      100     5.1e-3    9.8e-4     2.4
#     1000     4.5e-2    7.1e-3     2.7
# Beyond ~10 the error grows roughly in proportion to the ratio (4–6 times the ratio-1 error at 100, 28–50
# times at 1000), and the order rises above 2: the solution is pre-asymptotic, as under locking.
LOCKING_RATIO = 100.0


class ForceBalanceError(ValueError):
    """A T9 model that cannot be solved as given: the message starts with the refusal's name."""


@dataclass
class CellForceBalance:
    """A T9 force balance assembled on a realized geometry: the problem, its mesh and tags, and the
    coefficient checks `solve()` runs first."""

    problem: DiscreteProblem
    members: MembersMesh
    member_tags: dict[str, int]
    surface_tags: dict[str, int]
    face_tags: dict[str, int]
    geometry: MultiCompartmentGeometry
    time: fem.Constant
    _checks: list[Callable[[], None]] = field(default_factory=list)

    @property
    def velocity(self) -> fem.Function:
        return self.problem.unknown

    def solve(self, t: float = 0.0) -> fem.Function:
        """The velocity of the current configuration at time `t` (the coefficients' `sim.t`), after checking
        the coefficients' signs."""

        self.problem.set_time(t)
        for check in self._checks:
            check()
        self.problem.step()
        v = self.problem.unknown
        n_owned = v.function_space.dofmap.index_map.size_local * v.function_space.dofmap.index_map_bs
        finite = bool(np.isfinite(v.x.array[:n_owned]).all())
        if not self.problem.V.mesh.comm.allreduce(finite, op=MPI.LAND):
            raise ForceBalanceError(f"SolveFailed: the velocity at t = {t:.6g} is not finite")
        return v


def solve_cell_force_balance(
    md: MathDescription,
    description: GeometryDescription,
    *,
    h: float,
    t: float = 0.0,
    semi_implicit_dt: float | None = None,
    degree: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> CellForceBalance:
    """Validate `md`, realize `description` at mesh size `h`, assemble its T9 force balance and solve it once."""

    validate_or_raise(md)
    geometry = realize_multi_compartment(description, h=h, comm=comm)
    balance = assemble_cell_force_balance(md, geometry, semi_implicit_dt=semi_implicit_dt, degree=degree)
    balance.solve(t)
    return balance


def assemble_cell_force_balance(
    md: MathDescription,
    geometry: MultiCompartmentGeometry,
    *,
    semi_implicit_dt: float | None = None,
    degree: int = 1,
    prebuilt: tuple[MembersMesh, dict[str, int]] | None = None,
    fields: Mapping[str, UflExpr] | None = None,
    time: fem.Constant | None = None,
) -> CellForceBalance:
    """The T9 equation of `md` on `geometry`, assembled but not solved. `md` must already be valid
    (`validate_or_raise`; `solve_cell_force_balance` does it).

    `prebuilt` (the members' mesh from `build_members_mesh`), `fields` and `time` are for a run with species on
    the members (`cell_force_balance_run.py`, ADR 020): `fields` binds each member species to the Function
    holding its current values, which the volume slots `viscosity`, `bulk_viscosity`, `drag` and
    `active_stress` may read (decision 4); `time` is the `sim.t` Constant the species share."""

    eq = t9_equation(md)
    members = eq.members or (eq.subdomain,)
    fields = dict(fields or {})
    _refuse_fields(md, eq, set(fields))
    missing = [m for m in members if m not in geometry.compartments]
    if missing:
        raise ForceBalanceError(f"member(s) {missing} have no cells in the realized geometry {geometry.name!r}")
    bcs = [bc for bc in md.boundary_conditions if bc.variable == eq.variable]
    unheld = _check_geometry(md, eq, members, bcs, geometry)

    members_mesh, face_tags = prebuilt if prebuilt is not None else _members_mesh(geometry, members)
    mesh = members_mesh.mesh
    if time is None:
        time = fem.Constant(mesh, PETSc.ScalarType(0.0))  # type: ignore[operator]
    # The velocity is a zero placeholder here: no slot may read it (ADR 018 decision 3), but a parameter that
    # reads it (one a species' source uses) must still compile.
    placeholder = ufl.as_vector([0.0] * mesh.geometry.dim)
    symbols = compile_parameters(md, mesh, geometry.sizes, time, {eq.variable: placeholder, **fields})
    ctx = CompileContext(mesh=mesh, symbols=symbols)

    def compiled(text: str | None, context: CompileContext = ctx) -> UflExpr | None:
        return None if text is None else compile_expression(parse(text), context)

    def on_surface(name: str) -> CompileContext:
        """`geom.normal` on a free membrane: the formalism's normal points out of the surface's `inside`
        (formalism §1.8.4), while the members' mesh's facet normal points out of the member."""

        sign = 1.0 if geometry.membranes[name].inside in members else -1.0
        return CompileContext(mesh=mesh, symbols={**symbols, "__normal__": sign * ufl.FacetNormal(mesh)})

    member_tags = {m: geometry.compartments[m].region_tag for m in members}
    member_laws = []
    for m in members:
        slots = eq.volumes.get(m, {})
        viscosity = compiled(slots["viscosity"])
        assert viscosity is not None  # the validator requires it
        member_laws.append(
            MemberLaw(
                member_tags[m],
                viscosity=viscosity,
                bulk_viscosity=compiled(slots.get("bulk_viscosity")),
                drag=compiled(slots.get("drag")),
                pressure=compiled(slots.get("pressure")),
                active_stress=compiled(slots.get("active_stress")),
            )
        )
    surface_tags = {name: geometry.membranes[name].facet_tag for name in eq.surfaces}
    surface_laws = [
        SurfaceLaw(
            surface_tags[name],
            tension=compiled(slots.get("tension")),
            traction=compiled(slots.get("traction"), on_surface(name)),
        )
        for name, slots in eq.surfaces.items()
    ]

    space = fem.functionspace(mesh, ("Lagrange", degree, (mesh.geometry.dim,)))
    conditions: list[VelocityCondition] = []
    for bc in bcs:
        assert isinstance(bc, BCDirichlet)  # the validator allows only dirichlet on a T9 velocity
        tree = parse(bc.expression)
        value = None if _is_zero(tree) else compile_expression(tree, ctx)
        tag = geometry.membranes[bc.boundary].facet_tag if bc.boundary in geometry.membranes else face_tags[bc.boundary]
        conditions.append(velocity_dirichlet(space, members_mesh.facet_tags, tag, value))

    problem = assemble_force_balance(
        members_mesh,
        member_laws,
        surface_laws,
        viscous_operator=eq.viscous_operator or "symmetric",
        semi_implicit_dt=semi_implicit_dt,
        conditions=conditions,
        space=space,
        variable_name=eq.variable,
    )
    problem.time = time
    balance = CellForceBalance(
        problem=problem,
        members=members_mesh,
        member_tags=member_tags,
        surface_tags=surface_tags,
        face_tags=face_tags,
        geometry=geometry,
        time=time,
    )
    balance._checks = _coefficient_checks(members_mesh, member_laws, surface_laws, member_tags, surface_tags, time)
    balance._checks += [_drag_check(members_mesh, member_laws, member_tags, group, time) for group in unheld]
    return balance


def compile_parameters(
    md: MathDescription,
    mesh: Mesh,
    sizes: Mapping[str, float],
    time: fem.Constant,
    extra: Mapping[str, UflExpr] | None = None,
) -> dict[str, UflExpr]:
    """The model's parameters on `mesh`, in declaration order, with `geom.x`, `sim.t`, `region_size(...)` and
    the `extra` symbols (member species, the velocity) in scope, so a parameter may read a member species, as
    VCell's lowered math does through function parameters. Returns them with those symbols."""

    symbols: dict[str, UflExpr] = {
        region_size_symbol(name): fem.Constant(mesh, PETSc.ScalarType(value))  # type: ignore[operator]
        for name, value in sizes.items()
    }
    symbols["geom.x"] = ufl.SpatialCoordinate(mesh)
    symbols["sim.t"] = time
    symbols.update(extra or {})
    ctx = CompileContext(mesh=mesh, symbols=symbols)
    for p in md.parameters:
        if isinstance(p, ParameterConstant):
            symbols[p.name] = fem.Constant(mesh, PETSc.ScalarType(p.value))  # type: ignore[operator]
        elif isinstance(p, ParameterExpression):
            symbols[p.name] = compile_expression(parse(p.expression), ctx)
        else:
            raise ForceBalanceError(f"a force balance cannot bind parameter {p.name!r} of type {type(p).__name__}")
    return symbols


def t9_equation(md: MathDescription) -> TemplateEquation:
    """The model's one `cell_force_balance` equation."""

    equations = [eq for eq in md.equations if isinstance(eq, TemplateEquation) and eq.template == CELL_FORCE_BALANCE]
    if len(equations) != 1:
        raise ForceBalanceError(
            f"a force-balance solve takes one {CELL_FORCE_BALANCE!r} equation, got {len(equations)}"
        )
    return equations[0]


def build_members_mesh(md: MathDescription, geometry: MultiCompartmentGeometry) -> tuple[MembersMesh, dict[str, int]]:
    """The members' mesh of `md`'s force balance on `geometry`, and the tag of each box face a member touches."""

    eq = t9_equation(md)
    return _members_mesh(geometry, eq.members or (eq.subdomain,))


def term_powers(problem: DiscreteProblem) -> dict[str, float]:
    """Each term kind's power on the solved velocity: a bilinear term at `(v, v)`, a load at `v`. Testing the
    residual with `φ = v` makes them sum to zero (to solver precision, while every `dirichlet` value is zero):
    the energy identity of ADR 018 decision 5 and ADR 019 decision 2,
    `δE[v] = −2R(v) − Δt∫γ|∇_Γv|² − ∫σ_a∇·v + ∫t·v`, with `δE[v]` the pressure's and the tension's powers."""

    v = problem.unknown
    powers: dict[str, float] = {}
    for term in problem.terms:
        assert term.integrand is not None and term.measure is not None
        form = term.integrand * term.measure
        functional = ufl.action(form, v)  # the trial (or, for a load, the test) becomes v
        if len(form.arguments()) == 2:
            functional = ufl.action(functional, v)
        local = float(np.real(fem.assemble_scalar(fem.form(functional))))
        powers[term.kind.value] = powers.get(term.kind.value, 0.0) + problem.V.mesh.comm.allreduce(local, op=MPI.SUM)
    return powers


# -- refusals ---------------------------------------------------------------


def _names(node: Expr) -> set[str]:
    if isinstance(node, Name):
        return {node.name}
    if isinstance(node, Number):
        return set()
    if isinstance(node, FunctionCall):
        return set().union(*(_names(a) for a in node.args)) if node.args else set()
    if isinstance(node, UnaryOp):
        return _names(node.operand)
    if isinstance(node, BinaryOp):
        return _names(node.left) | _names(node.right)
    if isinstance(node, IndexAccess):
        return _names(node.base) | _names(node.index)
    if isinstance(node, VectorLiteral):
        return set().union(*(_names(c) for c in node.components)) if node.components else set()
    if isinstance(node, TensorLiteral):
        return set().union(*(_names(r) for r in node.rows)) if node.rows else set()
    return set()


_FIELD_SLOTS = frozenset({"viscosity", "bulk_viscosity", "drag", "active_stress"})


def _refuse_fields(md: MathDescription, eq: TemplateEquation, allowed: set[str]) -> None:
    """A slot, a parameter it reads, or a membrane velocity that reads a species or a region variable is refused
    by name (`FieldsNotYetSupported`), except a member species in `allowed` read by a volume slot other than
    `pressure` (ADR 020 decision 4; the rest is stage 5). So is `geom.normal`
    anywhere but in a `traction` (`NormalNotYetSupported`): a volume slot has no normal, a membrane velocity is
    interpolated at nodes, and a parameter is compiled once for every surface, without the side's sign."""

    variables = {v.name for v in md.variables} - {eq.variable}
    bodies = {p.name: p.expression for p in md.parameters if hasattr(p, "expression")}

    def reads(text: str, seen: set[str]) -> set[str]:
        found = set()
        for name in _names(parse(text)):
            if name in variables:
                found.add(name)
            elif name in bodies and name not in seen:
                seen.add(name)
                found |= reads(bodies[name], seen)
        return found

    def reads_normal(parameter: str, seen: set[str]) -> bool:
        seen.add(parameter)
        names = _names(parse(bodies[parameter]))
        return "geom.normal" in names or any(n in bodies and n not in seen and reads_normal(n, seen) for n in names)

    texts = [(f"volumes.{m}.{s}", t) for m, slots in eq.volumes.items() for s, t in slots.items()]
    texts += [(f"surfaces.{n}.{s}", t) for n, slots in eq.surfaces.items() for s, t in slots.items()]
    texts += [
        (f"the {bc.kind} BC on {bc.boundary!r}", bc.expression)
        for bc in md.boundary_conditions
        if bc.variable == eq.variable and isinstance(bc, BCDirichlet)
    ]
    for where, text in texts:
        found = reads(text, set())
        if where.startswith("volumes.") and where.rsplit(".", 1)[1] in _FIELD_SLOTS:
            found -= allowed
        if found:
            raise ForceBalanceError(
                f"FieldsNotYetSupported: {where} reads {sorted(found)}; a force balance that reads species or "
                f"region variables waits for chemistry on the cells (ADR 018 staging step 5)"
            )
        names = _names(parse(text))
        if "geom.normal" in names and not where.endswith(".traction"):
            raise ForceBalanceError(
                f"NormalNotYetSupported: {where} reads geom.normal, which a force balance reads only in a 'traction'"
            )
        through = sorted(n for n in names if n in bodies and reads_normal(n, set()))
        if through:
            raise ForceBalanceError(
                f"NormalNotYetSupported: {where} reads geom.normal through the parameter(s) {through}; write the "
                f"normal in the 'traction' itself, where its sign follows the surface"
            )


def _is_zero(tree: Expr) -> bool:
    if isinstance(tree, Number):
        return tree.value == 0.0
    if isinstance(tree, VectorLiteral):
        return all(_is_zero(c) for c in tree.components)
    return False


def _check_geometry(
    md: MathDescription,
    eq: TemplateEquation,
    members: tuple[str, ...],
    bcs: Sequence[object],
    geometry: MultiCompartmentGeometry,
) -> list[list[str]]:
    """ADR 018 decisions 2 and 4 and ADR 019 decisions 4 and 5, on the realized geometry (the formalism's T9
    rules that need it). Returns the connected groups of members that no `dirichlet` holds, whose drag each
    solve checks."""

    member_set = set(members)
    membranes = geometry.membranes
    motion_of = {s.name: s.motion for s in md.subdomains}

    def sides(name: str) -> set[str]:
        return {membranes[name].inside, membranes[name].outside}

    for name in eq.surfaces:
        if name not in membranes:
            raise ForceBalanceError(f"surface {name!r} has no facets in the realized geometry {geometry.name!r}")
        if not sides(name) & member_set:
            raise ForceBalanceError(f"surface {name!r} bounds no member of {list(members)}; it cannot load the cells")
        if "traction" in eq.surfaces[name] and sides(name) <= member_set:
            raise ForceBalanceError(f"surface {name!r} lies between two members; a traction loads a free membrane")
    # motion, both ways: a membrane bounding a member moves with v or is a growing front; one moving with v
    # bounds a member; a growing front is a free membrane that meets no other membrane or the box
    for name in membranes:
        motion = motion_of.get(name)
        moves_with_v = isinstance(motion, MotionUnknown) and motion.variable == eq.variable
        front = isinstance(motion, MotionPrescribedVelocity)
        if sides(name) & member_set:
            if not (moves_with_v or front):
                raise ForceBalanceError(
                    f"surface {name!r} bounds a member, so it must have motion {{kind: unknown, variable: "
                    f"{eq.variable}}} or be a growing front with a prescribed velocity (ADR 018 decision 2, ADR 019 "
                    f"decision 5)"
                )
            if front and sides(name) <= member_set:
                raise ForceBalanceError(
                    f"the growing front {name!r} lies between two members; a front is a free membrane"
                )
            if front and _meets_another_boundary(geometry, name):
                raise ForceBalanceError(
                    f"the growing front {name!r} meets another membrane or the box; a front ending at a junction "
                    f"is not supported yet (ADR 019 decision 5, first scope)"
                )
        elif moves_with_v:
            raise ForceBalanceError(
                f"surface {name!r} moves with {eq.variable!r} but bounds no member of {list(members)}"
            )
    held: set[str] = set()  # members with a dirichlet facet
    walled_faces: set[str] = set()
    for bc in bcs:
        assert isinstance(bc, BCDirichlet)
        if bc.boundary in membranes:
            touching = sides(bc.boundary) & member_set
            if len(touching) != 1:
                raise ForceBalanceError(
                    f"a velocity is given on {bc.boundary!r}, which is not a free membrane of the cells (a member "
                    f"against a non-member); it bounds {sorted(sides(bc.boundary))}"
                )
            held |= touching
        elif bc.boundary in motion_of:
            raise ForceBalanceError(
                f"a velocity is given on {bc.boundary!r}, which has no facets in the realized geometry "
                f"{geometry.name!r}"
            )
        else:
            touching = {m for m in members if bc.boundary in geometry.compartments[m].faces}
            if not touching:
                raise ForceBalanceError(f"the wall {bc.boundary!r} is not a box face any member touches")
            walled_faces.add(bc.boundary)
            held |= touching
    for m in members:
        free = sorted(set(geometry.compartments[m].faces) - walled_faces)
        if free:
            raise ForceBalanceError(
                f"MemberOnFreeBox: member {m!r} touches the box at {free} with no wall holding it there; the box "
                f"does not move, so give each such face `dirichlet` {eq.variable} = 0 (ADR 018 decision 4)"
            )
    # connected groups of members, joined by the membranes between two members
    group = {m: m for m in members}

    def root(m: str) -> str:
        while group[m] != m:
            m = group[m]
        return m

    for name, membrane in membranes.items():
        if sides(name) <= member_set and membrane.inside != membrane.outside:
            group[root(membrane.inside)] = root(membrane.outside)
    dragged = {m for m in members if "drag" in eq.volumes.get(m, {})}
    unheld: list[list[str]] = []
    for leader in sorted({root(m) for m in members}):
        component = [m for m in members if root(m) == leader]
        if set(component) & held:
            continue
        if not set(component) & dragged:
            raise ForceBalanceError(
                f"UnconstrainedRigidMotion: nothing resists a rigid motion of {component} — none has a 'drag' and "
                f"no 'dirichlet' condition holds it (ADR 018 decision 4)"
            )
        unheld.append(component)
    return unheld


def _meets_another_boundary(geometry: MultiCompartmentGeometry, name: str) -> bool:
    """Whether membrane `name` shares a vertex with another membrane or a box face (a junction)."""

    parent = geometry.parent_mesh
    fdim = parent.topology.dim - 1
    parent.topology.create_connectivity(fdim, 0)
    f2v = parent.topology.connectivity(fdim, 0)

    def vertices(tags: list[int]) -> set[int]:
        out: set[int] = set()
        for tag in tags:
            for f in geometry.facet_tags.find(tag):
                out.update(int(x) for x in f2v.links(f))
        return out

    own = vertices([geometry.membranes[name].facet_tag])
    others = [m.facet_tag for n, m in geometry.membranes.items() if n != name]
    others += [tag for part in geometry.compartments.values() for tag in part.faces.values()]
    meets = bool(own & vertices(others))
    return bool(parent.comm.allreduce(meets, op=MPI.LOR))


# -- the members' mesh -------------------------------------------------------


def _members_mesh(geometry: MultiCompartmentGeometry, members: tuple[str, ...]) -> tuple[MembersMesh, dict[str, int]]:
    parent = geometry.parent_mesh
    tdim = parent.topology.dim
    region_tags = [geometry.compartments[m].region_tag for m in members]
    cells = np.unique(np.concatenate([geometry.cell_tags.find(t) for t in region_tags])).astype(np.int32)
    sub, cell_map, *_ = dmesh.create_submesh(parent, tdim, cells)

    sub_index = sub.topology.index_map(tdim)
    sub_cells = np.arange(sub_index.size_local + sub_index.num_ghosts, dtype=np.int32)
    parent_cells = np.asarray(cell_map.sub_topology_to_topology(sub_cells, False), dtype=np.intp)
    parent_index = parent.topology.index_map(tdim)
    region_of_cell = np.zeros(parent_index.size_local + parent_index.num_ghosts, dtype=np.int32)
    region_of_cell[geometry.cell_tags.indices] = geometry.cell_tags.values
    cell_tags = dmesh.meshtags(sub, tdim, sub_cells, region_of_cell[parent_cells])

    face_names = _FACE_NAMES_3D if tdim == 3 else _FACE_NAMES_2D
    tagged: list[tuple[int, NDArray[np.int32]]] = []
    for membrane in geometry.membranes.values():
        tagged.append((membrane.facet_tag, geometry.facet_tags.find(membrane.facet_tag)))
    face_tags: dict[str, int] = {}
    for i, face in enumerate(face_names):
        facets = [
            geometry.facet_tags.find(geometry.compartments[m].faces[face])
            for m in members
            if face in geometry.compartments[m].faces
        ]
        if facets:
            face_tags[face] = _FACE_TAG + i
            tagged.append((_FACE_TAG + i, np.concatenate(facets).astype(np.int32)))
    facet_tags = _transfer_facet_tags(parent, sub, tagged)
    return MembersMesh(sub, cell_tags, facet_tags), face_tags


def _transfer_facet_tags(parent: Mesh, sub: Mesh, tagged: list[tuple[int, NDArray[np.int32]]]) -> MeshTags:
    """Re-locate tagged parent facets on the submesh by their midpoints (`create_submesh` copies the geometry, so
    a facet's twin has the same midpoint), keeping only facets the submesh has."""

    fdim = parent.topology.dim - 1
    sub.topology.create_entities(fdim)
    sub.topology.create_connectivity(fdim, fdim + 1)
    parent.topology.create_connectivity(fdim, fdim + 1)
    sub_index = sub.topology.index_map(fdim)
    sub_facets = np.arange(sub_index.size_local + sub_index.num_ghosts, dtype=np.int32)
    tree = cKDTree(dmesh.compute_midpoints(sub, fdim, sub_facets))
    scale = float(np.ptp(sub.geometry.x, axis=0).max()) or 1.0
    indices, values = [], []
    for tag, facets in tagged:
        if facets.size == 0:
            continue
        distance, nearest = tree.query(dmesh.compute_midpoints(parent, fdim, facets))
        found = sub_facets[nearest[distance < 1e-10 * scale]]
        indices.append(found)
        values.append(np.full(found.size, tag, dtype=np.int32))
    all_indices: NDArray[np.int32] = np.concatenate(indices) if indices else np.empty(0, dtype=np.int32)
    all_values: NDArray[np.int32] = np.concatenate(values) if values else np.empty(0, dtype=np.int32)
    unique, first = np.unique(all_indices, return_index=True)
    return dmesh.meshtags(sub, fdim, unique.astype(np.int32), all_values[first])


# -- the coefficient checks (ADR 018 decision 3) ------------------------------


def _coefficient_checks(
    members: MembersMesh,
    member_laws: list[MemberLaw],
    surface_laws: list[SurfaceLaw],
    member_tags: dict[str, int],
    surface_tags: dict[str, int],
    time: fem.Constant,
) -> list[Callable[[], None]]:
    mesh = members.mesh
    name_of_member = {tag: name for name, tag in member_tags.items()}
    name_of_surface = {tag: name for name, tag in surface_tags.items()}
    checks: list[Callable[[], None]] = []

    def volume_check(law: MemberLaw, slot: str, expr: UflExpr, strict: bool, refusal: str) -> Callable[[], None]:
        minimum = _CellMinimum(mesh, expr, members.cell_tags.find(law.tag))

        def check() -> None:
            low = minimum()
            if low < 0.0 or (strict and low <= 0.0):
                bound = "> 0" if strict else ">= 0"
                raise ForceBalanceError(
                    f"{refusal}: the {slot} of member {name_of_member[law.tag]!r} reaches {low:.6g} at "
                    f"t = {float(time.value):.6g}; it must stay {bound} (ADR 018 decision 3)"
                )

        return check

    for law in member_laws:
        checks.append(volume_check(law, "viscosity", law.viscosity, True, "NonPositiveViscosity"))
        if law.bulk_viscosity is not None:
            checks.append(volume_check(law, "bulk_viscosity", law.bulk_viscosity, False, "NegativeBulkViscosity"))
        if law.drag is not None:
            checks.append(volume_check(law, "drag", law.drag, False, "NegativeDrag"))
        if law.bulk_viscosity is not None:
            checks.append(_locking_check(mesh, members.cell_tags.find(law.tag), law, name_of_member[law.tag]))

    def tension_check(tag: int, gamma: UflExpr) -> Callable[[], None]:
        minimum = _FacetMinimum(mesh, gamma, members, tag)

        def check() -> None:
            low = minimum()
            if low <= 0.0:
                raise ForceBalanceError(
                    f"NonPositiveTension: the tension of {name_of_surface[tag]!r} reaches {low:.6g} at "
                    f"t = {float(time.value):.6g}; a given tension must stay > 0 (ADR 018 decision 3)"
                )

        return check

    checks += [tension_check(s.tag, s.tension) for s in surface_laws if s.tension is not None]
    return checks


def _drag_check(
    members: MembersMesh,
    member_laws: list[MemberLaw],
    member_tags: dict[str, int],
    group: list[str],
    time: fem.Constant,
) -> Callable[[], None]:
    """ADR 018 decision 4, at every step: a group of members that no `dirichlet` holds needs a drag that is
    positive somewhere in it, or its rigid motions are unresisted."""

    mesh = members.mesh
    laws = {law.tag: law for law in member_laws}
    negated = [
        _CellMinimum(mesh, -drag, members.cell_tags.find(member_tags[m]))
        for m in group
        if (drag := laws[member_tags[m]].drag) is not None
    ]

    def check() -> None:
        highest = max((-minimum() for minimum in negated), default=-np.inf)
        if not highest > 0.0:
            raise ForceBalanceError(
                f"UnconstrainedRigidMotion: the drag of {group} is nowhere positive at t = {float(time.value):.6g} "
                f"and no 'dirichlet' condition holds it, so its rigid motions are unresisted (ADR 018 decision 4)"
            )

    return check


def _locking_check(mesh: Mesh, cells: NDArray[np.int32], law: MemberLaw, name: str) -> Callable[[], None]:
    assert law.bulk_viscosity is not None
    negated = _CellMinimum(mesh, -law.bulk_viscosity / law.viscosity, cells)

    def check() -> None:
        high = -negated()
        if high > LOCKING_RATIO:
            warnings.warn(
                f"member {name!r}: bulk_viscosity / viscosity reaches {high:.3g}, above {LOCKING_RATIO:g}; a "
                f"low-order velocity locks as the interior nears incompressibility, so the solution may be too "
                f"stiff (ADR 018 decision 6; a mixed form is later work)",
                stacklevel=3,
            )

    return check


class _CellMinimum:
    """The minimum of a scalar `expr` over the owned `cells`, at degree-2 quadrature points and the vertices. The
    expression is compiled once: building one every step costs a JIT-cache lookup each (~50 ms). Bound to the
    mesh and its cells at construction, so a moving or remeshed mesh (stage 4) must rebuild it."""

    def __init__(self, mesh: Mesh, expr: UflExpr, cells: NDArray[np.int32]) -> None:
        cell_type = mesh.basix_cell()
        points, _ = basix.make_quadrature(cell_type, 2)
        self._mesh = mesh
        self._cells = cells[cells < mesh.topology.index_map(mesh.topology.dim).size_local]
        self._expression = fem.Expression(expr, np.vstack([points, basix.geometry(cell_type)]))

    def __call__(self) -> float:
        local = np.inf
        if self._cells.size:
            local = float(np.min(np.real(self._expression.eval(self._mesh, self._cells))))
        return float(self._mesh.comm.allreduce(local, op=MPI.MIN))


class _FacetMinimum:
    """The minimum of a scalar `expr` (one that reads no normal) at the midpoints of the owned facets tagged
    `tag`, evaluated in a cell beside each; compiled once per local facet index."""

    def __init__(self, mesh: Mesh, expr: UflExpr, members: MembersMesh, tag: int) -> None:
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(fdim, tdim)
        mesh.topology.create_connectivity(tdim, fdim)
        f2c = mesh.topology.connectivity(fdim, tdim)
        c2f = mesh.topology.connectivity(tdim, fdim)
        facets = members.facet_tags.find(tag)
        facets = facets[facets < mesh.topology.index_map(fdim).size_local]
        cell_type = mesh.basix_cell()
        reference = basix.geometry(cell_type)
        facet_vertices = basix.topology(cell_type)[fdim]
        by_local: dict[int, list[int]] = {}
        for f in facets:
            c = int(f2c.links(f)[0])
            local = int(np.flatnonzero(c2f.links(c) == f)[0])
            by_local.setdefault(local, []).append(c)
        self._mesh = mesh
        self._groups = [
            (
                fem.Expression(expr, reference[facet_vertices[local]].mean(axis=0, keepdims=True)),
                np.asarray(cells, dtype=np.int32),
            )
            for local, cells in by_local.items()
        ]

    def __call__(self) -> float:
        low = np.inf
        for expression, cells in self._groups:
            low = min(low, float(np.min(np.real(expression.eval(self._mesh, cells)))))
        return float(self._mesh.comm.allreduce(low, op=MPI.MIN))


__all__ = [
    "LOCKING_RATIO",
    "CellForceBalance",
    "ForceBalanceError",
    "assemble_cell_force_balance",
    "solve_cell_force_balance",
    "term_powers",
]

"""The T9 cell force balance in the IR (ADR 018 staging step 2, with ADR 019's terms): the tagged terms and the
quasi-static scheme, their operator invariants, the 3D deviator, contract C against vertex forces, and small
fixed-mesh solves.

Layers of ADR 004's verification: structural (which tagged terms a problem has), operator invariants (what
each assembled term does to a rigid motion, independent of any solve), and conformance with an independent
vertex evaluator (ADR 017 decision 6 (b)), plus vcell-core's vectors when its sibling checkout is present
(decision 6 (a)). The exact and manufactured solutions are stage 3's.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

import basix.ufl
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

from vcell_fenics.backend import BackwardEuler, DiscreteProblem, QuasiStatic, Term, TermKind
from vcell_fenics.backend.force_balance import (
    MemberLaw,
    MembersMesh,
    SurfaceLaw,
    ViscousOperator,
    assemble_force_balance,
    deviatoric_inner,
    force_balance_terms,
    surface_is_interior,
    velocity_dirichlet,
)

LEFT, RIGHT = 1, 2  # members
FREE_LEFT, FREE_RIGHT, CONTACT = 11, 12, 20  # surfaces

# ---------------------------------------------------------------------------
# Fixtures: two members side by side, in 2D and 3D
# ---------------------------------------------------------------------------


def _tags(mesh: dmesh.Mesh, dim: int, groups: list[tuple[int, np.ndarray]]) -> dmesh.MeshTags:
    indices = np.concatenate([idx for _, idx in groups]).astype(np.int32)
    values = np.concatenate([np.full(len(idx), tag, dtype=np.int32) for tag, idx in groups])
    order = np.argsort(indices)
    return dmesh.meshtags(mesh, dim, indices[order], values[order])


def _two_members(dim: int) -> MembersMesh:
    """`[0, 2] × [0, 1]` (× `[0, 1]`), split at `x = 1` into two members; each member's outer boundary is its
    free membrane and the plane `x = 1` is their contact."""

    if dim == 2:
        mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([2.0, 1.0])], [8, 4])
    else:
        mesh = dmesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0])], [4, 2, 2])
    tdim = mesh.topology.dim
    eps = 1e-10
    left = dmesh.locate_entities(mesh, tdim, lambda x: x[0] <= 1.0 + eps)
    right = dmesh.locate_entities(mesh, tdim, lambda x: x[0] >= 1.0 - eps)
    cell_tags = _tags(mesh, tdim, [(LEFT, left), (RIGHT, right)])
    mesh.topology.create_connectivity(tdim - 1, tdim)
    free_left = dmesh.locate_entities_boundary(mesh, tdim - 1, lambda x: x[0] <= 1.0 + eps)
    free_right = dmesh.locate_entities_boundary(mesh, tdim - 1, lambda x: x[0] >= 1.0 - eps)
    contact = dmesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 1.0))
    exterior = set(dmesh.exterior_facet_indices(mesh.topology).tolist())
    contact = np.array([f for f in contact if f not in exterior], dtype=np.int32)
    free_left = np.setdiff1d(free_left, contact)
    free_right = np.setdiff1d(free_right, np.union1d(contact, free_left))
    facet_tags = _tags(mesh, tdim - 1, [(FREE_LEFT, free_left), (FREE_RIGHT, free_right), (CONTACT, contact)])
    return MembersMesh(mesh, cell_tags, facet_tags)


def _space(members: MembersMesh) -> fem.FunctionSpace:
    mesh = members.mesh
    return fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))


def _field(V: fem.FunctionSpace, f: Callable[[np.ndarray], np.ndarray]) -> fem.Function:
    u = fem.Function(V)
    u.interpolate(f)
    return u


def _rigid_motions(V: fem.FunctionSpace) -> tuple[list[fem.Function], list[fem.Function]]:
    """(translations, rotations about the origin) as P1 fields, exact since they are linear."""

    gdim = V.mesh.geometry.dim

    def unit(k: int) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: np.vstack([np.full(x.shape[1], float(i == k)) for i in range(gdim)])

    translations = [_field(V, unit(k)) for k in range(gdim)]
    if gdim == 2:
        rotations = [_field(V, lambda x: np.vstack([-x[1], x[0]]))]
    else:
        rotations = [
            _field(V, lambda x: np.vstack([np.zeros(x.shape[1]), -x[2], x[1]])),
            _field(V, lambda x: np.vstack([x[2], np.zeros(x.shape[1]), -x[0]])),
            _field(V, lambda x: np.vstack([-x[1], x[0], np.zeros(x.shape[1])])),
        ]
    return translations, rotations


def _matrix(term: Term) -> np.ndarray:
    assert term.integrand is not None and term.measure is not None
    A = assemble_matrix(fem.form(term.integrand * term.measure))
    A.assemble()
    dense: np.ndarray = A.convert("dense").getDenseArray()
    return dense


def _load(term: Term) -> np.ndarray:
    assert term.integrand is not None and term.measure is not None
    b = assemble_vector(fem.form(term.integrand * term.measure))
    return np.array(b.array)


def _terms_of(terms: tuple[Term, ...], kind: TermKind) -> list[Term]:
    return [t for t in terms if t.kind is kind]


def _constant(mesh: dmesh.Mesh, value: float) -> fem.Constant:
    return fem.Constant(mesh, PETSc.ScalarType(value))  # type: ignore[operator]


def _laws(members: MembersMesh, *, fields: bool = True) -> tuple[list[MemberLaw], list[SurfaceLaw]]:
    """Every slot, with spatially varying coefficients where a slot may vary."""

    mesh = members.mesh
    x = ufl.SpatialCoordinate(mesh)
    eta = 1.0 + 0.5 * x[0] * x[1] if fields else _constant(mesh, 1.0)
    member_laws = [
        MemberLaw(
            LEFT,
            viscosity=eta,
            bulk_viscosity=0.3 + 0.1 * x[1],
            drag=2.0 + x[0],
            pressure=_constant(mesh, 0.7),
            active_stress=1.0 + x[0] * x[0] + 0.5 * x[1],
        ),
        MemberLaw(RIGHT, viscosity=2.0 * eta, drag=_constant(mesh, 1.5), pressure=_constant(mesh, -0.4)),
    ]
    surface_laws = [
        SurfaceLaw(FREE_LEFT, tension=1.0 + 0.2 * x[1]),
        SurfaceLaw(FREE_RIGHT, tension=_constant(mesh, 0.5)),
        SurfaceLaw(CONTACT, tension=0.8 + 0.3 * x[1]),
    ]
    return member_laws, surface_laws


# ---------------------------------------------------------------------------
# Structure (ADR 004 layer 1)
# ---------------------------------------------------------------------------


def test_the_problem_has_one_tagged_term_per_slot_and_the_quasi_static_scheme() -> None:
    members = _two_members(2)
    member_laws, surface_laws = _laws(members)
    surface_laws.append(SurfaceLaw(FREE_RIGHT, traction=ufl.as_vector([0.1, 0.0])))
    problem = assemble_force_balance(members, member_laws, surface_laws, semi_implicit_dt=0.01)
    assert isinstance(problem.scheme, QuasiStatic)
    assert problem.term_kinds() == {
        TermKind.VISCOUS_STRESS,
        TermKind.DRAG,
        TermKind.CELL_PRESSURE,
        TermKind.ACTIVE_STRESS,
        TermKind.SURFACE_TENSION,
        TermKind.TENSION_SEMI_IMPLICIT,
        TermKind.SURFACE_TRACTION,
    }
    kinds = [t.kind for t in problem.terms]
    assert kinds.count(TermKind.VISCOUS_STRESS) == 2  # one per member
    assert kinds.count(TermKind.SURFACE_TENSION) == kinds.count(TermKind.TENSION_SEMI_IMPLICIT) == 3


def test_the_laplacian_operator_is_its_own_kind_and_no_dt_means_no_semi_implicit_term() -> None:
    members = _two_members(2)
    member_laws, surface_laws = _laws(members)
    problem = assemble_force_balance(members, member_laws, surface_laws, viscous_operator="laplacian")
    assert TermKind.VISCOUS_LAPLACIAN in problem.term_kinds()
    assert TermKind.VISCOUS_STRESS not in problem.term_kinds()
    assert TermKind.TENSION_SEMI_IMPLICIT not in problem.term_kinds()


def test_a_contact_is_interior_and_a_free_membrane_exterior() -> None:
    members = _two_members(2)
    assert surface_is_interior(members, CONTACT)
    assert not surface_is_interior(members, FREE_LEFT)
    with pytest.raises(ValueError, match="no facets"):
        surface_is_interior(members, 999)


def test_the_quasi_static_scheme_refuses_a_time_derivative_and_backward_euler_honours_a_measure() -> None:
    members = _two_members(2)
    mesh = members.mesh
    V = _space(members)
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=members.cell_tags)
    terms = (Term(TermKind.TIME_DERIVATIVE), Term(TermKind.DIFFUSION, ufl.inner(ufl.grad(u), ufl.grad(w)), dx(LEFT)))

    def problem(scheme: QuasiStatic | BackwardEuler) -> DiscreteProblem:
        return DiscreteProblem(
            variable_name="v",
            V=V,
            trial=u,
            test=w,
            dx=ufl.dx(domain=mesh),
            unknown=fem.Function(V),
            previous=fem.Function(V),
            dt=_constant(mesh, 0.1),
            terms=terms,
            scheme=scheme,
            bcs=[],
        )

    with pytest.raises(TypeError, match="no time derivative"):
        _ = problem(QuasiStatic()).bilinear_form
    # backward Euler integrates the diffusion over LEFT only: for u = (x, 0), a(u, u) = ∫_Ω x² + Δt |LEFT|
    field = _field(V, lambda x: np.vstack([x[0], np.zeros(x.shape[1])]))
    a = problem(BackwardEuler()).bilinear_form
    value = fem.assemble_scalar(fem.form(ufl.action(ufl.action(a, field), field)))
    assert value == pytest.approx(8.0 / 3.0 + 0.1 * 1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Operator invariants (ADR 004 layer 3; ADR 018 decision 6, ADR 019 decisions 2–4)
# ---------------------------------------------------------------------------


def _terms(dim: int, **kwargs: object) -> tuple[MembersMesh, fem.FunctionSpace, tuple[Term, ...]]:
    members = _two_members(dim)
    V = _space(members)
    member_laws, surface_laws = _laws(members)
    surface_laws.append(SurfaceLaw(FREE_RIGHT, traction=ufl.as_vector([0.1] + [0.0] * (dim - 1))))
    terms = force_balance_terms(
        members,
        ufl.TrialFunction(V),
        ufl.TestFunction(V),
        member_laws,
        surface_laws,
        **kwargs,  # type: ignore[arg-type]
    )
    return members, V, terms


@pytest.mark.parametrize("dim", [2, 3])
def test_the_viscous_matrix_annihilates_every_rigid_motion(dim: int) -> None:
    _, V, terms = _terms(dim)
    translations, rotations = _rigid_motions(V)
    for term in _terms_of(terms, TermKind.VISCOUS_STRESS):
        A = _matrix(term)
        assert np.allclose(A, A.T, atol=1e-13)
        scale = np.abs(A).max()
        for r in translations + rotations:
            assert np.abs(A @ r.x.array).max() < 1e-12 * scale


@pytest.mark.parametrize("dim", [2, 3])
def test_the_loads_exert_no_net_force_or_torque(dim: int) -> None:
    """A pressure, an active stress and a tension, uniform or not, vanish on every rigid motion (ADR 018
    decision 6; ADR 019 decision 2): `∇·φ = 0` and `∇_Γ·φ = P:W = 0` for `φ = c + Wx`."""

    _, V, terms = _terms(dim)
    translations, rotations = _rigid_motions(V)
    for kind in (TermKind.CELL_PRESSURE, TermKind.ACTIVE_STRESS, TermKind.SURFACE_TENSION):
        loads = _terms_of(terms, kind)
        assert loads
        for term in loads:
            b = _load(term)
            for r in translations + rotations:
                assert abs(b @ r.x.array) < 1e-12 * max(np.abs(b).sum(), 1.0), kind


def test_a_traction_exerts_its_net_force() -> None:
    _, V, terms = _terms(2)
    (traction,) = _terms_of(terms, TermKind.SURFACE_TRACTION)
    translations, _ = _rigid_motions(V)
    free_right_length = 1.0 + 1.0 + 1.0  # three sides of the right unit square
    # the residual term is −t·φ, so its load on a translation is −∫t
    assert _load(traction) @ translations[0].x.array == pytest.approx(-0.1 * free_right_length, abs=1e-13)
    assert _load(traction) @ translations[1].x.array == pytest.approx(0.0, abs=1e-13)


def test_the_drag_matrix_is_symmetric_positive_definite_where_drag_acts() -> None:
    _, V, terms = _terms(2)
    for term in _terms_of(terms, TermKind.DRAG):
        A = _matrix(term)
        assert np.allclose(A, A.T, atol=1e-14)
        assert np.linalg.eigvalsh(A).min() > -1e-14
    translations, _ = _rigid_motions(V)
    left = _terms_of(terms, TermKind.DRAG)[0]
    x = ufl.SpatialCoordinate(V.mesh)
    expected = fem.assemble_scalar(
        fem.form((2.0 + x[0]) * ufl.Measure("dx", subdomain_data=_two_members(2).cell_tags, domain=V.mesh)(LEFT))
    )
    t = translations[0].x.array
    assert t @ _matrix(left) @ t == pytest.approx(expected, rel=1e-12)


def test_the_laplacian_annihilates_translations_and_damps_rotations() -> None:
    """ADR 019 decision 3: `η∇v:∇φ` is not frame-indifferent; a rotation `ω = 1` costs `2η|Ω|`."""

    members = _two_members(2)
    V = _space(members)
    law = MemberLaw(LEFT, viscosity=_constant(members.mesh, 1.0))
    (term,) = force_balance_terms(
        members, ufl.TrialFunction(V), ufl.TestFunction(V), [law], [], viscous_operator="laplacian"
    )
    assert term.kind is TermKind.VISCOUS_LAPLACIAN
    A = _matrix(term)
    translations, (rotation,) = _rigid_motions(V)
    for t in translations:
        assert np.abs(A @ t.x.array).max() < 1e-12
    r = rotation.x.array
    assert r @ A @ r == pytest.approx(2.0 * 1.0, rel=1e-12)  # |Ω_LEFT| = 1


def test_the_viscous_stress_uses_the_3d_deviator_in_2d() -> None:
    """ADR 018 decision 4: on a uniform dilation `v = x` (`ε = I`), `2η dev₃ε:ε = 2η(2 − 4/3)`, so the form
    costs `(4/3)η|Ω|`; the 2D deviator (`ufl.dev`) would give zero and lose coercivity with `η_b = 0`."""

    members = _two_members(2)
    V = _space(members)
    law = MemberLaw(LEFT, viscosity=_constant(members.mesh, 1.0))
    (term,) = force_balance_terms(members, ufl.TrialFunction(V), ufl.TestFunction(V), [law], [])
    dilation = _field(V, lambda x: np.vstack([x[0], x[1]])).x.array
    assert dilation @ _matrix(term) @ dilation == pytest.approx(4.0 / 3.0, rel=1e-12)
    # the negative control: UFL's 2D deviator has the dilation in its kernel
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    two_d = Term(term.kind, 2 * ufl.inner(ufl.dev(ufl.sym(ufl.grad(u))), ufl.sym(ufl.grad(w))), term.measure)
    assert abs(dilation @ _matrix(two_d) @ dilation) < 1e-12


def test_the_deviatoric_inner_product_matches_the_embedded_3d_deviator() -> None:
    members = _two_members(2)
    V = _space(members)
    a = ufl.sym(ufl.grad(_field(V, lambda x: np.vstack([x[0] * x[1], x[0] - x[1] ** 2]))))
    b = ufl.sym(ufl.grad(_field(V, lambda x: np.vstack([np.sin(x[1]), x[0] ** 2]))))

    def embed(e: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        rows = [[e[0, 0], e[0, 1], 0.0], [e[1, 0], e[1, 1], 0.0], [0.0, 0.0, 0.0]]
        return ufl.as_matrix(rows)  # type: ignore[no-any-return]

    ours = fem.assemble_scalar(fem.form(deviatoric_inner(a, b) * ufl.dx))
    embedded = fem.assemble_scalar(fem.form(ufl.inner(ufl.dev(embed(a)), embed(b)) * ufl.dx))
    assert ours == pytest.approx(embedded, rel=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_semi_implicit_tension_annihilates_translations_and_damps_rotations(dim: int) -> None:
    """ADR 018 decision 5: for `v = Wx`, `∇_Γv = WP`, which dissipates `Δt∫γ|WP|²`. With `|ω| = 1`,
    `|WP|² = 1 + (ω̂·n)²`: `1` in 2D (where `ω̂ ⟂ n`), and between 1 and 2 in 3D."""

    dt = 0.01
    members, V, terms = _terms(dim, semi_implicit_dt=_constant(_two_members(dim).mesh, dt))
    translations, rotations = _rigid_motions(V)
    n = ufl.FacetNormal(V.mesh)
    ds = ufl.Measure("ds", domain=V.mesh, subdomain_data=members.facet_tags)
    dS = ufl.Measure("dS", domain=V.mesh, subdomain_data=members.facet_tags)
    x = ufl.SpatialCoordinate(V.mesh)
    tensions = {FREE_LEFT: 1.0 + 0.2 * x[1], FREE_RIGHT: 0.5 + 0.0 * x[0], CONTACT: 0.8 + 0.3 * x[1]}
    axes = [ufl.as_vector([0.0, 0.0, 1.0])] if dim == 2 else [ufl.as_vector(list(e)) for e in np.eye(3)]
    stiff = _terms_of(terms, TermKind.TENSION_SEMI_IMPLICIT)
    assert len(stiff) == 3
    A = sum(_matrix(t) for t in stiff)
    for t in translations:
        assert np.abs(A @ t.x.array).max() < 1e-12
    for rotation, axis in zip(rotations, axes, strict=True):
        pieces = []
        for tag, g in tensions.items():
            weight = 1.0 if dim == 2 else 1 + ufl.dot(axis, n) ** 2
            integrand = dt * g * weight
            pieces.append(integrand("+") * dS(tag) if tag == CONTACT else integrand * ds(tag))
        expected = fem.assemble_scalar(fem.form(sum(pieces[1:], pieces[0])))
        r = rotation.x.array
        assert r @ A @ r == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Contract C: the FEM loads against vertex forces (ADR 017 decisions 1 and 6)
# ---------------------------------------------------------------------------


def _shoelace(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _area_gradient(p: np.ndarray) -> np.ndarray:
    """∂A/∂x_i of a counter-clockwise polygon: ½ J(x_{i+1} − x_{i−1}), J the rotation by −90°."""

    d = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    return 0.5 * np.column_stack([d[:, 1], -d[:, 0]])


def _length_gradient(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = (b - a) / np.linalg.norm(b - a)
    return -t, t


class _Complex:
    """A 2D cell complex (counter-clockwise faces over shared points), its independent vertex energies and
    forces, and a P1 mesh of it: each face fanned about its centroid, edges optionally split at midpoints."""

    def __init__(self, points: list[list[float]], faces: list[list[int]], *, split: bool = False) -> None:
        self.points = np.array(points, dtype=float)
        self.faces = faces
        self.edges: dict[tuple[int, int], list[int]] = {}
        for f, face in enumerate(faces):
            for a, b in zip(face, face[1:] + face[:1], strict=True):
                self.edges.setdefault((min(a, b), max(a, b)), []).append(f)
        nodes = [list(p) for p in points]
        weights: list[dict[int, float]] = [{i: 1.0} for i in range(len(points))]  # node -> {corner: ∂x/∂x_c}
        midpoint: dict[tuple[int, int], int] = {}
        if split:
            for a, b in self.edges:
                midpoint[(a, b)] = len(nodes)
                nodes.append(list(0.5 * (self.points[a] + self.points[b])))
                weights.append({a: 0.5, b: 0.5})
        cells, cell_face = [], []
        for f, face in enumerate(faces):
            centroid = len(nodes)
            nodes.append(list(self.points[face].mean(axis=0)))
            weights.append({i: 1.0 / len(face) for i in face})
            for a, b in zip(face, face[1:] + face[:1], strict=True):
                key = (min(a, b), max(a, b))
                chain = [a, midpoint[key], b] if key in midpoint else [a, b]
                for p, q in itertools.pairwise(chain):
                    cells.append([p, q, centroid])
                    cell_face.append(f)
        element = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        self.mesh = dmesh.create_mesh(MPI.COMM_SELF, np.array(cells, dtype=np.int64), element, np.array(nodes))
        self.nodes = np.array(nodes)
        self.weights = weights
        tdim = 2
        n_cells = self.mesh.topology.index_map(tdim).size_local
        face_of_cell = np.array(cell_face, dtype=np.int32)[self.mesh.topology.original_cell_index]
        cell_tags = dmesh.meshtags(self.mesh, tdim, np.arange(n_cells, dtype=np.int32), face_of_cell)
        self.mesh.topology.create_connectivity(1, 2)
        f2c = self.mesh.topology.connectivity(1, 2)
        indices, values = [], []
        for facet in range(self.mesh.topology.index_map(1).size_local):
            adjacent = sorted({int(face_of_cell[c]) for c in f2c.links(facet)})
            n_adjacent = len(f2c.links(facet))
            if n_adjacent == 1:
                indices.append(facet)
                values.append(self.free_tag(adjacent[0]))
            elif len(adjacent) == 2:
                indices.append(facet)
                values.append(self.contact_tag(adjacent[0], adjacent[1]))
        facet_tags = dmesh.meshtags(self.mesh, 1, np.array(indices, dtype=np.int32), np.array(values, dtype=np.int32))
        self.members = MembersMesh(self.mesh, cell_tags, facet_tags)
        self.V = fem.functionspace(self.mesh, ("Lagrange", 1, (2,)))
        coords = self.V.tabulate_dof_coordinates()[:, :2]
        self.dof_of_node = np.array([int(np.argmin(np.linalg.norm(coords - p, axis=1))) for p in self.nodes])

    @staticmethod
    def free_tag(f: int) -> int:
        return 100 + f

    @staticmethod
    def contact_tag(f: int, g: int) -> int:
        return 1000 + 10 * f + g

    def surface_tags(self) -> list[int]:
        return sorted(set(self.members.facet_tags.values.tolist()))

    # -- the independent vertex model -----------------------------------------

    def area(self, f: int) -> float:
        return _shoelace(self.points[self.faces[f]])

    def perimeter(self, f: int) -> float:
        p = self.points[self.faces[f]]
        return float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)))

    def area_force(self, f: int, factor: float) -> np.ndarray:
        """−factor · ∂A_f/∂x at every corner."""

        out = np.zeros_like(self.points)
        out[self.faces[f]] -= factor * _area_gradient(self.points[self.faces[f]])
        return out

    def length_force(self, edges: list[tuple[int, int]], factor: float) -> np.ndarray:
        out = np.zeros_like(self.points)
        for a, b in edges:
            ga, gb = _length_gradient(self.points[a], self.points[b])
            out[a] -= factor * ga
            out[b] -= factor * gb
        return out

    def fem_area(self, f: int) -> float:
        dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.members.cell_tags)
        return float(fem.assemble_scalar(fem.form(_constant(self.mesh, 1.0) * dx(f))).real)

    def fem_perimeter(self, f: int) -> float:
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.members.facet_tags)
        dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.members.facet_tags)
        one = _constant(self.mesh, 1.0)
        total = 0.0
        for tag in self.surface_tags():
            owners = [tag - 100] if tag < 1000 else [(tag - 1000) // 10, (tag - 1000) % 10]
            if f in owners:
                measure = ds(tag) if tag < 1000 else dS(tag)
                total += float(fem.assemble_scalar(fem.form(one * measure)).real)
        return total

    def face_edges(self, f: int) -> list[tuple[int, int]]:
        return [e for e, owners in self.edges.items() if f in owners]

    # -- the FEM side ------------------------------------------------------

    def corner_forces(
        self, member_laws: list[MemberLaw], surface_laws: list[SurfaceLaw]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The generalized forces −(load) at the corners, by the chain rule through every node, and the largest
        force at a node that is not a corner and moves no corner (a fan centre on its own)."""

        v, phi = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        loads = [
            t
            for t in force_balance_terms(self.members, v, phi, member_laws, surface_laws)
            if t.kind in (TermKind.CELL_PRESSURE, TermKind.SURFACE_TENSION)
        ]
        b = np.sum([_load(t) for t in loads], axis=0).reshape(-1, 2)
        nodal = -b[self.dof_of_node]
        corners = np.zeros_like(self.points)
        for node, w in enumerate(self.weights):
            for c, weight in w.items():
                corners[c] += weight * nodal[node]
        return corners, nodal


def _member_laws(cx: _Complex, pressures: dict[int, float]) -> list[MemberLaw]:
    one = _constant(cx.mesh, 1.0)
    return [
        MemberLaw(f, viscosity=one, pressure=_constant(cx.mesh, pressures.get(f, 0.0))) for f in range(len(cx.faces))
    ]


def _surface_laws(cx: _Complex, tension: Callable[[int], float]) -> list[SurfaceLaw]:
    return [SurfaceLaw(tag, tension=_constant(cx.mesh, tension(tag))) for tag in cx.surface_tags()]


_HEXAGON = [[math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)]
_COMPLEXES: dict[str, tuple[list[list[float]], list[list[int]], float]] = {
    "square": ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0, 1, 2, 3]], 1.0),
    "hexagon": (_HEXAGON, [[0, 1, 2, 3, 4, 5]], 1.0),
    "sheared": ([[0.0, 0.0], [1.0, 0.2], [1.3, 1.1], [0.1, 0.9]], [[0, 1, 2, 3]], 1.0),
    "pair": (
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
        [[0, 1, 4, 5], [1, 2, 3, 4]],
        1.0,
    ),
    "depth": ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0, 1, 2, 3]], 2.0),
}


@pytest.mark.parametrize("split", [False, True], ids=["corners", "split-edges"])
@pytest.mark.parametrize("case", sorted(_COMPLEXES))
def test_contract_c_area_and_surface_elasticity_and_cortical_and_interfacial_tension(case: str, split: bool) -> None:
    """ADR 017 decisions 1 and 6 (b): on a polygonal complex the P1 loads are the exact gradients of the vertex
    energies. Per unit depth `d`: `AreaElasticity` is the pressure `P = K(V0 − dA)`; `SurfaceElasticity` the
    tension `K_S(dS − S0)` on the cell's whole boundary; `CorticalTension` `σ`; `InterfacialTension` `Λ` on a
    contact. A contact carries the sum of its two cells' tensions (the lowering's `Λ + T_i + T_j`). With split
    edges, the corner force is the chain rule of the nodal forces."""

    points, faces, d = _COMPLEXES[case]
    cx = _Complex(points, faces, split=split)
    n_faces = len(faces)
    K, V0, KS, S0, sigma, Lam = 1.3, 0.6 * d, 0.7, 2.5 * d, 0.4, 0.25

    def owners_of(tag: int) -> list[int]:
        return [tag - 100] if tag < 1000 else [(tag - 1000) // 10, (tag - 1000) % 10]

    expected = np.zeros_like(cx.points)
    pressures = {}
    for f in range(n_faces):
        excess = K * (d * cx.area(f) - V0)
        pressures[f] = -excess
        expected += cx.area_force(f, excess)
    tensions = {f: KS * (d * cx.perimeter(f) - S0) + sigma for f in range(n_faces)}
    for f in range(n_faces):
        expected += cx.length_force(cx.face_edges(f), tensions[f])
    contacts = [e for e, owners in cx.edges.items() if len(owners) == 2]
    expected += cx.length_force(contacts, Lam)

    def tension(tag: int) -> float:
        return sum(tensions[f] for f in owners_of(tag)) + (Lam if tag >= 1000 else 0.0)

    corners, nodal = cx.corner_forces(_member_laws(cx, pressures), _surface_laws(cx, tension))
    assert np.abs(corners - expected).max() < 1e-13
    centres = [i for i, w in enumerate(cx.weights) if len(w) >= 3]
    assert np.abs(nodal[centres]).max() < 1e-13  # a fan centre is inside one cell: its forces cancel


_VECTORS = Path(__file__).resolve().parents[2] / "vcell-core" / "conformance" / "mechanics" / "terms.jsonl"


@pytest.mark.skipif(not _VECTORS.exists(), reason="vcell-core's conformance vectors need the sibling checkout")
def test_contract_c_against_vcell_core_vectors() -> None:
    """ADR 017 decision 6 (a): vcell-core's `conformance/mechanics/terms.jsonl`, read from the sibling checkout
    (vcell-core is private and unlicensed today, so the vectors are not copied here). Every force, per law, from
    the T9 loads. The energies use the FEM's measured areas and perimeters, so they check that the mesh
    measures the complex exactly and that the conventions (depth, the per-unit-depth energy) agree. Friction's
    per-vertex values are not comparable (ADR 017 decision 4)."""

    for line in _VECTORS.read_text().splitlines():
        case = json.loads(line)
        cx = _Complex(case["points"], case["faces"])
        d, terms, expected = case["depth"], case["terms"], case["expected"]
        laws: dict[str, tuple[dict[int, float], Callable[[int], float], float]] = {}

        def none(tag: int) -> float:
            return 0.0

        if "area" in expected:
            pressures, energy = {}, 0.0
            for key, p in terms["area_elasticity"].items():
                f = int(key)
                excess = p["K"] * (d * cx.fem_area(f) - p["V0"])
                pressures[f] = -excess
                energy += 0.5 * p["K"] * (d * cx.fem_area(f) - p["V0"]) ** 2 / d
            laws["area"] = (pressures, none, energy)
        per_face: dict[str, dict[int, float]] = {}
        energies: dict[str, float] = {}
        if "surface" in expected:
            per_face["surface"] = {}
            energies["surface"] = 0.0
            for key, p in terms["surface_elasticity"].items():
                f, S = int(key), d * cx.fem_perimeter(int(key))
                per_face["surface"][f] = p["K_S"] * (S - p["S0"])
                energies["surface"] += 0.5 * p["K_S"] * (S - p["S0"]) ** 2 / d
        if "cortical" in expected:
            per_face["cortical"] = {int(k): p["sigma"] for k, p in terms["cortical_tension"].items()}
            energies["cortical"] = sum(s * cx.fem_perimeter(f) for f, s in per_face["cortical"].items())
        for law, values in per_face.items():

            def face_tension(tag: int, values: dict[int, float] = values) -> float:
                owners = [tag - 100] if tag < 1000 else [(tag - 1000) // 10, (tag - 1000) % 10]
                return sum(values.get(f, 0.0) for f in owners)

            laws[law] = ({}, face_tension, energies[law])
        if "interfacial" in expected:
            lam = {tuple(sorted(map(int, k.split(",")))): p["Lambda"] for k, p in terms["interfacial_tension"].items()}

            def contact_tension(tag: int, lam: dict[tuple[int, ...], float] = lam) -> float:
                if tag < 1000:
                    return 0.0
                return float(lam.get(((tag - 1000) // 10, (tag - 1000) % 10), 0.0))

            contact_energy = 0.0
            for (a, b), owners in cx.edges.items():
                if len(owners) == 2:
                    pair = (min(owners), max(owners))
                    contact_energy += lam.get(pair, 0.0) * float(np.linalg.norm(cx.points[a] - cx.points[b]))
            laws["interfacial"] = ({}, contact_tension, contact_energy)
        assert set(laws) == set(expected) - {"friction"}, case["case"]
        for law, (pressures, tension, energy) in laws.items():
            corners, _ = cx.corner_forces(_member_laws(cx, pressures), _surface_laws(cx, tension))
            assert np.abs(corners - np.array(expected[law]["forces"])).max() < 1e-12, (case["case"], law)
            assert energy == pytest.approx(expected[law]["energy"], abs=1e-12), (case["case"], law)


# ---------------------------------------------------------------------------
# Small solves on a fixed mesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["symmetric", "laplacian"])
def test_the_drag_transmits_no_net_force_without_a_load_that_has_one(operator: ViscousOperator) -> None:
    """ADR 019 decision 2: testing the solved balance with a constant `φ` leaves `∫ζv dx = ∫t dS`. With only
    pressures, active stresses and tensions (which exert no net force), `∫ζv = 0`."""

    members = _two_members(2)
    member_laws, surface_laws = _laws(members)
    problem = assemble_force_balance(
        members,
        member_laws,
        surface_laws,
        viscous_operator=operator,
        semi_implicit_dt=0.05,
    )
    problem.step()
    v = problem.unknown
    assert np.abs(v.x.array).max() > 1e-3  # the loads do move it
    x = ufl.SpatialCoordinate(members.mesh)
    dx = ufl.Measure("dx", domain=members.mesh, subdomain_data=members.cell_tags)
    drag_force = [
        fem.assemble_scalar(fem.form((2.0 + x[0]) * v[k] * dx(LEFT) + 1.5 * v[k] * dx(RIGHT))) for k in range(2)
    ]
    assert np.abs(drag_force).max() < 1e-11

    traction = SurfaceLaw(FREE_RIGHT, traction=ufl.as_vector([0.1, -0.05]))
    problem = assemble_force_balance(members, member_laws, [*surface_laws, traction], viscous_operator=operator)
    problem.step()
    v = problem.unknown
    drag_force = [
        fem.assemble_scalar(fem.form((2.0 + x[0]) * v[k] * dx(LEFT) + 1.5 * v[k] * dx(RIGHT))) for k in range(2)
    ]
    assert drag_force == pytest.approx([0.1 * 3.0, -0.05 * 3.0], abs=1e-11)


def test_a_zero_velocity_membrane_holds_the_cell_against_an_active_stress() -> None:
    """ADR 019 decision 4 (iii): the zero-velocity edge (an adhesion band) holds `v = 0` on the membrane, with
    no drag; a non-uniform contractile stress still drives an interior flow."""

    members = _two_members(2)
    mesh = members.mesh
    x = ufl.SpatialCoordinate(mesh)
    V = _space(members)
    laws = [
        MemberLaw(
            LEFT, viscosity=_constant(mesh, 1.0), active_stress=ufl.exp(-((x[0] - 0.5) ** 2) - (x[1] - 0.5) ** 2)
        ),
        MemberLaw(RIGHT, viscosity=_constant(mesh, 1.0)),
    ]
    walls = [velocity_dirichlet(V, members.facet_tags, tag) for tag in (FREE_LEFT, FREE_RIGHT)]
    problem = assemble_force_balance(members, laws, conditions=walls, space=V)
    problem.step()
    v = problem.unknown.x.array.reshape(-1, 2)
    held = np.concatenate(
        [fem.locate_dofs_topological(V, 1, members.facet_tags.find(t)) for t in (FREE_LEFT, FREE_RIGHT)]
    )
    assert np.abs(v[held]).max() == 0.0
    assert np.abs(v).max() > 1e-4


def test_a_membranes_given_velocity_is_imposed() -> None:
    members = _two_members(2)
    V = _space(members)
    x = ufl.SpatialCoordinate(members.mesh)
    t = _constant(members.mesh, 0.0)
    condition = velocity_dirichlet(V, members.facet_tags, FREE_LEFT, ufl.as_vector([0.2 * x[1] * (1 + t), 0.0]))
    laws = [
        MemberLaw(LEFT, viscosity=_constant(members.mesh, 1.0)),
        MemberLaw(RIGHT, viscosity=_constant(members.mesh, 1.0)),
    ]
    problem = assemble_force_balance(members, laws, conditions=[condition], space=V)
    problem.step()
    dofs = fem.locate_dofs_topological(V, 1, members.facet_tags.find(FREE_LEFT))
    v, given = problem.unknown.x.array.reshape(-1, 2), condition.value.x.array.reshape(-1, 2).copy()
    assert np.allclose(v[dofs], given[dofs], atol=1e-14)
    assert np.abs(given[dofs]).max() > 0.1
    # a time-dependent velocity is refreshed by set_time through the problem's Dirichlet refreshers
    problem.time = t
    problem.set_time(1.0)
    problem.step()
    v = problem.unknown.x.array.reshape(-1, 2)
    assert np.allclose(v[dofs], 2.0 * given[dofs], atol=1e-14)


def test_a_3d_force_balance_solves_with_a_wall() -> None:
    members = _two_members(3)
    member_laws, surface_laws = _laws(members)
    V = _space(members)
    wall = velocity_dirichlet(V, members.facet_tags, FREE_LEFT)
    problem = assemble_force_balance(members, member_laws, surface_laws, conditions=[wall], space=V)
    problem.step()
    assert np.isfinite(problem.unknown.x.array).all()
    assert np.abs(problem.unknown.x.array).max() > 1e-4


# ---------------------------------------------------------------------------
# Values and signs the invariants cannot see (the review's mutation runs)
# ---------------------------------------------------------------------------


def _one_member(members: MembersMesh, **slots: object) -> list[Term]:
    V = _space(members)
    law = MemberLaw(LEFT, **slots)
    return list(force_balance_terms(members, ufl.TrialFunction(V), ufl.TestFunction(V), [law], []))


def test_an_active_stress_is_a_negative_pressure_and_pulls_a_cell_inward() -> None:
    """ADR 019 decision 2: `σ_a` enters as `+σ_a ∇·φ`, the pressure as `−P ∇·φ`, so `σ_a = c` is `P = −c`. Its
    load on the dilation `φ = x` is `+c d |Ω|`: the residual resists expansion, i.e. the stress contracts."""

    members = _two_members(2)
    mesh = members.mesh
    c = 0.7
    one = _constant(mesh, 1.0)
    (_, active) = _one_member(members, viscosity=one, active_stress=_constant(mesh, c))
    (_, pressure) = _one_member(members, viscosity=one, pressure=_constant(mesh, -c))
    assert active.kind is TermKind.ACTIVE_STRESS and pressure.kind is TermKind.CELL_PRESSURE
    assert np.allclose(_load(active), _load(pressure), atol=1e-14)
    dilation = _field(_space(members), lambda x: np.vstack([x[0], x[1]])).x.array
    assert _load(active) @ dilation == pytest.approx(c * 2 * 1.0, rel=1e-12)  # d = 2, |LEFT| = 1


@pytest.mark.parametrize("operator", ["symmetric", "laplacian"])
def test_the_bulk_viscosity_costs_its_value_on_a_dilation(operator: ViscousOperator) -> None:
    """`η_b ∇·v ∇·φ` on `v = x`: `η_b d² |Ω|`, with the shear viscosity 0 (only the bulk term left)."""

    members = _two_members(2)
    mesh = members.mesh
    V = _space(members)
    law = MemberLaw(LEFT, viscosity=_constant(mesh, 0.0), bulk_viscosity=_constant(mesh, 1.5))
    (term,) = force_balance_terms(
        members, ufl.TrialFunction(V), ufl.TestFunction(V), [law], [], viscous_operator=operator
    )
    dilation = _field(V, lambda x: np.vstack([x[0], x[1]])).x.array
    assert dilation @ _matrix(term) @ dilation == pytest.approx(1.5 * 4 * 1.0, rel=1e-12)


def test_a_two_sided_contact_tension_is_passed_restricted() -> None:
    """A coefficient that differs between a contact's sides is the caller's to combine: `ufl.avg` of a DG0
    tension that is 1 on LEFT and 3 on RIGHT loads the contact as a uniform 2 does."""

    members = _two_members(2)
    mesh = members.mesh
    V = _space(members)
    dg0 = fem.functionspace(mesh, ("DG", 0))
    g = fem.Function(dg0)
    g.x.array[members.cell_tags.find(LEFT)] = 1.0
    g.x.array[members.cell_tags.find(RIGHT)] = 3.0
    v, phi = ufl.TrialFunction(V), ufl.TestFunction(V)
    (two_sided,) = force_balance_terms(members, v, phi, [], [SurfaceLaw(CONTACT, tension=ufl.avg(g))])
    (uniform,) = force_balance_terms(members, v, phi, [], [SurfaceLaw(CONTACT, tension=_constant(mesh, 2.0))])
    assert np.abs(_load(uniform)).max() > 0.1
    assert np.allclose(_load(two_sided), _load(uniform), atol=1e-14)


def test_a_traction_on_a_contact_is_refused() -> None:
    members = _two_members(2)
    V = _space(members)
    with pytest.raises(ValueError, match="a traction is a load on a free membrane"):
        force_balance_terms(
            members,
            ufl.TrialFunction(V),
            ufl.TestFunction(V),
            [],
            [SurfaceLaw(CONTACT, traction=ufl.as_vector([1.0, 0.0]))],
        )


def test_every_cell_must_belong_to_a_member() -> None:
    members = _two_members(2)
    with pytest.raises(ValueError, match="belong to no member"):
        assemble_force_balance(members, [MemberLaw(LEFT, viscosity=_constant(members.mesh, 1.0))])


def test_the_quasi_static_scheme_refuses_a_foreign_term_and_a_problem_with_no_operator() -> None:
    members = _two_members(2)
    V = _space(members)
    v, w = ufl.TrialFunction(V), ufl.TestFunction(V)

    def problem(terms: tuple[Term, ...]) -> DiscreteProblem:
        return DiscreteProblem(
            variable_name="v",
            V=V,
            trial=v,
            test=w,
            dx=ufl.dx(domain=members.mesh),
            unknown=fem.Function(V),
            previous=fem.Function(V),
            dt=_constant(members.mesh, 0.0),
            terms=terms,
            scheme=QuasiStatic(),
            bcs=[],
        )

    source = Term(TermKind.SOURCE, ufl.inner(ufl.as_vector([1.0, 0.0]), w))
    with pytest.raises(TypeError, match="force-balance terms only"):
        _ = problem((source,)).bilinear_form
    (load,) = _one_member(members, viscosity=_constant(members.mesh, 1.0), pressure=_constant(members.mesh, 1.0))[1:]
    with pytest.raises(TypeError, match="needs a term in the unknown"):
        _ = problem((Term(load.kind, load.integrand, load.measure),)).bilinear_form


_GHOSTS = r"""
import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI
from vcell_fenics.backend.force_balance import MembersMesh, surface_is_interior

m = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([2.0, 1.0])], [8, 4],
                           ghost_mode=dmesh.GhostMode.shared_facet)
tdim = m.topology.dim
m.topology.create_connectivity(tdim - 1, tdim)
cells = np.arange(m.topology.index_map(tdim).size_local + m.topology.index_map(tdim).num_ghosts, dtype=np.int32)
cell_tags = dmesh.meshtags(m, tdim, cells, np.ones_like(cells))
bottom = dmesh.locate_entities(m, tdim - 1, lambda x: np.isclose(x[1], 0.0))  # owned and ghost facets
middle = dmesh.locate_entities(m, tdim - 1, lambda x: np.isclose(x[0], 1.0))
idx = np.concatenate([bottom, middle]).astype(np.int32)
val = np.concatenate([np.full(len(bottom), 11), np.full(len(middle), 20)]).astype(np.int32)
order = np.argsort(idx)
members = MembersMesh(m, cell_tags, dmesh.meshtags(m, tdim - 1, idx[order], val[order]))
print("PROBE", surface_is_interior(members, 11), surface_is_interior(members, 20))
"""


@pytest.mark.integration
def test_free_and_contact_surfaces_are_classified_alike_on_every_rank() -> None:
    """Facet tags that include ghost facets (as a mesher's or a file's do) classify as in serial: a ghost facet
    of a free membrane is not counted as interior (the review's finding, reproduced at `mpiexec -n 3`)."""

    import subprocess
    import sys

    mpiexec = Path(sys.executable).parent / "mpiexec"
    result = subprocess.run(
        [str(mpiexec), "-n", "3", sys.executable, "-c", _GHOSTS], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-3000:]
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("PROBE ")]
    assert lines == ["PROBE False True"] * 3

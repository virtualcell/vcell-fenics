"""Realization v1, step 1: the trivial (dim-0 / well-mixed) case of `backend.realize.realize`.

Checks that a non-spatial GeometryDescription becomes a backend Geometry with one lumped mesh per
compartment (and per membrane), that it satisfies the §1.11.10 cross-check against a matching
MathDescription, and that the unimplemented spatial path and the malformed cases raise clearly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import geometry as dgeom
from dolfinx.mesh import Mesh
from numpy.typing import NDArray

from vcell_fenics.backend.geometry import Geometry, cross_validate
from vcell_fenics.backend.realize import RealizationError, realize
from vcell_fenics.formalism.expr import BinaryOp, Expr, FunctionCall, IndexAccess, Number, UnaryOp
from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.rvachev import subvolume_implicit_functions
from vcell_fenics.formalism.schema import MathDescription, Subdomain, TemplateEquation, Variable


def _measure(geometry: Geometry, subdomain: str) -> float:
    """The integral of 1 over a subdomain mesh — its area (volume) or, for the membrane, length."""
    mesh = geometry.mesh_of(subdomain)
    return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=mesh))).real)


def _eval(expr: Expr, x: float, y: float) -> float | bool:
    """Scalar evaluator over the geom.x predicate / implicit-function subset — used as an
    independent oracle (the boolean predicate and its R-function) against the realized mesh."""
    if isinstance(expr, Number):
        return expr.value
    if isinstance(expr, IndexAccess):
        assert isinstance(expr.index, Number)
        return (x, y)[int(expr.index.value)]
    if isinstance(expr, UnaryOp):
        v = _eval(expr.operand, x, y)
        return -float(v) if expr.op == "-" else (float(v) if expr.op == "+" else not v)
    if isinstance(expr, BinaryOp):
        a, b = float(_eval(expr.left, x, y)), float(_eval(expr.right, x, y))
        op = expr.op
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "**":
            return math.pow(a, b)
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        if op == "&&":
            return bool(a) and bool(b)
        if op == "||":
            return bool(a) or bool(b)
        raise AssertionError(f"unhandled operator {op!r}")
    if isinstance(expr, FunctionCall):
        args = [float(_eval(a, x, y)) for a in expr.args]
        if expr.callee == "min":
            return min(args)
        if expr.callee == "max":
            return max(args)
        raise AssertionError(f"unhandled function {expr.callee!r}")
    raise AssertionError(f"unhandled node {type(expr).__name__}")


def _in_mesh(mesh: Mesh, points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Whether each (N, 3) point lies in a cell of `mesh` (via DOLFINx collision queries)."""
    tree = dgeom.bb_tree(mesh, mesh.topology.dim)
    candidates = dgeom.compute_collisions_points(tree, points)
    colliding = dgeom.compute_colliding_cells(mesh, candidates, points)
    return np.array([colliding.links(np.int32(i)).size > 0 for i in range(len(points))])


def _compartmental(name: str, *subvolumes: str, surfaces: tuple[SurfaceClass, ...] = ()) -> GeometryDescription:
    return GeometryDescription(
        name=name,
        dim=0,
        subvolumes=tuple(SubVolume(name=s, type="compartmental") for s in subvolumes),
        surfaces=surfaces,
    )


def test_trivial_single_compartment() -> None:
    geom = realize(_compartmental("cell", "cytosol"))
    assert geom.name == "cell"
    assert set(geom.subdomains) == {"cytosol"}
    assert geom.kind_of("cytosol") == "volume"
    assert geom.mesh_of("cytosol") is not None


def test_trivial_two_compartments_with_membrane() -> None:
    geom = realize(
        _compartmental(
            "cell",
            "cytosol",
            "extracellular",
            surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
        )
    )
    assert geom.kind_of("cytosol") == "volume"
    assert geom.kind_of("extracellular") == "volume"
    assert geom.kind_of("pm") == "surface"


def test_realized_geometry_passes_cross_check() -> None:
    geom = realize(_compartmental("cell", "cytosol"))
    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cytosol", kind="volume")],
        variables=[Variable(name="u", subdomain="cytosol")],
        equations=[
            TemplateEquation(template="lumped_ode", variable="u", subdomain="cytosol", temporality="time_dependent")
        ],
    )
    assert cross_validate(md, geom) == []


def test_2d_single_subvolume_is_the_whole_box() -> None:
    # A geometry with one subvolume is the whole bounding box (VCell's analytic background '1.0'):
    # a plain box mesh, one volume region, the four box faces named — no contours or membranes.
    spatial = GeometryDescription(
        name="square",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(SubVolume(name="domain", type="analytic", expression="1.0"),),
    )
    geom = realize(spatial, h=0.1)
    assert set(geom.subdomains) == {"domain"}
    assert geom.kind_of("domain") == "volume"
    assert set(geom.boundaries) == {"x_minus", "x_plus", "y_minus", "y_plus"}
    for face in ("x_minus", "x_plus", "y_minus", "y_plus"):
        boundary = geom.boundary_of(face)
        assert boundary is not None and not boundary.is_internal and boundary.subdomains == ("domain",)
        assert boundary.facets.size > 0
    # The four box faces are the subdomain mesh's own facets (single-compartment pattern).
    assert geom.parent_mesh is None


def test_non_compartmental_subvolume_in_dim0_rejected() -> None:
    bad = GeometryDescription(
        name="cell", dim=0, subvolumes=(SubVolume(name="cyto", type="analytic", expression="geom.x[0] < 1"),)
    )
    with pytest.raises(RealizationError, match="must be 'compartmental'"):
        realize(bad)


def test_no_subvolumes_rejected() -> None:
    with pytest.raises(RealizationError, match="no subvolumes"):
        realize(GeometryDescription(name="empty", dim=0))


# -- 2D body-fitted, box-partitioned --------------------------------------------


def _disk_in_box(radius: float = 0.5) -> GeometryDescription:
    """A 2x2 box centred at the origin with a disk `cytosol` inside an `extracellular` background,
    meeting at the `pm` membrane."""
    r2 = radius * radius
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="cytosol", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 < {r2}"),
            SubVolume(name="extracellular", type="analytic", expression=f"geom.x[0]**2 + geom.x[1]**2 > {r2}"),
        ),
        surfaces=(SurfaceClass(name="pm", inside="cytosol", outside="extracellular"),),
    )


def test_2d_disk_in_box_partition() -> None:
    radius = 0.5
    geom = realize(_disk_in_box(radius), h=0.06, resolution=161)

    assert geom.kind_of("cytosol") == "volume"
    assert geom.kind_of("extracellular") == "volume"
    assert geom.kind_of("pm") == "surface"

    cyto, ext = _measure(geom, "cytosol"), _measure(geom, "extracellular")
    # Body-fitted regions partition the box; the disk area matches pi r^2 within the polygonal
    # discretization of the marched contour (a few percent at this resolution).
    assert cyto + ext == pytest.approx(4.0, abs=1e-9)
    assert cyto == pytest.approx(math.pi * radius**2, rel=0.05)
    assert _measure(geom, "pm") == pytest.approx(2 * math.pi * radius, rel=0.05)


def test_realize_interface_coupled_builds_a_solvable_coupled_geometry() -> None:
    # The realize -> InterfaceCoupledGeometry bridge: the imported geometry is realized into the
    # two-bulk + membrane object the coupled solver consumes, retaining the entity maps `realize`
    # discards. A flux-balance permeability coupling then solves end-to-end on it, reaching the
    # disk-in-box mass-weighted equilibrium u_eq = A_disk/A_box.
    from vcell_fenics.backend import integrate_interface_coupled
    from vcell_fenics.backend.realize import realize_interface_coupled
    from vcell_fenics.formalism.schema import BCInterfaceFluxBalance, ParameterConstant

    radius = 0.5
    geometry = realize_interface_coupled(
        _disk_in_box(radius),
        inner_subdomain="cytosol",
        outer_subdomain="extracellular",
        membrane_subdomain="pm",
        interface="pm",
        h=0.1,
    )
    assert geometry.kind_of("cytosol") == "volume"
    assert geometry.kind_of("pm") == "surface"
    # The three entity maps relating the submeshes to the shared parent are retained.
    for subdomain in ("cytosol", "extracellular"):
        assert geometry.entity_map_of(subdomain) is not None

    md = MathDescription(
        geometry="cell",
        subdomains=[Subdomain(name="cytosol", kind="volume"), Subdomain(name="extracellular", kind="volume")],
        variables=[Variable(name="u", subdomain="cytosol"), Variable(name="v", subdomain="extracellular")],
        parameters=[ParameterConstant(name="P", value=0.5)],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="u",
                subdomain="cytosol",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="1.0",
            ),
            TemplateEquation(
                template="bulk_radv_diff",
                variable="v",
                subdomain="extracellular",
                temporality="time_dependent",
                terms={"diffusion": "1.0"},
                initial_condition="0.0",
            ),
        ],
        boundary_conditions=[
            BCInterfaceFluxBalance(variable="u", partner_variable="v", boundary="pm", expression="P * (v - u)")
        ],
    )
    result = integrate_interface_coupled(md, geometry, t_final=4.0)
    u_eq = (math.pi * radius**2) / 4.0  # A_disk / A_box (the box is 2x2 = 4)
    assert result.inner.x.array.mean() == pytest.approx(u_eq, rel=5e-2)
    assert result.outer.x.array.mean() == pytest.approx(u_eq, rel=5e-2)


def test_2d_point_membership_agrees_across_predicate_rfunction_and_mesh() -> None:
    """Random points are classified inside/outside the cell by three independent representations —
    the original boolean predicate, the Rvachev implicit field's sign, and the realized gmsh mesh —
    and all three agree. Points whose R-function value is near 0 (the membrane band, where the mesh
    necessarily approximates the contour) are skipped; the seed is fixed and can be changed if a rare
    near-boundary point slips past the skip."""

    radius = 0.5
    description = _disk_in_box(radius)
    h = 0.06
    geom = realize(description, h=h, resolution=161)

    assert description.subvolumes[0].expression is not None
    predicate = parse(description.subvolumes[0].expression)  # the cytosol boolean
    field = subvolume_implicit_functions(description)["cytosol"]  # its R-function

    rng = np.random.default_rng(0)
    pts2d = rng.uniform(-0.95, 0.95, size=(400, 2))
    points = np.column_stack([pts2d, np.zeros(len(pts2d))]).astype(np.float64)

    in_cyto = _in_mesh(geom.mesh_of("cytosol"), points)
    in_ext = _in_mesh(geom.mesh_of("extracellular"), points)

    phi_skip = 2.5 * h  # |phi| ~ distance-to-membrane here; skip the band the mesh can't resolve
    checked = 0
    for i, (x, y, _z) in enumerate(points):
        phi = float(_eval(field, float(x), float(y)))
        if abs(phi) < phi_skip:
            continue
        predicate_inside = bool(_eval(predicate, float(x), float(y)))
        assert (phi < 0) == predicate_inside, f"R-function vs predicate at {(x, y)}"
        assert bool(in_cyto[i]) == predicate_inside, f"mesh vs predicate at {(x, y)}"
        assert bool(in_ext[i]) == (not predicate_inside), f"complementary region at {(x, y)}"
        checked += 1
    assert checked > 100  # the skip band did not swallow the sample


def test_2d_membrane_is_internal_and_faces_external() -> None:
    geom = realize(_disk_in_box(), h=0.08, resolution=121)
    pm = geom.boundary_of("pm")
    assert pm is not None and pm.is_internal and set(pm.subdomains) == {"cytosol", "extracellular"}
    for face in ("x_minus", "x_plus", "y_minus", "y_plus"):
        boundary = geom.boundary_of(face)
        assert boundary is not None and not boundary.is_internal
        assert boundary.subdomains == ("extracellular",)
        assert boundary.facets.size > 0


def test_2d_realized_geometry_passes_cross_check() -> None:
    geom = realize(_disk_in_box(), h=0.08, resolution=121)
    md = MathDescription(
        geometry="cell",
        subdomains=[
            Subdomain(name="cytosol", kind="volume"),
            Subdomain(name="extracellular", kind="volume"),
            Subdomain(name="pm", kind="surface"),
        ],
        variables=[Variable(name="u", subdomain="cytosol")],
        equations=[
            TemplateEquation(template="bulk_radv_diff", variable="u", subdomain="cytosol", temporality="steady_state")
        ],
    )
    assert cross_validate(md, geom) == []


def _nested_three_region() -> GeometryDescription:
    """A nucleus (r=0.3) inside a cytosol (r=0.7) inside an ecm background, in a 2x2 box."""
    return GeometryDescription(
        name="cell",
        dim=2,
        extent=(2.0, 2.0, 1.0),
        origin=(-1.0, -1.0, 0.0),
        subvolumes=(
            SubVolume(name="nucleus", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.09"),
            SubVolume(name="cytosol", type="analytic", expression="geom.x[0]**2 + geom.x[1]**2 < 0.49"),
            SubVolume(name="ecm", type="analytic", expression="1.0"),
        ),
        surfaces=(
            SurfaceClass(name="ne", inside="nucleus", outside="cytosol"),
            SurfaceClass(name="pm", inside="cytosol", outside="ecm"),
        ),
    )


def test_2d_nested_three_region_partition() -> None:
    geom = realize(_nested_three_region(), h=0.05, resolution=201)

    assert {n: sg.kind for n, sg in geom.subdomains.items()} == {
        "nucleus": "volume",
        "cytosol": "volume",
        "ecm": "volume",
        "ne": "surface",
        "pm": "surface",
    }
    nucleus, cytosol, ecm = (_measure(geom, n) for n in ("nucleus", "cytosol", "ecm"))
    assert nucleus + cytosol + ecm == pytest.approx(4.0, abs=1e-9)  # partition the box exactly
    assert nucleus == pytest.approx(math.pi * 0.3**2, rel=0.05)
    assert cytosol == pytest.approx(math.pi * (0.7**2 - 0.3**2), rel=0.05)


def test_2d_nested_membranes_named_internal_and_paired() -> None:
    geom = realize(_nested_three_region(), h=0.06, resolution=161)
    ne = geom.boundary_of("ne")
    assert ne is not None and ne.is_internal and set(ne.subdomains) == {"nucleus", "cytosol"}
    assert _measure(geom, "ne") == pytest.approx(2 * math.pi * 0.3, rel=0.06)
    pm = geom.boundary_of("pm")
    assert pm is not None and pm.is_internal and set(pm.subdomains) == {"cytosol", "ecm"}
    # Only the background (ecm) touches the box faces in a nested geometry.
    for face in ("x_minus", "x_plus", "y_minus", "y_plus"):
        boundary = geom.boundary_of(face)
        assert boundary is not None and boundary.subdomains == ("ecm",)


def test_2d_nested_point_membership_mesh_matches_rfunction() -> None:
    """Stronger than the area checks: every sampled point lands in the same region per the realized
    multi-region **mesh** (which submesh contains it) and per the **R-function** partition (which
    subvolume's φ is negative). Points the mesh leaves ambiguous (in zero or several submeshes — the
    near-membrane band) are skipped; the seed is fixed and reseedable."""

    description = _nested_three_region()
    geom = realize(description, h=0.05, resolution=201)
    fields = subvolume_implicit_functions(description)
    names = list(fields)  # nucleus, cytosol, ecm (priority order)

    rng = np.random.default_rng(0)
    pts2d = rng.uniform(-0.95, 0.95, size=(500, 2))
    points = np.column_stack([pts2d, np.zeros(len(pts2d))]).astype(np.float64)
    in_mesh = {name: _in_mesh(geom.mesh_of(name), points) for name in names}

    checked = 0
    for i, (x, y, _z) in enumerate(points):
        mesh_regions = [name for name in names if in_mesh[name][i]]
        rfunc_regions = [name for name in names if _eval(fields[name], float(x), float(y)) < 0]
        if len(mesh_regions) != 1 or len(rfunc_regions) != 1:
            continue  # near a membrane: the mesh approximates the contour, sign is undefined
        assert mesh_regions[0] == rfunc_regions[0], f"at {(x, y)}: mesh={mesh_regions} rfunc={rfunc_regions}"
        checked += 1
    assert checked > 200  # the skip band did not swallow the sample


def test_2d_interior_must_be_analytic() -> None:
    bad = GeometryDescription(
        name="g",
        dim=2,
        subvolumes=(
            SubVolume(name="a", type="compartmental"),
            SubVolume(name="bg", type="analytic", expression="geom.x[0] > 0"),
        ),
    )
    with pytest.raises(RealizationError, match="must be 'analytic'"):
        realize(bad)


def test_3d_not_implemented() -> None:
    g = GeometryDescription(name="g", dim=3, subvolumes=(SubVolume(name="c", type="compartmental"),))
    with pytest.raises(NotImplementedError, match="v1 supports dim 0 and dim 2"):
        realize(g)

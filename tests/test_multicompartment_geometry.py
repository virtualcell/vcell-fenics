"""Verification of the multi-compartment cell geometry (internal interface).

`create_cell_extracellular` builds an inner disk (cytosol) inside an outer annulus
(extracellular) meeting at the membrane, and `make_cell_extracellular_geometry`
exposes it as a `Geometry` with two volume compartments, the membrane as a surface
subdomain, an internal interface boundary, and the external outer boundary. The
checks establish the defining multi-compartment properties:

1. **Compartments partition the disk** — cytosol and extracellular submesh areas
   match πr_in² and π(r_out²−r_in²), and sum to the whole disk.
2. **The membrane is an internal interface** — it integrates under the interior-facet
   measure `dS` (incident to both compartments), not the exterior `ds`; the outer
   circle is the reverse.
3. **The Geometry models incidence** — the interface boundary lists both compartments
   (`is_internal`), the outer boundary one, and the membrane is a `surface` subdomain.
4. **The cross-check accepts a matching multi-compartment model and rejects a kind
   mismatch.**
"""

from __future__ import annotations

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh

from tests.gmsh_meshers.multicompartment.geometry import (
    MEMBRANE_TAG,
    OUTER_TAG,
    create_cell_extracellular,
    make_cell_extracellular_geometry,
)
from vcell_fenics.backend import Geometry, cross_validate, make_disk_geometry
from vcell_fenics.formalism import load_yaml

_R_IN, _R_OUT = 0.6, 1.0


def _area(mesh: dmesh.Mesh) -> float:
    return float(fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * ufl.dx(domain=mesh))).real)


def _geom() -> Geometry:
    return make_cell_extracellular_geometry(
        "cell",
        cytosol="cyto",
        extracellular="ext",
        membrane="mem",
        interface="membrane_interface",
        outer="outer",
        inner_radius=_R_IN,
        outer_radius=_R_OUT,
        h=0.08,
    )


# ---------------------------------------------------------------------------
# 1. compartments partition the disk
# ---------------------------------------------------------------------------


def test_compartments_partition_the_disk() -> None:
    cell = create_cell_extracellular(inner_radius=_R_IN, outer_radius=_R_OUT, h=0.06)

    cyto = _area(cell.cytosol_mesh)
    ext = _area(cell.extracellular_mesh)
    assert cyto == pytest.approx(np.pi * _R_IN**2, rel=2e-2)
    assert ext == pytest.approx(np.pi * (_R_OUT**2 - _R_IN**2), rel=2e-2)
    assert cyto + ext == pytest.approx(np.pi * _R_OUT**2, rel=2e-2)  # the whole disk


# ---------------------------------------------------------------------------
# 2. the membrane is an internal interface
# ---------------------------------------------------------------------------


def test_membrane_is_internal_and_outer_is_external() -> None:
    cell = create_cell_extracellular(inner_radius=_R_IN, outer_radius=_R_OUT, h=0.06)
    parent = cell.parent_mesh
    dS = ufl.Measure("dS", domain=parent, subdomain_data=cell.facet_tags)
    ds = ufl.Measure("ds", domain=parent, subdomain_data=cell.facet_tags)
    one = fem.Constant(parent, 1.0)

    # The membrane is interior — it integrates under the interior-facet measure dS
    # (with a restriction) and contributes nothing to the exterior dS; the outer
    # circle is the reverse.
    membrane_internal = float(fem.assemble_scalar(fem.form(one("+") * dS(MEMBRANE_TAG))).real)
    outer_internal = float(fem.assemble_scalar(fem.form(one("+") * dS(OUTER_TAG))).real)
    outer_external = float(fem.assemble_scalar(fem.form(one * ds(OUTER_TAG))).real)
    assert membrane_internal == pytest.approx(2 * np.pi * _R_IN, rel=2e-2)
    assert outer_internal == pytest.approx(0.0, abs=1e-12)  # the outer circle is not interior
    assert outer_external == pytest.approx(2 * np.pi * _R_OUT, rel=2e-2)

    # The topological truth: membrane facets each border two cells and are absent from
    # the exterior set; outer facets border one and are entirely exterior.
    tdim = parent.topology.dim
    parent.topology.create_connectivity(tdim - 1, tdim)
    f2c = parent.topology.connectivity(tdim - 1, tdim)
    membrane_facets, outer_facets = cell.facet_tags.find(MEMBRANE_TAG), cell.facet_tags.find(OUTER_TAG)
    exterior = dmesh.exterior_facet_indices(parent.topology)
    assert all(f2c.links(f).size == 2 for f in membrane_facets)
    assert all(f2c.links(f).size == 1 for f in outer_facets)
    assert np.intersect1d(membrane_facets, exterior).size == 0
    assert np.intersect1d(outer_facets, exterior).size == outer_facets.size


def test_membrane_submesh_is_the_inner_circle() -> None:
    cell = create_cell_extracellular(inner_radius=_R_IN, outer_radius=_R_OUT, h=0.06)
    membrane = cell.membrane_mesh

    assert membrane.topology.dim == 1  # codim-1
    radii = np.linalg.norm(membrane.geometry.x[:, :2], axis=1)
    assert np.allclose(radii, _R_IN, atol=0.02)  # every node on the inner circle
    assert _area(membrane) == pytest.approx(2 * np.pi * _R_IN, rel=2e-2)  # its length


# ---------------------------------------------------------------------------
# 3. the Geometry models compartments + incidence
# ---------------------------------------------------------------------------


def test_geometry_exposes_compartments_and_incidence() -> None:
    geom = _geom()

    assert geom.kind_of("cyto") == "volume"
    assert geom.kind_of("ext") == "volume"
    assert geom.kind_of("mem") == "surface"

    interface = geom.boundary_of("membrane_interface")
    outer = geom.boundary_of("outer")
    assert interface is not None and outer is not None
    assert interface.is_internal  # incident to both compartments
    assert set(interface.subdomains) == {"cyto", "ext"}
    assert not outer.is_internal
    assert outer.subdomains == ("ext",)

    # The multi-compartment substrate is present (single-compartment geometries omit it).
    assert geom.parent_mesh is not None
    assert geom.cell_tags is not None and geom.facet_tags is not None
    assert make_disk_geometry("d", volume_subdomain="c").parent_mesh is None


# ---------------------------------------------------------------------------
# 4. cross-check against a multi-compartment MathDescription
# ---------------------------------------------------------------------------


def _model(*, membrane_kind: str = "surface") -> str:
    return f"""
math_description:
  geometry: cell
  subdomains:
    - {{ name: cyto, kind: volume, motion: {{ kind: none }} }}
    - {{ name: ext, kind: volume, motion: {{ kind: none }} }}
    - {{ name: mem, kind: {membrane_kind}, motion: {{ kind: none }} }}
  variables:
    - {{ name: a, subdomain: cyto }}
  equations:
    - template: bulk_radv_diff
      variable: a
      subdomain: cyto
      temporality: time_dependent
      terms: {{ diffusion: "0.1" }}
      initial_condition: "1.0"
"""


def test_cross_validate_accepts_the_matching_model() -> None:
    assert cross_validate(load_yaml(_model()), _geom()) == []


def test_cross_validate_rejects_a_kind_mismatch() -> None:
    # The model declares the membrane a volume, but the geometry provides a surface.
    diagnostics = cross_validate(load_yaml(_model(membrane_kind="volume")), _geom())
    assert any("mem" in d.message and "surface" in d.message for d in diagnostics)

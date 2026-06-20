"""End-to-end inc-0 test: a static bulk-diffusion MathDescription, assembled
through the formalism and solved, plus the geometry cross-check and loader.

This closes the pipeline `YAML → validate → assemble → DiscreteProblem → solve`
for the simplest case. The compiler's inc-0 subset only produces spatially
constant expressions, so the reproduced model is the constant-IC bulk-diffusion
case; richer ICs and physics arrive in later increments.
"""

from __future__ import annotations

import numpy as np
import pytest

from vcell_fenics.backend import (
    TermKind,
    assemble,
    clear_geometries,
    cross_validate,
    load_geometry,
    make_disk_geometry,
    register_geometry,
)
from vcell_fenics.formalism import MathDescription, load_yaml
from vcell_fenics.formalism.schema import Subdomain, Variable

_BULK_DIFFUSION = """
math_description:
  geometry: disk_2d
  subdomains:
    - name: cytoplasm
      kind: volume
  variables:
    - { name: c, subdomain: cytoplasm }
  equations:
    - template: bulk_radv_diff
      variable: c
      subdomain: cytoplasm
      temporality: time_dependent
      terms:
        diffusion: "0.5"
      initial_condition: "3.0"
"""


def _cross_md(*, geometry: str = "disk_2d", kind: str = "volume", subdomain: str = "cytoplasm") -> MathDescription:
    # Minimal MathDescription for exercising cross_validate, which reads only
    # the geometry name and the subdomains.
    return MathDescription(
        geometry=geometry,
        subdomains=[Subdomain(name=subdomain, kind=kind)],  # type: ignore[arg-type]
        variables=[Variable(name="c", subdomain=subdomain)],
        equations=[],
    )


# ---------------------------------------------------------------------------
# End-to-end.
# ---------------------------------------------------------------------------


def test_static_bulk_diffusion_end_to_end() -> None:
    md = load_yaml(_BULK_DIFFUSION)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.2)
    dp = assemble(md, geometry, dt=0.05)

    # The assembler built the expected IR structure...
    assert dp.term_kinds() == {TermKind.TIME_DERIVATIVE, TermKind.DIFFUSION}
    # ...applied the IC from the MathDescription...
    assert np.allclose(dp.unknown.x.array, 3.0)
    # ...and a constant stays constant under no-flux diffusion.
    for _ in range(20):
        dp.step()
    assert np.allclose(dp.unknown.x.array, 3.0, atol=1e-10)


def test_bulk_diffusion_conserves_mass_with_spatial_ic() -> None:
    # A non-uniform IC diffuses, but no-flux (natural Neumann) means ∫c is
    # conserved exactly. (Covers the conservation check of the retired bespoke
    # test_bulk_diffusion_static, now driven through the formalism.)
    spatial_ic = _BULK_DIFFUSION.replace('initial_condition: "3.0"', 'initial_condition: "1.0 + 0.3 * geom.x[0]"')
    md = load_yaml(spatial_ic)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", radius=1.0, h=0.1)
    dp = assemble(md, geometry, dt=0.01)

    mass0 = dp.total_mass()
    for _ in range(50):
        dp.step()
    assert abs(dp.total_mass() - mass0) / abs(mass0) < 1e-10


def test_expression_parameter_binds_against_constants() -> None:
    # The backend binds a ParameterExpression by compiling it against the symbol table (constants,
    # coordinates, time, earlier parameters), not just ParameterConstant — VCell models carry unit
    # factors like KFlux = Area/Volume or UnitFactor = pow(KMOLE, 1) as expression parameters. Here a
    # decay rate k = 2*k0 = 0.5 is an expression parameter; uniform first-order decay must reproduce
    # c0*exp(-k t) (no-flux keeps it spatially uniform), proving k bound to 0.5 (not rejected, not 0).
    from vcell_fenics.formalism.schema import ParameterConstant, ParameterExpression, TemplateEquation

    md = MathDescription(
        geometry="disk_2d",
        subdomains=[Subdomain(name="cyto", kind="volume")],
        variables=[Variable(name="c", subdomain="cyto")],
        parameters=[ParameterConstant(name="k0", value=0.25), ParameterExpression(name="k", expression="2 * k0")],
        equations=[
            TemplateEquation(
                template="bulk_radv_diff",
                variable="c",
                subdomain="cyto",
                temporality="time_dependent",
                terms={"diffusion": "0.1", "source": "-k * c"},
                initial_condition="2.0",
            )
        ],
    )
    dp = assemble(md, make_disk_geometry("disk_2d", volume_subdomain="cyto", radius=1.0, h=0.15), dt=0.02)
    for _ in range(50):  # to t = 1.0
        dp.step()
    assert dp.unknown.x.array.max() == pytest.approx(2.0 * np.exp(-0.5), rel=1e-2)


# ---------------------------------------------------------------------------
# Loader registry.
# ---------------------------------------------------------------------------


def test_geometry_registry_round_trip() -> None:
    clear_geometries()
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.4)
    register_geometry(geometry)
    assert load_geometry("disk_2d") is geometry
    clear_geometries()
    with pytest.raises(KeyError):
        load_geometry("disk_2d")


# ---------------------------------------------------------------------------
# §1.11.10 geometry cross-check.
# ---------------------------------------------------------------------------


def test_cross_validate_accepts_matching_geometry() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.5)
    assert cross_validate(_cross_md(), geometry) == []


def test_cross_validate_rejects_kind_mismatch() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.5)  # provides volume
    errors = [d.message for d in cross_validate(_cross_md(kind="surface"), geometry)]
    assert any("declared kind 'surface'" in m for m in errors)


def test_cross_validate_rejects_missing_subdomain() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.5)
    errors = [d.message for d in cross_validate(_cross_md(subdomain="nucleus"), geometry)]
    assert any("no matching region" in m for m in errors)


def test_cross_validate_rejects_geometry_name_mismatch() -> None:
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.5)
    errors = [d.message for d in cross_validate(_cross_md(geometry="other"), geometry)]
    assert any("references geometry 'other'" in m for m in errors)


# ---------------------------------------------------------------------------
# Subset guard — out-of-subset models fail loudly, not silently.
# ---------------------------------------------------------------------------


def test_relative_advection_slot_is_supported() -> None:
    # `relative_advection` (the Eulerian drift term) is now assembled; full
    # verification of its physics lives in test_backend_advection.
    with_advection = _BULK_DIFFUSION.replace(
        'diffusion: "0.5"', 'diffusion: "0.5"\n        relative_advection: "[1.0, 0.0]"'
    )
    md = load_yaml(with_advection)
    geometry = make_disk_geometry("disk_2d", volume_subdomain="cytoplasm", h=0.4)
    dp = assemble(md, geometry, dt=0.05)
    assert TermKind.ADVECTION in dp.term_kinds()

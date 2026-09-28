"""Template normalizations the backends share.

**T4 `lumped_ode` on a spatial subdomain is a field without transport.** VCell writes a species that does not
diffuse (an immobile buffer, an ER-bound state, a gating variable) as an `OdeEquation` on a spatial
compartment or membrane: it has a value at every point, governed by ``du/dt = rate(u, x, t)`` there, and no
flux between points (§1.4.2 T4; a spatially *constant* unknown is a region variable, T5). That is exactly the
field template with no diffusion or advection and ``source = rate``, so the solvers assemble it through
their existing field machinery: a P1 block whose residual is ``u̇·w − rate·w`` (#186).
"""

from __future__ import annotations

import dataclasses

from vcell_fenics.formalism.schema import SubdomainKind, TemplateEquation

_FIELD_TEMPLATE: dict[SubdomainKind, str] = {"volume": "bulk_radv_diff", "surface": "surface_pde_with_dilution"}


def as_field_equation(equation: TemplateEquation, kind: SubdomainKind) -> TemplateEquation:
    """``equation`` as its field template: a `lumped_ode` on a volume or surface becomes a transport-free
    `bulk_radv_diff` / `surface_pde_with_dilution` with ``source = rate``; anything else is returned as is."""

    if equation.template != "lumped_ode" or kind not in _FIELD_TEMPLATE:
        return equation
    return dataclasses.replace(equation, template=_FIELD_TEMPLATE[kind], terms={"source": equation.terms["rate"]})

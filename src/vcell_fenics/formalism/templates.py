"""Operator-template registry for v1 (T1–T4).

The validator checks each `TemplateEquation` against the spec registered here:
the governed variable's type and home-subdomain kind, the allowed
temporalities, the legal slot names and their value types, and any
template-specific structural rule (T1's "at least one of diffusion / source").
Specs mirror docs/modeling/declarative-formalism.md §1.4.2 and the §1.2.3
subdomain-kind availability table.

`weak_form` is intentionally absent: it is the escape hatch (§1.5), not a
template, and carries a free-form `form` instead of typed slots. The validator
special-cases it rather than looking it up here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vcell_fenics.formalism.schema import SubdomainKind, Temporality, VariableType


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One named term slot of a template. `types` is the set of value types the
    slot expression may produce (e.g. diffusion accepts a scalar or a symmetric
    tensor); `required` is whether the slot must be filled."""

    name: str
    types: frozenset[VariableType]
    required: bool


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    """The structural contract of an operator template (§1.4.2).

    `require_any_of` lists slot names of which at least one must be present for
    the equation to be non-trivial (T1's diffusion/source rule); empty means no
    such constraint.
    """

    name: str
    governed_types: frozenset[VariableType]
    subdomain_kinds: frozenset[SubdomainKind]
    temporalities: frozenset[Temporality]
    slots: tuple[SlotSpec, ...]
    require_any_of: tuple[str, ...] = field(default=())

    def slot(self, name: str) -> SlotSpec | None:
        return next((s for s in self.slots if s.name == name), None)


_SCALAR: frozenset[VariableType] = frozenset({"scalar"})
_SCALAR_OR_TENSOR: frozenset[VariableType] = frozenset({"scalar", "symmetric_tensor"})
_VECTOR: frozenset[VariableType] = frozenset({"vector"})
_ANY_TYPE: frozenset[VariableType] = frozenset({"scalar", "vector", "symmetric_tensor"})
_BOTH_TEMPORALITIES: frozenset[Temporality] = frozenset({"time_dependent", "steady_state"})


# T1 / T2 share the same slot shape (diffusion, relative_advection, source);
# they differ only in the governed subdomain kind and operator (bulk vs
# surface), which is a backend concern, not a schema-validation one.
_RAD_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(name="diffusion", types=_SCALAR_OR_TENSOR, required=False),
    SlotSpec(name="relative_advection", types=_VECTOR, required=False),
    SlotSpec(name="source", types=_SCALAR, required=False),
)
# T1 alone also takes a lab-frame (Eulerian) carrier velocity, `advection`: the species' own velocity in
# the fixed frame, independent of how the mesh moves — VCell's semantics. The backend transports it
# relative to the mesh (c − w) in conservation form with zero total flux at the boundary; on a static
# mesh (w = 0) it is plain advection. Exclusive with `relative_advection`, which is drift relative to the
# substrate (the Lagrangian default: a species on a moving subdomain rides with it).
_BULK_SLOTS: tuple[SlotSpec, ...] = (*_RAD_SLOTS, SlotSpec(name="advection", types=_VECTOR, required=False))


REGISTRY: dict[str, TemplateSpec] = {
    "bulk_radv_diff": TemplateSpec(  # T1
        name="bulk_radv_diff",
        governed_types=_SCALAR,
        subdomain_kinds=frozenset({"volume"}),
        temporalities=_BOTH_TEMPORALITIES,
        slots=_BULK_SLOTS,
        require_any_of=("diffusion", "source"),
    ),
    "surface_pde_with_dilution": TemplateSpec(  # T2
        name="surface_pde_with_dilution",
        governed_types=_SCALAR,
        subdomain_kinds=frozenset({"surface"}),
        temporalities=_BOTH_TEMPORALITIES,
        slots=_RAD_SLOTS,
    ),
    "algebraic_constraint": TemplateSpec(  # T3
        name="algebraic_constraint",
        governed_types=_ANY_TYPE,
        subdomain_kinds=frozenset({"volume", "surface", "point"}),
        temporalities=frozenset({"steady_state"}),
        slots=(SlotSpec(name="constraint", types=_SCALAR, required=True),),
    ),
    "lumped_ode": TemplateSpec(  # T4
        name="lumped_ode",
        governed_types=_SCALAR,
        subdomain_kinds=frozenset({"volume", "surface", "point"}),
        temporalities=_BOTH_TEMPORALITIES,
        slots=(SlotSpec(name="rate", types=_SCALAR, required=True),),
    ),
    "cahn_hilliard": TemplateSpec(  # diffuse-interface phase separation
        # A 4th-order conserved order parameter ∂φ/∂t = ∇·(M∇μ), μ = f'(φ) − ε²∇²φ, with the
        # standard double-well f = W φ²(1−φ)². The backend expands it into a mixed (φ, μ) system
        # (backend/cahn_hilliard.py) — the auxiliary chemical potential μ stays internal, so the
        # modeller declares only φ and the three physical scales. Always time-dependent.
        name="cahn_hilliard",
        governed_types=_SCALAR,
        subdomain_kinds=frozenset({"volume"}),
        temporalities=frozenset({"time_dependent"}),
        slots=(
            SlotSpec(name="mobility", types=_SCALAR, required=False),  # M
            SlotSpec(name="interface_width", types=_SCALAR, required=False),  # ε
            SlotSpec(name="well_height", types=_SCALAR, required=False),  # W
        ),
    ),
}

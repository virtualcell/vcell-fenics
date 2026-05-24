"""Validation pass over a MathDescription (docs/modeling/declarative-formalism.md §2.5).

This module implements the *structural* checks of §1.11 — the ones that read
the dataclass tree's declared fields. It does **not** yet parse or walk the
expression strings; the checks that depend on the expression AST (name
resolution inside expressions, parameter-expression cycles, slot type-checking,
weak-form `partial_t` presence, the operator-usage narrow/smoothness rules, and
IC-content rules) are a following increment, as is the geometry cross-check
(§1.11.10), which needs a Geometry object the MathDescription does not carry.

Checks implemented here:

- Reserved-name shadowing and duplicate / colliding declarations (§1.11.3).
- Structural reference resolution — subdomains, governed variables, motion
  variables, BC variables (§1.11.3).
- Coverage — every variable governed by exactly one equation; the IC-presence
  ⇔ temporality rule, with the unknown-motion-variable exception (§1.11.4).
- Template conformance — known template, allowed temporality / subdomain kind /
  governed type, legal and required slots, T1's "at least one of
  diffusion / source" (§1.4.2).
- Boundary-condition consistency — conflicting kinds, interface-partner
  declaration, `interface_flux_balance` bulk-only, weak-form Dirichlet-only
  (§1.11.7).

Diagnostics are collected (not fail-fast) so a single pass reports every
structural problem in a large model. `validate` returns all of them;
`validate_or_raise` raises `FormalismValidationError` if any is an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vcell_fenics.formalism.schema import (
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    BoundaryCondition,
    Equation,
    MathDescription,
    MotionUnknown,
    ParameterExpression,
    ParameterRegionMap,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
)
from vcell_fenics.formalism.templates import REGISTRY
from vcell_fenics.formalism.vocabulary import RESERVED_NAMES

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One validation finding. `path` is an attribute path into the
    MathDescription (e.g. `equations[1].terms`); empty for whole-model
    findings."""

    severity: Severity
    path: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}: " if self.path else ""
        return f"[{self.severity}] {where}{self.message}"


class FormalismValidationError(Exception):
    """Raised by `validate_or_raise` when a MathDescription has validation
    errors. `errors` holds the error-severity diagnostics; the str() lists
    them one per line."""

    def __init__(self, errors: list[Diagnostic]) -> None:
        self.errors = errors
        body = "\n".join(f"  {d}" for d in errors)
        super().__init__(f"MathDescription failed validation with {len(errors)} error(s):\n{body}")


def validate(md: MathDescription) -> list[Diagnostic]:
    """Run the structural validation pass and return all diagnostics
    (errors and warnings), in document order."""

    return _Validator(md).run()


def validate_or_raise(md: MathDescription) -> list[Diagnostic]:
    """Validate and raise `FormalismValidationError` if any error is found.
    Returns the (possibly empty) list of warning diagnostics on success."""

    diagnostics = validate(md)
    errors = [d for d in diagnostics if d.severity == "error"]
    if errors:
        raise FormalismValidationError(errors)
    return [d for d in diagnostics if d.severity == "warning"]


class _Validator:
    def __init__(self, md: MathDescription) -> None:
        self._md = md
        self._diagnostics: list[Diagnostic] = []

        # Indices, built with duplicate detection.
        self._subdomain_by_name: dict[str, Subdomain] = {}
        self._var_by_key: dict[tuple[str, str], Variable] = {}
        self._var_subdomains: dict[str, set[str]] = {}
        self._param_names: set[str] = set()
        # (variable, subdomain) pairs whose subdomain has unknown motion driven
        # by that variable — the IC-on-steady-state exception (§1.7.7).
        self._unknown_motion_vars: set[tuple[str, str]] = set()

    # -- diagnostics ---------------------------------------------------------

    def _error(self, path: str, message: str) -> None:
        self._diagnostics.append(Diagnostic("error", path, message))

    # -- driver --------------------------------------------------------------

    def run(self) -> list[Diagnostic]:
        self._build_indices()
        self._check_reserved_and_collisions()
        self._check_parameter_subdomain_refs()
        self._check_motion_variables()
        self._check_equations()
        self._check_boundary_conditions()
        return self._diagnostics

    def _build_indices(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if s.name in self._subdomain_by_name:
                self._error(f"subdomains[{i}]", f"duplicate subdomain name {s.name!r}")
            else:
                self._subdomain_by_name[s.name] = s

        for i, v in enumerate(self._md.variables):
            key = (v.name, v.subdomain)
            if key in self._var_by_key:
                self._error(f"variables[{i}]", f"duplicate variable {v.name!r} on subdomain {v.subdomain!r}")
            else:
                self._var_by_key[key] = v
            self._var_subdomains.setdefault(v.name, set()).add(v.subdomain)

        for i, p in enumerate(self._md.parameters):
            if p.name in self._param_names:
                self._error(f"parameters[{i}]", f"duplicate parameter name {p.name!r}")
            else:
                self._param_names.add(p.name)

        for s in self._md.subdomains:
            if isinstance(s.motion, MotionUnknown):
                self._unknown_motion_vars.add((s.motion.variable, s.name))

    # -- §1.11.3 reserved names, shadowing, collisions -----------------------

    def _check_reserved_and_collisions(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if s.name in RESERVED_NAMES:
                self._error(f"subdomains[{i}]", f"subdomain name {s.name!r} is a reserved name (§2.4.1)")
        for i, v in enumerate(self._md.variables):
            if v.name in RESERVED_NAMES:
                self._error(f"variables[{i}]", f"variable name {v.name!r} is a reserved name (§2.4.1)")
            if v.subdomain not in self._subdomain_by_name:
                self._error(f"variables[{i}]", f"references undeclared subdomain {v.subdomain!r}")
        for i, p in enumerate(self._md.parameters):
            if p.name in RESERVED_NAMES:
                self._error(f"parameters[{i}]", f"parameter name {p.name!r} is a reserved name (§2.4.1)")
            elif p.name in self._var_subdomains:
                self._error(
                    f"parameters[{i}]",
                    f"parameter {p.name!r} shadows a variable of the same name; shadowing is an error (§1.8.6)",
                )

    def _check_parameter_subdomain_refs(self) -> None:
        for i, p in enumerate(self._md.parameters):
            scope: str | None = None
            if isinstance(p, ParameterExpression | ParameterRegionMap):
                scope = p.subdomain
            if scope is not None and scope not in self._subdomain_by_name:
                self._error(f"parameters[{i}]", f"references undeclared subdomain {scope!r}")

    def _check_motion_variables(self) -> None:
        for i, s in enumerate(self._md.subdomains):
            if not isinstance(s.motion, MotionUnknown):
                continue
            key = (s.motion.variable, s.name)
            var = self._var_by_key.get(key)
            if var is None:
                self._error(
                    f"subdomains[{i}].motion",
                    f"motion variable {s.motion.variable!r} is not a declared variable on subdomain {s.name!r}",
                )
            elif var.type != "vector":
                self._error(
                    f"subdomains[{i}].motion",
                    f"motion variable {s.motion.variable!r} must be vector-typed (§1.10.3), got {var.type!r}",
                )

    # -- §1.11.4 coverage + §1.4.2 template conformance ----------------------

    def _check_equations(self) -> None:
        governed_count: dict[tuple[str, str], int] = {}

        for i, eq in enumerate(self._md.equations):
            path = f"equations[{i}]"
            subdomain = self._subdomain_by_name.get(eq.subdomain)
            if subdomain is None:
                self._error(path, f"references undeclared subdomain {eq.subdomain!r}")

            key = (eq.variable, eq.subdomain)
            var = self._var_by_key.get(key)
            if var is None:
                self._error(path, f"governs variable {eq.variable!r}, not declared on subdomain {eq.subdomain!r}")
            governed_count[key] = governed_count.get(key, 0) + 1

            self._check_ic_temporality(eq, key, path)

            if not isinstance(eq, WeakFormEquation):
                self._check_template_equation(eq, var, subdomain, path)

        for v in self._md.variables:
            count = governed_count.get((v.name, v.subdomain), 0)
            where = f"variable {v.name!r} on subdomain {v.subdomain!r}"
            if count == 0:
                self._error("equations", f"{where} is not governed by any equation (undetermined)")
            elif count > 1:
                self._error("equations", f"{where} is governed by {count} equations (overdetermined)")

    def _check_ic_temporality(self, eq: Equation, key: tuple[str, str], path: str) -> None:
        if eq.temporality == "time_dependent":
            if eq.initial_condition is None:
                self._error(path, "time_dependent equation requires an initial_condition (§1.9.5)")
        elif eq.initial_condition is not None and key not in self._unknown_motion_vars:
            self._error(
                path,
                "steady_state equation must not have an initial_condition (§1.9.5); the only exception is an "
                "unknown-motion variable, whose IC sets the t=0 configuration (§1.7.7)",
            )

    def _check_template_equation(
        self, eq: TemplateEquation, var: Variable | None, subdomain: Subdomain | None, path: str
    ) -> None:
        template = eq.template
        terms = eq.terms
        temporality = eq.temporality
        spec = REGISTRY.get(template)
        if spec is None:
            options = sorted(REGISTRY)
            self._error(path, f"unknown template {template!r} (v1 templates: {options}, or 'weak_form')")
            return

        if temporality not in spec.temporalities:
            self._error(
                path,
                f"template {template!r} does not support temporality {temporality!r} "
                f"(allowed: {sorted(spec.temporalities)})",
            )
        if subdomain is not None and subdomain.kind not in spec.subdomain_kinds:
            self._error(
                path,
                f"template {template!r} requires subdomain kind in {sorted(spec.subdomain_kinds)}; "
                f"subdomain {subdomain.name!r} is {subdomain.kind!r}",
            )
        if var is not None and var.type not in spec.governed_types:
            self._error(
                path,
                f"template {template!r} governs a {sorted(spec.governed_types)} variable; {var.name!r} is {var.type!r}",
            )

        valid_slots = {s.name for s in spec.slots}
        for slot_name in terms:
            if slot_name not in valid_slots:
                self._error(
                    f"{path}.terms",
                    f"unknown slot {slot_name!r} for template {template!r} (slots: {sorted(valid_slots)})",
                )
        for slot in spec.slots:
            if slot.required and slot.name not in terms:
                self._error(f"{path}.terms", f"template {template!r} requires slot {slot.name!r}")
        if spec.require_any_of and not any(name in terms for name in spec.require_any_of):
            self._error(
                f"{path}.terms",
                f"template {template!r} requires at least one of {list(spec.require_any_of)}",
            )

    # -- §1.11.7 boundary-condition consistency ------------------------------

    def _check_boundary_conditions(self) -> None:
        weak_form_vars = {(eq.variable, eq.subdomain) for eq in self._md.equations if isinstance(eq, WeakFormEquation)}
        groups: dict[tuple[str, str], set[str]] = {}

        for i, bc in enumerate(self._md.boundary_conditions):
            path = f"boundary_conditions[{i}]"
            if bc.variable not in self._var_subdomains:
                self._error(path, f"references undeclared variable {bc.variable!r}")
            if (
                isinstance(bc, BCInterfaceValueEquality | BCInterfaceFluxBalance)
                and bc.partner_variable not in self._var_subdomains
            ):
                self._error(path, f"partner_variable {bc.partner_variable!r} is not declared")
            if isinstance(bc, BCInterfaceFluxBalance):
                self._check_flux_balance_bulk_only(bc, path)
            self._check_weak_form_dirichlet_only(bc, path, weak_form_vars)
            groups.setdefault((bc.variable, bc.boundary), set()).add(bc.kind)

        for (variable, boundary), kinds in groups.items():
            if len(kinds) > 1:
                self._error(
                    "boundary_conditions",
                    f"conflicting BC kinds {sorted(kinds)} for variable {variable!r} on boundary {boundary!r}",
                )

    def _check_flux_balance_bulk_only(self, bc: BCInterfaceFluxBalance, path: str) -> None:
        for role, name in (("variable", bc.variable), ("partner_variable", bc.partner_variable)):
            hosts = self._var_subdomains.get(name)
            if not hosts:
                continue
            kinds = {self._subdomain_by_name[s].kind for s in hosts if s in self._subdomain_by_name}
            if kinds and "volume" not in kinds:
                self._error(
                    path,
                    f"interface_flux_balance requires {role} {name!r} to live on a volume subdomain; it lives on "
                    f"{sorted(kinds)}. Bulk-surface coupling uses the §1.6.5 Neumann + source pattern instead",
                )

    def _check_weak_form_dirichlet_only(
        self, bc: BoundaryCondition, path: str, weak_form_vars: set[tuple[str, str]]
    ) -> None:
        variable = bc.variable
        hosts = self._var_subdomains.get(variable)
        if not hosts or bc.kind == "dirichlet":
            return
        # Only flag when the variable is weak-form-governed on *every* subdomain
        # it lives on; otherwise the offending subdomain is ambiguous without a
        # Geometry, and a template-governed home could legitimately take this BC.
        if all((variable, s) in weak_form_vars for s in hosts):
            self._error(
                path,
                f"variable {variable!r} is governed by a weak-form equation; only Dirichlet BCs are allowed on it "
                f"(§1.5.6) — encode natural BCs in the form itself",
            )

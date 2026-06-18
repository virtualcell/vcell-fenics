"""MathDescription → YAML / JSON dumper.

Inverse of :mod:`vcell_fenics.formalism.loader`. The contract is round-trip
identity: ``load_dict(to_dict(md)) == md`` for any well-formed MathDescription.

Defaults are omitted on output: ``Variable(type="scalar", space="lagrange_p1")``
serialises to ``{name, subdomain}`` without the noise of redundant default
fields. The loader re-applies the same defaults on the way back in.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFluxBalance,
    BCInterfaceValueEquality,
    BCNeumann,
    BCRobin,
    BoundaryCondition,
    Equation,
    MathDescription,
    Motion,
    MotionNone,
    MotionPrescribedDisplacement,
    MotionPrescribedVelocity,
    MotionUnknown,
    Parameter,
    ParameterConstant,
    ParameterExpression,
    ParameterRegionMap,
    Subdomain,
    TemplateEquation,
    Variable,
    WeakFormEquation,
)

# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def to_dict(md: MathDescription) -> dict[str, object]:
    """Serialise a MathDescription to a JSON/YAML-ready nested dict.

    Defaults are omitted. The result satisfies ``load_dict(to_dict(md)) == md``.
    """

    return {"math_description": _math_description_to_dict(md)}


def dump_yaml(md: MathDescription, path: str | Path | None = None) -> str:
    """Emit a MathDescription as YAML text. If `path` is given, also writes
    to that file. The YAML uses block style for readability and preserves
    key insertion order (matching the doc's worked examples).
    """

    text = yaml.safe_dump(
        to_dict(md),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def dump_json(md: MathDescription, path: str | Path | None = None, *, indent: int = 2) -> str:
    """Emit a MathDescription as JSON text. If `path` is given, also writes
    to that file.
    """

    text = json.dumps(to_dict(md), indent=indent, ensure_ascii=False)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Per-entity serialisers. Each emits an insertion-ordered dict that round-trips
# cleanly through the loader.
# ---------------------------------------------------------------------------


def _math_description_to_dict(md: MathDescription) -> dict[str, object]:
    out: dict[str, object] = {
        "geometry": md.geometry,
        "subdomains": [_subdomain_to_dict(s) for s in md.subdomains],
        "variables": [_variable_to_dict(v) for v in md.variables],
        "equations": [_equation_to_dict(e) for e in md.equations],
    }
    # Defaults omitted: empty parameter / BC lists do not need to appear.
    if md.parameters:
        out["parameters"] = [_parameter_to_dict(p) for p in md.parameters]
    if md.boundary_conditions:
        out["boundary_conditions"] = [_bc_to_dict(b) for b in md.boundary_conditions]
    return out


def _subdomain_to_dict(s: Subdomain) -> dict[str, object]:
    out: dict[str, object] = {"name": s.name, "kind": s.kind}
    if not isinstance(s.motion, MotionNone):
        out["motion"] = _motion_to_dict(s.motion)
    return out


def _motion_to_dict(m: Motion) -> dict[str, object]:
    if isinstance(m, MotionNone):
        return {"kind": "none"}
    if isinstance(m, MotionPrescribedVelocity):
        return {"kind": "prescribed", "velocity": m.velocity}
    if isinstance(m, MotionPrescribedDisplacement):
        return {"kind": "prescribed", "displacement": m.displacement}
    if isinstance(m, MotionUnknown):
        return {"kind": "unknown", "variable": m.variable}
    raise AssertionError(f"unhandled Motion variant: {type(m).__name__}")


def _variable_to_dict(v: Variable) -> dict[str, object]:
    out: dict[str, object] = {"name": v.name, "subdomain": v.subdomain}
    if v.type != "scalar":
        out["type"] = v.type
    if v.space != "lagrange_p1":
        out["space"] = v.space
    return out


def _parameter_to_dict(p: Parameter) -> dict[str, object]:
    if isinstance(p, ParameterConstant):
        # Constant uses the {name, value} shorthand the loader recognises.
        # The explicit `kind: scalar` is redundant and omitted.
        return {"name": p.name, "value": p.value}
    if isinstance(p, ParameterExpression):
        out: dict[str, object] = {"name": p.name}
        if p.type != "scalar":
            out["type"] = p.type
        if p.subdomain is not None:
            out["subdomain"] = p.subdomain
        out["expression"] = p.expression
        return out
    if isinstance(p, ParameterRegionMap):
        return {
            "name": p.name,
            "kind": "region_map",
            "subdomain": p.subdomain,
            "values": dict(p.values),
        }
    raise AssertionError(f"unhandled Parameter variant: {type(p).__name__}")


def _equation_to_dict(e: Equation) -> dict[str, object]:
    if isinstance(e, WeakFormEquation):
        out: dict[str, object] = {
            "template": "weak_form",
            "variable": e.variable,
            "subdomain": e.subdomain,
            "temporality": e.temporality,
            "form": e.form,
        }
        if e.initial_condition is not None:
            out["initial_condition"] = e.initial_condition
        return out
    if isinstance(e, TemplateEquation):
        out = {
            "template": e.template,
            "variable": e.variable,
            "subdomain": e.subdomain,
            "temporality": e.temporality,
        }
        if e.terms:
            out["terms"] = dict(e.terms)
        if e.initial_condition is not None:
            out["initial_condition"] = e.initial_condition
        return out
    raise AssertionError(f"unhandled Equation variant: {type(e).__name__}")


def _bc_to_dict(b: BoundaryCondition) -> dict[str, object]:
    if isinstance(b, BCDirichlet):
        return {
            "kind": "dirichlet",
            "variable": b.variable,
            "boundary": b.boundary,
            "expression": b.expression,
        }
    if isinstance(b, BCNeumann):
        return {
            "kind": "neumann",
            "variable": b.variable,
            "boundary": b.boundary,
            "expression": b.expression,
        }
    if isinstance(b, BCRobin):
        return {
            "kind": "robin",
            "variable": b.variable,
            "boundary": b.boundary,
            "alpha": b.alpha,
            "beta": b.beta,
            "expression": b.expression,
        }
    if isinstance(b, BCInterfaceValueEquality):
        out: dict[str, object] = {
            "kind": "interface_value_equality",
            "variable": b.variable,
            "partner_variable": b.partner_variable,
            "boundary": b.boundary,
        }
        # Default partition coefficient "1" is omitted on output; the loader
        # re-applies it.
        if b.expression != "1":
            out["expression"] = b.expression
        return out
    if isinstance(b, BCInterfaceFluxBalance):
        return {
            "kind": "interface_flux_balance",
            "variable": b.variable,
            "partner_variable": b.partner_variable,
            "boundary": b.boundary,
            "expression": b.expression,
        }
    raise AssertionError(f"unhandled BoundaryCondition variant: {type(b).__name__}")

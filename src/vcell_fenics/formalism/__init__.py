"""Declarative formalism for cell-biology PDE/ODE systems.

Public surface for v1 (only the schema dataclasses ship so far; parser,
loader, validator, and backend translation are forthcoming):

>>> from vcell_fenics.formalism import MathDescription, Subdomain, Variable

The schema mirrors docs/modeling/declarative-formalism.md Part 2. See that
document for surface syntax, semantics, and design decisions; this package
is the in-memory representation those decisions resolve into.
"""

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
    SubdomainKind,
    TemplateEquation,
    Temporality,
    Variable,
    VariableType,
    WeakFormEquation,
)

__all__ = [
    "BCDirichlet",
    "BCInterfaceFluxBalance",
    "BCInterfaceValueEquality",
    "BCNeumann",
    "BCRobin",
    "BoundaryCondition",
    "Equation",
    "MathDescription",
    "Motion",
    "MotionNone",
    "MotionPrescribedDisplacement",
    "MotionPrescribedVelocity",
    "MotionUnknown",
    "Parameter",
    "ParameterConstant",
    "ParameterExpression",
    "ParameterRegionMap",
    "Subdomain",
    "SubdomainKind",
    "TemplateEquation",
    "Temporality",
    "Variable",
    "VariableType",
    "WeakFormEquation",
]

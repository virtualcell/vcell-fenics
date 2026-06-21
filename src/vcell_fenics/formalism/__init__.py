"""Declarative formalism for cell-biology PDE/ODE systems.

Public surface for v1 (schema dataclasses + YAML/JSON loader and dumper
ship now; parser, validator, and backend translation are forthcoming):

>>> from vcell_fenics.formalism import MathDescription, load_yaml, dump_yaml

The schema mirrors docs/modeling/declarative-formalism.md Part 2. See that
document for surface syntax, semantics, and design decisions; this package
is the in-memory representation those decisions resolve into.
"""

from vcell_fenics.formalism.dumper import dump_json, dump_yaml, to_dict
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
from vcell_fenics.formalism.geometry_io import (
    dump_geometry_json,
    dump_geometry_yaml,
    geometry_to_dict,
    load_geometry_dict,
    load_geometry_json,
    load_geometry_yaml,
)
from vcell_fenics.formalism.geometry_schema import (
    GeometryDescription,
    GeometryImage,
    PixelClass,
    SubVolume,
    SubVolumeType,
    SurfaceClass,
)
from vcell_fenics.formalism.geometry_validator import validate_geometry, validate_geometry_or_raise
from vcell_fenics.formalism.loader import FormalismLoadError, load_dict, load_json, load_yaml
from vcell_fenics.formalism.parser import ExpressionSyntaxError, parse
from vcell_fenics.formalism.rvachev import (
    RvachevLoweringError,
    is_boolean,
    lower_predicate,
    subvolume_implicit_functions,
)
from vcell_fenics.formalism.schema import (
    BCDirichlet,
    BCInterfaceFlux,
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
from vcell_fenics.formalism.validator import (
    Diagnostic,
    FormalismValidationError,
    validate,
    validate_or_raise,
)

__all__ = [
    "BCDirichlet",
    "BCInterfaceFlux",
    "BCInterfaceValueEquality",
    "BCNeumann",
    "BCRobin",
    "BinaryOp",
    "BoundaryCondition",
    "Diagnostic",
    "Equation",
    "Expr",
    "ExpressionSyntaxError",
    "FormalismLoadError",
    "FormalismValidationError",
    "FunctionCall",
    "GeometryDescription",
    "GeometryImage",
    "IndexAccess",
    "MathDescription",
    "Motion",
    "MotionNone",
    "MotionPrescribedDisplacement",
    "MotionPrescribedVelocity",
    "MotionUnknown",
    "Name",
    "Number",
    "Parameter",
    "ParameterConstant",
    "ParameterExpression",
    "ParameterRegionMap",
    "PixelClass",
    "RvachevLoweringError",
    "SubVolume",
    "SubVolumeType",
    "Subdomain",
    "SubdomainKind",
    "SurfaceClass",
    "TemplateEquation",
    "Temporality",
    "TensorLiteral",
    "UnaryOp",
    "Variable",
    "VariableType",
    "VectorLiteral",
    "WeakFormEquation",
    "dump_geometry_json",
    "dump_geometry_yaml",
    "dump_json",
    "dump_yaml",
    "geometry_to_dict",
    "is_boolean",
    "load_dict",
    "load_geometry_dict",
    "load_geometry_json",
    "load_geometry_yaml",
    "load_json",
    "load_yaml",
    "lower_predicate",
    "parse",
    "subvolume_implicit_functions",
    "to_dict",
    "validate",
    "validate_geometry",
    "validate_geometry_or_raise",
    "validate_or_raise",
]

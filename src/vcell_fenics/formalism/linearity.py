"""Which of a model's expressions are nonlinear in its unknowns (ADR 021 decision 2).

A pure function of the `MathDescription`, the same for every route, decided before anything is assembled. The
runner uses it to choose the time scheme: backward Euler assembles only terms it can split into a part linear in
the unknowns and a part free of them (UFL's `lhs`/`rhs`), so a model with any other expression runs by the method
of lines. The rules follow what that split accepts, not affinity alone: `u**1` is affine but holds the unknown
inside a power, which the split rejects, so it is nonlinear here.

**The unknowns** are the model's variables: species and region variables. A variable named by an
``unknown`` motion (the velocity of a cell force balance, or of an unknown-motion membrane) is excluded: it is
solved in its own staggered step and read lagged by the others (ADR 020 decision 3). Parameters cannot read a
variable (§1.8.3, the validator), so no closure through parameters is needed.

**Every expression that enters the time-stepped forms is checked:** each equation's slots, and the Neumann, Robin
and interface-flux conditions. A Dirichlet value is imposed, not assembled, and is compiled without the species on
every route, so it is not checked.
- A transport slot (``diffusion``, ``advection``, ``relative_advection``) multiplies ``∇u`` or ``u``, so it is
  nonlinear as soon as it reads an unknown.
- Any other expression is nonlinear unless it is affine in the unknowns: sums, products with at most one factor
  that reads an unknown, a quotient whose denominator reads none, and the linear operators (``grad``, ``div``,
  the Laplacians, ``partial_t``, ``trace``). An unknown inside any other function (``exp``, ``max``, a
  conditional, a comparison) or a power is nonlinear, and so is a vector or tensor literal in which some
  components read an unknown and others do not (UFL cannot split such a literal).

`trace(X)` reads `X`.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from vcell_fenics.formalism.parser import parse
from vcell_fenics.formalism.schema import (
    BCInterfaceFlux,
    BCNeumann,
    BCRobin,
    MathDescription,
    MotionUnknown,
    TemplateEquation,
)
from vcell_fenics.formalism.vocabulary import CALCULUS_OPERATORS, TENSOR_ALGEBRA, TIME_DERIVATIVE, TRACE

TRANSPORT_SLOTS = frozenset({"diffusion", "advection", "relative_advection"})

# operators linear in their (single) argument
_LINEAR_CALLS = CALCULUS_OPERATORS | TRACE | TIME_DERIVATIVE
# bilinear in their two arguments, like a product
_PRODUCT_CALLS = TENSOR_ALGEBRA


@dataclass(frozen=True)
class NonlinearExpression:
    """Where a nonlinear expression is: ``equation 'm' slot 'diffusion'`` or ``boundary condition …``."""

    where: str
    expression: str

    def __str__(self) -> str:
        return f"{self.where}: {self.expression}"


def unknowns(md: MathDescription) -> frozenset[str]:
    """The variables time-stepped together: every variable but those an ``unknown`` motion names."""

    lagged = {sd.motion.variable for sd in md.subdomains if isinstance(sd.motion, MotionUnknown)}
    return frozenset(v.name for v in md.variables) - lagged


def reads(node: Expr, names: frozenset[str]) -> bool:
    """Whether `node` reads any of `names`."""

    if isinstance(node, Name):
        return node.name in names
    if isinstance(node, Number):
        return False
    if isinstance(node, IndexAccess):
        return reads(node.base, names) or reads(node.index, names)
    if isinstance(node, FunctionCall):
        return any(reads(arg, names) for arg in node.args)
    if isinstance(node, UnaryOp):
        return reads(node.operand, names)
    if isinstance(node, BinaryOp):
        return reads(node.left, names) or reads(node.right, names)
    if isinstance(node, VectorLiteral):
        return any(reads(c, names) for c in node.components)
    if isinstance(node, TensorLiteral):
        return any(reads(row, names) for row in node.rows)
    raise TypeError(f"unknown expression node {node!r}")


def affine(node: Expr, names: frozenset[str]) -> bool:
    """Whether `node` is affine in `names` (the module docstring's rules)."""

    if not reads(node, names):
        return True
    if isinstance(node, Name):
        return True
    if isinstance(node, IndexAccess):
        return not reads(node.index, names) and affine(node.base, names)
    if isinstance(node, UnaryOp):
        return node.op in ("+", "-") and affine(node.operand, names)
    if isinstance(node, BinaryOp):
        left, right = node.left, node.right
        if node.op in ("+", "-"):
            return affine(left, names) and affine(right, names)
        if node.op == "*":
            return affine(left, names) and affine(right, names) and not (reads(left, names) and reads(right, names))
        if node.op == "/":
            return affine(left, names) and not reads(right, names)
        return False  # a power, a comparison or a logical operator of an unknown
    if isinstance(node, FunctionCall):
        args = node.args
        if node.callee in _LINEAR_CALLS:
            return all(affine(a, names) for a in args)
        if node.callee in _PRODUCT_CALLS:
            return all(affine(a, names) for a in args) and sum(reads(a, names) for a in args) <= 1
        return False  # an unknown inside exp, max, pow, if, …
    if isinstance(node, VectorLiteral):
        return _literal_splits(node.components, names)
    if isinstance(node, TensorLiteral):
        return _literal_splits(tuple(c for row in node.rows for c in row.components), names)
    return False


def _literal_splits(components: tuple[Expr, ...], names: frozenset[str]) -> bool:
    """A literal UFL can split: every component affine, and either all or none of them read an unknown."""

    return all(affine(c, names) for c in components) and len({reads(c, names) for c in components}) == 1


def nonlinear_expressions(md: MathDescription) -> list[NonlinearExpression]:
    """Every expression of `md` nonlinear in its unknowns (the module docstring), in document order."""

    names = unknowns(md)
    found: list[NonlinearExpression] = []

    def check(where: str, text: str, *, transport: bool = False) -> None:
        node = parse(text)
        if reads(node, names) if transport else not affine(node, names):
            found.append(NonlinearExpression(where, text))

    for eq in md.equations:
        if not isinstance(eq, TemplateEquation) or eq.variable not in names:
            continue
        for slot, text in eq.terms.items():
            check(f"equation {eq.variable!r} slot {slot!r}", text, transport=slot in TRANSPORT_SLOTS)
    for bc in md.boundary_conditions:
        if bc.variable not in names:
            continue
        where = f"{bc.kind} condition on {bc.variable!r} at {bc.boundary!r}"
        if isinstance(bc, BCRobin):
            check(f"{where} (alpha)", bc.alpha, transport=True)  # multiplies u
            check(f"{where} (beta)", bc.beta, transport=True)  # multiplies the normal flux
            check(where, bc.expression)
        elif isinstance(bc, BCNeumann | BCInterfaceFlux):
            check(where, bc.expression)
    return found


__all__ = ["TRANSPORT_SLOTS", "NonlinearExpression", "affine", "nonlinear_expressions", "reads", "unknowns"]

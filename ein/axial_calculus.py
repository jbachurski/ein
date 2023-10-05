import abc
from dataclasses import dataclass
from typing import TypeAlias

from .calculus import Index, Value, Variable


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    pass


Expr: TypeAlias = (
    "Sum | Get | Const | Range | Var | VarShape | Negate | Reciprocal | Add | Multiply"
)


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpr):
    index: Index
    body: Expr


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    target: Expr
    item: Expr
    axis: Index


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    index: Index
    size: Expr


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable


@dataclass(frozen=True, eq=False)
class VarShape(AbstractExpr):
    var: Variable
    axis: int


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    operands: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class Negate(AbstractScalarOperator):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Reciprocal(AbstractScalarOperator):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Add(AbstractScalarOperator):
    operands: tuple[Expr, Expr]


@dataclass(frozen=True, eq=False)
class Multiply(AbstractScalarOperator):
    operands: tuple[Expr, Expr]


__all__ = [
    "Index",
    "Variable",
    "Value",
    "Expr",
    "Sum",
    "Get",
    "Const",
    "Range",
    "Var",
    "VarShape",
    "Negate",
    "Reciprocal",
    "Add",
    "Multiply",
]

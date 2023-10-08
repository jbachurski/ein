import abc
from dataclasses import dataclass
from typing import TypeAlias

from .calculus import Index, Value, Variable


@dataclass(frozen=True)
class ValueAxis(Index):
    value: Value
    rank: int


@dataclass(frozen=True)
class VariableAxis(Index):
    variable: Variable
    rank: int


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    pass


Expr: TypeAlias = (
    "Sum | Get | Const | Range | Var | Dim | Negate | Reciprocal | Add | Multiply"
)


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpr):
    axis: Index
    body: Expr


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    operand: Expr
    item: Expr
    axis: Index


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    axis: Index
    size: Expr


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    operand: Expr
    axis: Index


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
    "Dim",
    "Negate",
    "Reciprocal",
    "Add",
    "Multiply",
]

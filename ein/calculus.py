import abc
from dataclasses import dataclass
from typing import TypeAlias

import numpy


class Index:
    pass


class Variable:
    pass


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    pass


Expr: TypeAlias = (
    "Vec | Sum | Get | Const | At | Var | Shape | Negate | Reciprocal | Add | Multiply"
)


@dataclass(frozen=True, eq=False)
class Value:
    array: numpy.ndarray


@dataclass(frozen=True, eq=False)
class Vec(AbstractExpr):
    index: Index
    size: Expr
    body: Expr


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpr):
    index: Index
    size: Expr
    body: Expr


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    operand: Expr
    item: Expr


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value


@dataclass(frozen=True, eq=False)
class At(AbstractExpr):
    index: Index


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable


@dataclass(frozen=True, eq=False)
class Shape(AbstractExpr):
    operand: Expr
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
    "Vec",
    "Sum",
    "Get",
    "Const",
    "At",
    "Var",
    "Shape",
    "Negate",
    "Reciprocal",
    "Add",
    "Multiply",
]

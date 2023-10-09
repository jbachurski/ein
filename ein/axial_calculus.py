import abc
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import numpy

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
    "Sum | Maximum | Get | Const | Range | Var | Dim | Switch | "
    "Negate | Reciprocal | LogicalNot | Add | Multiply | Less | LogicalAnd"
)


@dataclass(frozen=True, eq=False)
class AbstractScalarReduction(AbstractExpr):
    axis: Index
    body: Expr

    @property
    @abc.abstractmethod
    def ufunc(self) -> numpy.ufunc:
        ...


@dataclass(frozen=True, eq=False)
class Sum(AbstractScalarReduction):
    @property
    def ufunc(self) -> numpy.ufunc:
        return numpy.add


@dataclass(frozen=True, eq=False)
class Maximum(AbstractScalarReduction):
    @property
    def ufunc(self) -> numpy.ufunc:
        return numpy.maximum


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
class Switch(AbstractExpr):
    cond: Expr
    true: Expr
    false: Expr


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    operands: tuple[Expr, ...]
    ufunc: ClassVar[numpy.ufunc]


@dataclass(frozen=True, eq=False)
class AbstractUnaryScalarOperator(AbstractScalarOperator, abc.ABC):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Negate(AbstractUnaryScalarOperator):
    ufunc = numpy.negative


@dataclass(frozen=True, eq=False)
class Reciprocal(AbstractUnaryScalarOperator):
    ufunc = numpy.reciprocal


@dataclass(frozen=True, eq=False)
class LogicalNot(AbstractUnaryScalarOperator):
    ufunc = numpy.logical_not


@dataclass(frozen=True, eq=False)
class AbstractBinaryScalarOperator(AbstractScalarOperator, abc.ABC):
    operands: tuple[Expr, Expr]


@dataclass(frozen=True, eq=False)
class Add(AbstractBinaryScalarOperator):
    ufunc = numpy.add


@dataclass(frozen=True, eq=False)
class Multiply(AbstractBinaryScalarOperator):
    ufunc = numpy.multiply


@dataclass(frozen=True, eq=False)
class Less(AbstractBinaryScalarOperator):
    ufunc = numpy.less


@dataclass(frozen=True, eq=False)
class LogicalAnd(AbstractBinaryScalarOperator):
    ufunc = numpy.logical_and

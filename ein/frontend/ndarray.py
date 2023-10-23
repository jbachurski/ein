from typing import TypeAlias, cast

import numpy
import numpy.typing

from ein import calculus
from ein.calculus import AbstractExpr, Expr, Value

ArrayLike: TypeAlias = Expr | numpy.typing.ArrayLike | "Array"
ArrayItem: TypeAlias = ArrayLike


class Array:
    expr: Expr

    def __init__(self, array_like: ArrayLike):
        if isinstance(array_like, AbstractExpr):
            expr = cast(Expr, array_like)
        elif isinstance(array_like, Array):
            expr = array_like.expr
        else:
            array = numpy.array(array_like)
            assert array.dtype in (
                numpy.dtype(bool),
                numpy.dtype(float),
                numpy.dtype(int),
            ), array.dtype
            expr = calculus.Const(Value(array))
        self.expr = expr

    def __getitem__(self, item_like: ArrayItem | tuple[ArrayItem, ...]) -> "Array":
        item: tuple[ArrayItem, ...] = (
            (item_like,) if not isinstance(item_like, tuple) else item_like
        )
        expr = self.expr
        for axis_item in item:
            expr = calculus.Get(expr, Array(axis_item).expr)
        return Array(expr)

    def __bool__(self):
        raise TypeError(
            "Ein arrays don't have a boolean value and cannot be used in conditions - "
            "did you accidentally include it in an if or while statement?"
        )

    def dim(self, axis: int) -> "Array":
        return Array(calculus.Dim(self.expr, axis))

    def __add__(self, other: ArrayLike) -> "Array":
        return Array(calculus.Add((self.expr, Array(other).expr)))

    def __mul__(self, other: ArrayLike) -> "Array":
        return Array(calculus.Multiply((self.expr, Array(other).expr)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> "Array":
        return Array(calculus.Negate((self.expr,)))

    def __sub__(self, other: ArrayLike) -> "Array":
        return self + -Array(other)

    def __rsub__(self, other: ArrayLike) -> "Array":
        return other + (-self)

    def __truediv__(self, other: ArrayLike) -> "Array":
        return self * calculus.Reciprocal((Array(other).expr,))

    def __rtruediv__(self, other: ArrayLike) -> "Array":
        return Array(other) / self

    def __invert__(self) -> "Array":
        return Array(calculus.LogicalNot((self.expr,)))

    def __and__(self, other: ArrayLike) -> "Array":
        return Array(calculus.LogicalAnd((self.expr, Array(other).expr)))

    def __or__(self, other: ArrayLike) -> "Array":
        return ~(~self & ~Array(other))

    def __lt__(self, other: ArrayLike) -> "Array":
        return Array(calculus.Less((self.expr, Array(other).expr)))

    def __ne__(self, other: ArrayLike) -> "Array":  # type: ignore
        other_ = Array(other)
        return (self < other_) | (other_ < self)

    def __eq__(self, other: ArrayLike) -> "Array":  # type: ignore
        return ~(self != other)

    def __gt__(self, other: ArrayLike) -> "Array":
        return Array(other) < self

    def __le__(self, other: ArrayLike) -> "Array":
        other_ = Array(other)
        return (self < other_) | (self == other_)

    def __ge__(self, other: ArrayLike) -> "Array":
        return Array(other) <= self

    def where(self, true: ArrayLike, false: ArrayLike) -> "Array":
        return Array(calculus.Where((self.expr, Array(true).expr, Array(false).expr)))

    def exp(self) -> "Array":
        return Array(calculus.Exp((self.expr,)))

    def tanh(self) -> "Array":
        a, b = self.exp(), (-self).exp()
        return (a - b) / (a + b)
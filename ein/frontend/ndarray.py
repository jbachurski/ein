from typing import TypeAlias, Union, cast

import numpy
import numpy.typing

from ein import calculus
from ein.backend import BACKENDS, DEFAULT_BACKEND, Backend
from ein.calculus import AbstractExpr, Expr, Value
from ein.symbols import Variable

ArrayLike: TypeAlias = Union["int | float | bool | numpy.ndarray | Array"]


class Array:
    expr: Expr

    def __init__(self, array_like: ArrayLike | Expr):
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

    def numpy(
        self,
        *,
        env: dict[Variable, numpy.ndarray] | None = None,
        backend: Backend = DEFAULT_BACKEND,
    ) -> numpy.ndarray:
        env = env if env is not None else {}
        if not self.expr.free_symbols <= set(env):
            raise ValueError(
                f"Cannot evaluate array, as it depends on free variables: {self.expr.free_symbols}"
            )
        return BACKENDS[backend](self.expr, env)

    def __getitem__(self, item_like: ArrayLike | tuple[ArrayLike, ...]) -> "Array":
        item: tuple[ArrayLike, ...] = (
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

    def to_float(self) -> "Array":
        return Array(calculus.CastToFloat((self.expr,)))

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
        return self * Array(calculus.Reciprocal((Array(other).expr,)))

    def __rtruediv__(self, other: ArrayLike) -> "Array":
        return Array(other) / self

    def __mod__(self, other: ArrayLike) -> "Array":
        return Array(calculus.Modulo((self.expr, Array(other).expr)))

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise NotImplementedError("Power modulo is not supported")
        if isinstance(power, int):
            if power < 0:
                return (1.0 / self) ** (-power)

            def go(k: int) -> Array:
                if k == 1:
                    return self
                elif k % 2:
                    return go(k - 1) * self
                sub = go(k // 2)
                return sub * sub

            return go(power)

        return Array(calculus.Power((self.expr, Array(power).expr)))

    def __invert__(self) -> "Array":
        return Array(calculus.LogicalNot((self.expr,)))

    def __and__(self, other: ArrayLike) -> "Array":
        return Array(calculus.LogicalAnd((self.expr, Array(other).expr)))

    def __or__(self, other: ArrayLike) -> "Array":
        return Array(calculus.LogicalOr((self.expr, Array(other).expr)))

    def __lt__(self, other: ArrayLike) -> "Array":
        return Array(calculus.Less((self.expr, Array(other).expr)))

    def __ne__(self, other: ArrayLike) -> "Array":  # type: ignore
        return Array(calculus.NotEqual((self.expr, Array(other).expr)))

    def __eq__(self, other: ArrayLike) -> "Array":  # type: ignore
        return Array(calculus.Equal((self.expr, Array(other).expr)))

    def __gt__(self, other: ArrayLike) -> "Array":
        return Array(other).__lt__(self)

    def __le__(self, other: ArrayLike) -> "Array":
        return Array(calculus.LessEqual((self.expr, Array(other).expr)))

    def __ge__(self, other: ArrayLike) -> "Array":
        return Array(other).__le__(self)

    def where(self, true: ArrayLike, false: ArrayLike) -> "Array":
        return Array(calculus.Where((self.expr, Array(true).expr, Array(false).expr)))

    def min(self, other: ArrayLike) -> "Array":
        return Array(calculus.Min((self.expr, Array(other).expr)))

    def max(self, other: ArrayLike) -> "Array":
        return Array(calculus.Max((self.expr, Array(other).expr)))

    def exp(self) -> "Array":
        return Array(calculus.Exp((self.expr,)))

    def sin(self) -> "Array":
        return Array(calculus.Sin((self.expr,)))

    def tanh(self) -> "Array":
        a, b = self.exp(), (-self).exp()
        return (a - b) / (a + b)

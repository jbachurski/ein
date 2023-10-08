from typing import TypeAlias, cast

import numpy
import numpy.typing

from . import calculus
from .calculus import AbstractExpr, Expr, Value

TensorLike: TypeAlias = Expr | numpy.typing.ArrayLike | "Tensor"
TensorItem: TypeAlias = TensorLike


class Tensor:
    expr: Expr

    def __init__(self, tensor_like: TensorLike):
        if isinstance(tensor_like, AbstractExpr):
            expr = cast(Expr, tensor_like)
        elif isinstance(tensor_like, Tensor):
            expr = tensor_like.expr
        else:
            array = numpy.array(tensor_like)
            assert array.dtype in (
                numpy.dtype(bool),
                numpy.dtype(float),
                numpy.dtype(int),
            ), array.dtype
            expr = calculus.Const(Value(array))
        self.expr = expr

    def __getitem__(self, item_like: TensorItem | tuple[TensorItem, ...]) -> "Tensor":
        item: tuple[TensorItem, ...] = (
            (item_like,) if not isinstance(item_like, tuple) else item_like
        )
        expr = self.expr
        for axis_item in item:
            expr = calculus.Get(expr, Tensor(axis_item).expr)
        return Tensor(expr)

    def __bool__(self):
        raise TypeError(
            "Tensors don't have a boolean value and cannot be used in conditions - "
            "did you accidentally include it in an if or while statement?"
        )

    def __add__(self, other: TensorLike) -> "Tensor":
        return Tensor(calculus.Add((self.expr, Tensor(other).expr)))

    def __mul__(self, other: TensorLike) -> "Tensor":
        return Tensor(calculus.Multiply((self.expr, Tensor(other).expr)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> "Tensor":
        return Tensor(calculus.Negate((self.expr,)))

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + -Tensor(other)

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return other + (-self)

    def __truediv__(self, other: TensorLike) -> "Tensor":
        return self * calculus.Reciprocal((Tensor(other).expr,))

    def __rtruediv__(self, other: TensorLike) -> "Tensor":
        return Tensor(other) / self

    def __lt__(self, other: TensorLike) -> "Tensor":
        return Tensor(calculus.Less((self.expr, Tensor(other).expr)))

    def dim(self, axis: int) -> "Tensor":
        return Tensor(calculus.Dim(self.expr, axis))

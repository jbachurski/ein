import inspect
from dataclasses import dataclass
from typing import Callable, Protocol, Self, TypeAlias, assert_never, cast

import numpy
import numpy.typing

from . import calculus
from .calculus import AbstractExpr, Expr, Index, Value

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


class _TensorConstructor(Protocol):
    def __call__(self, *args: Index) -> TensorLike:
        ...


@dataclass
class TensorComprehension:
    application: Callable[[Index, Expr, Expr], Expr]
    sizes: tuple[TensorLike, ...] | None

    def __init__(self, *, application, sizes=None):
        self.application = application
        self.sizes = sizes

    def __getitem__(self, sizes) -> Self:
        if self.sizes is not None:
            raise TypeError("Already specified the sizes for the tensor comprehension.")
        if not isinstance(sizes, tuple):
            sizes = (sizes,)
        return type(self)(application=self.application, sizes=sizes)

    @staticmethod
    def _size_of(expr: calculus.Expr) -> calculus.Shape:
        match expr:
            case calculus.Get(operand):
                sub = TensorComprehension._size_of(operand)
                return calculus.Shape(sub.operand, sub.axis + 1)
            case calculus.Vec() | calculus.Const() | calculus.Var():
                return calculus.Shape(expr, 0)
            case (
                calculus.At()
                | calculus.Sum()
                | calculus.Shape()
                | calculus.Negate()
                | calculus.Reciprocal()
                | calculus.Add()
                | calculus.Multiply()
            ):
                assert False, f"Cannot take size of type-wise scalar {expr}"
            case _:
                assert_never(expr)

    def __call__(self, constructor: _TensorConstructor) -> Tensor:
        n = (
            len(inspect.signature(constructor).parameters)
            if self.sizes is None
            else len(self.sizes)
        )
        indices = [Index() for _ in range(n)]
        wrapped_indices = [Tensor(calculus.At(index)) for index in indices]
        base_body = constructor(*wrapped_indices)  # type: ignore
        body = Tensor(base_body).expr
        if self.sizes is None:
            size_of: dict[Index, Expr] = {}
            for rank, index in enumerate(indices):
                candidates = (
                    self._size_of(expr)
                    for expr in body.direct_indices.get(index, set())
                )
                candidates = (
                    shape_expr
                    for shape_expr in candidates
                    if not shape_expr.free_indices
                )
                if not candidates:
                    raise ValueError(
                        f"Cannot infer bounds for index without direct gets: {index}"
                    )
                else:
                    # TODO: Handle the ignored cases here by requiring equivalence.
                    shape_expr, *_ = candidates
                    size_of[index] = shape_expr
        else:
            size_of = {
                index: Tensor(size).expr
                for index, size in zip(indices, self.sizes, strict=True)
            }
        for index in reversed(indices):
            print("apply", index, size_of[index])
            body = self.application(index, size_of[index], body)
        return Tensor(body)


array = TensorComprehension(application=calculus.Vec)
sum = TensorComprehension(application=calculus.Sum)

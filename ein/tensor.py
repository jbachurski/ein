import inspect
from dataclasses import dataclass
from typing import Protocol, Self, TypeAlias, cast

import numpy
import numpy.typing

from . import calculus

TensorLike: TypeAlias = calculus.Expr | numpy.typing.ArrayLike | "Tensor"


class Tensor:
    expr: calculus.Expr

    def __init__(self, tensor_like: TensorLike):
        if isinstance(tensor_like, calculus.AbstractExpr):
            expr = cast(calculus.Expr, tensor_like)
        elif isinstance(tensor_like, Tensor):
            expr = tensor_like.expr
        else:
            array = numpy.array(tensor_like)
            assert array.dtype in (
                numpy.dtype(bool),
                numpy.dtype(float),
                numpy.dtype(int),
            )
            expr = calculus.Const(calculus.Value(array))
        self.expr = expr


class _TensorConstructor(Protocol):
    def __call__(self, *args: calculus.Index) -> TensorLike:
        ...


@dataclass
class TensorComprehension:
    sizes: tuple[Tensor, ...] | None

    def __init__(self, sizes=None):
        self.sizes = sizes

    def __getitem__(self, item) -> Self:
        return type(self)(self.sizes)

    def __call__(self, constructor: _TensorConstructor) -> Tensor:
        if self.sizes is None:
            _n = len(inspect.signature(constructor).parameters)
            raise NotImplementedError("Bounds inference is not implemented.")
        else:
            sizes = [size.expr for size in self.sizes]
        indices = [calculus.Index() for _ in range(len(sizes))]
        body = Tensor(constructor(*indices)).expr
        for index, size in reversed(list(zip(indices, sizes, strict=True))):
            body = calculus.Vec(index, size, body)
        return Tensor(body)


tensor = TensorComprehension()

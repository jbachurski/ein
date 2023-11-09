import abc
import inspect
from dataclasses import dataclass
from typing import Callable, Iterable, Self, TypeVar

import numpy

from ein import calculus
from ein.calculus import Expr, Var
from ein.symbols import Index, Variable
from ein.type_system import Scalar, Type

from .ndarray import Array, ArrayLike

T = TypeVar("T")


def identity(x: T) -> T:
    return x


@dataclass
class BaseComprehension(abc.ABC):
    sizes: tuple[ArrayLike, ...] | None

    def __init__(self, *, sizes=None):
        self.sizes = sizes

    def __getitem__(self, sizes) -> Self:
        if self.sizes is not None:
            raise TypeError("Already specified the sizes for the array comprehension.")
        if not isinstance(sizes, tuple):
            sizes = (sizes,)
        return type(self)(sizes=sizes)

    @staticmethod
    def _size_of(expr: calculus.Expr) -> Iterable[calculus.Expr]:
        match expr:
            case calculus.Get(operand):
                for sub in ArrayComprehension._size_of(operand):
                    if isinstance(sub, calculus.Dim):
                        yield calculus.Dim(
                            sub.operand,
                            sub.axis + 1 if isinstance(sub.axis, int) else sub.axis,
                        )
            case calculus.Vec():
                yield calculus.Dim(expr, 0)
                yield expr.size
            case calculus.Const() | calculus.Var() | calculus.Fold():
                yield calculus.Dim(expr, 0)

    def _get_sized(self, body: Expr, indices: tuple[Index, ...]) -> dict[Index, Expr]:
        if self.sizes is None:
            size_of: dict[Index, Expr] = {}
            for rank, index in enumerate(indices):
                candidates = [
                    candidate
                    for expr in body.direct_indices.get(index, set())
                    for candidate in self._size_of(expr)
                ]
                candidates = [
                    shape_expr
                    for shape_expr in candidates
                    if not shape_expr.free_indices
                ]
                if not candidates:
                    raise ValueError(
                        f"Cannot infer bounds for index without direct gets: {index}"
                    )
                else:
                    # TODO: Handle the ignored cases here by requiring equivalence.
                    shape_expr: Expr
                    shape_expr, *_ = candidates
                    size_of[index] = calculus.AssertEq(shape_expr, tuple(candidates))
        else:
            size_of = {
                index: Array(size).expr
                for index, size in zip(indices, self.sizes, strict=True)
            }
        return size_of


class CommutativeComprehension(BaseComprehension):
    @staticmethod
    @abc.abstractmethod
    def application(index: Index, size: Expr, body: Expr) -> Expr:
        ...

    @staticmethod
    def pre(expr: Expr, /) -> Expr:
        return expr

    @staticmethod
    def post(expr: Expr, /) -> Expr:
        return expr

    def __call__(self, constructor: Callable[..., ArrayLike]) -> Array:
        n = (
            len(inspect.signature(constructor).parameters)
            if self.sizes is None
            else len(self.sizes)
        )
        indices = [Index() for _ in range(n)]
        wrapped_indices = [Array(calculus.At(index)) for index in indices]
        body: Expr = self.pre(Array(constructor(*wrapped_indices)).expr)
        size_of = self._get_sized(body, tuple(indices))
        for index in reversed(indices):
            body = self.application(index, size_of[index], body)
        return Array(self.post(Array(body).expr))


class ArrayComprehension(CommutativeComprehension):
    application = calculus.Vec


class SumComprehension(CommutativeComprehension):
    @staticmethod
    def application(index: Index, size: Expr, body: Expr) -> Expr:
        init = calculus.Const(calculus.Value(numpy.array(0.0)))
        acc = Var(Variable(), Scalar(float))
        return calculus.Fold(index, size, calculus.Add((acc, body)), init, acc)


class MaxComprehension(CommutativeComprehension):
    @staticmethod
    def application(index: Index, size: Expr, body: Expr) -> Expr:
        init = calculus.Const(calculus.Value(numpy.array(float("-inf"))))
        acc = Var(Variable(), Scalar(float))
        max_body = calculus.Where((calculus.Less((body, acc)), acc, body))
        return calculus.Fold(index, size, max_body, init, acc)


class MinComprehension(MaxComprehension):
    @staticmethod
    def application(index: Index, size: Expr, body: Expr) -> Expr:
        init = calculus.Const(calculus.Value(numpy.array(float("inf"))))
        acc = Var(Variable(), Scalar(float))
        min_body = calculus.Where((calculus.Less((acc, body)), acc, body))
        return calculus.Fold(index, size, min_body, init, acc)


class FoldComprehension(BaseComprehension):
    def __call__(
        self, init_: ArrayLike, constructor: Callable[[Array, Array], ArrayLike]
    ) -> Array:
        assert self.sizes is None or len(self.sizes) == 1
        index, acc = Index(), Variable()
        init = Array(init_)
        typed_acc = calculus.Var(acc, init.expr.type)
        wrapped_index, wrapped_acc = Array(calculus.At(index)), Array(typed_acc)
        body = Array(constructor(wrapped_index, wrapped_acc)).expr
        size_of = self._get_sized(body, (index,))
        return Array(calculus.Fold(index, size_of[index], body, init.expr, typed_acc))


class VariableArray(Array):
    expr: Var

    def __init__(self, type_: Type):
        super().__init__(Var(Variable(), type_))

    @property
    def var(self) -> Variable:
        return self.expr.var


def function(
    types: Iterable[Type], fun: Callable[..., ArrayLike]
) -> tuple[tuple[Variable, ...], Expr]:
    args = [VariableArray(typ) for typ in types]
    return tuple(arg.var for arg in args), Array(fun(*args)).expr


array = ArrayComprehension()
sum = SumComprehension()
max = MaxComprehension()
min = MinComprehension()
fold = FoldComprehension()

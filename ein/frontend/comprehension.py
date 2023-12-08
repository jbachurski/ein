import abc
import functools
import inspect
from dataclasses import dataclass
from typing import Callable, Iterable, NewType, TypeAlias, TypeVar, overload

from ein import calculus
from ein.calculus import Expr, Var
from ein.symbols import Index, Variable
from ein.type_system import Type

from .ndarray import Array, ArrayLike

T = TypeVar("T")
Idx = NewType("Idx", Array)
Size: TypeAlias = Array | int

_FromIndex: TypeAlias = Callable[[Idx], ArrayLike]
_FromIndices: TypeAlias = (
    Callable[[Idx], ArrayLike]
    | Callable[[Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx], ArrayLike]
)


def identity(x: T) -> T:
    return x


@dataclass
class BaseComprehension(abc.ABC):
    @classmethod
    def _dim_of(cls, expr: calculus.Expr, axis: int = 0) -> Iterable[calculus.Expr]:
        yield calculus.Dim(expr, axis)
        match expr:
            case calculus.Get(target, _item):
                yield from cls._dim_of(target, axis + 1)
            case calculus.Vec():
                yield expr.size

    def _get_sized(
        self, body: Expr, indices: tuple[Index, ...], sizes
    ) -> dict[Index, Expr]:
        if sizes is None:
            size_of: dict[Index, Expr] = {}
            for rank, index in enumerate(indices):
                candidates = [
                    candidate
                    for expr in body.direct_indices.get(index, set())
                    for candidate in self._dim_of(expr)
                    if not candidate.free_indices
                ]
                if not candidates:
                    raise ValueError(
                        f"Cannot infer bounds for index without direct gets: {index}"
                    )
                else:
                    # TODO: Handle the ignored cases here by requiring equivalence.
                    shape_expr: Expr
                    shape_expr, *_ = candidates
                    size_of[index] = (
                        calculus.AssertEq(shape_expr, tuple(candidates))
                        if len(candidates) > 1
                        else shape_expr
                    )
        else:
            size_of = {
                index: Array(size).expr
                for index, size in zip(indices, sizes, strict=True)
            }
        return size_of


class CommutativeComprehension(BaseComprehension):
    @staticmethod
    @abc.abstractmethod
    def application(index: Index, size: Expr, body: Expr) -> Expr:
        ...

    def __call__(self, constructor: Callable[..., ArrayLike], sizes) -> Array:
        n = (
            len(inspect.signature(constructor).parameters)
            if sizes is None
            else len(sizes)
        )
        indices = [Index() for _ in range(n)]
        wrapped_indices = [Array(calculus.At(index)) for index in indices]
        body: Expr = Array(constructor(*wrapped_indices)).expr
        size_of = self._get_sized(body, tuple(indices), sizes)
        for index in reversed(indices):
            body = self.application(index, size_of[index], body)
        return Array(Array(body).expr)


class ArrayComprehension(CommutativeComprehension):
    application = calculus.Vec


class FoldComprehension(BaseComprehension):
    def _apply(self, init_, constructor, size):
        if not isinstance(init_, tuple):
            init_ = (init_,)
        k = len(init_)
        init = functools.reduce(calculus.Cons, [Array(a).expr for a in init_])

        def take(e: calculus.Expr, i: int, n: int) -> calculus.Expr:
            assert 0 <= i < n
            if n <= 1:
                return e
            elif i + 1 == n:
                return calculus.Second(e)
            return take(calculus.First(e), i, n - 1)

        def untuple(e: calculus.Expr, n: int) -> Array | tuple[Array, ...]:
            if n == 1:
                return Array(e)
            return tuple(Array(take(e, i, n)) for i in range(n))

        index = Index()
        acc = calculus.Var(Variable(), init.type)
        arg_index = Array(calculus.At(index))
        arg_acc = untuple(acc, k)
        body_ = constructor(arg_index, arg_acc)
        if not isinstance(body_, tuple):
            body_ = (body_,)
        if len(init_) != len(body_):
            raise TypeError(
                f"Mismatched number of tuple elements in "
                f"accumulator initialiser and fold body: {len(init_)} != {len(body_)}"
            )
        body = functools.reduce(calculus.Cons, [Array(a).expr for a in body_])
        size_of = self._get_sized(body, (index,), (size,) if size is not None else None)
        return untuple(calculus.Fold(index, size_of[index], acc.var, init, body), k)

    @overload
    def __call__(
        self,
        init_: ArrayLike,
        constructor: Callable[[Array, Array], ArrayLike],
        size,
    ) -> Array:
        ...

    @overload
    def __call__(
        self,
        init_: tuple[ArrayLike, ArrayLike],
        constructor: Callable[
            [Array, tuple[Array, Array]], tuple[ArrayLike, ArrayLike]
        ],
        size,
    ) -> tuple[Array, ...]:
        ...

    @overload
    def __call__(
        self,
        init_: tuple[ArrayLike, ArrayLike, ArrayLike],
        constructor: Callable[
            [Array, tuple[Array, Array, Array]], tuple[ArrayLike, ArrayLike, ArrayLike]
        ],
        size,
    ) -> tuple[Array, Array, Array]:
        ...

    def __call__(self, init_, constructor, size):
        return self._apply(init_, constructor, size)


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


def array(f: _FromIndices, *, size: tuple[Size, ...] | Size | None = None) -> Array:
    if size is not None and not isinstance(size, tuple):
        size = (size,)
    return ArrayComprehension()(f, size)


def fold(init, f, *, count: Size | None = None):
    return FoldComprehension()(init, f, count)

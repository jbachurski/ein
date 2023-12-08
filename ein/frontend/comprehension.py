import functools
import inspect
from typing import Callable, Iterable, NewType, TypeAlias, TypeVar, overload

from ein import calculus
from ein.calculus import Expr, Var
from ein.symbols import Index, Variable
from ein.type_system import Type

from .ndarray import Array, ArrayLike

T = TypeVar("T")
Idx = NewType("Idx", Array)
Size: TypeAlias = Array | int

_FromIndex: TypeAlias = Callable[[Idx], Array]
_FromIndices: TypeAlias = (
    Callable[[Idx], ArrayLike]
    | Callable[[Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx], ArrayLike]
)
Arrays2 = tuple[Array, Array]
Arrays3 = tuple[Array, Array, Array]
ArrayLikes2 = tuple[ArrayLike, ArrayLike]
ArrayLikes3 = tuple[ArrayLike, ArrayLike, ArrayLike]
_WithIndex: TypeAlias = Callable[[Idx, T], T]


def identity(x: T) -> T:
    return x


def _dim_of(expr: calculus.Expr, axis: int = 0) -> Iterable[calculus.Expr]:
    yield calculus.Dim(expr, axis)
    match expr:
        case calculus.Get(target, _item):
            yield from _dim_of(target, axis + 1)
        case calculus.Vec():
            yield expr.size


def _infer_sizes(body: Expr, indices: tuple[Index, ...], sizes) -> dict[Index, Expr]:
    if sizes is None:
        size_of: dict[Index, Expr] = {}
        for rank, index in enumerate(indices):
            candidates = [
                candidate
                for expr in body.direct_indices.get(index, set())
                for candidate in _dim_of(expr)
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
            index: Array(size).expr for index, size in zip(indices, sizes, strict=True)
        }
    return size_of


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


def array(
    constructor: _FromIndices, *, size: tuple[Size, ...] | Size | None = None
) -> Array:
    if size is not None and not isinstance(size, tuple):
        size = (size,)
    n = len(inspect.signature(constructor).parameters) if size is None else len(size)
    indices = [Index() for _ in range(n)]
    wrapped_indices = [Idx(Array(calculus.At(index))) for index in indices]
    body: Expr = Array(constructor(*wrapped_indices)).expr
    size_of = _infer_sizes(body, tuple(indices), size)
    for index in reversed(indices):
        body = calculus.Vec(index, size_of[index], body)
    return Array(Array(body).expr)


@overload
def fold(init: ArrayLike, step: _WithIndex[Array], count: Size | None = None) -> Array:
    ...


@overload
def fold(
    init: ArrayLikes2, step: _WithIndex[Arrays2], count: Size | None = None
) -> Arrays2:
    ...


@overload
def fold(
    init: ArrayLikes2, step: _WithIndex[Arrays3], count: Size | None = None
) -> Arrays3:
    ...


def fold(init, step, count=None):
    if not isinstance(init, tuple):
        init = (init,)
    k = len(init)
    init_expr = functools.reduce(calculus.Cons, [Array(a).expr for a in init])

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
    acc = calculus.Var(Variable(), init_expr.type)
    arg_index = Array(calculus.At(index))
    arg_acc = untuple(acc, k)
    body = step(arg_index, arg_acc)
    if not isinstance(body, tuple):
        body = (body,)
    if len(init) != len(body):
        raise TypeError(
            f"Mismatched number of tuple elements in "
            f"accumulator initialiser and fold body: {len(init)} != {len(body)}"
        )
    body_expr = functools.reduce(calculus.Cons, [Array(a).expr for a in body])
    size_of = _infer_sizes(body_expr, (index,), (count,) if count is not None else None)
    return untuple(
        calculus.Fold(index, size_of[index], acc.var, init_expr, body_expr), k
    )

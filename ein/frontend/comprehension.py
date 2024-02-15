import functools
import inspect
from typing import Callable, Iterable, NewType, TypeAlias, TypeVar, cast, overload

from ein import calculus
from ein.calculus import Expr
from ein.frontend.layout import VecLayout, build_layout, fold_layout
from ein.midend.size_classes import _dim_of, _with_indices_at_zero
from ein.symbols import Index, Symbol, Variable
from ein.type_system import Type, scalar

from .ndarray import Array, ArrayLike

T = TypeVar("T")
Idx = NewType("Idx", Array)
Size: TypeAlias = Array | int

StructLike: TypeAlias = (
    list["StructLike"] | tuple["StructLike", ...] | dict[str, "StructLike"] | ArrayLike
)
_FromIndex: TypeAlias = Callable[[Idx], Array]
_FromIndices: TypeAlias = (
    Callable[[Idx], ArrayLike]
    | Callable[[Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx], ArrayLike]
)
_StructFromIndices: TypeAlias = (
    Callable[[Idx], StructLike]
    | Callable[[Idx, Idx], StructLike]
    | Callable[[Idx, Idx, Idx], StructLike]
    | Callable[[Idx, Idx, Idx, Idx], StructLike]
)
Arrays2 = tuple[Array, Array]
Arrays3 = tuple[Array, Array, Array]
ArrayLikes2 = tuple[ArrayLike, ArrayLike]
ArrayLikes3 = tuple[ArrayLike, ArrayLike, ArrayLike]
_WithIndex: TypeAlias = Callable[[Idx, T], T]


def identity(x: T) -> T:
    return x


def _infer_sizes(body: Expr, symbols: tuple[Symbol, ...], sizes) -> dict[Symbol, Expr]:
    if sizes is None:
        size_of: dict[Symbol, Expr] = {}
        direct_indices: dict[Symbol, dict[Expr, set[Symbol]]] = {}
        visited: set[Expr] = set()

        def go(sub: Expr, captured: set[Symbol]) -> Expr:
            if sub in visited:
                return sub
            match sub:
                case calculus.Get(target, calculus.Store(symbol)):
                    direct_indices.setdefault(symbol, {})[target] = captured
            sub.map(lambda sub1: go(sub1, captured | sub.captured_symbols))
            return sub

        go(body, set())

        for rank, index in enumerate(symbols):
            # TODO: Implement a lexical scope check to detect when
            #  a bad candidate is picked. Metaprogramming hard!
            # FIXME: This probably ends up non-deterministic in some way,
            #  and some candidates picked are worse than others.
            candidates = [
                _with_indices_at_zero(candidate)
                for expr, captured in direct_indices.get(index, {}).items()
                for candidate in [_dim_of(expr)]
                if not candidate.free_symbols & captured
            ]
            if not candidates:
                raise ValueError(f"Cannot infer bounds for index: {index}")
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
            index: Array(size).expr for index, size in zip(symbols, sizes, strict=True)
        }
    return size_of


class VariableArray(Array):
    expr: calculus.Store

    def __init__(self, type_: Type):
        super().__init__(calculus.Store(Variable(), type_))

    @property
    def var(self) -> Variable:
        return cast(Variable, self.expr.symbol)


def function(
    types: Iterable[Type], fun: Callable[..., ArrayLike]
) -> tuple[tuple[Variable, ...], Expr]:
    args = [VariableArray(typ) for typ in types]
    return tuple(arg.var for arg in args), Array(fun(*args)).expr


def structs(
    constructor: _StructFromIndices, *, size: tuple[Size, ...] | Size | None = None
) -> Array:
    if size is not None and not isinstance(size, tuple):
        size = (size,)
    n = len(inspect.signature(constructor).parameters) if size is None else len(size)
    indices = [Index() for _ in range(n)]
    wrapped_indices = [Idx(Array(calculus.at(index))) for index in indices]
    cons = constructor(*wrapped_indices)
    layout = build_layout(cons, lambda a: Array(a).layout)
    body: Expr = fold_layout(
        cons, lambda a: Array(a).expr, lambda a, b: calculus.Cons(a, b)
    )
    size_of = _infer_sizes(body, tuple(indices), size)
    for index in reversed(indices):
        body = calculus.Vec(index, size_of[index], body)
        layout = VecLayout(layout)
    return Array(body, layout)


def array(
    constructor: _FromIndices, *, size: tuple[Size, ...] | Size | None = None
) -> Array:
    return structs(constructor, size=size)


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
    init: ArrayLikes3, step: _WithIndex[Arrays3], count: Size | None = None
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

    counter = calculus.variable(Variable(), scalar(int))
    acc = calculus.variable(Variable(), init_expr.type)
    arg_index = Array(counter)
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
    size_of = _infer_sizes(
        body_expr, (counter.var,), (count,) if count is not None else None
    )
    e = calculus.Fold(counter.var, size_of[counter.var], acc.var, init_expr, body_expr)
    return untuple(e, k)

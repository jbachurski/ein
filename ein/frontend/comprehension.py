import inspect
from typing import Any, Callable, Protocol, TypeAlias, TypeVar, cast, overload

from ein.midend.size_classes import _dim_of, _with_indices_at_zero
from ein.phi import phi
from ein.phi.phi import Expr
from ein.phi.type_system import scalar_type
from ein.symbols import Index, Symbol, Variable

from .layout import VecLayout, build_layout
from .ndarray import (
    Array,
    ArrayLike,
    Scalar,
    Vec,
    _layout_struct_to_expr,
    _phi_to_yarr,
    wrap,
)

T = TypeVar("T")
Idx: TypeAlias = Scalar
Size: TypeAlias = int | Scalar


class _Dataclass(Protocol):
    __dataclass_fields__: Any


StructLike: TypeAlias = (
    list["StructLike"]
    | tuple["StructLike", ...]
    | dict[str, "StructLike"]
    | ArrayLike
    | Any
)
_FromIndex: TypeAlias = Callable[[Idx], Array]
_FromIndices: TypeAlias = (
    Callable[[Idx], ArrayLike]
    | Callable[[Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx, Idx], ArrayLike]
    | Callable[[Idx, Idx, Idx, Idx, Idx, Idx], ArrayLike]
)
_WithIndex: TypeAlias = Callable[[Idx, T], T]


def identity(x: T) -> T:
    return x


def _infer_sizes(
    body: Expr, symbols: tuple[Symbol, ...], initial_captured: set[Symbol], sizes
) -> dict[Symbol, Expr]:
    if sizes is not None:
        return {
            index: wrap(size).expr for index, size in zip(symbols, sizes, strict=True)
        }
    size_of: dict[Symbol, Expr] = {}
    direct_indices: dict[Symbol, dict[Expr, set[Symbol]]] = {}
    visited: set[Expr] = set()

    def go(sub: Expr, captured: set[Symbol]) -> Expr:
        if sub in visited:
            return sub
        match sub:
            case phi.Get(target, phi.Store(symbol)):
                direct_indices.setdefault(symbol, {})[target] = captured
        sub.map(lambda sub1: go(sub1, captured | sub.captured_symbols))
        return sub

    go(body, {*symbols, *initial_captured})

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
                phi.AssertEq(shape_expr, tuple(candidates))
                if len(candidates) > 1
                else shape_expr
            )
    return size_of


@overload
def array(constructor: Callable[[], T]) -> Array:
    ...


@overload
def array(constructor: Callable[[Idx], T], *, size: Size | None = None) -> Vec[T]:
    ...


@overload
def array(
    constructor: Callable[[Idx, Idx], T], *, size: tuple[Size, Size] | None = None
) -> Vec[Vec[T]]:
    ...


@overload
def array(
    constructor: Callable[[Idx, Idx, Idx], T],
    *,
    size: tuple[Size, Size, Size] | None = None,
) -> Vec[Vec[Vec[T]]]:
    ...


@overload
def array(
    constructor: Callable[[Idx, Idx, Idx, Idx], T],
    *,
    size: tuple[Size, Size, Size] | None = None,
) -> Vec[Vec[Vec[Vec[T]]]]:
    ...


def array(constructor, *, size=None):
    if size is not None and not isinstance(size, tuple):
        size = (size,)
    n = len(inspect.signature(constructor).parameters) if size is None else len(size)
    indices = [Index() for _ in range(n)]
    wrapped_indices = [cast(Idx, _phi_to_yarr(phi.at(index))) for index in indices]
    cons = constructor(*wrapped_indices)
    layout = build_layout(cons, lambda a: wrap(a).layout)
    body: Expr = _layout_struct_to_expr(layout, cons)
    size_of = _infer_sizes(body, tuple(indices), set(), size)
    for index in reversed(indices):
        body = phi.Vec(index, size_of[index], body)
        layout = VecLayout(layout)
    return _phi_to_yarr(body, layout)


def fold(init: T, step: _WithIndex[T], count: Size | None = None) -> T:
    layout = build_layout(init, lambda a: wrap(a).layout)
    init_expr: Expr = _layout_struct_to_expr(layout, init)
    counter = phi.variable(Variable(), scalar_type(int))
    arg_index = _phi_to_yarr(counter)
    acc = phi.variable(Variable(), init_expr.type)
    arg_acc = _phi_to_yarr(acc, layout)
    body = step(arg_index, arg_acc)
    layout_ = build_layout(init, lambda a: wrap(a).layout)
    if layout != layout_:
        raise ValueError(
            f"Expected initialiser and fold body layouts to match, "
            f"got {layout} and {layout_}"
        )
    body_expr: Expr = _layout_struct_to_expr(layout, body)
    size_of = _infer_sizes(
        body_expr, (counter.var,), {acc.var}, (count,) if count is not None else None
    )
    expr = phi.Fold(counter.var, size_of[counter.var], acc.var, init_expr, body_expr)
    return _phi_to_yarr(expr, layout)

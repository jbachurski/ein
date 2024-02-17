import inspect
from typing import (
    Any,
    Callable,
    Iterable,
    NewType,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

from ein import calculus
from ein.calculus import Expr
from ein.frontend.layout import Layout, VecLayout, build_layout, fold_layout
from ein.midend.size_classes import _dim_of, _with_indices_at_zero
from ein.symbols import Index, Symbol, Variable
from ein.type_system import Type, scalar_type

from .ndarray import Array, ArrayLike, Scalar, Vec, _to_array, wrap

T = TypeVar("T")
Idx = NewType("Idx", Scalar)
Size: TypeAlias = Array | int


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


def _layout_struct_to_expr(layout: Layout, struct) -> Expr:
    return fold_layout(
        layout, struct, lambda a: wrap(a).expr, lambda a, b: calculus.Cons(a, b)
    )


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
            index: wrap(size).expr for index, size in zip(symbols, sizes, strict=True)
        }
    return size_of


def function(
    types: Iterable[Type], fun: Callable[..., ArrayLike]
) -> tuple[tuple[Variable, ...], Expr]:
    arg_vars = [calculus.variable(Variable(), type_) for type_ in types]
    args = [_to_array(var) for var in arg_vars]
    return tuple(var.var for var in arg_vars), wrap(fun(*args)).expr


def structs(
    constructor: _StructFromIndices, *, size: tuple[Size, ...] | Size | None = None
) -> Any:
    if size is not None and not isinstance(size, tuple):
        size = (size,)
    n = len(inspect.signature(constructor).parameters) if size is None else len(size)
    indices = [Index() for _ in range(n)]
    wrapped_indices = [Idx(_to_array(calculus.at(index))) for index in indices]
    cons = constructor(*wrapped_indices)
    layout = build_layout(cons, lambda a: wrap(a).layout)
    body: Expr = _layout_struct_to_expr(layout, cons)
    size_of = _infer_sizes(body, tuple(indices), size)
    for index in reversed(indices):
        body = calculus.Vec(index, size_of[index], body)
        layout = VecLayout(layout)
    return _to_array(body, layout)


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
def array(constructor: _FromIndices, *, size: tuple[Size, ...] | None = None) -> Array:
    ...


def array(constructor, *, size=None) -> Array:
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


@overload
def fold(init: StructLike, step: _WithIndex[Any], count: Size | None = None) -> Any:
    ...


def fold(init, step, count=None):
    layout = build_layout(init, lambda a: wrap(a).layout)
    init_expr: Expr = _layout_struct_to_expr(layout, init)
    counter = calculus.variable(Variable(), scalar_type(int))
    arg_index = _to_array(counter)
    acc_var = Variable()
    arg_acc = _to_array(calculus.variable(acc_var, init_expr.type), layout)
    body = step(arg_index, arg_acc)
    layout_ = build_layout(init, lambda a: wrap(a).layout)
    if layout != layout_:
        raise ValueError(
            f"Expected initialiser and fold body layouts to match, "
            f"got {layout} and {layout_}"
        )
    body_expr: Expr = _layout_struct_to_expr(layout, body)
    size_of = _infer_sizes(
        body_expr, (counter.var,), (count,) if count is not None else None
    )
    body = calculus.Fold(
        counter.var, size_of[counter.var], acc_var, init_expr, body_expr
    )
    return _to_array(body, layout)

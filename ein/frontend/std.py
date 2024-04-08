import functools
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .comprehension import Idx, Size, _FromIndex, array, fold
from .layout import build_layout, map_layout
from .ndarray import Array, ArrayLike, Scalar, Vec, wrap

T = TypeVar("T", Array, tuple[Array, Array], tuple[Array, Array, Array])


def where(cond: bool | Scalar, true: T, false: T) -> T:
    layout_ = build_layout(true, lambda a: wrap(a).layout)
    layout = build_layout(false, lambda a: wrap(a).layout)
    assert layout == layout_
    c = wrap(cond)
    return map_layout(
        layout,
        [true, false],
        lambda t, f: c.where(t, f),
        lambda ts, fs: array(lambda i: where(cond, ts[i], fs[i])),
    )


def concat(arg: Vec[T], *args: Vec[T]) -> Vec[T]:
    return arg.concat(concat(*args)) if args else arg


@dataclass
class Monoid(Generic[T]):
    identity: T
    concat: Callable[[T, T], T]

    def reduce(self, f: Callable[[Idx], T], count: Size | None = None) -> T:
        return fold(self.identity, lambda i, acc: self.concat(acc, f(i)), count=count)


sum_monoid = Monoid(wrap(0.0), lambda a, b: a + b)
min_monoid = Monoid(wrap(float("+inf")), lambda a, b: a.min(b))
max_monoid = Monoid(wrap(float("-inf")), lambda a, b: a.max(b))


def min(*args: ArrayLike) -> Array:
    return functools.reduce(min_monoid.concat, map(wrap, args))


def max(*args: ArrayLike) -> Array:
    return functools.reduce(max_monoid.concat, map(wrap, args))


def reduce_sum(f: _FromIndex, count: Size | None = None) -> Array:
    return sum_monoid.reduce(f, count)


def reduce_min(f: _FromIndex, count: Size | None = None) -> Array:
    return min_monoid.reduce(f, count)


def reduce_max(f: _FromIndex, count: Size | None = None) -> Array:
    return max_monoid.reduce(f, count)


def reduce_argmin(f: _FromIndex, count: Size | None = None) -> Array:
    def concat(a: tuple[Array, Array], b: tuple[Array, Array]) -> tuple[Array, Array]:
        return where(a[0] <= b[0], a, b)

    return Monoid((wrap(float("+inf")), wrap(0)), concat).reduce(
        lambda i: (f(i), i), count
    )[1]

import functools
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .comprehension import Idx, Size, _FromIndex, fold
from .ndarray import Array, ArrayLike, wrap

T = TypeVar("T", Array, tuple[Array, Array], tuple[Array, Array, Array])


def where(cond: ArrayLike, true: ArrayLike, false: ArrayLike) -> Array:
    return wrap(cond).where(true, false)


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
        lt = a[0] <= b[0]
        return where(lt, a[0], b[0]), where(lt, a[1], b[1])

    return Monoid((wrap(float("+inf")), wrap(0)), concat).reduce(
        lambda i: (f(i), i), count
    )[1]

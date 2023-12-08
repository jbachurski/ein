from dataclasses import dataclass
from typing import Callable, TypeVar

from .comprehension import Size, _FromIndex, fold
from .ndarray import Array, ArrayLike

T = TypeVar("T")


def where(cond: ArrayLike, true: ArrayLike, false: ArrayLike) -> Array:
    return Array(cond).where(true, false)


@dataclass
class Monoid:
    identity: Array
    concat: Callable[[Array, Array], Array]

    def reduce(self, f: _FromIndex, count: Size | None = None) -> Array:
        return fold(
            self.identity, lambda i, acc: self.concat(acc, Array(f(i))), count=count
        )


sum_monoid = Monoid(Array(0.0), lambda a, b: a + b)
min_monoid = Monoid(Array(float("+inf")), lambda a, b: where(a < b, a, b))
max_monoid = Monoid(Array(float("-inf")), lambda a, b: where(a < b, b, a))


def sum(f: _FromIndex, count: Size | None = None) -> Array:
    return sum_monoid.reduce(f, count)


def min(f: _FromIndex, count: Size | None = None) -> Array:
    return min_monoid.reduce(f, count)


def max(f: _FromIndex, count: Size | None = None) -> Array:
    return max_monoid.reduce(f, count)

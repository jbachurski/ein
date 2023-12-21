from dataclasses import dataclass
from functools import cache
from typing import Callable, Self, TypeAlias, TypeVar

from ein import calculus
from ein.midend.size_classes import SizeEquivalence
from ein.symbols import Index, Symbol
from ein.type_system import PrimitiveArrayType

from . import array_calculus, axial
from .axial import Axial

T = TypeVar("T")


def _first_opt(*args: T | None) -> T | None:
    for x in args:
        if x is not None:
            return x
    return None


@dataclass(frozen=True)
class Builder:
    expr: array_calculus.Expr

    @classmethod
    def const(cls, value) -> Self:
        return cls(array_calculus.Const(calculus.Value(value)))

    def __neg__(self) -> "Builder":
        return Builder(
            array_calculus.UnaryElementwise(
                array_calculus.UnaryElementwise.Kind.negative, self.expr
            )
        )

    def __add__(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.add, self.expr, other.expr
            )
        )

    def __sub__(self, other: "Builder") -> "Builder":
        return self + (-other)

    def min(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.minimum, self.expr, other.expr
            )
        )

    def max(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.maximum, self.expr, other.expr
            )
        )


def transform_get(
    target_: calculus.Expr,
    item_: calculus.Expr,
    go: Callable[[calculus.Expr], Axial],
    size_class: SizeEquivalence,
    index_sizes: dict[Index, array_calculus.Expr],
    *,
    use_takes: bool,
    use_slice_pads: bool,
    use_slice_elision: bool,
) -> Axial:
    target = go(target_)
    rank = target.expr.type.single.rank
    ts = match_index_clipped_shift(item_)
    if use_slice_pads and ts is not None:
        index, (shift, low, high) = ts
        axes = target._axes + (index,)
        if index not in target_.free_indices:
            is_trivial = {shift, low, high} == {None}
            if (
                use_slice_elision
                and is_trivial
                and size_class.equiv(index, calculus.Dim(target_, 0))
            ):
                return Axial(axes, target.expr)
            if use_slice_pads:
                zero = Builder.const(0).expr
                size = index_sizes[index]
                size1 = (Builder(size) + Builder.const(-1)).expr
                return Axial(
                    axes,
                    pad_slice_get(
                        target.expr,
                        target.positional_axis(0),
                        go(shift).normal if shift is not None else zero,
                        go(low).normal if low is not None else zero,
                        go(high).normal if high is not None else size1,
                        size,
                    ),
                )
    item = go(item_)
    if use_takes and not (target.type.free_indices & item.type.free_indices):
        axes = target._axes + item._axes
        take_axes: list[array_calculus.Expr | None] = [None] * rank
        take_axes[target.positional_axis(0)] = item.expr
        return Axial(axes, array_calculus.Take(target.expr, tuple(take_axes)))
    return general_get(target, item)


def _put_at(n: int, p: int, x: T) -> tuple[T | None, ...]:
    return p * (None,) + (x,) + (n - p - 1) * (None,)


def pad_slice_get(
    target: array_calculus.Expr,
    axis: int,
    shift_: array_calculus.Expr,
    low_: array_calculus.Expr,
    high_: array_calculus.Expr,
    size_: array_calculus.Expr,
) -> array_calculus.Expr:
    rank = target.type.single.rank
    assert 0 <= axis < rank
    assert all(
        e.type.single == PrimitiveArrayType(0, int)
        for e in (shift_, low_, high_, size_)
    )
    dim = Builder(array_calculus.Dim(axis, target))

    shift = Builder(shift_)
    low = Builder(low_).max(Builder.const(0))
    high = Builder(high_).min(dim - Builder.const(1))
    size = Builder(size_)

    # We want to compute an array b such that
    #  b[i] = a[min(max(i + shift, low), high)]
    # using the operation
    #  b = pad(a[start:end], (left, right))
    # (where indexing is along the selected axis)
    start = shift.max(low).min(high)
    stop = (shift + size - Builder.const(1)).max(low).min(high) + Builder.const(1)
    left = low.max(-shift)
    right = size - (stop - start) - left

    return array_calculus.Pad(
        array_calculus.Slice(
            target,
            _put_at(rank, axis, start.expr),
            _put_at(rank, axis, stop.expr),
        ),
        _put_at(rank, axis, left.expr),
        _put_at(rank, axis, right.expr),
    )


def general_get(target: Axial, item: Axial) -> Axial:
    axes = axial._alignment(item._axes, target._axes)
    k = len(axes)
    expr = array_calculus.Squeeze(
        (k,),
        array_calculus.Gather(
            k,
            target.aligned(axes),
            array_calculus.Unsqueeze(
                tuple(
                    range(
                        len(axes),
                        len(axes) + target.type.type.single.rank,
                    )
                ),
                item.aligned(axes),
            ),
        ),
    )
    return Axial(axes, expr)


ClippedShift: TypeAlias = tuple[
    calculus.Expr | None, calculus.Expr | None, calculus.Expr | None
]


@cache
def match_index_clipped_shift(expr: calculus.Expr) -> tuple[Index, ClippedShift] | None:
    if len(expr.free_indices) == 1:
        (index,) = expr.free_indices
        t = match_clipped_shift(expr, index)
        if t is not None:
            return index, t
    return None


@cache
def match_clipped_shift(expr: calculus.Expr, symbol: Symbol) -> ClippedShift | None:
    # expr is equivalent to the form min(max(symbol + shift, low), high)
    # where symbol occurs in no subterm of shift/low/high
    if not (expr.free_symbols <= {symbol}):
        return None
    if symbol not in expr.free_symbols:
        return None, None, None
    match expr:
        case calculus.Store(symbol_, _inner_type):
            if symbol == symbol_:
                return None, None, None
        case calculus.Add((first, second)):
            return _first_opt(
                _clip_shift_add(first, second, symbol),
                _clip_shift_add(second, first, symbol),
            )
        case calculus.Min((first, second)):
            return _first_opt(
                _clip_shift_min(first, second, symbol),
                _clip_shift_min(second, first, symbol),
            )
        case calculus.Max((first, second)):
            return _first_opt(
                _clip_shift_max(first, second, symbol),
                _clip_shift_max(second, first, symbol),
            )
    return None


def _clip_shift_add(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        if lo is None and hi is None:
            return calculus.Add((d, other)) if d is not None else other, None, None
    return None


def _clip_shift_min(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        hi = calculus.Min((hi, other)) if hi is not None else other
        return d, lo, hi
    return None


def _clip_shift_max(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        lo = calculus.Max((lo, other)) if lo is not None else other
        hi = calculus.Max((hi, other)) if hi is not None else None
        return d, lo, hi
    return None


def match_shift(expr: calculus.Expr, symbol: Symbol) -> calculus.Expr | None:
    t = match_clipped_shift(expr, symbol)
    if t is not None:
        d, lo, hi = t
        return d if lo is None and hi is None else None
    return None

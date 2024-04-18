from dataclasses import dataclass
from functools import cache
from typing import Callable, Self, TypeAlias, TypeVar

from ein.codegen import yarr
from ein.midend.size_classes import SizeEquivalence, _dim_of
from ein.phi import phi
from ein.phi.type_system import PrimitiveArrayType
from ein.symbols import Index, Symbol

from . import axial
from .axial import Axial

T = TypeVar("T")


def _first_opt(*args: T | None) -> T | None:
    for x in args:
        if x is not None:
            return x
    return None


@dataclass(frozen=True)
class Builder:
    expr: yarr.Expr

    @classmethod
    def const(cls, value) -> Self:
        return cls(yarr.Const(phi.Value(value)))

    def __neg__(self) -> "Builder":
        return Builder(
            yarr.UnaryElementwise(yarr.UnaryElementwise.Kind.negative, self.expr)
        )

    def __add__(self, other: "Builder") -> "Builder":
        return Builder(
            yarr.BinaryElementwise(
                yarr.BinaryElementwise.Kind.add, self.expr, other.expr
            )
        )

    def __sub__(self, other: "Builder") -> "Builder":
        return Builder(
            yarr.BinaryElementwise(
                yarr.BinaryElementwise.Kind.subtract, self.expr, other.expr
            )
        )

    def min(self, other: "Builder") -> "Builder":
        return Builder(
            yarr.BinaryElementwise(
                yarr.BinaryElementwise.Kind.minimum, self.expr, other.expr
            )
        )

    def max(self, other: "Builder") -> "Builder":
        return Builder(
            yarr.BinaryElementwise(
                yarr.BinaryElementwise.Kind.maximum, self.expr, other.expr
            )
        )


def transform_get(
    target_: phi.Expr,
    item_: phi.Expr,
    go: Callable[[phi.Expr], Axial],
    size_class: SizeEquivalence,
    index_sizes: dict[Index, yarr.Expr],
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
            dim = _dim_of(target_, 0)
            if use_slice_elision and is_trivial and size_class.equiv(index, dim):
                return Axial(axes, target.expr)
            if use_slice_pads:
                zero = Builder.const(0).expr
                size = index_sizes[index]
                dim1 = (Builder(go(dim).normal) - Builder.const(1)).expr
                return Axial(
                    axes,
                    pad_slice_get(
                        target.expr,
                        target.positional_axis(0),
                        go(shift).normal if shift is not None else zero,
                        go(low).normal if low is not None else zero,
                        go(high).normal if high is not None else dim1,
                        size,
                    ),
                )
    item = go(item_)
    if use_takes and not (target.type.free_indices & item.type.free_indices):
        axes = target._axes + item._axes
        take_axes: list[yarr.Expr | None] = [None] * rank
        take_axes[target.positional_axis(0)] = item.expr
        return Axial(axes, yarr.Take(target.expr, tuple(take_axes)))
    return general_get(target, item)


def _put_at(n: int, p: int, x: T) -> tuple[T | None, ...]:
    return p * (None,) + (x,) + (n - p - 1) * (None,)


def pad_slice_get(
    target: yarr.Expr,
    axis: int,
    shift_: yarr.Expr,
    low_: yarr.Expr,
    high_: yarr.Expr,
    size_: yarr.Expr,
) -> yarr.Expr:
    rank = target.type.single.rank
    assert 0 <= axis < rank
    assert all(
        e.type.single == PrimitiveArrayType(0, int)
        for e in (shift_, low_, high_, size_)
    )
    zero, one = Builder.const(0), Builder.const(1)

    shift = Builder(shift_)
    low = Builder(low_)
    high = Builder(high_)
    size = Builder(size_)

    # We want to compute an array b such that
    #  b[i] = a[min(max(i + shift, low), high)]
    # using the operation
    #  b = pad(a[start:stop], (left, right))
    # (where indexing is along the selected axis)
    start = shift.max(low).min(high)
    stop = (shift + size - one).max(low).min(high) + one
    left = (low - shift).max(zero).min(size - one)
    right = size - (stop - start) - left

    return yarr.Pad(
        yarr.Slice(
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
    expr = yarr.Squeeze(
        (k,),
        yarr.Gather(
            k,
            target.aligned(axes),
            yarr.Unsqueeze(
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


ClippedShift: TypeAlias = tuple[phi.Expr | None, phi.Expr | None, phi.Expr | None]


@cache
def match_index_clipped_shift(expr: phi.Expr) -> tuple[Index, ClippedShift] | None:
    if len(expr.free_indices) == 1:
        (index,) = expr.free_indices
        t = match_clipped_shift(expr, index)
        if t is not None:
            return index, t
    return None


@cache
def match_clipped_shift(expr: phi.Expr, symbol: Symbol) -> ClippedShift | None:
    # expr is equivalent to the form min(max(symbol + shift, low), high)
    # where symbol occurs in no subterm of shift/low/high
    if not expr.free_symbols:
        return None, None, None
    match expr:
        case phi.Store(symbol_, _inner_type):
            if symbol == symbol_:
                return None, None, None
        case phi.Add((first, second)):
            return _first_opt(
                _clip_shift_add(first, second, symbol),
                _clip_shift_add(second, first, symbol),
            )
        case phi.Subtract((first, second)):
            second = phi.Negate((second,))
            return _first_opt(
                _clip_shift_add(first, second, symbol),
                _clip_shift_add(second, first, symbol),
            )
        case phi.Min((first, second)):
            return _first_opt(
                _clip_shift_min(first, second, symbol),
                _clip_shift_min(second, first, symbol),
            )
        case phi.Max((first, second)):
            return _first_opt(
                _clip_shift_max(first, second, symbol),
                _clip_shift_max(second, first, symbol),
            )
    return None


def _clip_shift_add(
    shifted: phi.Expr, other: phi.Expr, symbol: Symbol
) -> ClippedShift | None:
    if other.free_indices:
        return None
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        if lo is None and hi is None:
            return phi.Add((d, other)) if d is not None else other, None, None
    return None


def _clip_shift_min(
    shifted: phi.Expr, other: phi.Expr, symbol: Symbol
) -> ClippedShift | None:
    if other.free_indices:
        return None
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        hi = phi.Min((hi, other)) if hi is not None else other
        return d, lo, hi
    return None


def _clip_shift_max(shifted: phi.Expr, other: phi.Expr, symbol: Symbol):
    if other.free_indices:
        return None
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        lo = phi.Max((lo, other)) if lo is not None else other
        hi = phi.Max((hi, other)) if hi is not None else None
        return d, lo, hi
    return None


def match_shift(expr: phi.Expr, symbol: Symbol) -> phi.Expr | None:
    t = match_clipped_shift(expr, symbol)
    if t is not None:
        d, lo, hi = t
        return d if lo is None and hi is None else None
    return None

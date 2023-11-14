from dataclasses import dataclass
from typing import Iterable, ParamSpec, TypeAlias, cast

from ein.symbols import Index, Variable
from ein.type_system import PrimitiveType

from . import array_calculus


@dataclass(frozen=True)
class AxialType:
    type: PrimitiveType
    free_indices: set[Index]

    @property
    def pretty(self) -> str:
        return f"{sorted(self.free_indices, key=lambda x: int(str(x)[1:]))}:{self.type.pretty}"


P = ParamSpec("P")
Axis: TypeAlias = Index | int
Axes: TypeAlias = tuple[Axis, ...]


def tuple_maybe_singleton(*args: array_calculus.Expr) -> array_calculus.Expr:
    return array_calculus.Tuple(args) if len(args) != 1 else args[0]


def untuple_maybe_singleton(
    target: array_calculus.Expr, at: int
) -> array_calculus.Expr:
    k = len(target.type.elems)
    return array_calculus.Untuple(at, k, target) if k != 1 else target


class Axial:
    axes: Axes
    expr: array_calculus.Expr

    def __init__(self, axes: Iterable[Axis], array: array_calculus.Expr):
        self.axes = tuple(axes)
        self.expr = array
        assert all(len(self.axes) == arr.rank for arr in self.expr.type.elems)

    @property
    def type(self) -> AxialType:
        free_indices = {index for index in self.axes if isinstance(index, Index)}
        return AxialType(
            self.expr.type.with_rank_delta(-len(free_indices)),
            free_indices,
        )

    @property
    def normal(self) -> array_calculus.Expr:
        assert not self.type.free_indices
        assert self.type.type.single
        rank = self.expr.type.single.rank
        inv: list[int | None] = [None for _ in range(rank)]
        for i, p in enumerate(self.axes):
            assert isinstance(p, int)
            # Axes are numbered in reverse order
            inv[rank - p - 1] = i
        return array_calculus.Transpose(tuple(cast(list[int], inv)), self.expr)

    def within(self, *args: tuple[Variable, array_calculus.Expr]) -> "Axial":
        return Axial(self.axes, array_calculus.Let(args, self.expr))

    def cons(self, other: "Axial") -> "Axial":
        n = len(self.expr.type.elems)
        m = len(other.expr.type.elems)
        return Axial(
            self.axes,
            tuple_maybe_singleton(
                *(untuple_maybe_singleton(self.expr, i) for i in range(n)),
                *(untuple_maybe_singleton(other.expr, i) for i in range(m)),
            ),
        )

    def slice_tuple(self, start: int, end: int) -> "Axial":
        k = len(self.expr.type.elems)
        assert 0 <= start <= end <= k
        return Axial(
            self.axes,
            tuple_maybe_singleton(
                *(untuple_maybe_singleton(self.expr, i) for i in range(start, end))
            ),
        )


# TODO: This is a silly baseline alignment algorithm.
def alignment(*args: Axes) -> Axes:
    seen = []
    for axes in args:
        for axis in axes:
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def _align(
    expr: array_calculus.Expr, axes: Axes, into_axes: Axes, *, leftpad: bool
) -> array_calculus.Expr:
    transposition = tuple(
        axes.index(axis) for axis in sorted(axes, key=into_axes.index)
    )
    expands = tuple(i for i, axis in enumerate(into_axes) if axis not in axes)
    expr = array_calculus.Transpose(transposition, expr)
    if not leftpad:
        while expands and not expands[0]:
            expands = tuple(x - 1 for x in expands[1:])
    expr = array_calculus.Unsqueeze(expands, expr)
    return expr


def align(
    target: Axial, into_axes: Axes, *, leftpad: bool = True
) -> array_calculus.Expr:
    k = len(target.expr.type.elems)

    def local_align(expr: array_calculus.Expr):
        return _align(expr, target.axes, into_axes, leftpad=leftpad)

    if k != 1:
        return array_calculus.Tuple(
            tuple(
                local_align(array_calculus.Untuple(i, k, target.expr)) for i in range(k)
            )
        )
    return local_align(target.expr)

from dataclasses import dataclass
from typing import Iterable, ParamSpec, TypeAlias, cast

from ein.symbols import Index, Variable
from ein.type_system import PrimitiveArrayType, ScalarKind

from . import array_calculus


@dataclass(frozen=True)
class AxialType:
    array_type: PrimitiveArrayType
    free_indices: set[Index]

    @property
    def pretty(self) -> str:
        return f"{sorted(self.free_indices, key=lambda x: int(str(x)[1:]))}:{self.array_type.pretty}"


P = ParamSpec("P")
Axis: TypeAlias = Index | int
Axes: TypeAlias = tuple[Axis, ...]


class Axial:
    axes: Axes
    array: array_calculus.Expr
    kind: ScalarKind

    def __init__(
        self, axes: Iterable[Axis], array: array_calculus.Expr, kind: ScalarKind
    ):
        self.axes = tuple(axes)
        self.array = array
        self.kind = kind
        assert len(self.axes) == self.array.rank

    @property
    def type(self) -> AxialType:
        free_indices = {index for index in self.axes if isinstance(index, Index)}
        return AxialType(
            PrimitiveArrayType(rank=len(self.axes) - len(free_indices), kind=self.kind),
            free_indices,
        )

    @property
    def normal(self) -> array_calculus.Expr:
        assert not self.type.free_indices
        rank = self.type.array_type.rank
        inv: list[int | None] = [None for _ in range(rank)]
        for i, p in enumerate(self.axes):
            assert isinstance(p, int)
            # Axes are numbered in reverse order
            inv[rank - p - 1] = i
        return array_calculus.Transpose(tuple(cast(list[int], inv)), self.array)

    def within(self, *args: tuple[Variable, array_calculus.Expr]) -> "Axial":
        return Axial(self.axes, array_calculus.Let(args, self.array), self.kind)


# TODO: This is a silly baseline alignment algorithm.
def alignment(*args: Axes) -> Axes:
    seen = []
    for axes in args:
        for axis in axes:
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def align(target: Axial, into_axes: Axes) -> array_calculus.Expr:
    transposition = tuple(
        target.axes.index(axis) for axis in sorted(target.axes, key=into_axes.index)
    )
    expands = tuple(i for i, axis in enumerate(into_axes) if axis not in target.axes)
    array = target.array
    array = array_calculus.Transpose(transposition, array)
    array = array_calculus.Unsqueeze(expands, array)
    return array

from dataclasses import dataclass
from typing import Iterable, ParamSpec, TypeAlias

from ein.codegen import yarr
from ein.phi.type_system import PrimitiveType
from ein.symbols import Index, Variable

P = ParamSpec("P")
Axis: TypeAlias = Index
Axes: TypeAlias = tuple[Axis, ...]


@dataclass(frozen=True)
class AxialType:
    type: PrimitiveType
    free_indices: set[Axis]

    @property
    def pretty(self) -> str:
        free = ", ".join(
            map(str, sorted(self.free_indices, key=lambda x: int(str(x)[1:])))
        )
        return f"{{{free}}}:{self.type.pretty}"


def tuple_maybe_singleton(*args: yarr.Expr) -> yarr.Expr:
    return yarr.Tuple(args) if len(args) != 1 else args[0]


def untuple_maybe_singleton(target: yarr.Expr, at: int) -> yarr.Expr:
    k = len(target.type.elems)
    return yarr.Untuple(at, k, target) if k != 1 else target


class Axial:
    _axes: Axes
    expr: yarr.Expr

    def __init__(self, axes: Iterable[Axis], array: yarr.Expr):
        self._axes = tuple(axes)
        self.expr = array
        assert len(self._axes) == len(set(self._axes))
        assert all(arr.rank >= len(self._axes) for arr in self.expr.type.elems)

    def free_axis(self, index: Index) -> int | None:
        return self._axes.index(index) if index in self._axes else None

    def positional_axis(self, axis: int) -> int:
        assert 0 <= axis < self.type.type.single.rank
        return len(self._axes) + axis

    @property
    def type(self) -> AxialType:
        axes = set(self._axes)
        return AxialType(self.expr.type.with_rank_delta(-len(axes)), axes)

    @property
    def normal(self) -> yarr.Expr:
        assert not self._axes
        return self.expr

    def along(self, index: Index, size: yarr.Expr) -> "Axial":
        j = self.expr.type.single.rank - self.type.type.single.rank
        i = self.free_axis(index)
        if i is not None:
            pi: list[int] = list(range(self.expr.type.single.rank))
            j -= 1
            pi[i], pi[j] = pi[j], pi[i]
            return Axial(
                tuple(self._axes[pi[i]] for i in range(len(self._axes) - 1)),
                yarr.Transpose(tuple(pi), self.expr),
            )
        else:
            return Axial(
                self._axes,
                yarr.Repeat(j, size, yarr.Unsqueeze((j,), self.expr)),
            )

    def within(self, *args: tuple[Variable, yarr.Expr]) -> "Axial":
        if not args:
            return self
        *rest, (var, bind) = args
        return Axial(self._axes, yarr.Let(var, bind, self.expr)).within(*rest)

    def cons(self, other: "Axial") -> "Axial":
        n = len(self.expr.type.elems)
        m = len(other.expr.type.elems)
        return Axial(
            self._axes,
            tuple_maybe_singleton(
                *(untuple_maybe_singleton(self.expr, i) for i in range(n)),
                *(untuple_maybe_singleton(other.expr, i) for i in range(m)),
            ),
        )

    def slice_tuple(self, start: int, end: int) -> "Axial":
        k = len(self.expr.type.elems)
        assert 0 <= start <= end <= k
        return Axial(
            self._axes,
            tuple_maybe_singleton(
                *(untuple_maybe_singleton(self.expr, i) for i in range(start, end))
            ),
        )

    def aligned(
        self,
        into_axes: Axes,
        *,
        leftpad: bool = True,
        repeats: dict[Axis, yarr.Expr] | None = None,
    ) -> yarr.Expr:
        k = len(self.expr.type.elems)

        def local_align(e: yarr.Expr):
            return _align(
                e,
                self._axes,
                into_axes,
                leftpad=leftpad,
                repeats=repeats if repeats is not None else {},
            )

        if k != 1:
            expr = yarr.Tuple(
                tuple(local_align(yarr.Untuple(i, k, self.expr)) for i in range(k))
            )
        else:
            expr = local_align(self.expr)

        return expr


# TODO: This is a silly baseline alignment algorithm.
def _alignment(*args: Axes) -> Axes:
    seen = []
    for axes in args:
        for axis in axes:
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def _align(
    expr: yarr.Expr,
    axes: Axes,
    into_axes: Axes,
    *,
    leftpad: bool,
    repeats: dict[Axis, yarr.Expr],
) -> yarr.Expr:
    assert len(axes) <= len(into_axes)
    free_pi = [axes.index(axis) for axis in sorted(axes, key=into_axes.index)]
    pos_id = list(range(len(axes), expr.type.single.rank))
    expr = yarr.Transpose(tuple(free_pi + pos_id), expr)
    expands = [(i, axis) for i, axis in enumerate(into_axes) if axis not in axes]
    if not leftpad:
        while expands and not expands[0][0]:
            expands = [(x - 1, axis) for x, axis in expands[1:]]
    expanded_axes = tuple(i for i, _ in expands)
    expr = yarr.Unsqueeze(expanded_axes, expr)
    for i, axis in expands:
        if axis in repeats:
            expr = yarr.Repeat(i, repeats[axis], expr)
    return expr

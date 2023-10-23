import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable, ParamSpec, TypeAlias, cast

import numpy

from ein.symbols import Index, Variable
from ein.type_system import PrimitiveArrayType

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


@dataclass
class Env:
    var: dict[Variable, array_calculus.Expr]
    idx: dict[Index, int]

    def copy(self) -> "Env":
        return Env(self.var.copy(), self.idx.copy())


class Axial:
    axes: Axes
    array: array_calculus.Expr

    def __init__(self, axes: Iterable[Axis], value: array_calculus.Expr):
        self.axes = tuple(axes)
        self.array = value

    @property
    def type(self) -> AxialType:
        free_indices = {index for index in self.axes if isinstance(index, Index)}
        return AxialType(
            PrimitiveArrayType(rank=len(self.axes) - len(free_indices)), free_indices
        )

    @staticmethod
    def of_normal(array: array_calculus.Expr):
        return Axial(reversed(range(array.rank)), array)

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


Expr: TypeAlias = (
    "Const | Range | Var | At | Dim | Fold | Gather | Vector | Reduce | Elementwise"
)


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    def __post_init__(self):
        assert self.type

    @property
    @abc.abstractmethod
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        ...

    @property
    @abc.abstractmethod
    def type(self) -> AxialType:
        ...


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    array: numpy.ndarray

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"array": self.array}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(PrimitiveArrayType(self.array.ndim), set())


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    index: Index
    size: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, {self.size}

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        return AxialType(PrimitiveArrayType(rank=0), {self.index})


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable
    var_type: PrimitiveArrayType

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(self.var_type, set())


@dataclass(frozen=True, eq=False)
class At(AbstractExpr):
    index: Index

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(PrimitiveArrayType(rank=0), set())


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    pos: int
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"pos": self.pos}, {self.target}

    @cached_property
    def type(self) -> AxialType:
        assert 0 <= self.pos < self.target.type.array_type.rank
        assert not self.target.type.free_indices
        return AxialType(PrimitiveArrayType(rank=0), set())


@dataclass(frozen=True, eq=False)
class Fold(AbstractExpr):
    index: Index
    acc: Var
    init: Expr
    size: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc}, {self.init, self.size, self.body}

    @cached_property
    def type(self) -> AxialType:
        assert self.size.type.array_type == PrimitiveArrayType(
            rank=0
        ), "Expected scalar fold size"
        # TODO: This, similarly to loop-dependent arrays and reductions, is possible to implement.
        #  But it does not have any benefits from vectorisation, so maybe it doesn't have to be.
        assert not self.size.type.free_indices, "Expected loop-independent fold size"
        assert (
            self.acc.var_type == self.init.type.array_type
        ), "Mismatched accumulator and initialiser type"
        # FIXME: This should work?
        assert not self.body.type.free_indices, "Expected outer-loop-independent body"
        return self.body.type


@dataclass(frozen=True, eq=False)
class Gather(AbstractExpr):
    target: Expr
    item: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.target, self.item}

    @cached_property
    def type(self) -> AxialType:
        assert self.target.type.array_type.rank > 0, "Expected vector target"
        assert self.item.type.array_type.rank == 0, "Expected scalar item"
        return AxialType(
            PrimitiveArrayType(rank=self.target.type.array_type.rank - 1),
            {
                index
                for index in self.target.type.free_indices | self.item.type.free_indices
            },
        )


@dataclass(frozen=True, eq=False)
class Vector(AbstractExpr):
    index: Index
    size: Expr
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, {self.size, self.target}

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        return AxialType(
            self.target.type.array_type.in_vector,
            self.target.type.free_indices - {self.index},
        )


@dataclass(frozen=True, eq=False)
class Reduce(AbstractExpr):
    ufunc: numpy.ufunc
    index: Index
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"ufunc": numpy.ufunc, "index": self.index}, {self.target}

    @cached_property
    def type(self, *args: AxialType) -> AxialType:
        assert not self.target.type.array_type.rank, "Expected scalar reduction"
        assert (
            self.index in self.target.type.free_indices
        ), "Can only reduce over free index"
        return AxialType(
            PrimitiveArrayType(rank=0), self.target.type.free_indices - {self.index}
        )


@dataclass(frozen=True, eq=False)
class Elementwise(AbstractExpr):
    ufunc: numpy.ufunc
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"ufunc": self.ufunc}, set(self.operands)

    @cached_property
    def type(self) -> AxialType:
        assert all(
            not op.type.array_type.rank for op in self.operands
        ), "Expected scalar elementwise"
        return AxialType(
            PrimitiveArrayType(rank=0),
            {index for op in self.operands for index in op.type.free_indices},
        )

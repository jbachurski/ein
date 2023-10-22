import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, ParamSpec, TypeAlias

import numpy

from ein.symbols import Index, Variable
from ein.type_system import PrimitiveArrayType


@dataclass(frozen=True)
class AxialType:
    array_type: PrimitiveArrayType
    free_indices: set[Index]


P = ParamSpec("P")
Axis: TypeAlias = Index | int
Axes: TypeAlias = tuple[Axis, ...]


@dataclass
class Env:
    var: dict[Variable, numpy.ndarray]
    idx: dict[Index, int]

    def copy(self) -> "Env":
        return Env(self.var.copy(), self.idx.copy())


class Axial:
    axes: Axes
    value: numpy.ndarray

    def __init__(self, axes: Iterable[Axis], value: numpy.ndarray):
        self.axes = tuple(axes)
        self.value = value

    def __repr__(self):
        return f"{self.axes}:{self.value!r}"

    @property
    def type(self) -> AxialType:
        free_indices = {index for index in self.axes if isinstance(index, Index)}
        return AxialType(
            PrimitiveArrayType(rank=len(self.axes) - len(free_indices)), free_indices
        )

    @staticmethod
    def of_normal(array: numpy.ndarray):
        return Axial(reversed(range(array.ndim)), array)

    @property
    def normal(self) -> numpy.ndarray:
        assert not self.type.free_indices
        rank = self.type.array_type.rank
        inv: list[int | None] = [None for _ in range(rank)]
        for i, p in enumerate(self.axes):
            assert isinstance(p, int)
            # Axes are numbered in reverse order
            inv[rank - p - 1] = i
        return numpy.transpose(self.value, inv)


# TODO: This is a silly baseline alignment algorithm.
def alignment(*args: Axes) -> Axes:
    seen = []
    for axes in args:
        for axis in axes:
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def align(target: Axial, into_axes: Axes) -> numpy.ndarray:
    transposition = tuple(
        target.axes.index(axis) for axis in sorted(target.axes, key=into_axes.index)
    )
    expands = tuple(i for i, axis in enumerate(into_axes) if axis not in target.axes)
    array = target.value
    array = numpy.transpose(array, transposition)
    array = numpy.expand_dims(array, expands)
    return array


@dataclass(frozen=True, eq=False)
class Expr(abc.ABC):
    def __post_init__(self):
        assert self.type

    @property
    @abc.abstractmethod
    def dependencies(self) -> tuple["Expr", ...]:
        ...

    @property
    @abc.abstractmethod
    def type(self) -> AxialType:
        ...

    @abc.abstractmethod
    def apply(self, dependencies: dict["Expr", Axial], env: Env) -> Axial:
        ...

    def execute(self, env: Env, base: dict["Expr", Axial] | None = None):
        results: dict[Expr, Axial] = base.copy() if base is not None else {}

        def go(op: Expr) -> Axial:
            print(op)
            if op not in results:
                results[op] = op.apply(
                    {sub: go(sub) for sub in op.dependencies}, env=env
                )
            return results[op]

        return go(self)


@dataclass(frozen=True, eq=False)
class Const(Expr):
    array: numpy.ndarray

    @property
    def dependencies(self) -> tuple[Expr, ...]:
        return ()

    @property
    def type(self) -> AxialType:
        return AxialType(PrimitiveArrayType(self.array.ndim), set())

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        return Axial.of_normal(self.array)


@dataclass(frozen=True, eq=False)
class Range(Expr):
    index: Index
    size: Expr

    @property
    def dependencies(self) -> tuple[Expr, ...]:
        return (self.size,)

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        return AxialType(PrimitiveArrayType(rank=0), {self.index})

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        return Axial([self.index], numpy.arange(int(dependencies[self.size].value)))


@dataclass(frozen=True, eq=False)
class Var(Expr):
    var: Variable
    var_type: PrimitiveArrayType

    @property
    def dependencies(self) -> tuple[Expr, ...]:
        return ()

    @property
    def type(self) -> AxialType:
        return AxialType(self.var_type, set())

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        return Axial.of_normal(env.var[self.var])


@dataclass(frozen=True, eq=False)
class At(Expr):
    index: Index

    @property
    def dependencies(self) -> tuple["Expr", ...]:
        return ()

    @property
    def type(self) -> AxialType:
        return AxialType(PrimitiveArrayType(rank=0), set())

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        return Axial([], numpy.array(env.idx[self.index]))


@dataclass(frozen=True, eq=False)
class Dim(Expr):
    pos: int
    target: Expr

    @property
    def dependencies(self) -> tuple["Expr", ...]:
        return (self.target,)

    @cached_property
    def type(self) -> AxialType:
        assert 0 <= self.pos < self.target.type.array_type.rank
        assert not self.target.type.free_indices
        return AxialType(PrimitiveArrayType(rank=0), set())

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        target = dependencies[self.target]
        pos = target.type.array_type.rank - self.pos - 1
        return Axial([], target.value.shape[target.axes.index(pos)])


@dataclass(frozen=True, eq=False)
class Fold(Expr):
    index: Index
    acc: Var
    size: Expr
    init: Expr
    body: Expr

    @property
    def dependencies(self) -> tuple["Expr", ...]:
        return self.size, self.init

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

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        env = env.copy()
        env.var[self.acc.var] = dependencies[self.init].normal
        for i in range(int(dependencies[self.size].value)):
            env.idx[self.index] = i
            env.var[self.acc.var] = self.body.execute(env, dependencies).normal
        return Axial.of_normal(env.var[self.acc.var])


@dataclass(frozen=True, eq=False)
class Gather(Expr):
    target: Expr
    item: Expr

    @property
    def dependencies(self):
        return self.target, self.item

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

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        target, item = dependencies[self.target], dependencies[self.item]
        used_axes = alignment(target.axes, item.axes)
        k = used_axes.index(target.type.array_type.rank - 1)
        result = numpy.take_along_axis(
            align(target, used_axes), align(item, used_axes), axis=k
        )
        return Axial(used_axes[:k] + used_axes[k + 1 :], numpy.squeeze(result, axis=k))


@dataclass(frozen=True, eq=False)
class Vector(Expr):
    index: Index
    size: Expr
    target: Expr

    @property
    def dependencies(self):
        return self.size, self.target

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        return AxialType(
            self.target.type.array_type.in_vector,
            self.target.type.free_indices - {self.index},
        )

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        size, target = dependencies[self.size], dependencies[self.target]
        if self.index in target.axes:
            return Axial(
                (
                    target.type.array_type.rank if axis == self.index else axis
                    for axis in target.axes
                ),
                target.value,
            )
        else:
            return Axial(
                (target.type.array_type.rank, *target.axes),
                numpy.repeat(numpy.expand_dims(target.value, axis=0), int(size.value)),
            )


@dataclass(frozen=True, eq=False)
class Reduce(Expr):
    ufunc: numpy.ufunc
    index: Index
    target: Expr

    @property
    def dependencies(self):
        return (self.target,)

    @cached_property
    def type(self, *args: AxialType) -> AxialType:
        assert not self.target.type.array_type.rank, "Expected scalar reduction"
        assert (
            self.index in self.target.type.free_indices
        ), "Can only reduce over free index"
        return AxialType(
            PrimitiveArrayType(rank=0), self.target.type.free_indices - {self.index}
        )

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        target = dependencies[self.target]
        # FIXME: target might not depend on the reduction index, though this is a rather degenerate corner case.
        #  This manifests with a failing `axes.index`.
        return Axial(
            (axis for axis in target.axes if axis != self.index),
            self.ufunc.reduce(target.value, axis=target.axes.index(self.index)),
        )


@dataclass(frozen=True, eq=False)
class Elementwise(Expr):
    ufunc: numpy.ufunc
    operands: tuple[Expr, ...]

    @property
    def dependencies(self) -> tuple["Expr", ...]:
        return self.operands

    @cached_property
    def type(self) -> AxialType:
        assert all(
            not op.type.array_type.rank for op in self.operands
        ), "Expected scalar elementwise"
        return AxialType(
            PrimitiveArrayType(rank=0),
            {index for op in self.operands for index in op.type.free_indices},
        )

    def apply(self, dependencies: dict[Expr, Axial], env: Env) -> Axial:
        used_axes = alignment(*(dependencies[op].axes for op in self.operands))
        return Axial(
            used_axes,
            self.ufunc(*(align(dependencies[op], used_axes) for op in self.operands)),
        )

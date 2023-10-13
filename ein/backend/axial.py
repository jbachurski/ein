import abc
from dataclasses import dataclass
from typing import Iterable, ParamSpec, TypeAlias

import numpy

from ein.backend.staged import Staged
from ein.symbols import Index, Variable
from ein.type_system import Type


@dataclass(frozen=True)
class AxialType:
    type: Type
    free_indices: set[Index]


P = ParamSpec("P")
Axis: TypeAlias = Index | int
Axes: TypeAlias = tuple[Axis, ...]
StagedAxial: TypeAlias = Staged["Operation", AxialType]
Env: TypeAlias = dict[Variable, numpy.ndarray]


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
        return AxialType(Type(rank=len(self.axes) - len(free_indices)), free_indices)


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


@dataclass(frozen=True)
class Operation(abc.ABC):
    def stage(self, *args: StagedAxial) -> StagedAxial:
        return Staged(self, self.type(*(arg.type for arg in args)), args)

    @abc.abstractmethod
    def type(self, *args: AxialType) -> AxialType:
        ...

    @abc.abstractmethod
    def apply(self, *args: Axial, env: Env) -> Axial:
        ...


@dataclass(frozen=True, eq=False)
class Const(Operation):
    array: numpy.ndarray

    def type(self, *args: AxialType) -> AxialType:
        () = args
        return AxialType(Type(self.array.ndim), set())

    def apply(self, *args: Axial, env: Env) -> Axial:
        () = args
        return Axial(reversed(range(self.array.ndim)), self.array)


@dataclass(frozen=True)
class Range(Operation):
    index: Index

    def type(self, *args: AxialType) -> AxialType:
        (size,) = args
        assert not size.type.rank, "Expected scalar size"
        assert not size.free_indices, "Expected loop-independent size"
        return AxialType(Type(0), {self.index})

    def apply(self, *args: Axial, env: Env) -> Axial:
        (size,) = args
        return Axial([self.index], numpy.arange(int(size.value)))


@dataclass(frozen=True)
class Var(Operation):
    var: Variable
    var_type: Type

    def type(self, *args: AxialType) -> AxialType:
        () = args
        return AxialType(self.var_type, set())

    def apply(self, *args: Axial, env: Env) -> Axial:
        () = args
        return Axial(reversed(range(self.var_type.rank)), env[self.var])


@dataclass(frozen=True)
class Dim(Operation):
    pos: int

    def type(self, *args: AxialType) -> AxialType:
        (target,) = args
        assert 0 <= self.pos < target.type.rank
        assert not target.free_indices
        return AxialType(Type(rank=0), set())

    def apply(self, *args: Axial, env: Env) -> Axial:
        (target,) = args
        pos = target.type.type.rank - self.pos - 1
        return Axial([], target.value.shape[target.axes.index(pos)])


@dataclass(frozen=True)
class Gather(Operation):
    def type(self, *args: AxialType) -> AxialType:
        (target, item) = args
        assert target.type.rank > 0, "Expected vector target"
        assert item.type.rank == 0, "Expected scalar item"
        return AxialType(
            Type(rank=target.type.rank - 1),
            {index for index in target.free_indices | item.free_indices},
        )

    def apply(self, *args: Axial, env: Env) -> Axial:
        (target, item) = args
        used_axes = alignment(target.axes, item.axes)
        k = used_axes.index(target.type.type.rank - 1)
        result = numpy.take_along_axis(
            align(target, used_axes), align(item, used_axes), axis=k
        )
        return Axial(used_axes[:k] + used_axes[k + 1 :], numpy.squeeze(result, axis=k))


@dataclass(frozen=True)
class Vector(Operation):
    index: Index

    def type(self, *args: AxialType) -> AxialType:
        (size, target) = args
        assert not size.type.rank, "Expected scalar size"
        assert not size.free_indices, "Expected loop-independent size"
        return AxialType(
            Type(rank=target.type.rank + 1), target.free_indices - {self.index}
        )

    def apply(self, *args: Axial, env: Env) -> Axial:
        (size, target) = args
        if self.index in target.axes:
            return Axial(
                (
                    target.type.type.rank if axis == self.index else axis
                    for axis in target.axes
                ),
                target.value,
            )
        else:
            return Axial(
                (target.type.type.rank, *target.axes),
                numpy.repeat(numpy.expand_dims(target.value, axis=0), int(size.value)),
            )


@dataclass(frozen=True)
class Reduce(Operation):
    ufunc: numpy.ufunc
    index: Index

    def type(self, *args: AxialType) -> AxialType:
        (size, target) = args
        assert not size.type.rank, "Expected scalar size"
        assert not size.free_indices, "Expected loop-independent size"
        assert not target.type.rank, "Expected scalar reduction"
        return AxialType(Type(rank=0), target.free_indices - {self.index})

    def apply(self, *args: Axial, env: Env) -> Axial:
        (size, target) = args
        return Axial(
            (axis for axis in target.axes if axis != self.index),
            self.ufunc.reduce(target.value, axis=target.axes.index(self.index)),
        )


@dataclass(frozen=True)
class Elementwise(Operation):
    ufunc: numpy.ufunc

    def type(self, *args: AxialType) -> AxialType:
        assert all(not arg.type.rank for arg in args), "Expected scalar elementwise"
        return AxialType(
            Type(rank=0), {index for arg in args for index in arg.free_indices}
        )

    def apply(self, *args: Axial, env: Env) -> Axial:
        used_axes = alignment(*(arg.axes for arg in args))
        return Axial(used_axes, self.ufunc(*(align(arg, used_axes) for arg in args)))

import abc
import operator
from functools import cached_property
from typing import Any, Iterable, TypeAlias, cast

import numpy
import numpy.typing

from ein.symbols import Index, Variable

from .node import Node, identity, node

Axial: TypeAlias = "Const | Range | Var | Dim | Gather | Vector | Elementwise | Reduce"
Axis = Index | int


def _all_unique(xs: Iterable[Any]) -> bool:
    xs = list(xs)
    return len(set(xs)) == len(xs)


def alignment(*args: tuple[Axis, ...]) -> tuple[Axis, ...]:
    # TODO: This is a silly baseline alignment algorithm.
    seen = []
    for axes in args:
        for axis in axes:
            if axis not in seen:
                seen.append(axis)
    return tuple(seen)


def align(
    source_axes: tuple[Any, ...], target_axes: tuple[Any, ...], array: Node
) -> Node:
    transposition = tuple(
        source_axes.index(axis) for axis in sorted(source_axes, key=target_axes.index)
    )
    expands = tuple(i for i, axis in enumerate(target_axes) if axis not in source_axes)
    array = node(numpy.transpose)(array, transposition)
    array = node(numpy.expand_dims)(array, expands)
    return array


class AbstractAxial(abc.ABC):
    axes: tuple[Axis, ...]
    free_sizes: dict[Index, Axial]

    def __init__(self, axes: Iterable[Axis], free_sizes: dict[Index, Axial]):
        self.axes = tuple(axes)
        self.free_sizes = free_sizes
        assert _all_unique(axes), "Expected axes to be unique"
        assert self.free_axes <= set(
            free_sizes
        ), "Expected all free indices to have a size"
        assert (
            set(range(self.rank)) == self.pos_axes
        ), "Expected positional axes to form a permutation"

    @abc.abstractmethod
    @cached_property
    def graph(self) -> Node:
        ...

    @property
    def pos_axes(self) -> set[int]:
        return {pos for pos in self.axes if isinstance(pos, int)}

    @property
    def free_axes(self):
        return {index for index in self.axes if isinstance(index, Index)}

    @property
    def rank(self) -> int:
        return len(self.axes) - len(self.free_sizes)

    def unwrap_scalar_axial(self) -> tuple[Index, ...]:
        assert self.rank == 0, "A scalar axial was expected at this point"
        return cast(tuple[Index, ...], self.axes)


class Const(AbstractAxial):
    array: numpy.array

    def __init__(self, array: numpy.typing.ArrayLike):
        self.array = numpy.array(array)
        super().__init__(range(self.array.ndim), {})

    @cached_property
    def graph(self) -> Node:
        return node(identity)(self.array)


class Range(AbstractAxial):
    index: Index
    size: Axial

    def __init__(self, index: Index, size: Axial):
        self.index = index
        self.size = size
        super().__init__((index,), {index: size})

    @cached_property
    def graph(self) -> Node:
        return node(numpy.arange)(self.size.graph)


class Var(AbstractAxial):
    var: Variable

    def __init__(self, var: Variable, rank: int):
        self.var = var
        super().__init__(range(rank), {})

    @cached_property
    def graph(self) -> Node:
        return node(identity)(self.var)


class Dim(AbstractAxial):
    operand: Axial
    pos: int

    def __init__(self, operand: Axial, pos: int):
        self.operand = operand
        self.pos = pos
        super().__init__((), {})

    @cached_property
    def graph(self) -> Node:
        shape = node(numpy.shape)(self.operand.graph)
        return node(operator.getitem)(shape, self.operand.axes.index(self.pos))


class Gather(AbstractAxial):
    target: Axial
    item: Axial
    alignment: tuple[Axis, ...]

    def __init__(self, target: Axial, item: Axial):
        self.target = target
        self.item = item
        self.pos = self.target.rank - 1
        assert self.pos >= 0, "Cannot Gather from a scalar target"
        self.alignment = alignment(self.target.axes, self.item.axes)
        super().__init__(
            [axis for axis in self.alignment if axis != self.pos],
            self.target.free_sizes | self.item.free_sizes,
        )

    @cached_property
    def graph(self) -> Node:
        if self.pos not in self.target.axes:
            return self.target.graph
        k = self.target.axes.index(self.pos)
        target = align(self.target.axes, self.alignment, self.target.graph)
        item = align(self.item.axes, self.alignment, self.item.graph)
        array = node(numpy.take_along_axis)(target, item, axis=k)
        return node(numpy.squeeze)(array, axis=k)


class Vector(AbstractAxial):
    index: Index
    size: Axial
    target: Axial

    def __init__(self, index: Index, size: Axial, target: Axial):
        self.index = index
        self.size = size
        self.target = target
        super().__init__(
            [self.target.rank] + [axis for axis in self.axes if axis != index],
            {
                index_: size
                for index_, size in self.free_sizes.items()
                if index != index_
            },
        )

    @cached_property
    def graph(self) -> Node:
        if self.index in self.target.axes:
            return align(
                self.target.axes,
                (self.index, *(axis for axis in self.axes if axis != self.index)),
                self.target.graph,
            )
        else:
            return node(numpy.repeat)(
                node(numpy.expand_dims)(self.target.graph, axis=0),
                self.size.graph,
                axis=0,
            )


class Elementwise(AbstractAxial):
    ufunc: numpy.ufunc
    operands: tuple[Axial, ...]

    def __init__(self, ufunc: numpy.ufunc, operands: tuple[Axial, ...]):
        self.ufunc = ufunc
        self.operands = operands
        free_sizes: dict[Index, Axial] = {}
        for operand in operands:
            free_sizes |= operand.free_sizes
        super().__init__(
            alignment(*(operand.unwrap_scalar_axial() for operand in operands)),
            free_sizes,
        )

    @cached_property
    def graph(self) -> Node:
        return node(self.ufunc)(
            *(
                align(operand.axes, self.axes, operand.graph)
                for operand in self.operands
            )
        )


class Reduce(AbstractAxial):
    ufunc: numpy.ufunc
    index: Index
    size: Axial
    target: Axial

    def __init__(self, ufunc: numpy.ufunc, index: Index, size: Axial, target: Axial):
        self.ufunc = ufunc
        self.index = index
        self.size = size
        self.target = target
        super().__init__(
            (axis for axis in target.axes if axis != index),
            {
                index_: size
                for index_, size in self.target.free_sizes.items()
                if index_ != index
            },
        )

    @cached_property
    def graph(self) -> Node:
        if self.index not in self.target.axes:
            raise NotImplementedError("Reductions with unused axes are not implemented")
        return node(self.ufunc.reduce)(
            self.target.graph, axis=self.target.axes.index(self.index)
        )

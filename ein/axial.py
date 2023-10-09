import abc
import operator
from functools import cached_property
from typing import TypeAlias

import numpy
import numpy.typing

from .calculus import Index, Variable
from .node import Node, identity, node

Axial: TypeAlias = (
    "Constant | FromVariable | Dim | Range | Ufunc | Where | UfuncReduce | Gather"
)


def align(
    source_axes: tuple[Index, ...], target_axes: tuple[Index, ...], array: Node
) -> Node:
    transposition = tuple(
        source_axes.index(axis) for axis in sorted(source_axes, key=target_axes.index)
    )
    expands = tuple(i for i, axis in enumerate(target_axes) if axis not in source_axes)
    array = node(numpy.transpose)(array, transposition)
    array = node(numpy.expand_dims)(array, expands)
    return array


class AbstractAxial(abc.ABC):
    axes: tuple[Index, ...]
    shape: dict[Index, Axial]

    @abc.abstractmethod
    @cached_property
    def graph(self) -> Node:
        ...


class Constant(AbstractAxial):
    array: numpy.array

    def __init__(self, array: numpy.array, axes: tuple[Index, ...]):
        self.array = array
        self.axes = axes
        self.shape = {
            axis: Constant(numpy.array(d), ()) for axis, d in zip(axes, array.shape)
        }

    @cached_property
    def graph(self) -> Node:
        return node(identity)(self.array)


class FromVariable(AbstractAxial):
    var: Variable

    def __init__(self, var: Variable, axes: tuple[Index, ...]):
        self.var = var
        self.axes = axes
        self.shape = {axis: Dim(self, axis) for axis in axes}

    @cached_property
    def graph(self) -> Node:
        return node(identity)(self.var)


class Dim(AbstractAxial):
    operand: Axial
    axis: Index

    def __init__(self, operand: Axial, axis: Index):
        self.operand = operand
        self.axis = axis
        self.axes = ()
        self.shape = {}

    @cached_property
    def graph(self) -> Node:
        shape = node(numpy.shape)(self.operand.graph)
        return node(operator.getitem)(shape, self.operand.axes.index(self.axis))


class Range(AbstractAxial):
    axis: Index
    size: Axial

    def __init__(self, axis: Index, size: Axial):
        self.size = size
        self.axis = axis
        self.axes = (axis,)
        self.shape = {axis: size}

    @cached_property
    def graph(self) -> Node:
        return node(numpy.arange)(self.size.graph)


class Ufunc(AbstractAxial):
    ufunc: numpy.ufunc
    operands: tuple[Axial, ...]

    # FIXME: Alignment could be decided globally rather than locally
    @staticmethod
    def alignment(*args: tuple[Index, ...]) -> tuple[Index, ...]:
        seen = []
        for axes in args:
            for axis in axes:
                if axis not in seen:
                    seen.append(axis)
        return tuple(seen)

    def __init__(self, ufunc: numpy.ufunc, operands: tuple[Axial, ...]):
        self.ufunc = ufunc
        self.operands = operands
        self.axes = self.alignment(*(operand.axes for operand in operands))
        self.shape = {}
        for operand in operands:
            self.shape |= operand.shape

    @cached_property
    def graph(self) -> Node:
        return node(self.ufunc)(
            *(
                align(operand.axes, self.axes, operand.graph)
                for operand in self.operands
            )
        )


class Where(AbstractAxial):
    cond: Axial
    true: Axial
    false: Axial

    def __init__(self, cond: Axial, true: Axial, false: Axial):
        self.cond, self.true, self.false = cond, true, false
        self.axes = Ufunc.alignment(cond.axes, true.axes, false.axes)
        self.shape = cond.shape | true.shape | false.shape

    @cached_property
    def graph(self) -> Node:
        return node(numpy.where)(
            align(self.cond.axes, self.axes, self.cond.graph),
            align(self.true.axes, self.axes, self.true.graph),
            align(self.false.axes, self.axes, self.false.graph),
        )


class UfuncReduce(AbstractAxial):
    ufunc: numpy.ufunc
    axis: Index
    operand: Axial

    def __init__(self, ufunc: numpy.ufunc, axis: Index, operand: Axial):
        self.ufunc = ufunc
        self.axis = axis
        self.operand = operand
        self.axes = tuple(
            existing_axis for existing_axis in operand.axes if existing_axis != axis
        )
        self.shape = {
            existing_axis: size
            for existing_axis, size in operand.shape.items()
            if existing_axis != axis
        }

    @cached_property
    def graph(self) -> Node:
        if self.axis not in self.operand.axes:
            raise NotImplementedError("Reductions with unused axes are not implemented")
        return node(self.ufunc.reduce)(
            self.operand.graph, axis=self.operand.axes.index(self.axis)
        )


class Gather(AbstractAxial):
    axis: Index
    operand: Axial
    item: Axial

    def __init__(self, axis: Index, operand: Axial, item: Axial):
        self.axis = axis
        self.operand = operand
        self.item = item
        self.pre_axes = Ufunc.alignment(self.operand.axes, self.item.axes)
        self.axes = tuple(
            existing_axis for existing_axis in self.pre_axes if existing_axis != axis
        )
        self.shape = {
            existing_axis: size
            for existing_axis, size in (self.operand.shape | self.item.shape).items()
            if existing_axis != axis
        }

    @cached_property
    def graph(self) -> Node:
        if self.axis not in self.operand.axes:
            return self.operand.graph
        if (
            isinstance(self.item, Range)
            and self.item.size == self.operand.shape[self.axis]
        ):
            return self.operand.graph
        k = self.operand.axes.index(self.axis)
        operand = align(self.operand.axes, self.pre_axes, self.operand.graph)
        item = align(self.item.axes, self.pre_axes, self.item.graph)
        array = node(numpy.take_along_axis)(operand, item, axis=k)
        return node(numpy.squeeze)(array, axis=k)


class Expand(AbstractAxial):
    operand: Axial

    def __init__(self, operand: Axial, axes: tuple[Index, ...]):
        self.operand = operand
        self.axes = axes
        self.shape = operand.shape

    @cached_property
    def graph(self) -> Node:
        return node(numpy.reshape)(
            align(self.operand.axes, self.axes, self.operand.graph),
            tuple(self.shape[axis].graph for axis in self.axes),
        )

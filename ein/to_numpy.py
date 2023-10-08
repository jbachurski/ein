from functools import cache
from typing import Any, assert_never

import numpy

from . import axial, axial_calculus, to_axial
from .axial_calculus import ValueAxis, VariableAxis
from .calculus import Index, Variable
from .node import Node


def transform(
    program: axial_calculus.Expr,
    program_axes: tuple[Index, ...],
    ranks: dict[Variable, int],
) -> Node:
    @cache
    def go(expr: axial_calculus.Expr) -> axial.Axial:
        match expr:
            case axial_calculus.AbstractScalarReduction(axis, body):
                return axial.UfuncReduce(expr.ufunc, axis, go(body))
            case axial_calculus.Get(operand, item, axis):
                return axial.Gather(axis, go(operand), go(item))
            case axial_calculus.Const(value):
                value_axes = tuple(
                    ValueAxis(value, rank) for rank in range(value.array.ndim)
                )
                return axial.Constant(value.array, value_axes)
            case axial_calculus.Range(axis, size):
                return axial.Range(axis, go(size))
            case axial_calculus.Var(var):
                var_axes = tuple(VariableAxis(var, rank) for rank in range(ranks[var]))
                return axial.FromVariable(var, var_axes)
            case axial_calculus.Dim(operand, axis):
                return axial.Dim(go(operand), axis)
            case axial_calculus.Switch(cond, false, true):
                return axial.Where(go(cond), go(false), go(true))
            case axial_calculus.Negate(operands):
                (operand,) = operands
                return axial.Ufunc(numpy.negative, (go(operand),))
            case axial_calculus.Reciprocal(operands):
                (operand,) = operands
                return axial.Ufunc(numpy.reciprocal, (go(operand),))
            case axial_calculus.Add(operands):
                (first, second) = operands
                return axial.Ufunc(numpy.add, (go(first), go(second)))
            case axial_calculus.Multiply(operands):
                (first, second) = operands
                return axial.Ufunc(numpy.multiply, (go(first), go(second)))
            case _:
                assert_never(expr)

    result = axial.Expand(go(program), program_axes)
    return result.graph


def repr_node(nodes, i):
    node = nodes[i]
    args, kwargs = node.mapped_args(
        lambda var: str(var), lambda arg: f"%{nodes.index(arg)}", lambda x: str(x)
    )
    return f"%{i} = {node.fun.__name__}({', '.join(map(str, args))}, {', '.join(f'{k}={v}' for k, v in kwargs.items())})"


def stage(program, ranks: dict[Variable, int]):
    axial_program, axial_program_axes = to_axial.transform(program, ranks)
    return transform(axial_program, axial_program_axes, ranks)


def interpret(program, env: dict[Variable, numpy.ndarray], debug: bool = False):
    numpy_program = stage(program, {var: array.ndim for var, array in env.items()})
    nodes = list(numpy_program.linearize())
    results: list[Any] = []
    for i, node in enumerate(nodes):
        if debug:
            print(repr_node(nodes, i))
        args, kwargs = node.mapped_args(
            lambda var: env[var], lambda arg: results[nodes.index(arg)], lambda x: x
        )
        results.append(node.fun(*args, **kwargs))
        if debug:
            print(results[-1])
    return results[-1]

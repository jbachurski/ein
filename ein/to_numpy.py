from functools import cache
from typing import Any, assert_never

import numpy

from . import axial, calculus, to_pure_axial
from .calculus import Index, ValueAxis, Variable, VariableAxis
from .node import Node


def transform(
    program: calculus.Expr,
    program_axes: tuple[Index, ...],
    ranks: dict[Variable, int],
) -> Node:
    @cache
    def go(expr: calculus.Expr) -> axial.Axial:
        match expr:
            case calculus.Get(operand, item, axis):
                assert isinstance(axis, Index)
                return axial.Gather(axis, go(operand), go(item))
            case calculus.Const(value):
                value_axes = tuple(
                    ValueAxis(value, rank) for rank in range(value.array.ndim)
                )
                return axial.Constant(value.array, value_axes)
            case calculus.Range(axis, size):
                return axial.Range(axis, go(size))
            case calculus.Var(var):
                var_axes = tuple(VariableAxis(var, rank) for rank in range(ranks[var]))
                return axial.FromVariable(var, var_axes)
            case calculus.Dim(operand, axis):
                assert isinstance(axis, Index)
                return axial.Dim(go(operand), axis)
            case calculus.Where(cond, false, true):
                return axial.Where(go(cond), go(false), go(true))
            case calculus.AbstractScalarAxisReduction(axis, body):
                return axial.UfuncReduce(expr.ufunc, axis, go(body))
            case calculus.AbstractScalarOperator(operands):
                return axial.Ufunc(
                    expr.ufunc, tuple(go(operand) for operand in operands)
                )
            case calculus.Vec(_, _, _) | calculus.At(
                _
            ) | calculus.AbstractScalarReduction(_, _, _):
                raise NotImplementedError(
                    "Positional axes are not supported in the numpy backend"
                )
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
    axial_program, axial_program_axes = to_pure_axial.transform(program, ranks)
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

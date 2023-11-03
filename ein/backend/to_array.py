from functools import cache
from typing import assert_never

import numpy

from ein import calculus
from ein.symbols import Index, Variable
from ein.type_system import to_float

from . import array_calculus, axial
from .axial import Axial


def transform(program: calculus.Expr) -> array_calculus.Expr:
    transformed: dict[calculus.Expr, Axial] = {}

    def _go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_vars: dict[Index, Variable],
    ) -> Axial:
        match expr:
            case calculus.Const(value):
                return Axial.of_normal(
                    array_calculus.Const(value.array),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.At(index):
                if index in index_sizes:  # vector index
                    return Axial(
                        [index],
                        array_calculus.Range(index_sizes[index]),
                        expr.type.primitive_type.single.kind,
                    )
                else:  # fold index
                    return Axial(
                        [],
                        array_calculus.Var(index_vars[index], 0),
                        expr.type.primitive_type.single.kind,
                    )
            case calculus.Var(var, var_type):
                return Axial.of_normal(
                    array_calculus.Var(var, var_type.primitive_type.single.rank),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Dim(target_, pos):
                target = go(target_, index_sizes, index_vars)
                pos = target.type.array_type.rank - pos - 1
                return Axial(
                    [],
                    array_calculus.Dim(target.axes.index(pos), target.array),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Fold(index, size_, body_, init_, acc):
                init = go(init_, index_sizes, index_vars)
                size = go(size_, index_sizes, index_vars)
                index_var = Variable()
                body = go(body_, index_sizes, index_vars | {index: index_var})
                return Axial.of_normal(
                    array_calculus.Fold(
                        index_var, acc.var, init.normal, size.normal, body.normal
                    ),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Get(target_, item_):
                target, item = go(target_, index_sizes, index_vars), go(
                    item_, index_sizes, index_vars
                )
                used_axes = axial.alignment(target.axes, item.axes)
                k = used_axes.index(target.type.array_type.rank - 1)
                result = array_calculus.Gather(
                    k,
                    axial.align(target, used_axes),
                    axial.align(item, used_axes),
                )
                return Axial(
                    used_axes[:k] + used_axes[k + 1 :],
                    array_calculus.Squeeze((k,), result),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Vec(index, size_, target_):
                size = go(size_, index_sizes, index_vars)
                target = go(target_, index_sizes | {index: size.normal}, index_vars)
                if index in target.axes:
                    return Axial(
                        (
                            target.type.array_type.rank if axis == index else axis
                            for axis in target.axes
                        ),
                        target.array,
                        expr.type.primitive_type.single.kind,
                    )
                else:
                    return Axial(
                        (target.type.array_type.rank, *target.axes),
                        array_calculus.Repeat(
                            0, size.array, array_calculus.Unsqueeze((0,), target.array)
                        ),
                        expr.type.primitive_type.single.kind,
                    )
            case calculus.AbstractScalarReduction(index, size_, target_):
                size = go(size_, index_sizes, index_vars)
                target = go(target_, index_sizes | {index: size.normal}, index_vars)
                # FIXME: target might not depend on the reduction index, though this is a rather degenerate corner case.
                #  This manifests with a failing `axes.index`.
                return Axial(
                    (axis for axis in target.axes if axis != index),
                    array_calculus.REDUCE[expr.ufunc](
                        target.axes.index(index), target.array
                    ),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.AbstractScalarOperator(operands):
                ops = [go(op, index_sizes, index_vars) for op in operands]
                used_axes = axial.alignment(*(op.axes for op in ops))
                if expr.ufunc == to_float:
                    (target,) = ops
                    return Axial(
                        used_axes,
                        array_calculus.Cast(numpy.dtype(float), target.array),
                        expr.type.primitive_type.single.kind,
                    )
                else:
                    return Axial(
                        used_axes,
                        array_calculus.ELEMENTWISE[expr.ufunc](
                            *(axial.align(op, used_axes) for op in ops)
                        ),
                        expr.type.primitive_type.single.kind,
                    )
            case _:
                assert_never(expr)

    def go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_vars: dict[Index, Variable],
    ) -> Axial:
        if expr not in transformed:
            transformed[expr] = _go(expr, index_sizes, index_vars)
        return transformed[expr]

    return go(program, {}, {}).normal


def optimize(program: array_calculus.Expr) -> array_calculus.Expr:
    @cache
    def go(expr: array_calculus.Expr) -> array_calculus.Expr:
        match expr:
            case array_calculus.Transpose(permutation, target):
                if permutation == tuple(range(len(permutation))):
                    return go(target)
            case array_calculus.Unsqueeze(axes, target):
                if not axes:
                    return go(target)
            case array_calculus.Squeeze(axes, target):
                if not axes:
                    return go(target)
        return expr.map(go)

    return go(program)

from typing import assert_never

from ein.symbols import Index, Variable

from . import array_calculus, axial_calculus
from .axial_calculus import Axial


def transform(program: axial_calculus.Expr) -> array_calculus.Expr:
    transformed: dict[axial_calculus.Expr, Axial] = {}

    def _go(expr: axial_calculus.Expr, index_vars: dict[Index, Variable]) -> Axial:
        match expr:
            case axial_calculus.Const(array):
                return Axial.of_normal(array_calculus.Const(array))
            case axial_calculus.Range(index, size_):
                size = go(size_, index_vars)
                return Axial([index], array_calculus.Range(size.array))
            case axial_calculus.Var(var, var_type):
                return Axial.of_normal(array_calculus.Var(var, var_type.rank))
            case axial_calculus.At(index):
                return Axial([], array_calculus.Var(index_vars[index], 0))
            case axial_calculus.Dim(pos, target_):
                target = go(target_, index_vars)
                pos = target.type.array_type.rank - pos - 1
                return Axial(
                    [], array_calculus.Dim(target.axes.index(pos), target.array)
                )
            case axial_calculus.Fold(index, acc, init_, size_, body_):
                init, size = go(init_, index_vars), go(size_, index_vars)
                index_var = Variable()
                body = go(body_, index_vars | {index: index_var})
                return Axial.of_normal(
                    array_calculus.Fold(
                        index_var, acc.var, init.normal, size.normal, body.normal
                    ),
                )
            case axial_calculus.Gather(target_, item_):
                target, item = go(target_, index_vars), go(item_, index_vars)
                used_axes = axial_calculus.alignment(target.axes, item.axes)
                k = used_axes.index(target.type.array_type.rank - 1)
                result = array_calculus.Gather(
                    k,
                    axial_calculus.align(target, used_axes),
                    axial_calculus.align(item, used_axes),
                )
                return Axial(
                    used_axes[:k] + used_axes[k + 1 :],
                    array_calculus.Squeeze((k,), result),
                )
            case axial_calculus.Vector(index, size_, target_):
                size, target = go(size_, index_vars), go(target_, index_vars)
                if index in target.axes:
                    return Axial(
                        (
                            target.type.array_type.rank if axis == index else axis
                            for axis in target.axes
                        ),
                        target.array,
                    )
                else:
                    return Axial(
                        (target.type.array_type.rank, *target.axes),
                        array_calculus.Repeat(
                            0, size.array, array_calculus.Unsqueeze((0,), target.array)
                        ),
                    )
            case axial_calculus.Reduce(ufunc, index, target_):
                target = go(target_, index_vars)
                # FIXME: target might not depend on the reduction index, though this is a rather degenerate corner case.
                #  This manifests with a failing `axes.index`.
                return Axial(
                    (axis for axis in target.axes if axis != index),
                    array_calculus.REDUCE[ufunc](
                        target.axes.index(index), target.array
                    ),
                )
            case axial_calculus.Elementwise(ufunc, operands):
                ops = [go(op, index_vars) for op in operands]
                used_axes = axial_calculus.alignment(*(op.axes for op in ops))
                return Axial(
                    used_axes,
                    array_calculus.ELEMENTWISE[ufunc](
                        *(axial_calculus.align(op, used_axes) for op in ops)
                    ),
                )
            case _:
                assert_never(expr)

    def go(
        expr: axial_calculus.Expr, index_vars: dict[Index, Variable]
    ) -> axial_calculus.Axial:
        if expr not in transformed:
            transformed[expr] = _go(expr, index_vars)
        return transformed[expr]

    return go(program, {}).normal

from typing import assert_never

from ein import calculus
from ein.symbols import Index

from . import axial_calculus


def transform(program: calculus.Expr) -> axial_calculus.Expr:
    transformed: dict[calculus.Expr, axial_calculus.Expr] = {}

    def _go(
        expr: calculus.Expr, sizes: dict[Index, "calculus.Expr | None"]
    ) -> axial_calculus.Expr:
        match expr:
            case calculus.Const(value):
                return axial_calculus.Const(value.array)
            case calculus.At(index):
                size = sizes[index]
                if size is not None:
                    return axial_calculus.Range(index, go(size, {}))
                else:
                    return axial_calculus.At(index)
            case calculus.Var(var, type_):
                return axial_calculus.Var(var, type_.primitive_type.single)
            case calculus.Dim(operand, axis):
                return axial_calculus.Dim(axis, go(operand, sizes))
            case calculus.Get(operand, item):
                return axial_calculus.Gather(go(operand, sizes), go(item, sizes))
            case calculus.Vec(index, size, body):
                return axial_calculus.Vector(
                    index, go(size, {}), go(body, {index: size} | sizes)
                )
            case calculus.Fold(index, size, body, init, acc):
                # TODO: Support accumulators with free indices (axial body/init).
                return axial_calculus.Fold(
                    index,
                    axial_calculus.Var(acc.var, acc.type.primitive_type.single),
                    go(init, {}),
                    go(size, {}),
                    go(body, {index: None}),
                )
            case calculus.AbstractScalarReduction(index, size, body):
                return axial_calculus.Reduce(
                    expr.ufunc, index, go(body, {index: size, **sizes})
                )
            case calculus.AbstractScalarOperator(operands):
                return axial_calculus.Elementwise(
                    expr.ufunc, tuple(go(operand, sizes) for operand in operands)
                )
            case _:
                assert_never(expr)

    def go(
        expr: calculus.Expr, sizes: dict[Index, "calculus.Expr | None"]
    ) -> axial_calculus.Expr:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    return go(program, {})

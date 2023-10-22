from typing import assert_never

import numpy.typing

from ein import calculus
from ein.backend import axial
from ein.backend.axial import Env, Expr
from ein.symbols import Index, Variable


def to_axial(program: calculus.Expr) -> Expr:
    transformed: dict[calculus.Expr, Expr] = {}

    def _go(expr: calculus.Expr, sizes: dict[Index, "calculus.Expr | None"]) -> Expr:
        match expr:
            case calculus.Const(value):
                return axial.Const(value.array)
            case calculus.At(index):
                size = sizes[index]
                if size is not None:
                    return axial.Range(index, go(size, {}))
                else:
                    return axial.At(index)
            case calculus.Var(var, type_):
                return axial.Var(var, type_.primitive_type.single)
            case calculus.Dim(operand, axis):
                return axial.Dim(axis, go(operand, sizes))
            case calculus.Get(operand, item):
                return axial.Gather(go(operand, sizes), go(item, sizes))
            case calculus.Vec(index, size, body):
                return axial.Vector(
                    index, go(size, {}), go(body, {index: size} | sizes)
                )
            case calculus.Fold(index, size, body, init, acc):
                # TODO: Support accumulators with free indices (axial body/init).
                return axial.Fold(
                    index,
                    axial.Var(acc.var, acc.type.primitive_type.single),
                    go(size, {}),
                    go(init, {}),
                    go(body, {index: None}),
                )
            case calculus.AbstractScalarReduction(index, size, body):
                return axial.Reduce(expr.ufunc, index, go(body, {index: size, **sizes}))
            case calculus.AbstractScalarOperator(operands):
                return axial.Elementwise(
                    expr.ufunc, tuple(go(operand, sizes) for operand in operands)
                )
            case _:
                assert_never(expr)

    def go(expr: calculus.Expr, sizes: dict[Index, "calculus.Expr | None"]) -> Expr:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    return go(program, {})


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return to_axial(program).execute(Env(var=env, idx={})).normal


__all__ = ["interpret"]

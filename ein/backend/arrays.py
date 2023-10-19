from typing import assert_never

import numpy.typing

from ein import calculus
from ein.backend import axial
from ein.backend.axial import Env, StagedAxial
from ein.calculus import Expr
from ein.symbols import Index, Variable


def to_axial(program: Expr) -> StagedAxial:
    transformed: dict[Expr, StagedAxial] = {}

    def _go(expr: Expr, sizes: dict[Index, "Expr | None"]) -> StagedAxial:
        match expr:
            case calculus.Const(value):
                return axial.Const(value.array).stage()
            case calculus.At(index):
                size = sizes[index]
                if size is not None:
                    return axial.Range(index).stage(go(size, {}))
                else:
                    return axial.At(index).stage()
            case calculus.Var(var, type_):
                return axial.Var(var, type_).stage()
            case calculus.Dim(operand, axis):
                return axial.Dim(axis).stage(go(operand, sizes))
            case calculus.Get(operand, item):
                return axial.Gather().stage(go(operand, sizes), go(item, sizes))
            case calculus.Vec(index, size, body):
                return axial.Vector(index).stage(
                    go(size, {}), go(body, {index: size, **sizes})
                )
            case calculus.Fold(index, size, body, init, acc):
                # TODO: Support accumulators with free indices (axial body/init).
                return axial.Fold(
                    go(body, {index: None}), index, axial.Var(acc.var, acc.type)
                ).stage(go(size, {}), go(init, {}))
            case calculus.AbstractScalarReduction(index, size, body):
                return axial.Reduce(expr.ufunc, index).stage(
                    go(size, {}), go(body, {index: size, **sizes})
                )
            case calculus.AbstractScalarOperator(operands):
                return axial.Elementwise(expr.ufunc).stage(
                    *(go(operand, sizes) for operand in operands)
                )
            case _:
                assert_never(expr)

    def go(expr: Expr, sizes: dict[Index, "Expr | None"]) -> StagedAxial:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    return go(program, {})


def interpret(program: Expr, env: dict[Variable, numpy.ndarray]) -> numpy.ndarray:
    return to_axial(program).execute(Env(var=env, idx={})).normal


__all__ = ["interpret"]

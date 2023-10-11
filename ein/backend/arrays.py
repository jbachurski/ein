from typing import assert_never

import numpy

from ein import calculus
from ein.calculus import Expr
from ein.symbols import Index, Variable

from . import axial


def to_axial(program: Expr, ranks: dict[Variable, int]) -> axial.Axial:
    transformed: dict[Expr, axial.Axial] = {}

    def _go(expr: Expr, sizes: dict[Index, Expr]) -> axial.Axial:
        match expr:
            case calculus.Const(value):
                return axial.Const(value.array)
            case calculus.At(index):
                return axial.Range(index, go(sizes[index], sizes))
            case calculus.Var(var):
                return axial.Var(var, ranks[var])
            case calculus.Dim(operand, axis):
                return axial.Dim(go(operand, sizes), axis)
            case calculus.Get(operand, item):
                return axial.Gather(go(operand, sizes), go(item, sizes))
            case calculus.Vec(index, size, body):
                return axial.vector(index, go(body, {index: size, **sizes}))
            case calculus.AbstractScalarReduction(index, size, body):
                return axial.Reduce(
                    index, size, go(body, {index: size, **sizes}), expr.ufunc
                )
            case calculus.Where(cond, true, false):
                return axial.Where(cond, true, false)
            case calculus.AbstractScalarOperator(operands):
                return axial.Elementwise(operands, expr.ufunc)
            case _:
                assert_never(expr)

    def go(expr: Expr, sizes: dict[Index, Expr]) -> axial.Axial:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    return go(program, {})


def interpret(expr: Expr, env: dict[Variable, numpy.ndarray]) -> numpy.ndarray:
    from .naive import interpret

    return interpret(expr, env)


__all__ = ["interpret"]

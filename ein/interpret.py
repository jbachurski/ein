from typing import assert_never

import numpy

from . import calculus
from .calculus import Expr, Index, Value, Variable


def _interpret(
    expr: Expr,
    env: dict[Variable, Value],
    idx: dict[Index, int],
) -> Value:
    operands: tuple[Expr, ...]
    match expr:
        case calculus.Vec(index, size, body):
            evaluated_size = int(_interpret(size, env, idx).array)
            return Value(
                numpy.array(
                    [
                        _interpret(body, env, idx | {index: i}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case calculus.AbstractScalarReduction(index, size, body):
            evaluated_size = int(_interpret(size, env, idx).array)
            return Value(
                expr.ufunc.reduce(
                    [
                        _interpret(body, env, idx | {index: i}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case calculus.Get(target, item):
            return Value(
                numpy.take(
                    _interpret(target, env, idx).array,
                    _interpret(item, env, idx).array,
                    axis=0,
                )
            )
        case calculus.Const(value):
            return value
        case calculus.At(index):
            return Value(numpy.array(idx[index]))
        case calculus.Var(var):
            return env[var]
        case calculus.Dim(operand, axis):
            return Value(_interpret(operand, env, idx).array.shape[axis])
        case calculus.Negate(operands):
            (target,) = operands
            return Value(numpy.negative(_interpret(target, env, idx).array))
        case calculus.Reciprocal(operands):
            (target,) = operands
            return Value(numpy.reciprocal(_interpret(target, env, idx).array))
        case calculus.Add(operands):
            first, second = operands
            return Value(
                numpy.add(
                    _interpret(first, env, idx).array,
                    _interpret(second, env, idx).array,
                )
            )
        case calculus.Multiply(operands):
            first, second = operands
            return Value(
                numpy.multiply(
                    _interpret(first, env, idx).array,
                    _interpret(second, env, idx).array,
                )
            )
        case _:
            assert_never(expr)


def interpret(
    program: Expr,
    env: dict[Variable, numpy.ndarray],
) -> numpy.ndarray:
    return _interpret(
        program, {var: Value(array) for var, array in env.items()}, {}
    ).array

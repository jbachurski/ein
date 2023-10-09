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
        case calculus.Where(cond, true, false):
            return Value(
                numpy.where(
                    _interpret(cond, env, idx).array,
                    _interpret(true, env, idx).array,
                    _interpret(false, env, idx).array,
                )
            )
        case calculus.AbstractScalarOperator(operands):
            return Value(
                expr.ufunc(
                    *(_interpret(operand, env, idx).array for operand in operands)
                )
            )
        case calculus.AbstractScalarAxisReduction(_, _) | calculus.Range(_, _):
            raise NotImplementedError("Axials are not supported in the interpreter")
        case _:
            assert_never(expr)


def interpret(
    program: Expr,
    env: dict[Variable, numpy.ndarray],
) -> numpy.ndarray:
    return _interpret(
        program, {var: Value(array) for var, array in env.items()}, {}
    ).array

from typing import assert_never

import numpy

from .calculus import (
    Add,
    At,
    Const,
    Dim,
    Expr,
    Get,
    Index,
    Multiply,
    Negate,
    Reciprocal,
    Sum,
    Value,
    Var,
    Variable,
    Vec,
)


def _interpret(
    program: Expr,
    env: dict[Variable, Value],
    idx: dict[Index, int],
) -> Value:
    operands: tuple[Expr, ...]
    match program:
        case Vec(index, size, body):
            evaluated_size = int(_interpret(size, env, idx).array)
            return Value(
                numpy.array(
                    [
                        _interpret(body, env, idx | {index: i}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case Sum(index, size, body):
            evaluated_size = int(_interpret(size, env, idx).array)
            return Value(
                numpy.sum(
                    [
                        _interpret(body, env, idx | {index: i}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case Get(target, item):
            return Value(
                numpy.take(
                    _interpret(target, env, idx).array,
                    _interpret(item, env, idx).array,
                    axis=0,
                )
            )
        case Const(value):
            return value
        case At(index):
            return Value(numpy.array(idx[index]))
        case Var(var):
            return env[var]
        case Dim(operand, axis):
            return Value(_interpret(operand, env, idx).array.shape[axis])
        case Negate(operands):
            (target,) = operands
            return Value(numpy.negative(_interpret(target, env, idx).array))
        case Reciprocal(operands):
            (target,) = operands
            return Value(numpy.reciprocal(_interpret(target, env, idx).array))
        case Add(operands):
            first, second = operands
            return Value(
                numpy.add(
                    _interpret(first, env, idx).array,
                    _interpret(second, env, idx).array,
                )
            )
        case Multiply(operands):
            first, second = operands
            return Value(
                numpy.multiply(
                    _interpret(first, env, idx).array,
                    _interpret(second, env, idx).array,
                )
            )
        case _:
            assert_never(program)


def interpret(
    program: Expr,
    env: dict[Variable, numpy.ndarray],
) -> numpy.ndarray:
    return _interpret(
        program, {var: Value(array) for var, array in env.items()}, {}
    ).array

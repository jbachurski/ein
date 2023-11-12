from typing import assert_never

import numpy

from ein import calculus
from ein.calculus import Expr, Value
from ein.symbols import Index, Variable


def _interpret(expr: Expr, env: dict[Variable, Value], idx: dict[Index, int]) -> Value:
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
        case calculus.Fold(index, size, acc, init, body):
            evaluated_size = int(_interpret(size, env, idx).array)
            env = env.copy()
            env[acc.var] = _interpret(init, env, idx)
            for i in range(evaluated_size):
                env[acc.var] = _interpret(body, env, idx | {index: i})
            return env.pop(acc.var)
        case calculus.Get(target, item):
            return Value(
                numpy.take(
                    _interpret(target, env, idx).array,
                    _interpret(item, env, idx).array,
                    axis=0,
                )
            )
        case calculus.AssertEq(target, operands):
            assert len({int(_interpret(op, env, idx).array) for op in operands}) == 1
            return _interpret(target, env, idx)
        case calculus.Const(value):
            return value
        case calculus.At(index):
            return Value(numpy.array(idx[index]))
        case calculus.Var(var):
            return env[var]
        case calculus.Dim(operand, axis):
            return Value(_interpret(operand, env, idx).array.shape[axis])
        case calculus.Let(bindings, body):
            to_env = {var: _interpret(binding, env, idx) for var, binding in bindings}
            return _interpret(body, env | to_env, idx)
        case calculus.Cons(first, second):
            return Value((_interpret(first, env, idx), _interpret(second, env, idx)))
        case calculus.First(target):
            return _interpret(target, env, idx).pair[0]
        case calculus.Second(target):
            return _interpret(target, env, idx).pair[1]
        case calculus.AbstractScalarOperator(operands):
            return Value(
                expr.ufunc(
                    *(_interpret(operand, env, idx).array for operand in operands)
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


__all__ = ["interpret"]

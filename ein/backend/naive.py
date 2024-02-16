from typing import assert_never, cast

import numpy

from ein import calculus
from ein.calculus import Expr, Value
from ein.midend.lining import outline
from ein.midend.structs import struct_of_arrays_transform
from ein.symbols import Symbol, Variable


def _interpret(expr: Expr, env: dict[Symbol, Value]) -> Value:
    operands: tuple[Expr, ...]
    match expr:
        case calculus.Vec(index, size, body):
            evaluated_size = int(_interpret(size, env).array)
            return Value(
                numpy.array(
                    [
                        _interpret(body, env | {index: Value(numpy.array(i))}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case calculus.Fold(counter, size, acc, init, body):
            n = int(_interpret(size, env).array)
            env = env.copy()
            env[acc] = _interpret(init, env)
            for i in range(n):
                env[counter] = Value(numpy.array(i))
                env[acc] = _interpret(body, env)
            if n:
                del env[counter]
            return env.pop(acc)
        case calculus.Get(target, item):
            return Value(
                numpy.take(
                    _interpret(target, env).array,
                    _interpret(item, env).array,
                    axis=0,
                    mode="clip",
                )
            )
        case calculus.AssertEq(target, operands):
            assert len({int(_interpret(op, env).array) for op in operands}) == 1
            return _interpret(target, env)
        case calculus.Const(value):
            return value
        case calculus.Store(symbol, _inner_type):
            return env[symbol]
        case calculus.Dim(operand, axis):
            return Value(_interpret(operand, env).array.shape[axis])
        case calculus.Let(var, bind, body):
            return _interpret(body, env | {var: _interpret(bind, env)})
        case calculus.Cons(first, second):
            return Value((_interpret(first, env), _interpret(second, env)))
        case calculus.First(target):
            return _interpret(target, env).pair[0]
        case calculus.Second(target):
            return _interpret(target, env).pair[1]
        case calculus.AbstractScalarOperator(operands):
            return Value(
                expr.ufunc(*(_interpret(operand, env).array for operand in operands))
            )
        case calculus.Extrinsic(_, fun, operands):
            return Value(fun(*(_interpret(operand, env).array for operand in operands)))
        case _:
            assert_never(expr)


def interpret(
    program: Expr,
    env: dict[Variable, numpy.ndarray],
) -> numpy.ndarray:
    program = cast(Expr, outline(struct_of_arrays_transform(program)))
    return _interpret(program, {var: Value(array) for var, array in env.items()}).array


__all__ = ["interpret"]

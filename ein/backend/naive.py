import functools
from typing import assert_never, cast

import numpy

from ein.midend.lining import outline
from ein.midend.structs import struct_of_arrays_transform
from ein.phi import phi
from ein.phi.phi import Expr
from ein.symbols import Symbol, Variable
from ein.value import Value


def _interpret(expr: Expr, env: dict[Symbol, Value]) -> Value:
    operands: tuple[Expr, ...]
    match expr:
        case phi.Vec(index, size, body):
            evaluated_size = int(_interpret(size, env).array)
            return Value(
                numpy.array(
                    [
                        _interpret(body, env | {index: Value(numpy.array(i))}).array
                        for i in range(evaluated_size)
                    ]
                )
            )
        case phi.Fold(counter, size, acc_v, init, body):
            n = int(_interpret(size, env).array)
            env[acc_v] = _interpret(init, env)
            for i in range(n):
                env[counter] = Value(numpy.array(i))
                env[acc_v] = _interpret(body, env)
            if n:
                del env[counter]
            return env.pop(acc_v)
        case phi.Reduce(init, x, y, xy, vecs_):
            vecs = [_interpret(vec_, env).array for vec_ in vecs_]
            (n,) = {int(vec.shape[0]) for vec in vecs}
            acc = _interpret(init, env)
            for i in range(n):
                env[x] = acc
                env[y] = functools.reduce(
                    lambda a, b: Value((a, b)), [Value(vec[i]) for vec in vecs]
                )
                acc = _interpret(xy, env)
            if n:
                del env[x], env[y]
            return acc
        case phi.Get(target, item):
            tt = _interpret(target, env).array
            it = _interpret(item, env).array
            if not tt.size:
                return Value(numpy.empty(tt.shape[1:], tt.dtype))
            return Value(numpy.take(tt, it, axis=0, mode="clip"))
        case phi.Concat(first, second):
            return Value(
                numpy.concatenate(
                    (_interpret(first, env).array, _interpret(second, env).array)
                )
            )
        case phi.AssertEq(target, operands):
            assert len({int(_interpret(op, env).array) for op in operands}) == 1
            return _interpret(target, env)
        case phi.Const(value):
            return value
        case phi.Store(symbol, _inner_type):
            return env[symbol]
        case phi.Dim(operand, axis):
            return Value(_interpret(operand, env).array.shape[axis])
        case phi.Let(var, bind, body):
            return _interpret(body, env | {var: _interpret(bind, env)})
        case phi.Cons(first, second):
            return Value((_interpret(first, env), _interpret(second, env)))
        case phi.First(target):
            return _interpret(target, env).pair[0]
        case phi.Second(target):
            return _interpret(target, env).pair[1]
        case phi.AbstractScalarOperator(operands):
            return Value(
                expr.ufunc(*(_interpret(operand, env).array for operand in operands))
            )
        case phi.Extrinsic(_, fun, operands):
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

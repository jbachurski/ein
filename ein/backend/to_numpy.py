from functools import cache
from typing import Callable, TypeAlias, TypeVar, assert_never, cast

import numpy

from ein import calculus
from ein.symbols import Variable

from . import array_calculus, to_array

Env: TypeAlias = dict[Variable, numpy.ndarray]

T = TypeVar("T")
S = TypeVar("S")


def maybe(f: Callable[[T], S]) -> Callable[[T | None], S | None]:
    return lambda x: f(x) if x is not None else None


def stage_in_array(
    program: array_calculus.Expr,
) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
    def go(expr: array_calculus.Expr) -> Callable[[Env], numpy.ndarray]:
        return cast(Callable[[Env], numpy.ndarray], go_either(expr))

    @cache
    def go_either(
        expr: array_calculus.Expr,
    ) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
        match expr:
            case array_calculus.Const(array):
                return lambda env: array.array
            case array_calculus.Var(var, _var_rank):
                return lambda env: env[var]
            case array_calculus.Let(var, bind_, body_):
                bind, body = go(bind_), go(body_)

                def with_let(env: Env):
                    env[var] = bind(env)
                    ret = body(env)
                    del env[var]
                    return ret

                return with_let
            case array_calculus.Dim(axis, target_):
                target = go(target_)
                return lambda env: numpy.array(target(env).shape[axis])
            case array_calculus.Range(size_):
                size = go(size_)
                return lambda env: numpy.arange(size(env))
            case array_calculus.Transpose(permutation, target_):
                target = go(target_)
                return lambda env: numpy.transpose(target(env), permutation)
            case array_calculus.Squeeze(axes, target_):
                target = go(target_)
                return lambda env: numpy.squeeze(target(env), axes)
            case array_calculus.Unsqueeze(axes, target_):
                target = go(target_)
                return lambda env: numpy.expand_dims(target(env), axes)
            case array_calculus.Gather(axis, target_, item_):
                target, item = go(target_), go(item_)

                def apply_gather(env: Env) -> numpy.ndarray:
                    arr = target(env)
                    return numpy.take_along_axis(
                        arr, numpy.clip(item(env), 0, arr.shape[axis] - 1), axis
                    )

                return apply_gather
            case array_calculus.Take(target_, items_):
                target = go(target_)
                items = [maybe(go)(item_) for item_ in items_]
                return lambda env: target(env)[
                    tuple(
                        item(env) if item is not None else slice(None) for item in items
                    )
                ]
            case array_calculus.Slice(target_, shifts_, sizes_):
                target = go(target_)
                shifts = [maybe(go)(shift_) for shift_ in shifts_]
                sizes = [maybe(go)(size_) for size_ in sizes_]

                def apply_slice(env: Env) -> numpy.ndarray:
                    arr = target(env)

                    shape = []
                    read_slices = []
                    written_slices = []
                    for dim, shift_of_env, size_of_env in zip(arr.shape, shifts, sizes):
                        shift = (
                            int(shift_of_env(env)) if shift_of_env is not None else 0
                        )
                        size = (
                            max(0, int(size_of_env(env)))
                            if size_of_env is not None
                            else dim
                        )
                        shape.append(size)
                        if shift < 0:
                            k = -shift
                            read_slices.append(slice(0, max(0, size - k)))
                            written_slices.append(slice(min(k, size), size))
                        else:
                            k = shift
                            read_slices.append(slice(min(k, size), size))
                            written_slices.append(slice(0, max(0, size - k)))

                    result = numpy.empty(shape)
                    result[tuple(written_slices)] = arr[tuple(read_slices)]
                    return result

                return lambda env: apply_slice(env)

            case array_calculus.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: numpy.repeat(target(env), count(env), axis=axis)
            case array_calculus.Reduce(kind, axis, target_):
                target = go(target_)
                call = array_calculus.ELEMENTWISE_KINDS[kind]
                return lambda env: call(target(env), axis=axis)
            case array_calculus.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).astype(dtype)
            case array_calculus.UnaryElementwise(kind, target_):
                target = go(target_)
                call = array_calculus.ELEMENTWISE_KINDS[kind]
                return lambda env: call(target(env))
            case array_calculus.BinaryElementwise(kind, first_, second_, inplace):
                first, second = go(first_), go(second_)
                call = array_calculus.ELEMENTWISE_KINDS[kind]
                if inplace == 0:

                    def call_to_first(env: Env) -> numpy.ndarray:
                        out = first(env)
                        return call(out, second(env), out=out)

                    return call_to_first
                elif inplace == 1:

                    def call_to_second(env: Env) -> numpy.ndarray:
                        out = second(env)
                        return call(first(env), out, out=out)

                    return call_to_second
                return lambda env: call(first(env), second(env))
            case array_calculus.TernaryElementwise(kind, first_, second_, third_):
                first, second, third = go(first_), go(second_), go(third_)
                call = array_calculus.ELEMENTWISE_KINDS[kind]
                return lambda env: call(first(env), second(env), third(env))
            case array_calculus.Fold(index_var, size_, acc_var, init_, body_):
                init, size, body = go_either(init_), go(size_), go_either(body_)

                def fold(env: Env) -> numpy.ndarray | tuple[numpy.ndarray, ...]:
                    acc, n = init(env), size(env)
                    n = max(int(n), 0)
                    for i in range(n):
                        env[acc_var] = acc
                        env[index_var] = numpy.array(i)
                        acc = body(env)
                    if n:
                        del env[acc_var], env[index_var]
                    return acc

                return fold
            case array_calculus.Tuple(operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: tuple(op(env) for op in operands)
            case array_calculus.Untuple(at, _arity, target_):
                target = go_either(target_)
                return lambda env: target(env)[at]
            case array_calculus.Einsum(subs, operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: numpy.einsum(subs, *(op(env) for op in operands))
            case _:
                assert_never(expr)

    return go(program)


def stage(
    program: calculus.Expr,
) -> Callable[[dict[Variable, numpy.ndarray]], numpy.ndarray]:
    return stage_in_array(to_array.transform(program))


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return stage(program)(env)

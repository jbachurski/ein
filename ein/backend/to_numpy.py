from functools import cache
from typing import Callable, TypeAlias, assert_never, cast

import numpy

from ein import calculus
from ein.symbols import Variable

from . import array_calculus, to_array

Env: TypeAlias = dict[Variable, numpy.ndarray]

reduce_kind = {
    array_calculus.Reduce.Kind.sum: numpy.sum,
    array_calculus.Reduce.Kind.max: numpy.max,
}
unary_kind = {
    array_calculus.UnaryElementwise.Kind.negative: numpy.negative,
    array_calculus.UnaryElementwise.Kind.reciprocal: numpy.reciprocal,
    array_calculus.UnaryElementwise.Kind.exp: numpy.exp,
    array_calculus.UnaryElementwise.Kind.sin: numpy.sin,
    array_calculus.UnaryElementwise.Kind.logical_not: numpy.logical_not,
}
binary_kind = {
    array_calculus.BinaryElementwise.Kind.add: numpy.add,
    array_calculus.BinaryElementwise.Kind.multiply: numpy.multiply,
    array_calculus.BinaryElementwise.Kind.mod: numpy.mod,
    array_calculus.BinaryElementwise.Kind.less: numpy.less,
    array_calculus.BinaryElementwise.Kind.logical_and: numpy.logical_and,
}
ternary_kind = {
    array_calculus.TernaryElementwise.Kind.where: numpy.where,
}

NEVER_ALIAS = (
    array_calculus.UnaryElementwise,
    array_calculus.BinaryElementwise,
    array_calculus.TernaryElementwise,
)


def stage(
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
                return lambda env: array
            case array_calculus.Var(var, _var_rank):
                return lambda env: env[var]
            case array_calculus.Let(bindings, body_):
                body = go(body_)
                return lambda env: body(
                    env | {bound: go(binding)(env) for bound, binding in bindings}
                )
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
                return lambda env: numpy.take_along_axis(target(env), item(env), axis)
            case array_calculus.Take(target_, items_):
                target = go(target_)
                items = [go(item) if item is not None else None for item in items_]
                return lambda env: target(env)[
                    tuple(
                        item(env) if item is not None else slice(None) for item in items
                    )
                ]
            case array_calculus.Slice(target_, stops_):
                target = go(target_)
                stops = [go(stop) if stop is not None else None for stop in stops_]

                def apply_slice(env: Env) -> numpy.ndarray:
                    arr = target(env)
                    stops_env = [
                        stop(env) if stop is not None else None for stop in stops
                    ]
                    slices = [
                        slice(stop) if stop is not None and d != stop else slice(None)
                        for d, stop in zip(arr.shape, stops_env)
                    ]
                    cop = any(s is not None for s in slices)
                    return arr[tuple(slices)] if cop else arr

                return lambda env: apply_slice(env)

            case array_calculus.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: numpy.repeat(target(env), count(env), axis=axis)
            case array_calculus.Reduce(kind, axis, target_):
                target = go(target_)
                call = reduce_kind[kind]
                return lambda env: call(target(env), axis=axis)
            case array_calculus.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).astype(dtype)
            case array_calculus.UnaryElementwise(kind, target_):
                target = go(target_)
                call = unary_kind[kind]
                return lambda env: call(target(env))
            case array_calculus.BinaryElementwise(kind, first_, second_, inplace):
                first, second = go(first_), go(second_)
                call = binary_kind[kind]
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
                call = ternary_kind[kind]
                return lambda env: call(first(env), second(env), third(env))
            case array_calculus.Fold(index_var, size_, acc_var, init_, body_):
                init, size, body = go_either(init_), go(size_), go_either(body_)

                def fold(env: Env) -> numpy.ndarray | tuple[numpy.ndarray, ...]:
                    acc, n = init(env), size(env)
                    for i in range(n):
                        acc = body(env | {acc_var: acc, index_var: numpy.array(i)})
                    return acc

                return fold
            case array_calculus.Tuple(operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: tuple(op(env) for op in operands)
            case array_calculus.Untuple(at, _arity, target_):
                target = go_either(target_)
                return lambda env: target(env)[at]
            case _:
                assert_never(expr)

    return go(program)


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return stage(to_array.transform(program))(env)

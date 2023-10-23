from dataclasses import dataclass
from functools import cache
from typing import Callable, TypeAlias, assert_never, cast

import numpy

from ein import calculus
from ein.symbols import Variable

from . import array_calculus, to_array, to_axial

Env: TypeAlias = dict[Variable, numpy.ndarray]

reduce_kind = {
    array_calculus.Reduce.Kind.sum: numpy.sum,
    array_calculus.Reduce.Kind.max: numpy.max,
}
unary_kind = {
    array_calculus.UnaryElementwise.Kind.negative: numpy.negative,
    array_calculus.UnaryElementwise.Kind.reciprocal: numpy.reciprocal,
    array_calculus.UnaryElementwise.Kind.exp: numpy.exp,
    array_calculus.UnaryElementwise.Kind.logical_not: numpy.logical_not,
}
binary_kind = {
    array_calculus.BinaryElementwise.Kind.add: numpy.add,
    array_calculus.BinaryElementwise.Kind.multiply: numpy.multiply,
    array_calculus.BinaryElementwise.Kind.less: numpy.less,
    array_calculus.BinaryElementwise.Kind.logical_and: numpy.logical_and,
}
ternary_kind = {
    array_calculus.TernaryElementwise.Kind.where: numpy.where,
}


@dataclass
class Cell:
    f: Callable[[Env], numpy.ndarray]
    _last: tuple[Env, numpy.ndarray] | None = None

    def __call__(self, env: Env) -> numpy.ndarray:
        if self._last is None or self._last[0] is not env:
            self._last = (env, self.f(env))
        return cast(numpy.ndarray, self._last[1])


def stage(program: array_calculus.Expr) -> Callable[[Env], numpy.ndarray]:
    @cache
    def go(expr: array_calculus.Expr) -> Callable[[Env], numpy.ndarray]:
        match expr:
            case array_calculus.Const(array):
                return lambda env: array
            case array_calculus.Var(var, _var_rank):
                return lambda env: env[var]
            case array_calculus.Dim(axis, target_):
                target = go(target_)
                return lambda env: target(env).shape[axis]
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
            case array_calculus.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: numpy.repeat(target(env), count(env), axis=axis)
            case array_calculus.Reduce(kind, axis, target_):
                target = go(target_)
                call = reduce_kind[kind]
                return lambda env: call(target(env), axis=axis)
            case array_calculus.UnaryElementwise(kind, target_):
                target = go(target_)
                call = unary_kind[kind]
                return lambda env: call(target(env))
            case array_calculus.BinaryElementwise(kind, first_, second_):
                first, second = go(first_), go(second_)
                call = binary_kind[kind]
                return lambda env: call(first(env), second(env))
            case array_calculus.TernaryElementwise(kind, first_, second_, third_):
                first, second, third = go(first_), go(second_), go(third_)
                call = ternary_kind[kind]
                return lambda env: call(first(env), second(env), third(env))
            case array_calculus.Fold(index_var, acc_var, init_, size_, body_):
                init, size, body = go(init_), go(size_), go(body_)

                def fold(env: Env) -> numpy.ndarray:
                    acc, n = init(env), size(env)
                    for i in range(n):
                        acc = body(env | {acc_var: acc, index_var: numpy.array(i)})
                    return acc

                return fold
            case _:
                assert_never(expr)

    return go(program)


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return stage(to_array.transform(to_axial.transform(program)))(env)

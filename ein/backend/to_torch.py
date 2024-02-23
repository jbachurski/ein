import itertools
from functools import cache
from typing import Any, Callable, TypeAlias, assert_never, cast

import numpy

from ein import calculus
from ein.midend.lining import outline
from ein.symbols import Variable

try:
    import torch
except ImportError:
    torch = None  # type: ignore


from . import array_calculus, to_array
from .array_calculus import (
    BinaryElementwise,
    Reduce,
    TernaryElementwise,
    UnaryElementwise,
)
from .to_numpy import maybe

Env: TypeAlias = dict[Variable, torch.Tensor | tuple[torch.Tensor, ...]]


def squeeze_axes(tensor, axes):
    for axis in axes:
        tensor = tensor.squeeze(axis)
    return tensor


def unsqueeze_axes(tensor, axes):
    for axis in axes:
        tensor = tensor.unsqueeze(axis)
    return tensor


def stage_in_array(
    program: array_calculus.Expr,
) -> Callable[[Env], torch.Tensor | tuple[torch.Tensor, ...]]:
    def go(expr: array_calculus.Expr) -> Callable[[Env], torch.Tensor]:
        return cast(Callable[[Env], torch.Tensor], go_either(expr))

    @cache
    def go_either(
        expr: array_calculus.Expr,
    ) -> Callable[[Env], torch.Tensor | tuple[torch.Tensor, ...]]:
        match expr:
            case array_calculus.Const(array):
                # Do this to avoid torch complaining about lack of support
                # - we ought to never modify constants anyway
                array.array.flags.writeable = True
                arr = torch.from_numpy(array.array)
                array.array.flags.writeable = False
                return lambda env: arr
            case array_calculus.Var(var, _var_rank):
                return lambda env: env[var]
            case array_calculus.Let(var, bind_, body_):
                bind, body = go_either(bind_), go_either(body_)

                def with_let(env: Env):
                    env[var] = bind(env)
                    ret = body(env)
                    del env[var]
                    return ret

                return with_let
            case array_calculus.Dim(axis, target_):
                target = go(target_)
                return lambda env: torch.scalar_tensor(
                    target(env).shape[axis], dtype=torch.int64
                )
            case array_calculus.Range(size_):
                size = go(size_)
                return lambda env: torch.arange(int(size(env)))
            case array_calculus.Transpose(permutation, target_):
                target = go(target_)
                return lambda env: target(env).permute(*permutation)
            case array_calculus.Squeeze(axes, target_):
                target = go(target_)
                return lambda env: squeeze_axes(target(env), axes)
            case array_calculus.Unsqueeze(axes, target_):
                target = go(target_)
                return lambda env: unsqueeze_axes(target(env), axes)
            case array_calculus.Gather(axis, target_, item_):
                target, item = go(target_), go(item_)

                def apply_gather(env: Env) -> torch.Tensor:
                    arr = target(env)
                    return torch.gather(
                        arr, axis, torch.clip(item(env), 0, arr.shape[axis] - 1)
                    )

                return apply_gather
            case array_calculus.Take(target_, items_):
                target = go(target_)
                items = [maybe(go)(item_) for item_ in items_]

                def apply_take(env: Env) -> torch.Tensor:
                    arr = target(env)
                    it = (
                        torch.clip(item(env), 0, dim - 1)
                        if item is not None
                        else slice(None)
                        for dim, item in zip(arr.shape, items)
                    )
                    return arr[*it]

                return apply_take
            case array_calculus.Slice(target_, starts_, stops_):
                target = go(target_)
                starts = [maybe(go)(start_) for start_ in starts_]
                stops = [maybe(go)(stop_) for stop_ in stops_]

                def apply_slice(env: Env) -> torch.Tensor:
                    slices = [
                        slice(
                            start(env) if start is not None else None,
                            stop(env) if stop is not None else None,
                        )
                        for start, stop in zip(starts, stops)
                    ]
                    return target(env)[*slices]

                return apply_slice
            case array_calculus.Pad(target_, lefts_, rights_):
                target = go(target_)
                lefts = [maybe(go)(left_) for left_ in lefts_]
                rights = [maybe(go)(right_) for right_ in rights_]
                its = tuple(zip(lefts, rights))[::-1]

                def apply_pad(env: Env) -> torch.Tensor:
                    pads = tuple(
                        itertools.chain.from_iterable(
                            (
                                int(left(env)) if left is not None else 0,
                                int(right(env)) if right is not None else 0,
                            )
                            for left, right in its
                        )
                    )
                    return (
                        torch.nn.functional.pad(
                            target(env).unsqueeze(0).unsqueeze(0),
                            pads,
                            mode="replicate",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                return apply_pad
            case array_calculus.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: torch.repeat_interleave(
                    target(env), count(env), axis
                )
            case array_calculus.Reduce(kind, axis, target_):
                target = go(target_)
                call = REDUCE[kind]
                return lambda env: call(target(env), axis=axis)
            case array_calculus.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).to(dtype)
            case array_calculus.UnaryElementwise(kind, target_):
                target = go(target_)
                call = ELEMENTWISE[kind]
                return lambda env: call(target(env))
            case array_calculus.BinaryElementwise(kind, first_, second_, _inplace):
                first, second = go(first_), go(second_)
                call = ELEMENTWISE[kind]
                return lambda env: call(first(env), second(env))
            case array_calculus.TernaryElementwise(kind, first_, second_, third_):
                first, second, third = go(first_), go(second_), go(third_)
                call = ELEMENTWISE[kind]
                return lambda env: call(first(env), second(env), third(env))
            case array_calculus.Fold(index_var, size_, acc_var, init_, body_):
                init, size, body = go_either(init_), go(size_), go_either(body_)

                def fold(env: Env) -> torch.Tensor | tuple[torch.Tensor, ...]:
                    acc, n = init(env), size(env)
                    n = max(int(n), 0)
                    for i in range(n):
                        env[acc_var] = acc
                        env[index_var] = torch.scalar_tensor(i, dtype=torch.int64)
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
                return lambda env: torch.einsum(subs, *[op(env) for op in operands])
            case array_calculus.Extrinsic(_, fun, operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: fun(*(op(env) for op in operands))
            case _:
                assert_never(expr)

    program = cast(array_calculus.Expr, outline(program))
    program = to_array.apply_inplace_on_temporaries(program)
    return go(program)


def stage(
    program: calculus.Expr,
) -> Callable[[dict[Variable, torch.Tensor]], torch.Tensor | tuple[torch.Tensor, ...]]:
    return stage_in_array(to_array.transform(program))


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray | torch.Tensor]
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    return stage(program)(
        {
            var: torch.from_numpy(numpy.asarray(arr))
            if not isinstance(arr, torch.Tensor)
            else arr
            for var, arr in env.items()
        }
    )


REDUCE: Any = {
    Reduce.Kind.add: torch.sum,
    Reduce.Kind.minimum: torch.amin,
    Reduce.Kind.maximum: torch.amax,
}
ELEMENTWISE: Any = {
    UnaryElementwise.Kind.negative: torch.negative,
    UnaryElementwise.Kind.reciprocal: torch.reciprocal,
    UnaryElementwise.Kind.exp: torch.exp,
    UnaryElementwise.Kind.sin: torch.sin,
    UnaryElementwise.Kind.logical_not: torch.logical_not,
    BinaryElementwise.Kind.add: torch.add,
    BinaryElementwise.Kind.subtract: torch.subtract,
    BinaryElementwise.Kind.multiply: torch.multiply,
    BinaryElementwise.Kind.mod: numpy.mod,  # Need to use the wrapper for mod
    BinaryElementwise.Kind.power: torch.pow,
    BinaryElementwise.Kind.minimum: torch.minimum,
    BinaryElementwise.Kind.maximum: torch.maximum,
    BinaryElementwise.Kind.less: torch.less,
    BinaryElementwise.Kind.less_equal: torch.less_equal,
    BinaryElementwise.Kind.equal: torch.eq,
    BinaryElementwise.Kind.not_equal: torch.ne,
    BinaryElementwise.Kind.logical_and: torch.logical_and,
    BinaryElementwise.Kind.logical_or: torch.logical_or,
    TernaryElementwise.Kind.where: torch.where,
}

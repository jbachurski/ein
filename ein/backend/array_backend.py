import abc
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy

from ein.codegen import yarr
from ein.value import Value

T = TypeVar("T")
S = TypeVar("S")
Z = TypeVar("Z")


def maybe(f: Callable[[T], S], x: T | None) -> S | None:
    return f(x) if x is not None else None


def maybe_call(f: Callable[[T], S] | None, x: T) -> S | None:
    return f(x) if f is not None else None


def maybe_call_or(f: Callable[[T], S] | None, x: T, y: S) -> S:
    return f(x) if f is not None else y


class AbstractArrayBackend(abc.ABC, Generic[T]):
    @classmethod
    @abc.abstractmethod
    def constant(cls, value: Value) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def preprocess_bound(cls, target: T) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def dim(cls, target: T, axis: int) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def range(cls, size: T) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def concat(cls, *args: T, axis: int) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def transpose(cls, target: T, permutation: tuple[int, ...]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def squeeze(cls, target: T, axes: tuple[int, ...]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def unsqueeze(cls, target: T, axes: tuple[int, ...]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def gather(cls, target: T, item: T, axis: int) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def take(cls, target: T, items: Sequence[T | None]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def slice(cls, target: T, slices: Sequence[slice]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def pad(cls, target: T, slices: Sequence[tuple[int, int]]) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def repeat(cls, target: T, count: T, axis: int) -> T:
        ...

    @classmethod
    @abc.abstractmethod
    def prepare_einsum(cls, subs: str) -> Callable[..., T]:
        ...

    @classmethod
    def stage(
        cls,
        expr: yarr.Expr,
        go: Callable[[Any], Callable[[Any], Any]],
    ) -> Callable[..., Any] | None:
        match expr:
            case yarr.Const(value):
                a = cls.constant(value)
                return lambda env: a
            case yarr.Var(var, _var_rank):
                return lambda env: env[var]
            case yarr.Let(var, bind_, body_):
                bind, body = go(bind_), go(body_)

                def with_let(env):
                    bound = bind(env)
                    env[var] = (
                        tuple(map(cls.preprocess_bound, bound))
                        if isinstance(bound, tuple)
                        else cls.preprocess_bound(bound)
                    )
                    del bound
                    ret = body(env)
                    del env[var]
                    return ret

                return with_let
            case yarr.Dim(axis, target_):
                target = go(target_)
                return lambda env: cls.dim(target(env), axis)
            case yarr.Range(size_):
                size = go(size_)
                return lambda env: cls.range(size(env))
            case yarr.Concat(operands_, axis):
                ops = [go(op_) for op_ in operands_]
                return lambda env: cls.concat(*(op(env) for op in ops), axis=axis)
            case yarr.Transpose(permutation, target_):
                target = go(target_)
                return lambda env: cls.transpose(target(env), permutation)
            case yarr.Squeeze(axes, target_):
                target = go(target_)
                return lambda env: cls.squeeze(target(env), axes)
            case yarr.Unsqueeze(axes, target_):
                target = go(target_)
                return lambda env: cls.unsqueeze(target(env), axes)
            case yarr.Gather(axis, target_, item_):
                target, item = go(target_), go(item_)
                return lambda env: cls.gather(target(env), item(env), axis)
            case yarr.Take(target_, items_):
                target = go(target_)
                items = tuple(maybe(go, item_) for item_ in items_)
                return lambda env: cls.take(
                    target(env), [maybe_call(item, env) for item in items]
                )
            case yarr.Slice(target_, starts_, stops_):
                target = go(target_)
                starts = tuple(maybe(go, start_) for start_ in starts_)
                stops = tuple(maybe(go, stop_) for stop_ in stops_)
                return lambda env: cls.slice(
                    target(env),
                    tuple(
                        slice(maybe_call(x, env), maybe_call(y, env))
                        for x, y in zip(starts, stops)
                    ),
                )
            case yarr.Pad(target_, lefts_, rights_):
                target = go(target_)
                lefts = tuple(maybe(go, left_) for left_ in lefts_)
                rights = tuple(maybe(go, right_) for right_ in rights_)
                return lambda env: cls.pad(
                    target(env),
                    tuple(
                        (maybe_call_or(x, env, 0), maybe_call_or(y, env, 0))
                        for x, y in zip(lefts, rights)
                    ),
                )
            case yarr.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: cls.repeat(target(env), count(env), axis)
            case yarr.Fold(index_var, size_, acc_var, init_, body_):
                init, size, body = go(init_), go(size_), go(body_)

                def fold(env):
                    acc, n = init(env), max(int(size(env)), 0)
                    for i in range(n):
                        env[acc_var] = acc
                        env[index_var] = cls.constant(Value(numpy.array(i)))
                        acc = body(env)
                    if n:
                        del env[acc_var], env[index_var]
                    return acc

                return fold
            case yarr.Reduce(init_, x, y, xy_, vecs_, axis):
                init, xy, vecs = go(init_), go(xy_), go(vecs_)

                def reduce(env):
                    acc = init(env)
                    vals = vecs(env)
                    singleton = False
                    if not isinstance(vals, tuple):
                        vals = (vals,)
                        acc = (acc,)
                        singleton = True

                    def wrap(o):
                        return (o,) if singleton else o

                    def unwrap(o):
                        if singleton:
                            (o,) = o
                        return o

                    (n,) = {val.shape[axis] for val in vals}
                    pivot = (slice(None),) * axis
                    # The accumulator is unsqueezed to align the batch axis
                    acc = tuple(cls.unsqueeze(ac, (axis,)) for ac in acc)

                    def idx(s):
                        nonlocal vals
                        return tuple(val[*pivot, s] for val in vals)

                    def get(x0, y0):
                        env[x] = unwrap(x0)
                        env[y] = unwrap(y0)
                        return wrap(xy(env))

                    while n > 1:
                        if n % 2:
                            acc = get(idx([-1]), acc)
                        vals = get(idx(slice(None, -1, 2)), idx(slice(1, None, 2)))
                        n //= 2
                    if n:
                        acc = get(idx([0]), acc)
                        del env[x], env[y]

                    return unwrap(tuple(cls.squeeze(ac, (axis,)) for ac in acc))

                return reduce
            case yarr.Tuple(operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: tuple(op(env) for op in operands)
            case yarr.Untuple(at, _arity, target_):
                tup = go(target_)
                return lambda env: tup(env)[at]
            case yarr.Einsum(subs, operands_):
                operands = tuple(go(op_) for op_ in operands_)
                einsum_fun = cls.prepare_einsum(subs=subs)
                return lambda env: einsum_fun(*(op(env) for op in operands))
            case yarr.Extrinsic(_, fun, operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: fun(*(op(env) for op in operands))
        return None

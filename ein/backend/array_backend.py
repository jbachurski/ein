import abc
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy

from ein.codegen import yarr
from ein.symbols import Variable
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
    @abc.abstractmethod
    def constant(self, value: Value) -> T:
        ...

    @abc.abstractmethod
    def preprocess_bound(self, target: T) -> T:
        ...

    @abc.abstractmethod
    def dim(self, target: T, axis: int) -> T:
        ...

    @abc.abstractmethod
    def range(self, size: T) -> T:
        ...

    @abc.abstractmethod
    def concat(self, *args: T, axis: int) -> T:
        ...

    @abc.abstractmethod
    def transpose(self, target: T, permutation: tuple[int, ...]) -> T:
        ...

    @abc.abstractmethod
    def squeeze(self, target: T, axes: tuple[int, ...]) -> T:
        ...

    @abc.abstractmethod
    def unsqueeze(self, target: T, axes: tuple[int, ...]) -> T:
        ...

    @abc.abstractmethod
    def gather(self, target: T, item: T, axis: int) -> T:
        ...

    @abc.abstractmethod
    def take(self, target: T, items: Sequence[T | None]) -> T:
        ...

    @abc.abstractmethod
    def slice(self, target: T, slices: Sequence[slice]) -> T:
        ...

    @abc.abstractmethod
    def pad(self, target: T, slices: Sequence[tuple[int, int]]) -> T:
        ...

    @abc.abstractmethod
    def repeat(self, target: T, count: T, axis: int) -> T:
        ...

    @abc.abstractmethod
    def prepare_einsum(self, subs: str) -> Callable[..., T]:
        ...

    def fold(
        self,
        index_var: Variable,
        count: Callable,
        acc_var: Variable,
        init: Callable,
        body: Callable,
    ) -> Callable:
        def fold_impl(env):
            acc, n = init(env), max(int(count(env)), 0)
            for i in range(n):
                env[acc_var] = acc
                env[index_var] = self.constant(Value(numpy.array(i)))
                acc = body(env)
            if n:
                del env[acc_var], env[index_var]
            return acc

        return fold_impl

    def reduce(
        self,
        init: Callable,
        x: Variable,
        y: Variable,
        xy: Callable,
        vecs: Callable,
        axis: int,
    ) -> Callable:
        def reduce_impl(env):
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
            acc = tuple(self.unsqueeze(ac, (axis,)) for ac in acc)

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

            return unwrap(tuple(self.squeeze(ac, (axis,)) for ac in acc))

        return reduce_impl

    def stage(
        self,
        expr: yarr.Expr,
        go: Callable[[Any], Callable[[Any], Any]],
    ) -> Callable[..., Any] | None:
        match expr:
            case yarr.Const(value):
                a = self.constant(value)
                return lambda env: a
            case yarr.Var(var, _var_rank):
                return lambda env: env[var]
            case yarr.Let(var, bind_, body_):
                bind, body = go(bind_), go(body_)

                def with_let(env):
                    bound = bind(env)
                    env[var] = (
                        tuple(map(self.preprocess_bound, bound))
                        if isinstance(bound, tuple)
                        else self.preprocess_bound(bound)
                    )
                    del bound
                    ret = body(env)
                    del env[var]
                    return ret

                return with_let
            case yarr.Dim(axis, target_):
                target = go(target_)
                return lambda env: self.dim(target(env), axis)
            case yarr.Range(size_):
                size = go(size_)
                return lambda env: self.range(size(env))
            case yarr.Concat(operands_, axis):
                ops = [go(op_) for op_ in operands_]
                return lambda env: self.concat(*(op(env) for op in ops), axis=axis)
            case yarr.Transpose(permutation, target_):
                target = go(target_)
                return lambda env: self.transpose(target(env), permutation)
            case yarr.Squeeze(axes, target_):
                target = go(target_)
                return lambda env: self.squeeze(target(env), axes)
            case yarr.Unsqueeze(axes, target_):
                target = go(target_)
                return lambda env: self.unsqueeze(target(env), axes)
            case yarr.Gather(axis, target_, item_):
                target, item = go(target_), go(item_)
                return lambda env: self.gather(target(env), item(env), axis)
            case yarr.Take(target_, items_):
                target = go(target_)
                items = tuple(maybe(go, item_) for item_ in items_)
                return lambda env: self.take(
                    target(env), [maybe_call(item, env) for item in items]
                )
            case yarr.Slice(target_, starts_, stops_):
                target = go(target_)
                starts = tuple(maybe(go, start_) for start_ in starts_)
                stops = tuple(maybe(go, stop_) for stop_ in stops_)
                return lambda env: self.slice(
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
                return lambda env: self.pad(
                    target(env),
                    tuple(
                        (maybe_call_or(x, env, 0), maybe_call_or(y, env, 0))
                        for x, y in zip(lefts, rights)
                    ),
                )
            case yarr.Repeat(axis, count_, target_):
                count, target = go(count_), go(target_)
                return lambda env: self.repeat(target(env), count(env), axis)
            case yarr.Fold(index_var, count_, acc_var, init_, body_):
                init, count, body = go(init_), go(count_), go(body_)
                return self.fold(index_var, count, acc_var, init, body)
            case yarr.Reduce(init_, x, y, xy_, vecs_, axis):
                init, xy, vecs = go(init_), go(xy_), go(vecs_)
                return self.reduce(init, x, y, xy, vecs, axis)
            case yarr.Tuple(operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: tuple(op(env) for op in operands)
            case yarr.Untuple(at, _arity, target_):
                tup = go(target_)
                return lambda env: tup(env)[at]
            case yarr.Einsum(subs, operands_):
                operands = tuple(go(op_) for op_ in operands_)
                einsum_fun = self.prepare_einsum(subs=subs)
                return lambda env: einsum_fun(*(op(env) for op in operands))
            case yarr.Extrinsic(_, fun, operands_):
                operands = tuple(go(op) for op in operands_)
                return lambda env: fun(*(op(env) for op in operands))
        return None

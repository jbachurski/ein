from functools import cache
from typing import Any, Callable, Optional, Sequence, TypeAlias, cast

import numpy

from ein.codegen import phi_to_yarr, yarr
from ein.codegen.yarr import (
    BinaryElementwise,
    ReduceAxis,
    TernaryElementwise,
    UnaryElementwise,
)
from ein.midend.lining import outline
from ein.phi import phi
from ein.symbols import Variable
from ein.value import Value

from .array_backend import AbstractArrayBackend

try:
    import jax
    from jax import config
    from jax import numpy as jnp

    config.update("jax_enable_x64", True)
except ImportError:

    class MagicGetter:
        def __getattr__(self, item) -> Any:
            return None

    jax = MagicGetter()  # type: ignore
    jnp = MagicGetter()  # type: ignore


Env: TypeAlias = dict[Variable, jnp.ndarray | tuple[jnp.ndarray, ...]]


class JaxBackend(AbstractArrayBackend[jnp.ndarray]):
    def constant(self, value: Value) -> jnp.ndarray:
        if isinstance(value.value, jnp.ndarray):
            return value.value
        array = value.array
        ret = jnp.asarray(array, dtype=array.dtype)
        return ret

    def preprocess_bound(self, target: jnp.ndarray) -> jnp.ndarray:
        return target

    def dim(self, target: jnp.ndarray, axis: int) -> jnp.ndarray:
        return target.shape[axis]

    def range(self, size: jnp.ndarray) -> jnp.ndarray:
        return jnp.arange(size)

    def concat(self, *args: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.concatenate(args, axis=axis)

    def transpose(
        self, target: jnp.ndarray, permutation: tuple[int, ...]
    ) -> jnp.ndarray:
        return jnp.transpose(target, permutation)

    def squeeze(self, target: jnp.ndarray, axes: tuple[int, ...]) -> jnp.ndarray:
        return jnp.squeeze(target, axes)

    def unsqueeze(self, target: jnp.ndarray, axes: tuple[int, ...]) -> jnp.ndarray:
        return jnp.expand_dims(target, axes)

    def gather(self, target: jnp.ndarray, item: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.take_along_axis(
            target, jnp.clip(item, 0, target.shape[axis] - 1), axis
        )

    def take(
        self, target: jnp.ndarray, items: Sequence[Optional[jnp.ndarray]]
    ) -> jnp.ndarray:
        SLICE_NONE = slice(None)
        it: Any = (
            jnp.clip(item, 0, dim - 1) if item is not None else SLICE_NONE
            for dim, item in zip(target.shape, items)
        )
        return target[*it]

    def slice(self, target: jnp.ndarray, slices: Sequence[slice]) -> jnp.ndarray:
        print(slices)
        return target[*slices]

    def pad(self, target: jnp.ndarray, pads: Sequence[tuple[int, int]]) -> jnp.ndarray:
        return jnp.pad(target, pads, mode="edge")  # type: ignore

    def repeat(self, target: jnp.ndarray, count: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.repeat(target, count, axis=axis)

    def prepare_einsum(self, subs: str) -> Callable[..., jnp.ndarray]:
        return lambda *args: jnp.einsum(subs, *args)

    def fold(
        self,
        index_var: Variable,
        count: Callable,
        acc_var: Variable,
        init: Callable,
        body: Callable,
    ) -> Callable:
        def jax_fold_impl(env):
            def jax_body(i, acc):
                env[acc_var] = acc
                env[index_var] = i
                ret = body(env)
                del env[acc_var], env[index_var]
                return ret

            return jax.lax.fori_loop(0, jnp.maximum(count(env), 0), jax_body, init(env))

        return jax_fold_impl

    def reduce(
        self,
        init: Callable,
        x: Variable,
        y: Variable,
        xy: Callable,
        vecs: Callable,
        axis: int,
    ) -> Callable:
        return super().reduce(init, x, y, xy, vecs, axis)
        # def jax_reduce_impl(env):
        #     vs, it = vecs(env), init(env)
        #     wraps = not isinstance(vs, tuple)
        #     if wraps:
        #         vs = (vs,)
        #         it = (it,)
        #
        #     def jax_computation(first, second):
        #         if wraps:
        #             (first,) = first
        #             (second,) = second
        #         env[x] = first
        #         env[y] = second
        #         ret = xy(env)
        #         print(first, second, "=>", ret)
        #         del env[x], env[y]
        #         if wraps:
        #             ret = (ret,)
        #         return ret
        #
        #     # FIXME: This is a silly heuristic to check if we actually can use XLA Reduce,
        #     #  which only accepts scalars as the reduced object.
        #     if any(i.ndim for i in it):
        #         return super(type(self), self).reduce(init, x, y, xy, vecs, axis)(env)
        #     print("vecs", vs, "init", it)
        #     result = jax.lax.reduce(vs, it, jax_computation, [axis])
        #     if wraps:
        #         (result,) = result
        #     return result
        #
        # return jax_reduce_impl


INSTANCE = JaxBackend()


def stage_in_array(
    program: yarr.Expr,
) -> Callable[[Env], jnp.ndarray | tuple[jnp.ndarray, ...]]:
    def go(expr: yarr.AbstractExpr) -> Callable[[Env], jnp.ndarray]:
        return cast(Callable[[Env], jnp.ndarray], go_either(expr))

    @cache
    def go_either(
        expr: yarr.AbstractExpr,
    ) -> Callable[[Env], jnp.ndarray | tuple[jnp.ndarray, ...]]:
        expr = cast(yarr.Expr, expr)
        match expr:
            case yarr.ReduceAxis(kind, axis, target_):
                target = go(target_)
                call = REDUCE[kind]
                return lambda env: call(target(env), axis=axis)
            case yarr.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).astype(dtype)
            case yarr.UnaryElementwise(kind, target_):
                target = go(target_)
                call = ELEMENTWISE[kind]
                return lambda env: call(target(env))
            case yarr.BinaryElementwise(kind, first_, second_, _inplace):
                first, second = go(first_), go(second_)
                call = ELEMENTWISE[kind]
                return lambda env: call(first(env), second(env))
            case yarr.TernaryElementwise(kind, first_, second_, third_):
                first, second, third = go(first_), go(second_), go(third_)
                call = ELEMENTWISE[kind]
                return lambda env: call(first(env), second(env), third(env))
            case _:
                from_default = INSTANCE.stage(expr, go)
                assert from_default is not None
                return from_default

    program = cast(yarr.Expr, outline(program))
    return go(program)


def stage(
    program: phi.Expr,
) -> Callable[[dict[Variable, jnp.ndarray]], jnp.ndarray | tuple[jnp.ndarray, ...]]:
    array_program = phi_to_yarr.transform(program)
    with jax.ensure_compile_time_eval():
        staged = stage_in_array(array_program)
    free_vars = list(array_program.free_variables)

    def wrapped_staged(*args):
        with jax.ensure_compile_time_eval():
            return staged({v: arg for v, arg in zip(free_vars, args)})

    # Some tinkering so Jax is happy to get a simple sequence of arguments
    def wrapper(
        env: dict[Variable, jnp.ndarray]
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        return jax.jit(wrapped_staged)(*(env[var] for var in free_vars))

    return wrapper


def interpret(
    program: phi.Expr, env: dict[Variable, numpy.ndarray | jnp.ndarray]
) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
    return stage(program)({var: jnp.asarray(arr) for var, arr in env.items()})


REDUCE: Any = {
    ReduceAxis.Kind.add: jnp.sum,
    ReduceAxis.Kind.minimum: jnp.amin,
    ReduceAxis.Kind.maximum: jnp.amax,
}
ELEMENTWISE: Any = {
    UnaryElementwise.Kind.negative: jnp.negative,
    UnaryElementwise.Kind.reciprocal: jnp.reciprocal,
    UnaryElementwise.Kind.exp: jnp.exp,
    UnaryElementwise.Kind.sin: jnp.sin,
    UnaryElementwise.Kind.cos: jnp.cos,
    UnaryElementwise.Kind.logical_not: jnp.logical_not,
    BinaryElementwise.Kind.add: jnp.add,
    BinaryElementwise.Kind.subtract: jnp.subtract,
    BinaryElementwise.Kind.multiply: jnp.multiply,
    BinaryElementwise.Kind.mod: jnp.mod,
    BinaryElementwise.Kind.power: jnp.power,
    BinaryElementwise.Kind.minimum: jnp.minimum,
    BinaryElementwise.Kind.maximum: jnp.maximum,
    BinaryElementwise.Kind.less: jnp.less,
    BinaryElementwise.Kind.less_equal: jnp.less_equal,
    BinaryElementwise.Kind.equal: jnp.equal,
    BinaryElementwise.Kind.not_equal: jnp.not_equal,
    BinaryElementwise.Kind.logical_and: jnp.logical_and,
    BinaryElementwise.Kind.logical_or: jnp.logical_or,
    TernaryElementwise.Kind.where: jnp.where,
}

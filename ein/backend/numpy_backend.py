from functools import cache
from typing import Any, Callable, Sequence, TypeAlias, cast

import numpy

from ein.codegen import phi_to_yarr, yarr
from ein.midend.lining import outline
from ein.phi import calculus
from ein.symbols import Variable
from ein.value import Value

from .array_backend import AbstractArrayBackend

Env: TypeAlias = dict[Variable, numpy.ndarray | tuple[numpy.ndarray, ...]]


class NumpyBackend(AbstractArrayBackend[numpy.ndarray]):
    @classmethod
    def constant(cls, value: Value) -> numpy.ndarray:
        return value.array

    @classmethod
    def preprocess_bound(cls, target: numpy.ndarray) -> numpy.ndarray:
        if target.base is not None:
            base = target.base.strides
            # Memory order non-monotonic
            if any(x < y for x, y in zip(base, base[1:])):
                return target.copy()
        return target

    @classmethod
    def dim(cls, target: numpy.ndarray, axis: int) -> numpy.ndarray:
        return numpy.array(target.shape[axis])

    @classmethod
    def range(cls, size: numpy.ndarray) -> numpy.ndarray:
        return numpy.arange(int(size))

    @classmethod
    def concat(cls, *args: numpy.ndarray, axis: int) -> numpy.ndarray:
        return numpy.concatenate(args, axis=axis)

    @classmethod
    def transpose(
        cls, target: numpy.ndarray, permutation: tuple[int, ...]
    ) -> numpy.ndarray:
        return numpy.transpose(target, permutation)

    @classmethod
    def squeeze(cls, target: numpy.ndarray, axes: tuple[int, ...]) -> numpy.ndarray:
        return numpy.squeeze(target, axes)

    @classmethod
    def unsqueeze(cls, target: numpy.ndarray, axes: tuple[int, ...]) -> numpy.ndarray:
        return numpy.expand_dims(target, axes)

    @classmethod
    def gather(
        cls, target: numpy.ndarray, item: numpy.ndarray, axis: int
    ) -> numpy.ndarray:
        return numpy.take_along_axis(
            target, numpy.clip(item, 0, target.shape[axis] - 1), axis
        )

    @classmethod
    def take(
        cls, target: numpy.ndarray, items: Sequence[numpy.ndarray | None]
    ) -> numpy.ndarray:
        SLICE_NONE = slice(None)
        it: Any = (
            numpy.clip(item, 0, dim - 1) if item is not None else SLICE_NONE
            for dim, item in zip(target.shape, items)
        )
        return target[*it]

    @classmethod
    def slice(cls, target: numpy.ndarray, slices: Sequence[slice]) -> numpy.ndarray:
        return target[*slices]

    @classmethod
    def pad(
        cls, target: numpy.ndarray, pads: Sequence[tuple[int, int]]
    ) -> numpy.ndarray:
        return fast_edge_pad(target, tuple(pads))

    @classmethod
    def repeat(
        cls, target: numpy.ndarray, count: numpy.ndarray, axis: int
    ) -> numpy.ndarray:
        return numpy.repeat(target, int(count), axis=axis)

    @classmethod
    def prepare_einsum(cls, subs: str) -> Callable[..., numpy.ndarray]:
        op_subs, res_subs = subs.split("->")
        ops_subs = op_subs.split(",")
        dummy_ops = [numpy.empty((4,) * len(curr_subs)) for curr_subs in ops_subs]
        path = numpy.einsum_path(subs, *dummy_ops, optimize="optimal")[0]
        return lambda *args: numpy.einsum(subs, *args, optimize=path)


INSTANCE = NumpyBackend()


def fast_edge_pad(
    target: numpy.ndarray, pads: tuple[tuple[int, int], ...]
) -> numpy.ndarray:
    if not target.size:
        return numpy.empty(tuple(lt + rt for lt, rt in pads))
    if sum(lt + rt for lt, rt in pads) == 0:
        return target.copy()
    if all(not (lt and rt) for lt, rt in pads):
        res = numpy.empty([d + lt + rt for d, (lt, rt) in zip(target.shape, pads)])
        edge_index: list[int | numpy.ndarray | slice] = []
        edge_unsqueeze: list[None | slice] = []
        dest: list[slice] = []
        dest_op: list[slice] = []
        skip = slice(None)
        for lt, rt in pads:
            if lt == rt == 0:
                edge_index.append(skip)
                edge_unsqueeze.append(skip)
                dest.append(skip)
                dest_op.append(skip)
            elif lt == 0:
                edge_index.append(-1)
                edge_unsqueeze.append(None)
                dest.append(slice(-rt))
                dest_op.append(slice(-rt, None))
            elif rt == 0:
                edge_index.append(0)
                edge_unsqueeze.append(None)
                dest.append(slice(lt, None))
                dest_op.append(slice(lt))
        res[*dest] = target
        res[*dest_op] = target[*edge_index][*edge_unsqueeze]
        return res
    return numpy.pad(target, pads, mode="edge")  # type: ignore


def stage_in_array(
    program: yarr.Expr,
) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
    def go(expr: yarr.Expr) -> Callable[[Env], numpy.ndarray]:
        return cast(Callable[[Env], numpy.ndarray], go_either(expr))

    @cache
    def go_either(
        expr: yarr.Expr,
    ) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
        match expr:
            case yarr.ReduceAxis(kind, axis, target_):
                target = go(target_)
                call = yarr.REDUCE_KINDS[kind].reduce
                return lambda env: call(target(env), axis=axis)
            case yarr.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).astype(dtype)
            case yarr.UnaryElementwise(kind, target_):
                target = go(target_)
                call = yarr.ELEMENTWISE_KINDS[kind]
                return lambda env: call(target(env))
            case yarr.BinaryElementwise(kind, first_, second_, inplace):
                first, second = go(first_), go(second_)
                call = yarr.ELEMENTWISE_KINDS[kind]
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
            case yarr.TernaryElementwise(kind, first_, second_, third_):
                first, second, third = go(first_), go(second_), go(third_)
                call = yarr.ELEMENTWISE_KINDS[kind]
                return lambda env: call(first(env), second(env), third(env))
            case _:
                from_default = INSTANCE.stage(expr, go)
                assert from_default is not None
                return from_default

    program = cast(yarr.Expr, outline(program))
    program = phi_to_yarr.apply_inplace_on_temporaries(program)

    return go(program)


def stage(
    program: calculus.Expr,
) -> Callable[[dict[Variable, numpy.ndarray]], numpy.ndarray]:
    array_program = phi_to_yarr.transform(program)
    return stage_in_array(array_program)  # type: ignore


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return stage(program)(env)

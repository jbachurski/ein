from functools import cache
from typing import Any, Callable, Sequence, TypeAlias, cast

import numpy

from ein import calculus
from ein.backend.array_backend import AbstractArrayBackend
from ein.midend.lining import outline
from ein.midend.structs import struct_of_arrays_transform
from ein.symbols import Variable
from ein.value import Value

from . import array_calculus, to_array

Env: TypeAlias = dict[Variable, numpy.ndarray | tuple[numpy.ndarray, ...]]


class NumpyBackend(AbstractArrayBackend[numpy.ndarray]):
    @classmethod
    def constant(cls, value: Value) -> numpy.ndarray:
        return value.array

    @classmethod
    def preprocess_bound(cls, target: numpy.ndarray) -> numpy.ndarray:
        return target.copy() if target.base is not None else target

    @classmethod
    def dim(cls, target: numpy.ndarray, axis: int) -> numpy.ndarray:
        return numpy.array(target.shape[axis])

    @classmethod
    def range(cls, size: numpy.ndarray) -> numpy.ndarray:
        return numpy.arange(int(size))

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
    program: array_calculus.Expr,
) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
    def go(expr: array_calculus.Expr) -> Callable[[Env], numpy.ndarray]:
        return cast(Callable[[Env], numpy.ndarray], go_either(expr))

    @cache
    def go_either(
        expr: array_calculus.Expr,
    ) -> Callable[[Env], numpy.ndarray | tuple[numpy.ndarray, ...]]:
        match expr:
            case array_calculus.Reduce(kind, axis, target_):
                target = go(target_)
                call = array_calculus.REDUCE_KINDS[kind].reduce
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
            case _:
                from_default = INSTANCE.stage(expr, go)
                assert from_default is not None
                return from_default

    program = cast(array_calculus.Expr, outline(program))
    program = to_array.apply_inplace_on_temporaries(program)

    return go(program)


def prepare(program: calculus.Expr) -> array_calculus.Expr:
    return to_array.transform(struct_of_arrays_transform(program))


def stage(
    program: calculus.Expr,
) -> Callable[[dict[Variable, numpy.ndarray]], numpy.ndarray]:
    return stage_in_array(prepare(program))  # type: ignore


def interpret(
    program: calculus.Expr, env: dict[Variable, numpy.ndarray]
) -> numpy.ndarray:
    return stage(program)(env)

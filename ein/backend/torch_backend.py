import itertools
from functools import cache
from typing import Any, Callable, Sequence, TypeAlias, cast

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
    import torch
except ImportError:

    class MagicGetter:
        def __getattr__(self, item) -> Any:
            return None

    torch = MagicGetter()  # type: ignore


Env: TypeAlias = dict[Variable, torch.Tensor | tuple[torch.Tensor, ...]]


class TorchBackend(AbstractArrayBackend[torch.Tensor]):
    def constant(self, value: Value) -> torch.Tensor:
        if isinstance(value.value, torch.Tensor):
            return value.value
        array = value.array
        ret = torch.from_numpy(array)
        return ret

    def preprocess_bound(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def dim(self, target: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.scalar_tensor(target.shape[axis], dtype=torch.int64)

    def range(self, size: torch.Tensor) -> torch.Tensor:
        return torch.arange(int(size))

    def concat(self, *args: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.concat(args, dim=axis)

    def transpose(
        self, target: torch.Tensor, permutation: tuple[int, ...]
    ) -> torch.Tensor:
        return target.permute(*permutation)

    def squeeze(self, target: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        return squeeze_axes(target, axes)

    def unsqueeze(self, target: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        return unsqueeze_axes(target, axes)

    def gather(
        self, target: torch.Tensor, item: torch.Tensor, axis: int
    ) -> torch.Tensor:
        return torch.gather(target, axis, torch.clip(item, 0, target.shape[axis] - 1))

    def take(
        self, target: torch.Tensor, items: Sequence[torch.Tensor | None]
    ) -> torch.Tensor:
        SLICE_NONE = slice(None)
        it = (
            torch.clip(item, 0, dim - 1) if item is not None else SLICE_NONE
            for dim, item in zip(target.shape, items)
        )
        return target[*it]

    def slice(self, target: torch.Tensor, slices: Sequence[slice]) -> torch.Tensor:
        return target[*slices]

    def pad(
        self, target: torch.Tensor, pads: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        return (
            torch.nn.functional.pad(
                target.unsqueeze(0).unsqueeze(0),
                tuple(itertools.chain(*pads)),
                mode="replicate",
            )
            .squeeze(0)
            .squeeze(0)
        )

    def repeat(
        self, target: torch.Tensor, count: torch.Tensor, axis: int
    ) -> torch.Tensor:
        return torch.repeat_interleave(target, int(count), axis)

    def prepare_einsum(self, subs: str) -> Callable[..., torch.Tensor]:
        return lambda *args: torch.einsum(subs, *args)


INSTANCE = TorchBackend()


def squeeze_axes(tensor, axes):
    for axis in axes:
        tensor = tensor.squeeze(axis)
    return tensor


def unsqueeze_axes(tensor, axes):
    for axis in axes:
        tensor = tensor.unsqueeze(axis)
    return tensor


def stage_in_array(
    program: yarr.Expr,
) -> Callable[[Env], torch.Tensor | tuple[torch.Tensor, ...]]:
    def go(expr: yarr.AbstractExpr) -> Callable[[Env], torch.Tensor]:
        return cast(Callable[[Env], torch.Tensor], go_either(expr))

    @cache
    def go_either(
        expr: yarr.AbstractExpr,
    ) -> Callable[[Env], torch.Tensor | tuple[torch.Tensor, ...]]:
        expr = cast(yarr.Expr, expr)
        match expr:
            case yarr.ReduceAxis(kind, axis, target_):
                target = go(target_)
                call = REDUCE[kind]
                return lambda env: call(target(env), axis=axis)
            case yarr.Cast(dtype, target_):
                target = go(target_)
                return lambda env: target(env).to(dtype)
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
) -> Callable[[dict[Variable, torch.Tensor]], torch.Tensor | tuple[torch.Tensor, ...]]:
    return stage_in_array(phi_to_yarr.transform(program))


def interpret(
    program: phi.Expr, env: dict[Variable, numpy.ndarray | torch.Tensor]
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
    ReduceAxis.Kind.add: torch.sum,
    ReduceAxis.Kind.minimum: torch.amin,
    ReduceAxis.Kind.maximum: torch.amax,
}
ELEMENTWISE: Any = {
    UnaryElementwise.Kind.negative: torch.negative,
    UnaryElementwise.Kind.reciprocal: torch.reciprocal,
    UnaryElementwise.Kind.exp: torch.exp,
    UnaryElementwise.Kind.sin: torch.sin,
    UnaryElementwise.Kind.cos: torch.cos,
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

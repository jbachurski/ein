from typing import Callable, Sequence

from numpy import ndarray

from ein.backend import STAGE_BACKENDS, Backend
from ein.phi.calculus import Expr, variable
from ein.phi.type_system import Type, type_from_ndarray
from ein.symbols import Variable

from .ndarray import ArrayLike, _phi_to_yarr, _TorchTensor, wrap


def with_varargs(
    types: Sequence[Type], fun: Callable[..., ArrayLike]
) -> tuple[tuple[Variable, ...], Expr]:
    arg_vars = [variable(Variable(), type_) for type_ in types]
    args = [_phi_to_yarr(var) for var in arg_vars]
    return tuple(var.var for var in arg_vars), wrap(fun(*args)).expr


class Function:
    types: Sequence[Type] | None = None
    _staged: dict[
        tuple[tuple[Type, ...], Backend], tuple[tuple[Variable, ...], Callable]
    ]

    def __init__(
        self, fun: Callable[..., ArrayLike], *, types: Sequence[Type] | None = None
    ):
        self.fun = fun
        self.types = tuple(types) if types is not None else None
        self._staged = {}

    def _types_from_args(self, *args) -> tuple[Type, ...]:
        types = tuple(type_from_ndarray(arg) for arg in args)
        if self.types is not None and types != self.types:
            raise TypeError(
                f"Mismatched type signature in call: got {types}, expected {self.types}."
            )
        return types

    def _interpret(self, args, backend: Backend):
        types = self._types_from_args(*args)
        key = (types, backend)
        if key not in self._staged:
            varargs, expr = self.phi(*types)
            self._staged[key] = (varargs, STAGE_BACKENDS[backend](expr))
        varargs, call = self._staged[key]
        return call({var: arg for var, arg in zip(varargs, args)})

    def phi(self, *types: Type) -> tuple[tuple[Variable, ...], Expr]:
        return with_varargs(types, self.fun)

    def numpy(self, *args: ndarray) -> ndarray:
        return self._interpret(args, backend="numpy")

    def torch(self, *args: ndarray | _TorchTensor) -> _TorchTensor:
        import torch

        args = tuple(
            arg if isinstance(arg, _TorchTensor) else torch.from_numpy(arg)
            for arg in args
        )
        return self._interpret(args, backend="torch")

    # @overload
    # def __call__(self, *args: ndarray | float | int) -> ndarray:
    #     ...
    #
    # @overload
    # def __call__(self, *args: _TorchTensor | ndarray | float | int) -> _TorchTensor:
    #     ...

    def __call__(self, *args):
        if any(isinstance(arg, _TorchTensor) for arg in args):
            return self.torch(*args)
        return self.numpy(*args)


def typed_function(*args: Type) -> Callable[[Callable[..., ArrayLike]], Function]:
    return lambda fun: Function(fun, types=args)


def function(fun: Callable[..., ArrayLike]) -> Function:
    return Function(fun)

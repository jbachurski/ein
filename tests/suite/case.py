import abc
from typing import Callable, ClassVar, Sequence

import numpy

from ein import Array, Type, function
from ein.calculus import Expr, Variable


class Case(abc.ABC):
    ein_types: ClassVar[Sequence[Type]]

    @staticmethod
    @abc.abstractmethod
    def in_ein(*args: Array) -> Array:
        ...

    @staticmethod
    @abc.abstractmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        ...

    @staticmethod
    @abc.abstractmethod
    def sample() -> tuple[numpy.ndarray, ...]:
        ...

    @classmethod
    def in_ein_function(
        cls,
        interpret: Callable[[Expr, dict[Variable, numpy.ndarray]], numpy.ndarray],
        *args: numpy.ndarray,
    ):
        arg_vars, expr = function(cls.ein_types, cls.in_ein)
        return interpret(expr, {v: a for v, a in zip(arg_vars, args)})

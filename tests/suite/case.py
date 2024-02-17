import abc
from typing import Callable, ClassVar, Sequence

import numpy

from ein import Array, calculus, function
from ein.calculus import Expr, Variable
from ein.type_system import Type


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
    def ein_function(cls) -> tuple[tuple[Variable, ...], calculus.Expr]:
        return function(cls.ein_types, cls.in_ein)

    @classmethod
    def in_ein_function(
        cls,
        interpret: Callable[[Expr, dict[Variable, numpy.ndarray]], numpy.ndarray],
        *args: numpy.ndarray,
    ) -> numpy.ndarray:
        arg_vars, expr = cls.ein_function()
        return interpret(expr, {v: a for v, a in zip(arg_vars, args)})

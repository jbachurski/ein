import abc
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import numpy

from ein import Scalar, Vec, array, fold, matrix_type, wrap
from ein.frontend import std
from ein.frontend.typed import array2

from ..case import Case

T = TypeVar("T", bound="Semiring")


class Semiring(abc.ABC):
    @property
    @abc.abstractmethod
    def zero(self) -> Self:
        ...

    @property
    @abc.abstractmethod
    def one(self) -> Self:
        ...

    @property
    @abc.abstractmethod
    def closure(self) -> Self:
        ...

    @abc.abstractmethod
    def __add__(self, other: Self) -> Self:
        ...

    @abc.abstractmethod
    def __mul__(self, other: Self) -> Self:
        ...


@dataclass
class SemiringMatrix(Generic[T], Semiring):
    elem: Vec[Vec[T]]

    @property
    def _any(self) -> T:
        return self.elem[0, 0]

    @property
    def zero(self) -> "SemiringMatrix[T]":
        size = (self.elem.size(0), self.elem.size(1))
        return SemiringMatrix(array(lambda i, j: self._any.zero, size=size))

    @property
    def one(self) -> "SemiringMatrix[T]":
        size = (self.elem.size(0), self.elem.size(1))
        r = SemiringMatrix(
            array(
                lambda i, j: std.where(i == j, self._any.one, self._any.zero), size=size
            )
        )
        return r

    @property
    def closure(self):
        # Algorithm 6.1 of https://dl.acm.org/doi/10.5555/6400.6411
        eliminated = fold(
            self.elem,
            lambda k, acc: array2(
                lambda i, j: acc[i, j] + acc[i, k] * acc[k, k].closure * acc[k, j]
            ).assume(self.elem[k, k]),
        )
        return SemiringMatrix(eliminated) + self.one

    def __add__(self, other: "SemiringMatrix[T]") -> "SemiringMatrix[T]":
        return SemiringMatrix(array2(lambda i, j: self.elem[i, j] + other.elem[i, j]))

    def __mul__(self, other: "SemiringMatrix[T]") -> "SemiringMatrix[T]":
        return SemiringMatrix(
            array2(
                lambda i, j: fold(
                    self._any.zero,
                    lambda k, acc: acc + self.elem[i, k] * other.elem[k, j],
                )
            )
        )


@dataclass
class Tropical(Semiring):
    x: Scalar

    @property
    def zero(self) -> "Tropical":
        return Tropical(wrap(float("+inf")))

    @property
    def one(self) -> "Tropical":
        return Tropical(wrap(0.0))

    @property
    def closure(self) -> "Tropical":
        return self.one

    def __add__(self, other: "Tropical") -> "Tropical":
        return Tropical(std.min(self.x, other.x))

    def __mul__(self, other: "Tropical") -> "Tropical":
        return Tropical(self.x + other.x)


class FunWithSemirings(Case):
    ein_types = [matrix_type(float)]

    @staticmethod
    def in_ein(d0: Vec[Vec[Scalar]]) -> Vec[Vec[Scalar]]:
        d = SemiringMatrix(array2(lambda i, j: Tropical(d0[i, j])))
        d_star = d.closure
        return array2(lambda i, j: d_star.elem[i, j].x)

    @staticmethod
    def in_numpy(d: numpy.ndarray) -> numpy.ndarray:
        d = d.copy()
        n, n_ = d.shape
        d[numpy.diag_indices(n)] = 0
        for k in range(n):
            d = numpy.minimum(d, d[:, k][:, None] + d[k, :][None, :])
        return d

    @staticmethod
    def in_python(d0: numpy.ndarray) -> numpy.ndarray:
        d = [[x for x in row] for row in d0]
        n, n_ = d0.shape
        for i in range(n):
            d[i][i] = 0.0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        return numpy.array(d)

    @staticmethod
    def sample(n: int = 10) -> tuple[numpy.ndarray]:
        return (numpy.abs(numpy.random.randn(n, n)),)

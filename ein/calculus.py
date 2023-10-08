import abc
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

import numpy


class Index:
    pass


class Variable:
    pass


@dataclass(frozen=True, eq=False)
class Value:
    array: numpy.ndarray


Expr: TypeAlias = (
    "Vec | Sum | Get | Const | At | Var | Shape | Negate | Reciprocal | Add | Multiply"
)


def _merge_adj(*args: dict[Index, set["Expr"]]):
    result: dict[Index, set["Expr"]] = {}
    for arg in args:
        for k, v in arg.items():
            result.setdefault(k, set()).update(v)
    return result


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    @property
    @abc.abstractmethod
    def dependencies(self) -> set[Expr]:
        ...

    @property
    def _indices(self) -> set[Index]:
        return set()

    @property
    def _captured_indices(self) -> set[Index]:
        return set()

    @cached_property
    def free_indices(self) -> set[Index]:
        deps = self.dependencies
        sub_free_indices = (
            set.union(*(sub.free_indices for sub in deps)) if deps else set()
        )
        indices = self._indices | sub_free_indices
        assert self._captured_indices <= indices
        return indices - self._captured_indices

    @cached_property
    def direct_indices(self) -> dict[Index, set[Expr]]:
        return _merge_adj(*(sub.direct_indices for sub in self.dependencies))


@dataclass(frozen=True, eq=False)
class Vec(AbstractExpr):
    index: Index
    size: Expr
    body: Expr

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.size, self.body}

    @property
    def _captured_indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpr):
    index: Index
    size: Expr
    body: Expr

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.size, self.body}

    @property
    def _captured_indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    operand: Expr
    item: Expr

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.operand, self.item}

    @property
    def direct_index(self) -> Index | None:
        return self.item.index if isinstance(self.item, At) else None

    @cached_property
    def direct_indices(self) -> dict[Index, set[Expr]]:
        index = self.direct_index
        return _merge_adj(
            super().direct_indices, {index: {self.operand}} if index is not None else {}
        )


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value

    @property
    def dependencies(self) -> set[Expr]:
        return set()


@dataclass(frozen=True, eq=False)
class At(AbstractExpr):
    index: Index

    @property
    def dependencies(self) -> set[Expr]:
        return set()

    @property
    def _indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable

    @property
    def dependencies(self) -> set[Expr]:
        return set()


@dataclass(frozen=True, eq=False)
class Shape(AbstractExpr):
    operand: Expr
    axis: int

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.operand}


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    operands: tuple[Expr, ...]

    @cached_property
    def dependencies(self) -> set[Expr]:
        return set(self.operands)


@dataclass(frozen=True, eq=False)
class Negate(AbstractScalarOperator):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Reciprocal(AbstractScalarOperator):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Add(AbstractScalarOperator):
    operands: tuple[Expr, Expr]


@dataclass(frozen=True, eq=False)
class Multiply(AbstractScalarOperator):
    operands: tuple[Expr, Expr]


__all__ = [
    "Index",
    "Variable",
    "Value",
    "Expr",
    "Vec",
    "Sum",
    "Get",
    "Const",
    "At",
    "Var",
    "Shape",
    "Negate",
    "Reciprocal",
    "Add",
    "Multiply",
]

import abc
import enum
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, TypeAlias

import numpy

from ein.symbols import Variable

Expr: TypeAlias = (
    "Const | Var | Dim | Range | Transpose | Squeeze | Unsqueeze | Gather | Repeat | "
    "Reduce | Cast | UnaryElementwise | BinaryElementwise | TernaryElementwise | Fold"
)


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    def __post_init__(self):
        assert self.rank >= 0

    @property
    @abc.abstractmethod
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        ...

    @property
    @abc.abstractmethod
    def rank(self) -> int:
        ...


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    array: numpy.ndarray

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"array": self.array}, set()

    @property
    def rank(self) -> int:
        return self.array.ndim


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable
    var_rank: int

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @property
    def rank(self) -> int:
        return self.var_rank


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    axis: int
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.target}

    @property
    def rank(self) -> int:
        return 0


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    size: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.size}

    @cached_property
    def rank(self) -> int:
        assert not self.size.rank
        return 1


@dataclass(frozen=True, eq=False)
class Transpose(AbstractExpr):
    permutation: tuple[int, ...]
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axes": self.permutation}, {self.target}

    @cached_property
    def rank(self) -> int:
        assert len(self.permutation) == self.target.rank
        return self.target.rank


@dataclass(frozen=True, eq=False)
class Squeeze(AbstractExpr):
    axes: tuple[int, ...]
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axes": self.axes}, {self.target}

    @cached_property
    def rank(self) -> int:
        return self.target.rank - len(self.axes)


@dataclass(frozen=True, eq=False)
class Unsqueeze(AbstractExpr):
    axes: tuple[int, ...]
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axes": self.axes}, {self.target}

    @cached_property
    def rank(self) -> int:
        return self.target.rank + len(self.axes)


@dataclass(frozen=True, eq=False)
class Gather(AbstractExpr):
    axis: int
    target: Expr
    item: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.target, self.item}

    @cached_property
    def rank(self) -> int:
        assert (
            self.target.rank == self.item.rank
        ), "Gather assumes broadcast of target and item"
        assert 0 <= self.axis < self.target.rank, "Gather axis not in range"
        return self.target.rank


@dataclass(frozen=True, eq=False)
class Repeat(AbstractExpr):
    axis: int
    count: Expr
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.count, self.target}

    @cached_property
    def rank(self) -> int:
        assert self.count.rank == 0, "Can only repeat scalar number of times"
        assert 0 <= self.axis < self.target.rank, "Repeat axis not in range"
        return self.target.rank


@dataclass(frozen=True, eq=False)
class Reduce(AbstractExpr):
    class Kind(enum.Enum):
        sum = enum.auto()
        max = enum.auto()

    kind: Kind
    axis: int
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"kind": self.kind.name, "axis": self.axis}, {self.target}

    @cached_property
    def rank(self) -> int:
        assert 0 <= self.axis <= self.target.rank, "Mismatched reduction axis"
        return self.target.rank - 1


@dataclass(frozen=True, eq=False)
class Cast(AbstractExpr):
    dtype: numpy.dtype
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"dtype": self.dtype}, {self.target}

    @cached_property
    def rank(self) -> int:
        return self.target.rank


@dataclass(frozen=True, eq=False)
class AbstractElementwise(AbstractExpr):
    @property
    @abc.abstractmethod
    def operands(self) -> tuple[Expr, ...]:
        ...

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"kind": getattr(self, "kind").name}, set(self.operands)

    @cached_property
    def rank(self) -> int:
        ranks = {op.rank for op in self.operands}
        assert (
            len(ranks) == 1
        ), f"Expected unique rank for elementwise broadcast, not {ranks}"
        (rank,) = ranks
        return rank


@dataclass(frozen=True, eq=False)
class UnaryElementwise(AbstractElementwise):
    class Kind(enum.Enum):
        negative = enum.auto()
        reciprocal = enum.auto()
        exp = enum.auto()
        logical_not = enum.auto()

    kind: Kind
    target: Expr

    @property
    def operands(self) -> tuple[Expr]:
        return (self.target,)


@dataclass(frozen=True, eq=False)
class BinaryElementwise(AbstractElementwise):
    class Kind(enum.Enum):
        add = enum.auto()
        multiply = enum.auto()
        less = enum.auto()
        logical_and = enum.auto()

    kind: Kind
    first: Expr
    second: Expr

    @property
    def operands(self) -> tuple[Expr, Expr]:
        return self.first, self.second


@dataclass(frozen=True, eq=False)
class TernaryElementwise(AbstractElementwise):
    class Kind(enum.Enum):
        where = enum.auto()

    kind: Kind
    first: Expr
    second: Expr
    third: Expr

    @property
    def operands(self) -> tuple[Expr, Expr, Expr]:
        return self.first, self.second, self.third


@dataclass(frozen=True, eq=False)
class Fold(AbstractExpr):
    index: Variable
    acc: Variable
    init: Expr
    size: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc}, {self.init, self.size, self.body}

    @cached_property
    def rank(self) -> int:
        assert self.size.rank == 0, "Expected scalar size"
        assert (
            self.init.rank == self.body.rank
        ), "Mismatched init and body accumulator rank"
        return self.body.rank


REDUCE: dict[Any, Any] = {
    numpy.add: partial(Reduce, Reduce.Kind.sum),
    numpy.maximum: partial(Reduce, Reduce.Kind.max),
}

ELEMENTWISE: dict[Any, Any] = {
    numpy.negative: partial(UnaryElementwise, UnaryElementwise.Kind.negative),
    numpy.reciprocal: partial(UnaryElementwise, UnaryElementwise.Kind.reciprocal),
    numpy.exp: partial(UnaryElementwise, UnaryElementwise.Kind.exp),
    numpy.logical_not: partial(UnaryElementwise, UnaryElementwise.Kind.logical_not),
    numpy.add: partial(BinaryElementwise, BinaryElementwise.Kind.add),
    numpy.multiply: partial(BinaryElementwise, BinaryElementwise.Kind.multiply),
    numpy.less: partial(BinaryElementwise, BinaryElementwise.Kind.less),
    numpy.logical_and: partial(BinaryElementwise, BinaryElementwise.Kind.logical_and),
    numpy.where: partial(TernaryElementwise, TernaryElementwise.Kind.where),
}

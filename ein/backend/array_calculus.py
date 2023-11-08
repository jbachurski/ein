import abc
import enum
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Callable, TypeAlias

import numpy

from ein.symbols import Variable

Expr: TypeAlias = (
    "Const | Var | Let | Dim | Range | Transpose | Squeeze | Unsqueeze | Gather | Repeat | "
    "Reduce | Cast | UnaryElementwise | BinaryElementwise | TernaryElementwise | Fold | Tuple | Untuple"
)


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    def __post_init__(self):
        assert self.rank >= 0

    @abc.abstractmethod
    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        ...

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

    def map(self, f: Callable[[Expr], Expr]) -> "Const":
        return self

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

    def map(self, f: Callable[[Expr], Expr]) -> "Var":
        return self

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @property
    def rank(self) -> int:
        return self.var_rank


@dataclass(frozen=True, eq=False)
class Let(AbstractExpr):
    bindings: tuple[tuple[Variable, Expr], ...]
    body: Expr

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Let(
            tuple((var, f(binding)) for var, binding in self.bindings), f(self.body)
        )

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"vars": [var for var, _ in self.bindings]}, {
            binding for _, binding in self.bindings
        } | {self.body}

    @cached_property
    def rank(self) -> int:
        return self.body.rank


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    axis: int
    target: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Dim":
        return Dim(self.axis, f(self.target))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.target}

    @property
    def rank(self) -> int:
        return 0


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    size: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Range":
        return Range(f(self.size))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Transpose":
        return Transpose(self.permutation, f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Squeeze":
        return Squeeze(self.axes, f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Unsqueeze":
        return Unsqueeze(self.axes, f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Gather":
        return Gather(self.axis, f(self.target), f(self.item))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Repeat":
        return Repeat(self.axis, f(self.count), f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Reduce":
        return Reduce(self.kind, self.axis, f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> "Cast":
        return Cast(self.dtype, f(self.target))

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

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return type(self)(self.kind, *(f(op) for op in self.operands))  # type: ignore

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
        sin = enum.auto()
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
        mod = enum.auto()
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

    def map(self, f: Callable[[Expr], Expr]) -> "Fold":
        return Fold(self.index, self.acc, f(self.init), f(self.size), f(self.body))

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


# FIXME: Typing tuples doesn't work. Should be using type: PrimitiveType, not rank: int.
@dataclass(frozen=True, eq=False)
class Tuple(AbstractExpr):
    operands: tuple[Expr, ...]

    def map(self, f: Callable[[Expr], Expr]) -> "Tuple":
        return Tuple(tuple(f(op) for op in self.operands))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, set(self.operands)

    @cached_property
    def rank(self):
        assert False


@dataclass(frozen=True, eq=False)
class Untuple(AbstractExpr):
    at: int
    arity: int
    target: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Untuple":
        return Untuple(self.at, self.arity, f(self.target))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"at": self.at, "arity": self.arity}, {self.target}

    @cached_property
    def rank(self):
        assert False


REDUCE: dict[Any, Any] = {
    numpy.add: partial(Reduce, Reduce.Kind.sum),
    numpy.maximum: partial(Reduce, Reduce.Kind.max),
}

ELEMENTWISE: dict[Any, Any] = {
    numpy.negative: partial(UnaryElementwise, UnaryElementwise.Kind.negative),
    numpy.reciprocal: partial(UnaryElementwise, UnaryElementwise.Kind.reciprocal),
    numpy.exp: partial(UnaryElementwise, UnaryElementwise.Kind.exp),
    numpy.sin: partial(UnaryElementwise, UnaryElementwise.Kind.sin),
    numpy.logical_not: partial(UnaryElementwise, UnaryElementwise.Kind.logical_not),
    numpy.add: partial(BinaryElementwise, BinaryElementwise.Kind.add),
    numpy.multiply: partial(BinaryElementwise, BinaryElementwise.Kind.multiply),
    numpy.mod: partial(BinaryElementwise, BinaryElementwise.Kind.mod),
    numpy.less: partial(BinaryElementwise, BinaryElementwise.Kind.less),
    numpy.logical_and: partial(BinaryElementwise, BinaryElementwise.Kind.logical_and),
    numpy.where: partial(TernaryElementwise, TernaryElementwise.Kind.where),
}

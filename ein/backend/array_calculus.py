import abc
import enum
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Callable, Optional, TypeAlias

import numpy

from ein.calculus import Value
from ein.symbols import Variable
from ein.type_system import (
    UFUNC_SIGNATURES,
    PrimitiveArrayType,
    PrimitiveType,
    Scalar,
    ScalarKind,
    resolve_scalar_signature,
)

Expr: TypeAlias = (
    "Const | Var | Let | Dim | Range | "
    "Transpose | Squeeze | Unsqueeze | Gather | Take | Slice | Repeat | "
    "Reduce | Cast | UnaryElementwise | BinaryElementwise | TernaryElementwise | Fold | Tuple | Untuple"
)


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    def __post_init__(self):
        assert self.type

    @abc.abstractmethod
    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        ...

    @property
    @abc.abstractmethod
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        ...

    @property
    @abc.abstractmethod
    def type(self) -> PrimitiveType:
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
    def type(self) -> PrimitiveType:
        return Value(self.array).type.primitive_type


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable
    var_type: PrimitiveType

    def map(self, f: Callable[[Expr], Expr]) -> "Var":
        return self

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @property
    def type(self) -> PrimitiveType:
        return self.var_type


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
    def type(self) -> PrimitiveType:
        return self.body.type


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    axis: int
    target: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Dim":
        return Dim(self.axis, f(self.target))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.target}

    @cached_property
    def type(self) -> PrimitiveType:
        assert 0 <= self.axis < self.target.type.single.rank
        return PrimitiveType.of_array(0, int)


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    size: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Range":
        return Range(f(self.size))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.size}

    @cached_property
    def type(self) -> PrimitiveType:
        assert not self.size.type.single.rank
        return PrimitiveType.of_array(1, int)


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
    def type(self) -> PrimitiveType:
        assert len(self.permutation) == self.target.type.single.rank
        return self.target.type


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
    def type(self) -> PrimitiveType:
        assert len(self.axes) == len(set(self.axes))
        return self.target.type.single_with_rank_delta(-len(self.axes))


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
    def type(self) -> PrimitiveType:
        assert len(self.axes) == len(set(self.axes))
        return self.target.type.single_with_rank_delta(+len(self.axes))


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
    def type(self) -> PrimitiveType:
        assert (
            self.target.type.single.rank == self.item.type.single.rank
        ), "Gather broadcasts target and item"
        assert 0 <= self.axis < self.target.type.single.rank, "Gather axis not in range"
        assert self.item.type.single.kind == int
        return self.target.type


@dataclass(frozen=True, eq=False)
class Take(AbstractExpr):
    target: Expr
    items: tuple[Optional[Expr], ...]

    def map(self, f: Callable[[Expr], Expr]) -> "Take":
        return Take(
            f(self.target),
            tuple(f(item) if item is not None else None for item in self.items),
        )

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {
            self.target,
            *(item for item in self.items if item is not None),
        }

    @cached_property
    def type(self) -> PrimitiveType:
        assert len(self.items) == self.target.type.single.rank
        return self.target.type.single_with_rank_delta(
            -sum(item is not None for item in self.items)
        )


@dataclass(frozen=True, eq=False)
class Slice(AbstractExpr):
    target: Expr
    stops: tuple[Optional[Expr], ...]

    def map(self, f: Callable[[Expr], Expr]) -> "Slice":
        return Slice(
            f(self.target),
            tuple(f(stop) if stop is not None else None for stop in self.stops),
        )

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {
            self.target,
            *(stop for stop in self.stops if stop is not None),
        }

    @cached_property
    def type(self) -> PrimitiveType:
        assert all(
            isinstance(stop, AbstractExpr) or stop is None for stop in self.stops
        )
        assert len(self.stops) == self.target.type.single.rank
        return self.target.type


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
    def type(self) -> PrimitiveType:
        assert (
            self.count.type.single.rank == 0
        ), "Can only repeat scalar number of times"
        assert 0 <= self.axis < self.target.type.single.rank, "Repeat axis not in range"
        return self.target.type


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
    def type(self) -> PrimitiveType:
        assert (
            0 <= self.axis <= self.target.type.single.rank
        ), "Mismatched reduction axis"
        return self.target.type.single_with_rank_delta(-1)


@dataclass(frozen=True, eq=False)
class Cast(AbstractExpr):
    dtype: ScalarKind
    target: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Cast":
        return Cast(self.dtype, f(self.target))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"dtype": self.dtype}, {self.target}

    @cached_property
    def type(self) -> PrimitiveType:
        return self.target.type.single_with_kind(self.dtype)


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
    def type(self) -> PrimitiveType:
        signature, constraints = UFUNC_SIGNATURES[ELEMENTWISE_KINDS[self.kind]]  # type: ignore
        rank = max(op.type.single.rank for op in self.operands)
        kind = resolve_scalar_signature(
            (Scalar(op.type.single.kind) for op in self.operands),
            signature,
            constraints,
        ).kind
        return PrimitiveType((PrimitiveArrayType(rank, kind),))


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
    inplace: int | None = None

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        base, deps = super().debug
        return (
            base | {"inplace": self.inplace} if self.inplace is not None else base,
            deps,
        )

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
    size: Expr
    acc: Variable
    init: Expr
    body: Expr

    def map(self, f: Callable[[Expr], Expr]) -> "Fold":
        return Fold(self.index, f(self.size), self.acc, f(self.init), f(self.body))

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc}, {self.init, self.size, self.body}

    @cached_property
    def type(self) -> PrimitiveType:
        assert self.size.type.single.rank == 0, "Expected scalar size"
        assert (
            self.init.type == self.body.type
        ), "Mismatched init and body accumulator types"
        return self.body.type


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
    def type(self) -> PrimitiveType:
        return PrimitiveType(tuple(op.type.single for op in self.operands))


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
    def type(self) -> PrimitiveType:
        assert len(self.target.type.elems) == self.arity
        return PrimitiveType((self.target.type.elems[self.at],))


REDUCE: dict[Any, Any] = {
    numpy.add: partial(Reduce, Reduce.Kind.sum),
    numpy.maximum: partial(Reduce, Reduce.Kind.max),
}

ELEMENTWISE_UFUNCS: dict[Any, Any] = {
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

ELEMENTWISE_KINDS = {
    Reduce.Kind.sum: numpy.sum,
    Reduce.Kind.max: numpy.max,
    UnaryElementwise.Kind.negative: numpy.negative,
    UnaryElementwise.Kind.reciprocal: numpy.reciprocal,
    UnaryElementwise.Kind.exp: numpy.exp,
    UnaryElementwise.Kind.sin: numpy.sin,
    UnaryElementwise.Kind.logical_not: numpy.logical_not,
    BinaryElementwise.Kind.add: numpy.add,
    BinaryElementwise.Kind.multiply: numpy.multiply,
    BinaryElementwise.Kind.mod: numpy.mod,
    BinaryElementwise.Kind.less: numpy.less,
    BinaryElementwise.Kind.logical_and: numpy.logical_and,
    TernaryElementwise.Kind.where: numpy.where,
}

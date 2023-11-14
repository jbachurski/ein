import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, ClassVar, TypeAlias, cast

import numpy
import numpy.typing

from ein.symbols import Index, Variable
from ein.type_system import (
    UFUNC_SIGNATURES,
    Pair,
    Scalar,
    Type,
    Vector,
    ndarray,
    resolve_scalar_signature,
    to_float,
)


class Value:
    value: numpy.ndarray | tuple["Value", "Value"]

    def __init__(
        self,
        value: "Value | tuple[Value, Value] | numpy.typing.ArrayLike",
    ):
        if isinstance(value, Value):
            self.value = value.value
        elif isinstance(value, tuple):
            first, second = value
            self.value = (Value(first), Value(second))
        else:
            self.value = numpy.array(value)

    def __eq__(self, other) -> bool:
        return isinstance(other, Value) and (
            self.value is other.value
            if isinstance(self.value, numpy.ndarray)
            else self.value == other.value
        )

    def __hash__(self) -> int:
        return (
            hash(id(self.value))
            if isinstance(self.value, numpy.ndarray)
            else hash(self.value)
        )

    def __repr__(self) -> str:
        return repr(self.value)

    @property
    def array(self) -> numpy.ndarray:
        if not isinstance(self.value, numpy.ndarray):
            raise TypeError(f"Value is not an array but one was expected: {self.value}")
        return self.value

    @property
    def pair(self) -> tuple["Value", "Value"]:
        if not isinstance(self.value, tuple):
            raise TypeError(f"Value is not a pair but one was expected: {self.value}")
        _, _ = self.value
        return cast(tuple[Value, Value], self.value)

    @property
    def type(self) -> Type:
        if isinstance(self.value, numpy.ndarray):
            return ndarray(self.array.ndim, Scalar.from_dtype(self.array.dtype).kind)
        elif isinstance(self.value, tuple):
            first, second = self.value
            return Pair(first.type, second.type)
        assert False


Expr: TypeAlias = (
    "Const | At | Var | Let | AssertEq | Dim | Get | Vec | Fold | "
    "Cons | First | Second |"
    "Negate | Reciprocal | Exp | Sin | LogicalNot | CastToFloat | "
    "Add | Multiply | Modulo | Less | LogicalAnd | Where"
)


def _merge_adj(*args: dict[Index, set["Expr"]]):
    result: dict[Index, set["Expr"]] = {}
    for arg in args:
        for k, v in arg.items():
            result.setdefault(k, set()).update(v)
    return result


@dataclass(frozen=True, eq=False)
class AbstractExpr(abc.ABC):
    def __post_init__(self):
        assert self.type

    @property
    @abc.abstractmethod
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        ...

    @property
    @abc.abstractmethod
    def type(self) -> Type:
        ...

    @property
    @abc.abstractmethod
    def dependencies(self) -> set[Expr]:
        ...

    @abc.abstractmethod
    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        ...

    @property
    def _indices(self) -> set[Index]:
        return set()

    @property
    def _variables(self) -> set[Variable]:
        return set()

    @property
    def _captured_indices(self) -> set[Index]:
        return set()

    @property
    def _captured_variables(self) -> set[Variable]:
        return set()

    @cached_property
    def free_indices(self) -> set[Index]:
        free_indices = set().union(*(sub.free_indices for sub in self.dependencies))
        return (self._indices | free_indices) - self._captured_indices

    @cached_property
    def free_variables(self) -> set[Variable]:
        free_variables = set().union(*(sub.free_variables for sub in self.dependencies))
        return (self._variables | free_variables) - self._captured_variables

    @cached_property
    def direct_indices(self) -> dict[Index, set[Expr]]:
        return _merge_adj(*(sub.direct_indices for sub in self.dependencies))


@dataclass(frozen=True)
class Const(AbstractExpr):
    value: Value

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"value": self.value.array}, set()

    @cached_property
    def type(self) -> Type:
        return self.value.type

    @property
    def dependencies(self) -> set[Expr]:
        return set()

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return self


@dataclass(frozen=True)
class At(AbstractExpr):
    index: Index

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, set()

    @cached_property
    def type(self) -> Type:
        return Scalar(int)

    @property
    def dependencies(self) -> set[Expr]:
        return set()

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return self

    @property
    def _indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True)
class Var(AbstractExpr):
    var: Variable
    var_type: Type

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @cached_property
    def type(self) -> Type:
        return self.var_type

    @property
    def dependencies(self) -> set[Expr]:
        return set()

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return self

    @property
    def _variables(self) -> set[Variable]:
        return {self.var}


@dataclass(frozen=True, eq=False)
class Let(AbstractExpr):
    bindings: tuple[tuple[Variable, Expr], ...]
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"vars": [var for var, _ in self.bindings]}, self.dependencies

    @cached_property
    def type(self) -> Type:
        return self.body.type

    @property
    def dependencies(self) -> set[Expr]:
        return {expr for _, expr in self.bindings} | {self.body}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Let(tuple((var, f(expr)) for var, expr in self.bindings), f(self.body))

    @property
    def _variables(self) -> set[Variable]:
        return {var for var, _ in self.bindings}


@dataclass(frozen=True, eq=False)
class AssertEq(AbstractExpr):
    target: Expr
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.target, *self.operands}

    @cached_property
    def type(self) -> Type:
        if any(op.type != self.operands[0].type for op in self.operands):
            raise TypeError(
                "Mismatched types in equality assertion: "
                f"{[op.type for op in self.operands]}"
            )
        return self.target.type

    @property
    def dependencies(self) -> set[Expr]:
        return {self.target, *self.operands}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return AssertEq(f(self.target), tuple(f(op) for op in self.operands))


@dataclass(frozen=True)
class Dim(AbstractExpr):
    operand: Expr
    axis: int

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.operand}

    @cached_property
    def type(self) -> Type:
        return Scalar(int)

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.operand}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Dim(f(self.operand), self.axis)


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    operand: Expr
    item: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.operand, self.item}

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.operand.type, Vector):
            raise TypeError(f"Cannot index a non-array type {self.operand.type}")
        if not isinstance(self.item.type, Scalar):
            raise TypeError(f"Cannot index with a non-scalar type {self.item.type}")
        if not isinstance(self.item.type, Scalar):
            raise TypeError(f"Cannot index with a non-integer type {self.item.type}")
        return self.operand.type.elem

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.operand, self.item}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Get(f(self.operand), f(self.item))

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
class Cons(AbstractExpr):
    first: Expr
    second: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.first, self.second}

    @cached_property
    def type(self) -> Type:
        return Pair(self.first.type, self.second.type)

    @property
    def dependencies(self) -> set[Expr]:
        return {self.first, self.second}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Cons(f(self.first), f(self.second))


@dataclass(frozen=True)
class AbstractDecons(AbstractExpr, abc.ABC):
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.target}

    def _unwrap_pair(self) -> Pair:
        if not isinstance(self.target.type, Pair):
            raise TypeError(f"Can only take First of a Pair, not {self.target.type}")
        return self.target.type

    @property
    def dependencies(self) -> set[Expr]:
        return {self.target}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return type(self)(f(self.target))  # type: ignore


@dataclass(frozen=True)
class First(AbstractDecons):
    @cached_property
    def type(self) -> Type:
        return self._unwrap_pair().first


@dataclass(frozen=True)
class Second(AbstractDecons):
    @cached_property
    def type(self) -> Type:
        return self._unwrap_pair().second


@dataclass(frozen=True, eq=False)
class Vec(AbstractExpr):
    index: Index
    size: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, {self.size, self.body}

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.size.type, Scalar):
            raise TypeError(f"Size must be a scalar, not {self.size.type}")
        if self.size.type.kind != int:
            raise TypeError(f"Size must be an integer, not {self.size.type}")
        return Vector(self.body.type)

    @property
    def dependencies(self) -> set[Expr]:
        return {self.size, self.body}

    @property
    def _captured_indices(self) -> set[Index]:
        return {self.index}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Vec(self.index, f(self.size), f(self.body))


@dataclass(frozen=True, eq=False)
class Fold(AbstractExpr):
    index: Index
    size: Expr
    acc: Var
    init: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc.var}, {
            self.size,
            self.init,
            self.body,
        }

    @property
    def dependencies(self) -> set[Expr]:
        return {self.size, self.init, self.body}

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.size.type, Scalar):
            raise TypeError(f"Size must be a scalar, not {self.size.type}")
        if self.size.type.kind != int:
            raise TypeError(f"Size must be an integer, not {self.size.type}")
        if self.init.type != self.acc.type:
            raise TypeError(
                f"Initial value and accumulator must be of the same type, got {self.init.type} != {self.acc.type}"
            )
        return self.acc.type

    @property
    def _captured_variables(self) -> set[Variable]:
        return {self.acc.var}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Fold(self.index, f(self.size), self.acc, f(self.init), f(self.body))


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    ufunc: ClassVar[numpy.ufunc]
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, set(self.operands)

    @cached_property
    def dependencies(self) -> set[Expr]:
        return set(self.operands)

    @cached_property
    def type(self) -> Type:
        signature, constraints = UFUNC_SIGNATURES[self.ufunc]
        return resolve_scalar_signature(
            [op.type for op in self.operands],
            signature,
            constraints,
            f"Operator {type(self).__name__} of ",
        )

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return type(self)(tuple(f(op) for op in self.operands))  # type: ignore


@dataclass(frozen=True, eq=False)
class AbstractUnaryScalarOperator(AbstractScalarOperator, abc.ABC):
    operands: tuple[Expr]


@dataclass(frozen=True, eq=False)
class Negate(AbstractUnaryScalarOperator):
    ufunc = numpy.negative


@dataclass(frozen=True, eq=False)
class Reciprocal(AbstractUnaryScalarOperator):
    ufunc = numpy.reciprocal


@dataclass(frozen=True, eq=False)
class Exp(AbstractUnaryScalarOperator):
    ufunc = numpy.exp


@dataclass(frozen=True, eq=False)
class Sin(AbstractUnaryScalarOperator):
    ufunc = numpy.sin


@dataclass(frozen=True, eq=False)
class LogicalNot(AbstractUnaryScalarOperator):
    ufunc = numpy.logical_not


@dataclass(frozen=True, eq=False)
class CastToFloat(AbstractUnaryScalarOperator):
    ufunc = staticmethod(to_float)  # type: ignore


@dataclass(frozen=True, eq=False)
class AbstractBinaryScalarOperator(AbstractScalarOperator, abc.ABC):
    operands: tuple[Expr, Expr]


@dataclass(frozen=True, eq=False)
class Add(AbstractBinaryScalarOperator):
    ufunc = numpy.add


@dataclass(frozen=True, eq=False)
class Multiply(AbstractBinaryScalarOperator):
    ufunc = numpy.multiply


@dataclass(frozen=True, eq=False)
class Modulo(AbstractBinaryScalarOperator):
    ufunc = numpy.mod


@dataclass(frozen=True, eq=False)
class Less(AbstractBinaryScalarOperator):
    ufunc = numpy.less


@dataclass(frozen=True, eq=False)
class LogicalAnd(AbstractBinaryScalarOperator):
    ufunc = numpy.logical_and


class AbstractTernaryScalarOperator(AbstractScalarOperator):
    operands: tuple[Expr, Expr, Expr]


class Where(AbstractTernaryScalarOperator):
    ufunc = staticmethod(numpy.where)  # type: ignore

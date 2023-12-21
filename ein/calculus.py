import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, ClassVar, TypeAlias, Union, cast

import numpy
import numpy.typing

from ein.symbols import Index, Symbol, Variable
from ein.term import Term
from ein.type_system import (
    UFUNC_SIGNATURES,
    Pair,
    Scalar,
    Type,
    Vector,
    ndarray,
    resolve_scalar_signature,
    scalar,
    to_float,
)

BIG_DATA_SIZE: int = 1024


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
            self.value = (
                numpy.array(value) if not isinstance(value, numpy.ndarray) else value
            )
            self.value.flags.writeable = False

    def __eq__(self, other) -> bool:
        if not isinstance(other, Value):
            return False
        if isinstance(self.value, numpy.ndarray):
            if not isinstance(other.value, numpy.ndarray):
                return False
            if len(self.value.data) != len(other.value.data):
                return False
            if self.value.dtype != other.value.dtype:
                return False
            if len(self.value.data) < BIG_DATA_SIZE:
                return self.value.data == other.value.data
            return self is other
        return self.value == other.value

    def __hash__(self) -> int:
        if isinstance(self.value, numpy.ndarray):
            if len(self.value.data) < BIG_DATA_SIZE:
                return hash((self.value.dtype, self.value.data.tobytes()))
            return hash(id(self.value))
        return hash(self.value)

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


Expr: TypeAlias = Union[
    "Const | Store | Let | AssertEq | Dim | Get | Vec | Fold | "
    "Cons | First | Second |"
    "Negate | Reciprocal | Exp | Sin | LogicalNot | CastToFloat | "
    "Add | Multiply | Modulo | Power | Less | LogicalAnd | LogicalOr | Where"
]


def _merge_adj(*args: dict[Index, set["Expr"]]):
    result: dict[Index, set["Expr"]] = {}
    for arg in args:
        for k, v in arg.items():
            result.setdefault(k, set()).update(v)
    return result


@dataclass(frozen=True, eq=False)
class AbstractExpr(Term):
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
    def subterms(self) -> tuple["Expr", ...]:
        ...

    @property
    def captured_symbols(self) -> set[Symbol]:
        return set()

    def wrap_let(self, var: Variable, bind: Term) -> Expr:
        return Let(var, cast(Expr, bind), cast(Expr, self))

    def wrap_var(self, var: Variable) -> Expr:
        return Store(var, self.type)

    def unwrap_let(self) -> tuple[Variable, "Term", "Term"] | None:
        return None

    def unwrap_symbol(self) -> Symbol | None:
        return None

    @property
    def is_atom(self) -> bool:
        return False

    @property
    def is_loop(self) -> bool:
        return False


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"value": self.value.array}, set()

    @cached_property
    def type(self) -> Type:
        return self.value.type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return ()

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return self


@dataclass(frozen=True, eq=False)
class Store(AbstractExpr):
    symbol: Symbol
    inner_type: Type

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"symbol": self.symbol}, set()

    @cached_property
    def type(self) -> Type:
        return self.inner_type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return ()

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return self

    def unwrap_symbol(self) -> Symbol | None:
        return self.symbol

    @property
    def is_atom(self) -> bool:
        return True

    @property
    def var(self) -> Variable:
        assert isinstance(self.symbol, Variable)
        return self.symbol

    @property
    def index(self) -> Index:
        assert isinstance(self.symbol, Index)
        return self.symbol


@dataclass(frozen=True, eq=False)
class Let(AbstractExpr):
    var: Variable
    bind: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set(self.subterms)

    @cached_property
    def type(self) -> Type:
        return self.body.type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.bind, self.body

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Let(self.var, f(self.bind), f(self.body))

    @property
    def captured_symbols(self) -> set[Symbol]:
        return {self.var}

    def unwrap_let(self) -> tuple[Variable, "Term", "Term"] | None:
        return self.var, self.bind, self.body


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
    def subterms(self) -> tuple[Expr, ...]:
        return self.target, *self.operands

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return AssertEq(f(self.target), tuple(f(op) for op in self.operands))


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    operand: Expr
    axis: int

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.operand}

    @cached_property
    def type(self) -> Type:
        return scalar(int)

    @cached_property
    def subterms(self) -> tuple[Expr, ...]:
        return (self.operand,)

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
    def subterms(self) -> tuple[Expr, ...]:
        return self.operand, self.item

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Get(f(self.operand), f(self.item))


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
    def subterms(self) -> tuple[Expr, ...]:
        return self.first, self.second

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Cons(f(self.first), f(self.second))


@dataclass(frozen=True, eq=False)
class AbstractDecons(AbstractExpr, abc.ABC):
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.target}

    def _unwrap_pair(self) -> Pair:
        if not isinstance(self.target.type, Pair):
            raise TypeError(f"Can only project Pairs, not {self.target.type}")
        return self.target.type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return (self.target,)

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return type(self)(f(self.target))  # type: ignore


@dataclass(frozen=True, eq=False)
class First(AbstractDecons):
    @cached_property
    def type(self) -> Type:
        return self._unwrap_pair().first


@dataclass(frozen=True, eq=False)
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
        free_indices = {
            index for index in self.size.free_symbols if isinstance(index, Index)
        }
        if free_indices:
            raise TypeError(
                f"Size cannot depend on comprehension indices: {free_indices}"
            )
        if self.size.type.kind != int:
            raise TypeError(f"Size must be an integer, not {self.size.type}")
        return Vector(self.body.type)

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.size, self.body

    @property
    def captured_symbols(self) -> set[Symbol]:
        return {self.index}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Vec(self.index, f(self.size), f(self.body))

    @property
    def is_loop(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class Fold(AbstractExpr):
    index: Index
    size: Expr
    acc: Variable
    init: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc}, {
            self.size,
            self.init,
            self.body,
        }

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.size, self.init, self.body

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.size.type, Scalar):
            raise TypeError(f"Size must be a scalar, not {self.size.type}")
        if self.size.type.kind != int:
            raise TypeError(f"Size must be an integer, not {self.size.type}")
        if self.init.type != self.body.type:
            raise TypeError(
                f"Initial value and body must be of the same type, got {self.init.type} != {self.body.type}"
            )
        return self.init.type

    @property
    def captured_symbols(self) -> set[Symbol]:
        return {self.index, self.acc}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Fold(self.index, f(self.size), self.acc, f(self.init), f(self.body))

    @property
    def is_loop(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    ufunc: ClassVar[numpy.ufunc]
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, set(self.operands)

    @cached_property
    def subterms(self) -> tuple[Expr, ...]:
        return self.operands

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
class Power(AbstractBinaryScalarOperator):
    ufunc = numpy.power


@dataclass(frozen=True, eq=False)
class Less(AbstractBinaryScalarOperator):
    ufunc = numpy.less


@dataclass(frozen=True, eq=False)
class LogicalAnd(AbstractBinaryScalarOperator):
    ufunc = numpy.logical_and


@dataclass(frozen=True, eq=False)
class LogicalOr(AbstractBinaryScalarOperator):
    ufunc = numpy.logical_or


@dataclass(frozen=True, eq=False)
class AbstractTernaryScalarOperator(AbstractScalarOperator):
    operands: tuple[Expr, Expr, Expr]


@dataclass(frozen=True, eq=False)
class Where(AbstractTernaryScalarOperator):
    ufunc = staticmethod(numpy.where)  # type: ignore


def variable(var: Variable, type_: Type) -> Store:
    return Store(var, type_)


def at(index: Index) -> Store:
    return Store(index, scalar(int))

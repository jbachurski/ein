import abc
import functools
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, ClassVar, TypeAlias, Union, cast

import numpy.typing

from ein.phi.type_system import (
    UFUNC_SIGNATURES,
    Pair,
    Scalar,
    Type,
    Vector,
    resolve_scalar_signature,
    scalar_type,
    to_float,
)
from ein.symbols import Index, Symbol, Variable
from ein.term import Term
from ein.value import Value

Expr: TypeAlias = Union[
    "Vec | Fold | Reduce |"
    "Dim | Get | Concat |"
    "Const | Store | Let | AssertEq |"
    "Cons | First | Second |"
    "Negate | Reciprocal | Exp | Sin | Cos | LogicalNot | CastToFloat | "
    "Add | Subtract | Multiply | Modulo | Power | Min | Max | "
    "Less | LessEqual | Equal | NotEqual | LogicalAnd | LogicalOr | Where |"
    "Extrinsic"
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
        return {"value": repr(self.value)}, set()

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
    target: Expr
    axis: int

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"axis": self.axis}, {self.target}

    @cached_property
    def type(self) -> Type:
        return scalar_type(int)

    @cached_property
    def subterms(self) -> tuple[Expr, ...]:
        return (self.target,)

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Dim(f(self.target), self.axis)


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
        if self.size.free_indices:
            raise TypeError(
                f"Size cannot depend on comprehension indices: {self.size.free_indices}"
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
    counter: Variable
    size: Expr
    acc: Variable
    init: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.counter, "acc": self.acc}, {
            self.size,
            self.init,
            self.body,
        }

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.size, self.init, self.body

    @cached_property
    def type(self) -> Type:
        assert isinstance(self.counter, Variable)
        assert isinstance(self.acc, Variable)
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
        return {self.counter, self.acc}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Fold(self.counter, f(self.size), self.acc, f(self.init), f(self.body))

    @property
    def is_loop(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class Reduce(AbstractExpr):
    init: Expr
    x: Variable
    y: Variable
    xy: Expr
    vecs: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"xs": self.x, "ys": self.y}, set(self.subterms)

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.init, *self.vecs, self.xy

    @cached_property
    def type(self) -> Type:
        if self.init.type != self.xy.type:
            raise TypeError(
                f"Reduction identity type mismatched: id {self.init.type} != {self.xy.type}."
            )
        for vec in self.vecs:
            if not isinstance(vec.type, Vector):
                raise TypeError(f"Must reduce over a vector, not {vec.type}.")
        elems_type = functools.reduce(
            Pair, (cast(Vector, vec.type).elem for vec in self.vecs)
        )
        if elems_type != self.xy.type:
            raise TypeError(
                f"Reduction must preserve type, but {self.xy.type} != elements {elems_type}."
            )
        return self.xy.type

    @property
    def captured_symbols(self) -> set[Symbol]:
        return {self.x, self.y}

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Reduce(
            f(self.init), self.x, self.y, f(self.xy), tuple(f(vec) for vec in self.vecs)
        )

    @property
    def is_loop(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    operands: tuple[Expr, ...]
    ufunc: ClassVar[Callable]

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
class Cos(AbstractUnaryScalarOperator):
    ufunc = numpy.cos


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
class Subtract(AbstractBinaryScalarOperator):
    ufunc = numpy.subtract


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
class Min(AbstractBinaryScalarOperator):
    ufunc = numpy.minimum


@dataclass(frozen=True, eq=False)
class Max(AbstractBinaryScalarOperator):
    ufunc = numpy.maximum


@dataclass(frozen=True, eq=False)
class Less(AbstractBinaryScalarOperator):
    ufunc = numpy.less


@dataclass(frozen=True, eq=False)
class LessEqual(AbstractBinaryScalarOperator):
    ufunc = numpy.less_equal


@dataclass(frozen=True, eq=False)
class Equal(AbstractBinaryScalarOperator):
    ufunc = numpy.equal


@dataclass(frozen=True, eq=False)
class NotEqual(AbstractBinaryScalarOperator):
    ufunc = numpy.not_equal


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


@dataclass(frozen=True, eq=False)
class Concat(AbstractExpr):
    first: Expr
    second: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.first, self.second}

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.first.type, Vector):
            raise TypeError(
                f"Concatenated values must be vectors, not {self.first.type}"
            )
        if self.first.type != self.second.type:
            raise TypeError(
                f"Concatenated values must be of the same (vector) type, got {self.first.type} != {self.second.type}"
            )
        return self.first.type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.first, self.second

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Concat(f(self.first), f(self.second))


@dataclass(frozen=True, eq=False)
class Extrinsic(AbstractExpr):
    _type: Type
    fun: Callable
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"fun": self.fun.__name__}, {*self.operands}

    @cached_property
    def type(self) -> Type:
        return self._type

    @property
    def subterms(self) -> tuple[Expr, ...]:
        return self.operands

    def map(self, f: Callable[[Expr], Expr]) -> Expr:
        return Extrinsic(self._type, self.fun, tuple(f(op) for op in self.operands))


def variable(var: Variable, type_: Type) -> Store:
    return Store(var, type_)


def at(index: Index) -> Store:
    return Store(index, scalar_type(int))

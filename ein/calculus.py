import abc
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, TypeAlias

import numpy

from ein.symbols import Index, Variable
from ein.type_system import Scalar, Type, Vector
from ein.type_system import ndarray as ndarray_type


@dataclass(frozen=True, eq=False)
class Value:
    array: numpy.ndarray

    @property
    def type(self) -> Type:
        return ndarray_type(self.array.ndim)


Expr: TypeAlias = (
    "Const | At | Var | Dim | Get | Vec | Fold | Sum | Maximum | "
    "Negate | Reciprocal | Exp | LogicalNot | Add | Multiply | Less | LogicalAnd | Where"
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
    def type(self) -> Type:
        ...

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
        return indices - self._captured_indices

    @cached_property
    def direct_indices(self) -> dict[Index, set[Expr]]:
        return _merge_adj(*(sub.direct_indices for sub in self.dependencies))


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    value: Value

    @cached_property
    def type(self) -> Type:
        return self.value.type

    @property
    def dependencies(self) -> set[Expr]:
        return set()


@dataclass(frozen=True, eq=False)
class At(AbstractExpr):
    index: Index

    @cached_property
    def type(self) -> Type:
        return Scalar()

    @property
    def dependencies(self) -> set[Expr]:
        return set()

    @property
    def _indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable
    var_type: Type

    @cached_property
    def type(self) -> Type:
        return self.var_type

    @property
    def dependencies(self) -> set[Expr]:
        return set()


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    operand: Expr
    axis: int

    @cached_property
    def type(self) -> Type:
        return Scalar()

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.operand}


@dataclass(frozen=True, eq=False)
class Get(AbstractExpr):
    operand: Expr
    item: Expr

    @cached_property
    def type(self) -> Type:
        if not isinstance(self.operand.type, Vector):
            raise TypeError(f"Cannot index a non-array type {self.operand.type}")
        if not isinstance(self.item.type, Scalar):
            raise TypeError(f"Cannot index with a non-scalar type {self.item.type}")
        return self.operand.type.elem

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
class AbstractVectorization(AbstractExpr):
    index: Index
    size: Expr
    body: Expr

    def _validate_size(self) -> None:
        if not isinstance(self.size.type, Scalar):
            raise TypeError(f"Size must be a scalar, not {self.size.type}")

    @cached_property
    def dependencies(self) -> set[Expr]:
        return {self.size, self.body}

    @property
    def _captured_indices(self) -> set[Index]:
        return {self.index}


@dataclass(frozen=True, eq=False)
class Vec(AbstractVectorization):
    @cached_property
    def type(self) -> Type:
        return Vector(self.body.type)


@dataclass(frozen=True, eq=False)
class Fold(AbstractVectorization):
    init: Expr
    acc: Var

    @cached_property
    def type(self) -> Type:
        self._validate_size()
        if self.init.type != self.acc.type:
            raise TypeError(
                f"Initial value and accumulator must be of the same type, got {self.init.type} != {self.acc.type}"
            )
        return self.acc.type


@dataclass(frozen=True, eq=False)
class AbstractScalarReduction(AbstractVectorization):
    ufunc: ClassVar[numpy.ufunc]

    @cached_property
    def type(self) -> Type:
        self._validate_size()
        if not isinstance(self.body.type, Scalar):
            raise TypeError(f"Can only reduce over scalar types, not {self.body.type}")
        return self.body.type


@dataclass(frozen=True, eq=False)
class Sum(AbstractScalarReduction):
    ufunc = numpy.add


@dataclass(frozen=True, eq=False)
class Maximum(AbstractScalarReduction):
    ufunc = numpy.maximum


@dataclass(frozen=True, eq=False)
class AbstractScalarOperator(AbstractExpr, abc.ABC):
    operands: tuple[Expr, ...]
    ufunc: ClassVar[numpy.ufunc]
    signature: ClassVar[tuple[tuple[numpy.dtype | str, ...], numpy.dtype | str]]

    @cached_property
    def dependencies(self) -> set[Expr]:
        return set(self.operands)

    @cached_property
    def type(self) -> Type:
        # TODO: This should also do some dtype checks.
        types = [operand.type for operand in self.operands]
        if not all(isinstance(typ, Scalar) for typ in types):
            raise TypeError(f"Scalar operator expected only scalars, got {types}")
        return Scalar()


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
class LogicalNot(AbstractUnaryScalarOperator):
    ufunc = numpy.logical_not


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
class Less(AbstractBinaryScalarOperator):
    ufunc = numpy.less


@dataclass(frozen=True, eq=False)
class LogicalAnd(AbstractBinaryScalarOperator):
    ufunc = numpy.logical_and


class AbstractTernaryScalarOperator(AbstractScalarOperator):
    operands: tuple[Expr, Expr, Expr]


class Where(AbstractTernaryScalarOperator):
    ufunc = staticmethod(numpy.where)  # type: ignore

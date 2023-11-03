import abc
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeAlias

import numpy

from ein.symbols import Index, Variable
from ein.type_system import (
    UFUNC_SIGNATURES,
    PrimitiveArrayType,
    Scalar,
    resolve_scalar_signature,
)

from .axial import AxialType

Expr: TypeAlias = (
    "Const | Range | Var | At | Dim | Fold | Gather | Vector | Reduce | Elementwise"
)


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
    def type(self) -> AxialType:
        ...


@dataclass(frozen=True, eq=False)
class Const(AbstractExpr):
    array: numpy.ndarray

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"array": self.array}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(
            PrimitiveArrayType(
                self.array.ndim, Scalar.from_dtype(self.array.dtype).kind
            ),
            set(),
        )


@dataclass(frozen=True, eq=False)
class Range(AbstractExpr):
    index: Index
    size: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, {self.size}

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert self.size.type.array_type.kind, "Expected integer size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        return AxialType(PrimitiveArrayType(rank=0, kind=int), {self.index})


@dataclass(frozen=True, eq=False)
class Var(AbstractExpr):
    var: Variable
    var_type: PrimitiveArrayType

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"var": self.var}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(self.var_type, set())


@dataclass(frozen=True, eq=False)
class At(AbstractExpr):
    index: Index

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, set()

    @property
    def type(self) -> AxialType:
        return AxialType(PrimitiveArrayType(rank=0, kind=int), set())


@dataclass(frozen=True, eq=False)
class Dim(AbstractExpr):
    pos: int
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"pos": self.pos}, {self.target}

    @cached_property
    def type(self) -> AxialType:
        assert 0 <= self.pos < self.target.type.array_type.rank
        assert not self.target.type.free_indices
        return AxialType(PrimitiveArrayType(rank=0, kind=int), set())


@dataclass(frozen=True, eq=False)
class Fold(AbstractExpr):
    index: Index
    acc: Var
    init: Expr
    size: Expr
    body: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index, "acc": self.acc}, {self.init, self.size, self.body}

    @cached_property
    def type(self) -> AxialType:
        assert self.size.type.array_type == PrimitiveArrayType(
            rank=0, kind=int
        ), "Expected scalar fold size"
        # TODO: This, similarly to loop-dependent arrays and reductions, is possible to implement.
        #  But it does not have any benefits from vectorisation, so maybe it doesn't have to be.
        assert not self.size.type.free_indices, "Expected loop-independent fold size"
        assert (
            self.acc.var_type == self.init.type.array_type
        ), "Mismatched accumulator and initialiser type"
        # FIXME: This should work?
        assert not self.body.type.free_indices, "Expected outer-loop-independent body"
        return self.body.type


@dataclass(frozen=True, eq=False)
class Gather(AbstractExpr):
    target: Expr
    item: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {}, {self.target, self.item}

    @cached_property
    def type(self) -> AxialType:
        assert self.target.type.array_type.rank > 0, "Expected vector target"
        assert self.item.type.array_type.rank == 0, "Expected scalar index"
        assert self.item.type.array_type.kind == int, "Expected integer index"
        return AxialType(
            self.target.type.array_type.item,
            {
                index
                for index in self.target.type.free_indices | self.item.type.free_indices
            },
        )


@dataclass(frozen=True, eq=False)
class Vector(AbstractExpr):
    index: Index
    size: Expr
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"index": self.index}, {self.size, self.target}

    @cached_property
    def type(self) -> AxialType:
        assert not self.size.type.array_type.rank, "Expected scalar size"
        assert not self.size.type.free_indices, "Expected loop-independent size"
        assert self.size.type.array_type.kind == int, "Expected integer size"
        return AxialType(
            self.target.type.array_type.in_vector,
            self.target.type.free_indices - {self.index},
        )


@dataclass(frozen=True, eq=False)
class Reduce(AbstractExpr):
    ufunc: numpy.ufunc
    index: Index
    target: Expr

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"ufunc": numpy.ufunc, "index": self.index}, {self.target}

    @cached_property
    def type(self, *args: AxialType) -> AxialType:
        assert not self.target.type.array_type.rank, "Expected scalar reduction"
        assert (
            self.index in self.target.type.free_indices
        ), "Can only reduce over free index"
        return AxialType(
            PrimitiveArrayType(rank=0, kind=self.target.type.array_type.kind),
            self.target.type.free_indices - {self.index},
        )


@dataclass(frozen=True, eq=False)
class Elementwise(AbstractExpr):
    ufunc: numpy.ufunc
    operands: tuple[Expr, ...]

    @property
    def debug(self) -> tuple[dict[str, Any], set[Expr]]:
        return {"ufunc": self.ufunc}, set(self.operands)

    @cached_property
    def type(self) -> AxialType:
        assert all(
            not op.type.array_type.rank for op in self.operands
        ), "Expected scalar elementwise"
        signature, constraints = UFUNC_SIGNATURES[self.ufunc]
        return AxialType(
            PrimitiveArrayType(
                rank=0,
                kind=resolve_scalar_signature(
                    [op.type.array_type.type for op in self.operands],
                    signature,
                    constraints,
                    f"Elementwise {self.ufunc} of ",
                ).kind,
            ),
            {index for op in self.operands for index in op.type.free_indices},
        )

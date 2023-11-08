import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, TypeAlias

import numpy

Type: TypeAlias = "Scalar | Vector | Pair"
ScalarKind: TypeAlias = type[bool] | type[int] | type[float]
SCALAR_TYPES: tuple[ScalarKind, ...] = (bool, int, float)


@dataclass(frozen=True)
class AbstractType(abc.ABC):
    @property
    @abc.abstractmethod
    def primitive_type(self) -> "PrimitiveType":
        ...


@dataclass(frozen=True)
class Scalar(AbstractType):
    kind: ScalarKind

    def __post_init__(self):
        assert self.kind in SCALAR_TYPES

    @staticmethod
    def from_dtype(dtype: numpy.dtype) -> "Scalar":
        t = dtype.type
        if issubclass(t, numpy.bool_):
            return Scalar(bool)
        elif issubclass(t, numpy.integer):
            return Scalar(int)
        elif issubclass(t, numpy.floating):
            return Scalar(float)
        raise TypeError(f"Datatype {dtype} cannot be converted to a legal Scalar type.")

    @property
    def pretty(self) -> str:
        return self.kind.__name__

    @property
    def primitive_type(self) -> "PrimitiveType":
        return PrimitiveType((PrimitiveArrayType(rank=0, kind=self.kind),))


@dataclass(frozen=True)
class Vector(AbstractType):
    elem: Type

    def __post_init__(self):
        assert isinstance(self.elem, AbstractType)

    @property
    def pretty(self) -> str:
        return f"[]{self.elem.pretty}"

    @property
    def primitive_type(self) -> "PrimitiveType":
        return PrimitiveType((self.elem.primitive_type.single.in_vector,))


@dataclass(frozen=True)
class Pair(AbstractType):
    first: Type
    second: Type

    def __post_init__(self):
        assert isinstance(self.first, AbstractType)
        assert isinstance(self.second, AbstractType)

    @property
    def pretty(self) -> str:
        return f"({self.first.pretty}, {self.second.pretty})"

    @property
    def primitive_type(self) -> "PrimitiveType":
        return PrimitiveType(
            (*self.first.primitive_type.elems, *self.second.primitive_type.elems)
        )


def vector(type_: ScalarKind) -> Vector:
    return Vector(Scalar(type_))


def matrix(type_: ScalarKind) -> Vector:
    return Vector(Vector(Scalar(type_)))


def ndarray(rank: int, type_: ScalarKind) -> Type:
    return Vector(ndarray(rank - 1, type_)) if rank else Scalar(type_)


@dataclass(frozen=True)
class PrimitiveArrayType:
    rank: int
    kind: ScalarKind

    def __post_init__(self):
        assert self.rank >= 0
        assert self.kind in SCALAR_TYPES

    @property
    def pretty(self) -> str:
        return self.type.pretty

    @property
    def in_vector(self) -> "PrimitiveArrayType":
        return PrimitiveArrayType(self.rank + 1, self.kind)

    @property
    def item(self) -> "PrimitiveArrayType":
        assert self.rank, "Expected non-scalar array to index"
        return PrimitiveArrayType(self.rank - 1, self.kind)

    @property
    def type(self) -> Type:
        return (
            Vector(PrimitiveArrayType(self.rank - 1, self.kind).type)
            if self.rank
            else Scalar(self.kind)
        )


@dataclass(frozen=True)
class PrimitiveType:
    elems: tuple[PrimitiveArrayType, ...]

    def __post_init__(self):
        assert self.elems, "No unit types in the language"

    @property
    def pretty(self) -> str:
        return "(" + ", ".join(elem.pretty for elem in self.elems) + ")"

    @property
    def single(self) -> PrimitiveArrayType:
        (elem,) = self.elems
        return elem

    @property
    def type(self) -> Type:
        match self.elems:
            case (elem,):
                return elem.type
            case (elem, *elems):
                return Pair(elem.type, PrimitiveType(tuple(elems)).type)
        assert False, "Unit is not a primitive type"


def resolve_scalar_signature(
    types: Iterable[Type],
    signature: tuple[tuple[str | ScalarKind, ...], str | ScalarKind],
    constraints: dict[str, set[ScalarKind]],
    _prefix_fail_msg: str = "",
) -> Scalar:
    signature_args, signature_result = signature
    type_vars: dict[str, ScalarKind] = {}
    _fail_msg = f"{_prefix_fail_msg}signature {signature_args}->{signature_result} ({constraints})"
    for i, (typ, sig) in enumerate(zip(types, signature_args)):
        if not isinstance(typ, Scalar):
            raise TypeError(
                f"{_fail_msg} expected only scalars, got {typ} at position {i}"
            )
        if isinstance(sig, str):
            if sig in constraints and typ.kind not in constraints[sig]:
                raise TypeError(
                    f"{_fail_msg} expected one of {constraints[sig]} for {sig} at position {i}, got {typ}"
                )
            if sig in type_vars and type_vars[sig] != typ.kind:
                raise TypeError(
                    f"{_fail_msg} found conflicting assignments for {sig} at position {i}, got {typ.kind} != {type_vars[sig]}"
                )
            type_vars[sig] = typ.kind
        elif typ.kind != sig:
            raise TypeError(f"{_fail_msg} expected {sig} at position {i}, got {typ}")
    return Scalar(
        type_vars[signature_result]
        if isinstance(signature_result, str)
        else signature_result
    )


number = {int, float}
to_float = partial(numpy.ndarray.astype, dtype=float)  # type: ignore
UFUNC_SIGNATURES: dict[
    Any,
    tuple[
        tuple[tuple[str | ScalarKind, ...], str | ScalarKind],
        dict[str, set[ScalarKind]],
    ],
] = {
    numpy.negative: ((("T",), "T"), {"T": number}),
    numpy.reciprocal: (((float,), float), {}),
    numpy.exp: (((float,), float), {}),
    numpy.sin: (((float,), float), {}),
    to_float: (((int,), float), {}),
    numpy.logical_not: (((bool,), bool), {}),
    numpy.add: ((("T", "T"), "T"), {"T": number}),
    numpy.multiply: ((("T", "T"), "T"), {"T": number}),
    numpy.mod: (((int, int), int), {}),
    numpy.less: ((("T", "T"), bool), {"T": number}),
    numpy.logical_and: (((bool, bool), bool), {}),
    numpy.where: (((bool, "T", "T"), "T"), {}),
}

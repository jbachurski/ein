import abc
from dataclasses import dataclass
from typing import TypeAlias

Type: TypeAlias = "Scalar | Vector | Pair"


@dataclass(frozen=True)
class AbstractType(abc.ABC):
    @property
    @abc.abstractmethod
    def primitive_type(self) -> "PrimitiveType":
        ...


@dataclass(frozen=True)
class Scalar(AbstractType):
    @property
    def pretty(self) -> str:
        return "*"

    @property
    def primitive_type(self) -> "PrimitiveType":
        return PrimitiveType((PrimitiveArrayType(rank=0),))


@dataclass(frozen=True)
class Vector(AbstractType):
    elem: Type

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

    @property
    def pretty(self) -> str:
        return f"({self.first.pretty}, {self.second.pretty})"

    @property
    def primitive_type(self) -> "PrimitiveType":
        return PrimitiveType(
            (*self.first.primitive_type.elems, *self.second.primitive_type.elems)
        )


def vector() -> Vector:
    return Vector(Scalar())


def matrix() -> Vector:
    return Vector(Vector(Scalar()))


def ndarray(rank: int) -> Type:
    return Vector(ndarray(rank - 1)) if rank else Scalar()


@dataclass(frozen=True)
class PrimitiveArrayType:
    rank: int

    def __post_init__(self):
        assert self.rank >= 0

    @property
    def pretty(self) -> str:
        return "[]" * self.rank + "*"

    @property
    def in_vector(self) -> "PrimitiveArrayType":
        return PrimitiveArrayType(self.rank + 1)

    @property
    def item(self) -> "PrimitiveArrayType":
        assert self.rank, "Expected non-scalar array to index"
        return PrimitiveArrayType(self.rank - 1)

    @property
    def type(self) -> Type:
        return Vector(PrimitiveArrayType(self.rank - 1).type) if self.rank else Scalar()


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

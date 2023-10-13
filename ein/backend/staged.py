from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar("S")
T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class Staged(Generic[S, T]):
    operation: S
    type: T
    operands: tuple["Staged[S, T]", ...]

import abc
import dataclasses
from functools import cached_property
from typing import Any, Callable, TypeVar

from ein.symbols import Index, Symbol, Variable

T = TypeVar("T")


def _maybe_set(x: T | None) -> set[T]:
    return {x} if x is not None else set()


@dataclasses.dataclass(frozen=True, eq=False)
class Term(abc.ABC):
    @property
    def _fields(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
        )

    @cached_property
    def hash(self) -> int:
        return hash((type(self), self._fields))

    def __eq__(self, other) -> bool:
        if type(self) != type(other):
            return False
        if self.hash != other.hash:
            return False
        if self is other:
            return True
        return self._fields == other._fields

    def __hash__(self) -> int:
        return self.hash

    @property
    @abc.abstractmethod
    def subterms(self) -> tuple["Term", ...]:
        ...

    @abc.abstractmethod
    def map(self, f: Callable) -> "Term":
        ...

    @abc.abstractmethod
    def wrap_let(self, var: Variable, bind: "Term") -> "Term":
        ...

    @abc.abstractmethod
    def wrap_var(self, var: Variable) -> "Term":
        ...

    @abc.abstractmethod
    def unwrap_let(self) -> tuple[Variable, "Term", "Term"] | None:
        ...

    @abc.abstractmethod
    def unwrap_symbol(self) -> Symbol | None:
        ...

    @property
    @abc.abstractmethod
    def is_atom(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def is_loop(self) -> bool:
        ...

    @property
    @abc.abstractmethod
    def captured_symbols(self) -> set[Symbol]:
        ...

    @cached_property
    def free_symbols(self) -> set[Symbol]:
        free = set().union(*(sub.free_symbols for sub in self.subterms))
        return (_maybe_set(self.unwrap_symbol()) | free) - self.captured_symbols

    @cached_property
    def free_indices(self) -> set[Index]:
        return {index for index in self.free_symbols if isinstance(index, Index)}

    @cached_property
    def free_variables(self) -> set[Variable]:
        return {index for index in self.free_symbols if isinstance(index, Variable)}

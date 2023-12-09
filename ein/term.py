import abc
import dataclasses
from functools import cached_property
from typing import Any, Callable, TypeVar

from ein.symbols import Index, Variable

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
    def unwrap_var(self) -> Variable | None:
        ...

    @abc.abstractmethod
    def unwrap_index(self) -> Index | None:
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
    def captured_indices(self) -> set[Index]:
        ...

    @property
    @abc.abstractmethod
    def captured_variables(self) -> set[Variable]:
        ...

    @cached_property
    def free_indices(self) -> set[Index]:
        free = set().union(*(sub.free_indices for sub in self.subterms))
        return (_maybe_set(self.unwrap_index()) | free) - self.captured_indices

    @cached_property
    def free_variables(self) -> set[Variable]:
        free = set().union(*(sub.free_variables for sub in self.subterms))
        return (_maybe_set(self.unwrap_var()) | free) - self.captured_variables

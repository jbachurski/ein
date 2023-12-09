import abc
from typing import Callable

from ein.symbols import Index, Variable


class Term(abc.ABC):
    @abc.abstractmethod
    def __hash__(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def dependencies(self) -> tuple["Term", ...]:
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

    @property
    @abc.abstractmethod
    def free_indices(self) -> set[Index]:
        ...

    @property
    @abc.abstractmethod
    def free_variables(self) -> set[Variable]:
        ...

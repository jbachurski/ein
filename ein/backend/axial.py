import abc
from typing import Any

from ein.calculus import Index


class Axial:
    @abc.abstractmethod
    def vector(self, axis: Index) -> "Axial":
        ...


Const: Any = None
Range: Any = None
Var: Any = None
Dim: Any = None
Gather: Any = None
vector: Any = None
Reduce: Any = None
Elementwise: Any = None

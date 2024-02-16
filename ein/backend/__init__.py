from typing import Callable, Literal, TypeAlias

import numpy

from ein.backend import naive, to_numpy, to_torch
from ein.calculus import Expr
from ein.symbols import Variable

Backend: TypeAlias = Literal["naive", "numpy", "to_torch"]
BACKENDS: dict[str, Callable[[Expr, dict[Variable, numpy.ndarray]], numpy.ndarray]] = {
    "naive": naive.interpret,
    "numpy": to_numpy.interpret,
    "torch": to_torch.interpret,
}

DEFAULT_BACKEND: Backend = "numpy"

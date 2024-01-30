from typing import Callable, Literal, TypeAlias

import numpy

from ein.calculus import Expr
from ein.symbols import Variable

from . import naive, to_array, to_numpy, to_torch  # noqa

Backend: TypeAlias = Literal["naive", "numpy"]
BACKENDS: dict[
    Backend,
    Callable[[Expr, dict[Variable, numpy.ndarray]], numpy.ndarray],
] = {
    "naive": naive.interpret,
    "numpy": to_numpy.interpret,
}

DEFAULT_BACKEND: Backend = "numpy"

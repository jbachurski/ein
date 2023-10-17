from .backend.arrays import interpret as interpret_with_arrays
from .backend.naive import interpret as interpret_with_naive
from .frontend.comprehension import Tensor, Type, array, function, max, min, sum

__all__ = [
    "Tensor",
    "array",
    "sum",
    "max",
    "min",
    "function",
    "Type",
    "interpret_with_naive",
    "interpret_with_arrays",
]

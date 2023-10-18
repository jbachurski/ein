from .backend.arrays import interpret as interpret_with_arrays
from .backend.naive import interpret as interpret_with_naive
from .frontend.comprehension import Array, Type, array, function, max, min, sum

__all__ = [
    "Array",
    "array",
    "sum",
    "max",
    "min",
    "function",
    "Type",
    "interpret_with_naive",
    "interpret_with_arrays",
]

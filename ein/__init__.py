from .backend.arrays import interpret as interpret_with_arrays
from .backend.naive import interpret as interpret_with_naive
from .frontend.comprehension import Array, array, fold, function, max, min, sum
from .type_system import Scalar, Type, Vector, matrix, ndarray, vector

__all__ = [
    "Array",
    "array",
    "sum",
    "max",
    "min",
    "fold",
    "function",
    "Type",
    "Scalar",
    "Vector",
    "vector",
    "matrix",
    "ndarray",
    "interpret_with_naive",
    "interpret_with_arrays",
]

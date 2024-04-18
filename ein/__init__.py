from ein.phi.type_system import matrix_type, ndarray_type, scalar_type, vector_type

from .backend.naive import interpret as interpret_with_naive
from .backend.numpy_backend import interpret as interpret_with_numpy
from .backend.torch_backend import interpret as interpret_with_torch
from .frontend.comprehension import array, fold
from .frontend.functions import function, with_varargs
from .frontend.ndarray import Array, Scalar, Vec, ext, wrap

Bool = Int = Float = Scalar

__all__ = [
    "array",
    "fold",
    "ext",
    "wrap",
    "function",
    "with_varargs",
    "Array",
    "Scalar",
    "Bool",
    "Int",
    "Float",
    "Vec",
    "scalar_type",
    "vector_type",
    "matrix_type",
    "ndarray_type",
    "interpret_with_naive",
    "interpret_with_numpy",
    "interpret_with_torch",
]

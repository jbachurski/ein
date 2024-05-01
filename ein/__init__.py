from typing import TypeAlias

from ein.phi.type_system import matrix_type, ndarray_type, scalar_type, vector_type

from .backend.jax_backend import interpret as interpret_with_jax
from .backend.naive import interpret as interpret_with_naive
from .backend.numpy_backend import interpret as interpret_with_numpy
from .backend.torch_backend import interpret as interpret_with_torch
from .frontend.comprehension import array, fold
from .frontend.functions import function, with_varargs
from .frontend.ndarray import Array, Scalar, Vec, ext, wrap
from .frontend.std import where

Bool: TypeAlias = Scalar
Int: TypeAlias = Scalar
Float: TypeAlias = Scalar

__all__ = [
    "array",
    "fold",
    "ext",
    "wrap",
    "function",
    "with_varargs",
    "where",
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
    "interpret_with_jax",
]

from typing import Callable, Literal, TypeAlias

from . import jax_backend, naive, numpy_backend, torch_backend

Backend: TypeAlias = Literal["naive", "numpy", "torch", "jax"]
BACKENDS: dict[str, Callable] = {
    "naive": naive.interpret,
    "numpy": numpy_backend.interpret,
    "torch": torch_backend.interpret,
    "jax": jax_backend.interpret,
}
STAGE_BACKENDS: dict[str, Callable] = {
    "numpy": numpy_backend.stage,
    "torch": torch_backend.stage,
    "jax": jax_backend.stage,
}

DEFAULT_BACKEND: Backend = "numpy"

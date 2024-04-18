from typing import Callable, Literal, TypeAlias

from ein.backend import naive, numpy_backend, torch_backend

Backend: TypeAlias = Literal["naive", "numpy", "torch"]
BACKENDS: dict[str, Callable] = {
    "naive": naive.interpret,
    "numpy": numpy_backend.interpret,
    "torch": torch_backend.interpret,
}
STAGE_BACKENDS: dict[str, Callable] = {
    "numpy": numpy_backend.stage,
    "torch": torch_backend.stage,
}

DEFAULT_BACKEND: Backend = "numpy"

from typing import Callable, Literal, TypeAlias

from ein.backend import naive, to_numpy, to_torch

Backend: TypeAlias = Literal["naive", "numpy", "torch"]
BACKENDS: dict[str, Callable] = {
    "naive": naive.interpret,
    "numpy": to_numpy.interpret,
    "torch": to_torch.interpret,
}
STAGE_BACKENDS: dict[str, Callable] = {
    "numpy": to_numpy.stage,
    "torch": to_torch.stage,
}

DEFAULT_BACKEND: Backend = "numpy"

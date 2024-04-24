import pytest

from ein import (
    interpret_with_jax,
    interpret_with_naive,
    interpret_with_numpy,
    interpret_with_torch,
)

with_backend = pytest.mark.parametrize("backend", ["naive", "numpy", "torch", "jax"])
with_backend_for_dynamic_sizes = pytest.mark.parametrize(
    "backend", ["naive", "numpy", "torch"]
)

with_interpret = pytest.mark.parametrize(
    "interpret",
    [
        interpret_with_naive,
        interpret_with_numpy,
        interpret_with_torch,
        interpret_with_jax,
    ],
    ids=["naive", "numpy", "torch", "jax"],
)


with_interpret_for_dynamic_sizes = pytest.mark.parametrize(
    "interpret",
    [interpret_with_naive, interpret_with_numpy, interpret_with_torch],
    ids=["naive", "numpy", "torch"],
)

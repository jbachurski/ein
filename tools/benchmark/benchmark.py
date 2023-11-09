import statistics
import timeit
from functools import partial
from typing import Callable, Iterable, Sequence

import numpy

from ein import interpret_with_numpy
from tests.suite.parboil.mri_q import MriQ

N = [(1, 2, 5)[i % 3] * 10 ** (i // 3) for i in range(3, 13)]


def benchmark(run: Callable[..., numpy.ndarray]) -> Iterable[tuple[int, list[float]]]:
    for n in N:
        times = []
        sample = MriQ.sample(n, n)
        seconds_passed = 0.0
        runs = 0
        while runs < 3 or (runs < 100 and seconds_passed < 2):
            times.append(timeit.timeit(lambda: run(*sample), number=1))
            seconds_passed += times[-1]
            runs += 1
        yield n, times


def mean_stdev(ts: Sequence[float]) -> float:
    return statistics.stdev(ts) / numpy.sqrt(len(ts))


executors: list[tuple[str, Callable]] = [
    ("python", MriQ.in_python),
    ("numpy", MriQ.in_numpy),
    ("ein-numpy", partial(MriQ.in_ein_function, interpret_with_numpy)),
    # Way too inefficient to be interesting
    # ("ein-naive", partial(MriQ.in_ein_function, interpret_with_naive)),
]

result: dict[str, dict[int, list[float]]] = {}

for name, fun in executors:
    print(f"=== {name} ===")
    result[name] = {}
    for n, ts in benchmark(fun):
        print(
            f"k = x = {n} -> {statistics.mean(ts)} ± {mean_stdev(ts)}  ({len(ts)} runs)"
        )
        result[name][n] = ts

print(result)

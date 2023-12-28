import cProfile
import gc
import statistics
import timeit
from typing import Any, Callable, Sequence, TypeAlias

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

import ein
from ein.calculus import Expr
from ein.symbols import Variable
from tests.suite.deep import GAT
from tests.suite.parboil import MriQ, Stencil
from tests.suite.rodinia import NN, Hotspot, KMeans, Pathfinder

MIN_RUNS = 5
MIN_CAPPED_RUNS = 20
CAP_RUNNING_SECONDS = 1.0


def benchmark(
    run: Callable[..., numpy.ndarray], sample: tuple, do_profile: bool = False
) -> list[float]:
    times = []
    seconds_passed = 0.0
    runs = 0
    gc.disable()
    while runs < MIN_RUNS or (
        runs < MIN_CAPPED_RUNS and seconds_passed < CAP_RUNNING_SECONDS
    ):
        times.append(timeit.timeit(lambda: run(*sample), number=1))
        seconds_passed += times[-1]
        runs += 1
    gc.enable()
    gc.collect()
    if do_profile:
        with cProfile.Profile() as prof:
            run(*sample)
        prof.print_stats(sort="tottime")
    return times


def mean_stdev(ts: Sequence[float]) -> float:
    return statistics.stdev(ts) / numpy.sqrt(len(ts))


def precompile(varargs: tuple[Variable, ...], program: Expr) -> Callable:
    staged = ein.backend.to_numpy.stage(program)
    return lambda *args: staged({var: arg for var, arg in zip(varargs, args)})


Executor: TypeAlias = tuple[str, Callable, Callable[[int], bool]]
Executors: TypeAlias = list[Executor]
Benchmark: TypeAlias = tuple[Callable[[int], tuple], list[int], Executors]

DEEP_GAT = "Deep: GAT"
PARBOIL_MRI_Q = "Parboil: MRI-Q"
PARBOIL_STENCIL = "Parboil: Stencil"
RODINIA_HOTSPOT = "Rodinia: Hotspot"
RODINIA_KMEANS = "Rodinia: KMeans"
RODINIA_NN = "Rodinia: NN"
RODINIA_PATHFINDER = "Rodinia: Pathfinder"

BENCHMARKS: dict[str, Benchmark] = {
    DEEP_GAT: (
        lambda n: GAT.sample(4, n, n, n),
        list(numpy.geomspace(50, 150, 20).astype(int)),
        [
            ("Ein", precompile(*GAT.ein_function()), lambda n: 2 <= n <= 150),
            ("NumPy", GAT.in_numpy, lambda n: 2 <= n <= 150),
        ],
    ),
    PARBOIL_MRI_Q: (
        lambda n: MriQ.sample(n, n),
        list(numpy.geomspace(50, 1e5, 20).astype(int)),
        [
            ("Ein", precompile(*MriQ.ein_function()), lambda n: 100 <= n < 5e4),
            # Uses over 40 GB RAM at 5e4
            ("NumPy", MriQ.in_numpy, lambda n: 100 <= n < 3e4),
            # Saves memory by a non-idiomatic Python loop
            # ("NumPy (loop)", MriQ.in_numpy_frugal, [n for n in N if n >= 100]),
            # Uses einsum, which is probably best but not general
            # ("NumPy (\\texttt{einsum})", MriQ.in_numpy_einsum, [n for n in N if n >= 100]),
            # Too slow at 1e4 (over 20 seconds)
            ("Python", MriQ.in_python, lambda n: 100 <= n < 2e3),
            # Way too inefficient to be interesting
            # ("ein-naive", partial(MriQ.in_ein_function, interpret_with_naive, N[4:7])),
        ],
    ),
    PARBOIL_STENCIL: (
        lambda n: Stencil.sample(5, n, n, n),
        list(numpy.geomspace(8, 250, 20).astype(int)),
        [
            ("Ein", precompile(*Stencil.ein_function()), lambda n: 2 <= n <= 250),
            ("NumPy", Stencil.in_numpy, lambda n: 2 <= n <= 250),
            ("Python", Stencil.in_python, lambda n: 2 <= n <= 30),
        ],
    ),
    RODINIA_HOTSPOT: (
        lambda n: Hotspot.sample(5, n, n),
        list(numpy.geomspace(50, 5e3, 20).astype(int)),
        [
            ("Ein", precompile(*Hotspot.ein_function()), lambda n: 2 <= n <= 5e3),
            ("NumPy", Hotspot.in_numpy, lambda n: 2 <= n <= 5e3),
            ("Python", Hotspot.in_python, lambda n: 2 <= n <= 500),
        ],
    ),
    RODINIA_KMEANS: (
        lambda n: KMeans.sample(n, n, n, 10),
        list(numpy.geomspace(50, 300, 20).astype(int)),
        [
            ("Ein", precompile(*KMeans.ein_function()), lambda n: 2 <= n <= 175),
            ("NumPy", KMeans.in_numpy, lambda n: 2 <= n <= 300),
            ("Python", KMeans.in_python, lambda n: 2 <= n <= 100),
        ],
    ),
    RODINIA_NN: (
        lambda n: NN.sample(n, 20),
        list(numpy.geomspace(50, 1e7, 20).astype(int)),
        [
            ("Ein", precompile(*NN.ein_function()), lambda n: 2 <= n <= 2e4),
            ("NumPy", NN.in_numpy, lambda n: 2 <= n <= 1e7),
            ("Python", NN.in_python, lambda n: 2 <= n <= 4e5),
        ],
    ),
    RODINIA_PATHFINDER: (
        lambda n: Pathfinder.sample(max(1, n // 10), n * (n // max(1, n // 10))),
        # Higher sizes of Pathfinder cause costs to no longer fit in cache, killing runtime
        list(numpy.geomspace(120, 1e4, 20).astype(int)),
        [
            ("Ein", precompile(*Pathfinder.ein_function()), lambda n: 2 <= n <= 1e4),
            ("NumPy", Pathfinder.in_numpy, lambda n: 2 <= n <= 1e4),
            ("Python", Pathfinder.in_python, lambda n: 2 <= n <= 2e3),
        ],
    ),
}


def perform(
    get_sample: Callable[[int], Any], params: list[int], executors: Executors
) -> dict[str, dict[int, list[float]]]:
    result: dict[str, dict[int, list[float]]] = {}

    for name, fun, pred in executors:
        print(f"=== {name} ===")
        result[name] = {}
        for n, ts in ((n, benchmark(fun, get_sample(n))) for n in params if pred(n)):
            print(
                f"k = x = {n} -> {min(ts)}  ({statistics.mean(ts)} ± {mean_stdev(ts)} across {len(ts)} runs)"
            )
            result[name][n] = ts

    return result


def plots(name: str, results: dict[str, dict[int, list[float]]]) -> None:
    seaborn.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-pastel")
    seaborn.set(rc={"text.usetex": True})

    for exec_name, exec_result in results.items():
        ax = seaborn.lineplot(
            x="$n$",
            y="runtime",
            data=pandas.DataFrame.from_records(
                [{"$n$": n, "runtime": t} for n, ts in exec_result.items() for t in ts]
            ),
            label=exec_name,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    plt.legend()
    plt.title(name)
    plt.show()


if __name__ == "__main__":
    benchmarks = [
        DEEP_GAT,
        PARBOIL_MRI_Q,
        PARBOIL_STENCIL,
        RODINIA_HOTSPOT,
        RODINIA_KMEANS,
        RODINIA_NN,
        RODINIA_PATHFINDER,
    ]

    for name in benchmarks:
        print(f"\n\n === ||| {name} ||| === ")
        get_sample, params, executors = BENCHMARKS[name]
        results = perform(get_sample, params, executors)
        plots(name, results)

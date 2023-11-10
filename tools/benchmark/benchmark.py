import cProfile
import gc
import statistics
import timeit
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

import ein
from tests.suite.parboil.mri_q import MriQ

N = [int(10 ** (i / 10)) for i in range(48)]


def benchmark(
    run: Callable[..., numpy.ndarray], sample_n: int, do_profile: bool = False
) -> list[float]:
    times = []
    sample = MriQ.sample(sample_n, sample_n)
    seconds_passed = 0.0
    runs = 0
    while runs < 7 or (runs < 100 and seconds_passed < 5):
        times.append(timeit.timeit(lambda: run(*sample), number=1))
        gc.collect()
        seconds_passed += times[-1]
        runs += 1
    if do_profile:
        with cProfile.Profile() as prof:
            run(*sample)
        prof.print_stats(sort="tottime")
    return times


def mean_stdev(ts: Sequence[float]) -> float:
    return statistics.stdev(ts) / numpy.sqrt(len(ts))


def in_precompiled_ein_function():
    varargs, program = MriQ.ein_function()
    staged = ein.backend.to_numpy.stage(ein.backend.to_array.transform(program))
    return lambda *args: staged({var: arg for var, arg in zip(varargs, args)})


executors: list[tuple[str, Callable, list[int]]] = [
    ("Ein $\\to$ NumPy", in_precompiled_ein_function(), [n for n in N if n >= 100]),
    # Uses over 40 GB RAM at 5e4
    ("NumPy", MriQ.in_numpy, [n for n in N if 100 <= n < 3 * 10**4]),
    # Saves memory by a non-idiomatic Python loop
    # ("NumPy (loop)", MriQ.in_numpy_frugal, [n for n in N if n >= 100]),
    # Uses einsum, which is probably best but not general
    # ("NumPy (\\texttt{einsum})", MriQ.in_numpy_einsum, [n for n in N if n >= 100]),
    # Too slow at 1e4 (over 20 seconds)
    ("Python", MriQ.in_python, [n for n in N if 100 <= n < 10**4]),
    # Way too inefficient to be interesting
    # ("ein-naive", partial(MriQ.in_ein_function, interpret_with_naive, N[4:7])),
]


def main() -> dict[str, dict[int, list[float]]]:
    result: dict[str, dict[int, list[float]]] = {}

    for name, fun, ns in executors:
        print(f"=== {name} ===")
        result[name] = {}
        for _ in range(3):
            benchmark(fun, ns[0])
        for n, ts in ((n, benchmark(fun, n, do_profile=n == ns[-1])) for n in ns):
            print(
                f"k = x = {n} -> {statistics.mean(ts)} ± {mean_stdev(ts)}  ({len(ts)} runs)"
            )
            result[name][n] = ts

    return result


def plots(result: dict[str, dict[int, list[float]]]) -> None:
    print(result)
    seaborn.set_style("whitegrid")
    plt.style.use("seaborn-v0_8-pastel")
    seaborn.set(rc={"text.usetex": True})

    for name, _, _ in executors:
        ax = seaborn.lineplot(
            x="$n$",
            y="runtime",
            data=pandas.DataFrame.from_records(
                [{"$n$": n, "runtime": t} for n, ts in result[name].items() for t in ts]
            ),
            label=name,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plots(main())

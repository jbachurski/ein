import numpy

from ein import Array, array, fold, matrix
from ein.frontend import std

from ..case import Case


class Pathfinder(Case):
    ein_types = [matrix(float)]

    @staticmethod
    def in_ein(*args: Array) -> Array:
        (costs,) = args
        n = costs.dim(1)
        return fold(
            array(lambda i: 0.0, size=n),
            lambda t, dist: array(
                lambda i: (
                    costs[t, i]
                    + std.min(
                        dist[i],
                        dist[std.max(i - 1, 0)],
                        dist[std.min(i + 1, n - 1)],
                    )
                )
            ),
        )

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        (costs,) = args
        t, n = costs.shape
        dist = numpy.zeros(n)
        for i in range(t):
            left, right = dist.copy(), dist.copy()
            left[1:] = dist[:-1]
            right[:-1] = dist[1:]
            dist = costs[i] + numpy.minimum(dist, numpy.minimum(left, right))
        return dist

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        (costs,) = args
        t, n = costs.shape
        dist = [0.0 for _ in range(n)]
        for i in range(t):
            dist = [
                costs[i, p] + min(dist[max(0, p - 1)], dist[p], dist[min(n - 1, p + 1)])
                for p in range(n)
            ]
        return numpy.array(dist)

    @staticmethod
    def sample(t: int = 11, n: int = 13) -> tuple[numpy.ndarray, ...]:
        costs = numpy.abs(numpy.random.randn(t, n))
        return (costs,)

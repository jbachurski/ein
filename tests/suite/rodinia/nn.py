from math import hypot

import numpy

from ein import Array, array, fold, scalar, vector
from ein.frontend.std import reduce_argmin, where

from ..case import Case


class NN(Case):
    ein_types = [scalar(int)] + [scalar(float)] * 2 + [vector(float)] * 2

    @staticmethod
    def in_ein(*args: Array) -> Array:
        k, x, y, xs, ys = args

        def update(vec, p, x):
            return array(lambda i: where(i != p, vec[i], x))

        def step_nn(i, acc):
            dist, res = acc
            p = reduce_argmin(lambda i: dist[i])
            return update(dist, p, float("+inf")), update(res, i, p)

        return fold(
            (
                array(lambda i: ((x - xs[i]) ** 2 + (y - ys[i]) ** 2) ** 0.5),
                array(lambda i: 0, size=k),
            ),
            step_nn,
            count=k,
        )[1]

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        k, x, y, xs, ys = args
        k = int(k)
        distances = numpy.sqrt((x - xs) ** 2 + (y - ys) ** 2)
        results = []
        for _ in range(k):
            results.append(distances.argmin())
            distances[results[-1]] = numpy.inf

        return numpy.array(results)

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        k, x, y, xs, ys = args
        k = int(k)

        assert len(xs) == len(ys)
        distances = [hypot(x - x1, y - y1) for x1, y1 in zip(xs, ys)]
        results = []
        for _ in range(k):
            results.append(min((d, i) for i, d in enumerate(distances))[1])
            distances[results[-1]] = float("inf")

        return numpy.array(results)

    @staticmethod
    def sample(n: int = 22, k: int = 5) -> tuple[numpy.ndarray, ...]:
        assert k < n
        x, y = numpy.random.randn(), numpy.random.randn()
        xs, ys = numpy.random.randn(n), numpy.random.randn(n)
        return numpy.array(k), x, y, xs, ys

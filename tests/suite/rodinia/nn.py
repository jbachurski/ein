from functools import partial
from math import hypot

import numpy

from ein import Scalar, Vec, array, ext, fold, scalar_type, vector_type, wrap
from ein.frontend.typed import array1

from ..case import Case


class NN(Case):
    ein_types = [scalar_type(int)] + [scalar_type(float)] * 2 + [vector_type(float)] * 2

    @staticmethod
    def in_ein(
        k: Scalar, x: Scalar, y: Scalar, xs: Vec[Scalar], ys: Vec[Scalar]
    ) -> Vec[Scalar]:
        def update(vec: Vec[Scalar], p: Scalar, x: Scalar) -> Vec[Scalar]:
            # return array1(lambda i: where(i != p, vec[i], x))
            # unsafe!
            def with_update(arr, pos, val):
                arr[..., pos] = val
                return arr

            return ext(with_update, vec.expr.type)(vec, p, x)

        def argmin(vec: Vec[Scalar]) -> Scalar:
            return ext(partial(numpy.argmin, axis=-1), scalar_type(int))(vec)

        def step_nn(i, acc):
            dist, res = acc
            p = argmin(dist)
            return update(dist, p, wrap(float("+inf"))), update(res, i, p)

        return fold(
            (
                array1(lambda i: ((x - xs[i]) ** 2 + (y - ys[i]) ** 2) ** 0.5),
                array(lambda i: wrap(0), size=k),
            ),
            step_nn,
            count=k,
        )[1]

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        k_, x, y, xs, ys = args
        k = int(k_)
        distances = numpy.sqrt((x - xs) ** 2 + (y - ys) ** 2)
        results = []
        for _ in range(k):
            results.append(distances.argmin())
            distances[results[-1]] = numpy.inf

        return numpy.array(results)

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        k_, x, y, xs, ys = args
        k = int(k_)

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
        return numpy.array(k), numpy.array(x), numpy.array(y), xs, ys

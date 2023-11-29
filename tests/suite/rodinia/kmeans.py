from typing import Callable

import numpy

from ein import Array, Scalar, array, fold, matrix
from ein import sum as fold_sum

from ..case import Case


class KMeans(Case):
    ein_types = [matrix(float), Scalar(int), Scalar(int)]

    @staticmethod
    def in_ein(*args: Array) -> Array:
        points, k, it = args

        def square(x: Array) -> Array:
            return x * x

        def dist(p1: Array, p2: Array) -> Array:
            return fold_sum(lambda d: square(p1[d] - p2[d]))

        def maximum(x: Array | int, y: Array) -> Array:
            return (x > y).where(x, y)

        def argmin_concat(
            a: tuple[Array, Array], b: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            lt = a[0] <= b[0]
            return lt.where(a[0], b[0]), lt.where(a[1], b[1])

        def fold_argmin(f: Callable[[Array], Array]) -> Array:
            return fold(
                (float("inf"), 0),
                lambda i, acc: argmin_concat(acc, (f(i), i)),
            )[1]

        def fold_centres(_i: Array, centres: Array) -> Array:
            ks, ds = centres.dim(0), centres.dim(1)
            members = array(
                lambda i: fold_argmin(lambda j: dist(points[i], centres[j]))
            )

            def members_of(j: Array) -> Array:
                return fold(0, lambda i, acc: acc + Array(members[i] == j).where(1, 0))

            return array[ks, ds](
                lambda j, d: (
                    fold_sum(lambda i: Array(members[i] == j).where(points[i][d], 0.0))
                    / maximum(1, members_of(j)).to_float()
                )
            )

        return fold[it](array[k](lambda i: points[i]), fold_centres)

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        points, k, it = args

        centres = points[:k]
        for _ in range(it):
            dist = (
                (points[:, numpy.newaxis, :] - centres[numpy.newaxis, :, :]) ** 2
            ).sum(axis=2)
            members = dist.argmin(axis=1)
            hist = numpy.zeros(k)
            numpy.add.at(hist, members, 1)
            new_centres = numpy.zeros_like(centres)
            numpy.add.at(new_centres, members, points)
            new_centres /= numpy.maximum(hist[:, numpy.newaxis], 1)
            centres = new_centres

        return centres

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        points_, k, it = args
        points: list[list[float]] = [[x for x in p] for p in points_]
        d = len(points[0])
        k, it = int(k), int(it)
        centres = points[:k]

        def dist(p1: list[float], p2: list[float]) -> float:
            return sum((x1 - x2) ** 2 for x1, x2 in zip(p1, p2))

        for _ in range(it):
            members = [
                min((dist(p, c), i) for i, c in enumerate(centres))[1] for p in points
            ]
            hist = [0 for _ in range(k)]
            new_centres = [[0.0 for _ in range(d)] for _ in range(k)]
            for i, p in enumerate(points):
                hist[members[i]] += 1
                for j in range(d):
                    new_centres[members[i]][j] += points[i][j]
            for i in range(k):
                h = max(hist[i], 1)
                for j in range(d):
                    new_centres[i][j] /= h
            centres = new_centres

        return numpy.array(centres)

    @staticmethod
    def sample(
        n: int = 6, d: int = 3, k: int = 2, it: int = 20
    ) -> tuple[numpy.ndarray, ...]:
        locs = numpy.random.randn(k, d)
        points = numpy.random.randn(n, d)
        for i in range(n):
            points[i] += locs[numpy.random.randint(k)]
        return points, numpy.array(k), numpy.array(it)

import numpy

from ein import Array, array, fold, ndarray, scalar
from ein.frontend.std import where

from ..case import Case

C0, C1 = 1 / 6, 1 / 6 / 6


class Stencil(Case):
    ein_types = [scalar(int), ndarray(3, float)]

    @staticmethod
    def in_ein(*args: Array) -> Array:
        it, A0 = args
        nx, ny, nz = A0.dim(0), A0.dim(1), A0.dim(2)
        return fold(
            A0,
            lambda _, A: array(
                lambda x, y, z: where(
                    (x == 0)
                    | (x == nx - 1)
                    | (y == 0)
                    | (y == ny - 1)
                    | (z == 0)
                    | (z == nz - 1),
                    A[x, y, z],
                    (
                        A[x - 1, y, z]
                        + A[x + 1, y, z]
                        + A[x, y - 1, z]
                        + A[x, y + 1, z]
                        + A[x, y, z - 1]
                        + A[x, y, z + 1]
                    )
                    * C1
                    + A[x, y, z] * C0,
                )
            ),
            count=it,
        )

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        it, A = args
        it = int(it)

        for _ in range(it):
            edge = A.copy()
            edge[1:-1, 1:-1, 1:-1] = 0
            inner = numpy.zeros_like(A)
            inner[1:-1, 1:-1, 1:-1] += (
                A[1:-1, 1:-1, :-2]
                + A[1:-1, 1:-1, 2:]
                + A[1:-1, :-2, 1:-1]
                + A[1:-1, 2:, 1:-1]
                + A[:-2, 1:-1, 1:-1]
                + A[2:, 1:-1, 1:-1]
            ) * C1
            inner[1:-1, 1:-1, 1:-1] += A[1:-1, 1:-1, 1:-1] * C0
            A = edge + inner

        return A

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        it, A = args
        it = int(it)
        nx, ny, nz = A.shape

        for _ in range(it):

            def f(x, y, z):
                if x == 0 or x == nx - 1:
                    return A[x, y, z]
                if y == 0 or y == ny - 1:
                    return A[x, y, z]
                if z == 0 or z == nz - 1:
                    return A[x, y, z]
                return (
                    A[x, y, z + 1]
                    + A[x, y, z - 1]
                    + A[x, y - 1, z]
                    + A[x, y + 1, z]
                    + A[x - 1, y, z]
                    + A[x + 1, y, z]
                ) * C1 + A[x, y, z] * C0

            A = numpy.array(
                [
                    [[f(x, y, z) for z in range(nz)] for y in range(ny)]
                    for x in range(nx)
                ]
            )

        return A

    @staticmethod
    def sample(
        it: int = 3, nx: int = 4, ny: int = 5, nz: int = 6
    ) -> tuple[numpy.ndarray, ...]:
        A = numpy.random.randn(nx, ny, nz)
        return numpy.array(it), A

from dataclasses import dataclass

import numpy

from ein import Float, Int, Vec, array, fold, scalar_type, wrap
from ein.frontend.std import where

from ..case import Case

# For fairness, this test case does not make use of NumPy complex numbers,
# as they are not directly implemented in Ein at the moment.


class Mandelbrot(Case):
    ein_types = [scalar_type(int), scalar_type(int), scalar_type(int)]

    @staticmethod
    def in_ein(w: Int, h: Int, t: Int) -> Vec[Vec[Int]]:
        @dataclass
        class Complex:
            x: Float
            y: Float

            def __add__(self, other: "Complex") -> "Complex":
                return Complex(self.x + other.x, self.y + other.y)

            def square(self) -> "Complex":
                return Complex(self.x * self.x - self.y * self.y, 2.0 * self.x * self.y)

            def magnitude(self) -> Float:
                return self.x * self.x + self.y * self.y

        def lerp(x: Float, a: Float, b: Float) -> Float:
            return x * b + (1.0 - x) * a

        def mapped(i: Int, j: Int) -> Complex:
            return Complex(
                lerp((i.float() / (h - 1).float()), wrap(-2.0), wrap(0.47)),
                lerp((j.float() / (w - 1).float()), wrap(-1.12), wrap(1.12)),
            )

        def divergence_count(c: Complex) -> Int:
            def step(i: Int, acc: tuple[Int, Complex]) -> tuple[Int, Complex]:
                out, z = acc
                return where(
                    (out > -1) & (z.magnitude() <= 4.0),
                    (i + 1, z.square() + c),
                    (out, z),
                )

            return fold((wrap(0), Complex(wrap(0.0), wrap(0.0))), step, count=t)[0]

        return array(lambda i, j: divergence_count(mapped(i, j)), size=(w, h))

    @staticmethod
    def in_numpy(w, h, t) -> numpy.ndarray:
        w, h, t = int(w), int(h), int(t)
        return numpy.zeros((w, h))

    @staticmethod
    def in_python(w, h, t) -> numpy.ndarray:
        w, h, t = int(w), int(h), int(t)
        arr = [[-1 for _ in range(h)] for _ in range(w)]

        def lerp(x, a, b):
            return x * b + (1.0 - x) * a

        for i in range(h):
            for j in range(w):
                c = lerp(i / (h - 1), -2, 0.47) + lerp(j / (w - 1), -1.12, 1.12) * 1j
                z = complex(0)
                for k in range(t):
                    if abs(z) > 2:
                        break
                    z = z**2 + c
                else:
                    k = t
                arr[i][j] = k
        return numpy.array(arr)

    @staticmethod
    def sample(n: int = 4, t: int = 20) -> tuple[numpy.ndarray, ...]:
        return numpy.array(n), numpy.array(n), numpy.array(t)

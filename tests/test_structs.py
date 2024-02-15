from dataclasses import dataclass

import numpy.testing

from ein import Array, array, structs
from ein.frontend.std import reduce_sum, where


def test_adhoc_structs():
    from ein import array, structs

    s = structs(lambda i: (i, i**2, {"+": i**3, "-": -(i**3)}), size=10)
    a = array(lambda j: s[j, 2, "-"])
    numpy.testing.assert_allclose(a.numpy(), [-(i**3) for i in range(10)])


def test_complex_scalars():
    @dataclass
    class C:
        real: Array
        imag: Array

        def __add__(self, other) -> "C":
            return C(self.real + other.real, self.imag + other.imag)

        def __mul__(self, other) -> "C":
            return C(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )

    def linspace(a, b, n):
        n = Array(n)
        return array(lambda i: i.to_float() * (b - a) / (n - 1).to_float() + a, size=n)

    p = linspace(0.0, 3.14, 11)
    c = structs(lambda i: C(p[i].cos(), p[i].sin()))
    cc = array(lambda i: (c[i] * c[i] + c[i]).real)

    def cis(x):
        return numpy.cos(x) + numpy.sin(x) * 1j

    c1 = cis(numpy.linspace(0, 3.14, 11))
    cc1 = numpy.real(c1 * (c1 + 1))
    numpy.testing.assert_allclose(cc.numpy(), cc1)


@dataclass
class Matrix:
    elem: Array

    @classmethod
    def of(cls, arr) -> "Matrix":
        return Matrix(Array(arr))

    @classmethod
    def square(cls, n, constructor):
        return Matrix(array(constructor, size=(n, n)))

    @classmethod
    def eye(cls, n) -> "Matrix":
        return Matrix(array(lambda i, j: where(i == j, 1.0, 0.0), size=(n, n)))

    def scale(self, a) -> "Matrix":
        return Matrix(array(lambda i, j: self.elem[i, j] * a))

    def __add__(self, other: "Matrix") -> "Matrix":
        return Matrix(array(lambda i, j: self.elem[i, j] + other.elem[i, j]))

    def __mul__(self, other: "Matrix") -> "Matrix":
        return Matrix(array(lambda i, j: self.elem[i, j] * other.elem[i, j]))

    def __matmul__(self, other: "Matrix") -> "Matrix":
        return Matrix(
            array(lambda i, j: reduce_sum(lambda k: self.elem[i, k] * other.elem[k, j]))
        )


def test_matrix_sanity():
    a = Matrix(array(lambda i, j: 2 * i + j, size=(2, 2)))
    numpy.testing.assert_allclose((a + a).elem.numpy(), numpy.array([[0, 2], [4, 6]]))


def test_matrix_batches():
    base = numpy.array([[1, 2], [3, 4]], dtype=float)
    scales = structs(
        lambda a: Matrix.eye(2).scale(a.to_float()),
        size=4,
    )
    matrices = structs(lambda a: scales[a] @ Matrix.of(base))

    got = array(lambda i: matrices[i].elem).numpy()
    exp = numpy.arange(4)[:, None, None] * numpy.array(base)
    numpy.testing.assert_allclose(got, exp)

from functools import partial

import numpy
import pytest

from ein import Array, array, ext, scalar_type, vector_type, wrap
from ein.frontend.std import concat


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_basic_extrinsic(backend):
    a = numpy.array([1, 2, 3])

    def vsum(x) -> Array:
        return ext(partial(numpy.sum, axis=-1), vector_type(float))(x)

    b = vsum(a).eval(backend=backend)
    numpy.testing.assert_allclose(b, 6)


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_extrinsic_broadcast(backend):
    def logaddexp(x, y) -> Array:
        return ext(numpy.logaddexp, scalar_type(float))(x, y)

    a = numpy.array([1, 2, 3])
    b = numpy.array([1, 1 / 2, 1 / 3])

    c = array(lambda i, j: logaddexp(wrap(a)[i], wrap(b)[j])).eval(backend=backend)
    exp = numpy.logaddexp(a[:, None], b[None, :])
    numpy.testing.assert_allclose(c, exp)


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_extrinsic_batched_reduction(backend):
    def argmin(x) -> Array:
        return ext(partial(numpy.argmin, axis=-1), vector_type(float))(x)

    a = numpy.arange(9).reshape(3, 3)
    numpy.testing.assert_allclose(
        array(lambda i: argmin(array(lambda j: wrap(a)[i, j]))).eval(backend=backend),
        numpy.argmin(a, axis=0),
    )
    numpy.testing.assert_allclose(
        array(lambda i: argmin(array(lambda j: wrap(a)[j, i]))).eval(backend=backend),
        numpy.argmin(a, axis=1),
    )


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_reduce_sort(backend):
    def sort(x) -> Array:
        return ext(partial(numpy.sort, axis=-1), vector_type(float))(x)

    a0 = numpy.random.randn(11)
    nil = array(lambda i: 0.0, size=0)

    a = wrap(a0)
    numpy.testing.assert_allclose(
        array(lambda i: array(lambda _: a[i], size=1))
        .reduce(nil, lambda x, y: sort(concat(x, y)))
        .eval(backend=backend),
        numpy.sort(a0),
    )

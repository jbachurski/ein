from functools import partial

import numpy
import pytest

from ein import Array, Scalar, Vector, arr, array, ext


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_basic_extrinsic(backend):
    a = numpy.array([1, 2, 3])

    def vsum(x) -> Array:
        return ext(partial(numpy.sum, axis=-1), Vector(Scalar(float)))(x)

    b = vsum(a).numpy(backend=backend)
    numpy.testing.assert_allclose(b, 6)


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_extrinsic_broadcast(backend):
    def logaddexp(x, y) -> Array:
        return ext(numpy.logaddexp, Scalar(float))(x, y)

    a = numpy.array([1, 2, 3])
    b = numpy.array([1, 1 / 2, 1 / 3])

    c = array(lambda i, j: logaddexp(arr(a)[i], arr(b)[j])).numpy(backend=backend)
    exp = numpy.logaddexp(a[:, None], b[None, :])
    numpy.testing.assert_allclose(c, exp)


@pytest.mark.parametrize("backend", ["naive", "numpy"])
def test_extrinsic_batched_reduction(backend):
    def argmin(x) -> Array:
        return ext(partial(numpy.argmin, axis=-1), Vector(Scalar(float)))(x)

    a = numpy.arange(9).reshape(3, 3)
    numpy.testing.assert_allclose(
        array(lambda i: argmin(array(lambda j: arr(a)[i, j]))).numpy(backend=backend),
        numpy.argmin(a, axis=0),
    )
    numpy.testing.assert_allclose(
        array(lambda i: argmin(array(lambda j: arr(a)[j, i]))).numpy(backend=backend),
        numpy.argmin(a, axis=1),
    )

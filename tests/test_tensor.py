import functools
from typing import Any

import numpy
import pytest

from ein import (
    Array,
    array,
    fold,
    function,
    interpret_with_naive,
    interpret_with_numpy,
    matrix,
    scalar,
    vector,
)
from ein.frontend.std import reduce_max, reduce_min, reduce_sum, where

with_backend = pytest.mark.parametrize("backend", ["naive", "numpy"])
with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_naive, interpret_with_numpy], ids=["naive", "numpy"]
)


def test_type_checks():
    with pytest.raises(TypeError):
        _ = function([scalar(int)], lambda n: 1 + array(lambda i: 0, size=n))


@with_interpret
def test_mul_grid(interpret):
    (n0,), grid_expr = function(
        [scalar(int)],
        lambda n: array(
            lambda i, j: (i.to_float() + 1.0) / (j.to_float() + 1.0), size=(n, n)
        ),
    )

    numpy.testing.assert_allclose(
        interpret(
            grid_expr,
            {n0: numpy.array(5)},
        ),
        numpy.array([[i / j for j in range(1, 6)] for i in range(1, 6)]),
    )


@with_interpret
@pytest.mark.parametrize(
    "with_bounds_inference", [False, True], ids=["give-sizes", "infer-sizes"]
)
def test_matmul(interpret, with_bounds_inference):
    def matmul(a: Array, b: Array):
        n, k = a.dim(0), a.dim(1)
        _k, m = b.dim(0), b.dim(1)
        array_: Any = (
            array if with_bounds_inference else functools.partial(array, size=(n, m))
        )
        sum_: Any = (
            reduce_sum
            if with_bounds_inference
            else functools.partial(reduce_sum, count=k)
        )
        return array_(lambda i, j: sum_(lambda t: a[i, t] * b[t, j]))

    (a0, b0), matmul_expr = function([matrix(float), matrix(float)], matmul)

    import ein.debug
    import ein.midend.lining

    print("args", a0, b0)
    print(ein.debug.pretty_print(matmul_expr))

    first = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    second = numpy.array([[1], [0], [-1]], dtype=float)
    numpy.testing.assert_allclose(
        interpret(
            matmul_expr,
            {
                a0: first,
                b0: second,
            },
        ),
        first @ second,
    )


@with_backend
@pytest.mark.parametrize(
    "with_bounds_inference", [False, True], ids=["give-sizes", "infer-sizes"]
)
def test_uv_loss(backend, with_bounds_inference):
    m, k, n = 12, 8, 10
    x_values = numpy.random.randn(m, n)
    u_values = numpy.random.randn(m, k)
    v_values = numpy.random.randn(n, k)
    x, u, v = Array(x_values), Array(u_values), Array(v_values)

    inner_sum: Any = (
        reduce_sum if with_bounds_inference else functools.partial(reduce_sum, count=k)
    )
    outer_sum: Any = (
        (lambda f: reduce_sum(lambda i: reduce_sum(lambda j: f(i, j))))
        if with_bounds_inference
        else (
            lambda f: reduce_sum(
                lambda i: reduce_sum(lambda j: f(i, j), count=n), count=m
            )
        )
    )
    loss = outer_sum(
        lambda i, j: (x[i, j] - inner_sum(lambda t: u[i, t] * v[j, t])) ** 2
    )

    numpy.testing.assert_allclose(
        loss.numpy(backend=backend), ((x_values - u_values @ v_values.T) ** 2).sum()
    )


@with_interpret
def test_max_minus_min(interpret):
    (a, b), expr = function(
        [vector(float), vector(float)],
        lambda a, b: reduce_max(lambda i: a[i] * b[i])
        - reduce_min(lambda i: a[i] * b[i]),
    )
    a_values, b_values = numpy.random.randn(256), numpy.random.randn(256)
    numpy.testing.assert_allclose(
        interpret(expr, {a: a_values, b: b_values}),
        (a_values * b_values).max() - (a_values * b_values).min(),
    )


@with_interpret
def test_switches(interpret):
    def sgn_max(a: Array, b: Array) -> Array:
        m = array(lambda i: (a[i] > b[i]).where(a[i], b[i]))
        return array(lambda i: (m[i] > 0.0).where(1, (a[i] == b[i]).where(0, -1)))

    (a0, b0), sgn_max_expr = function([vector(float), vector(float)], sgn_max)
    a_values, b_values = numpy.random.randn(16), numpy.random.randn(16)
    numpy.testing.assert_allclose(
        interpret(sgn_max_expr, {a0: a_values, b0: b_values}),
        numpy.sign(numpy.maximum(a_values, b_values)),
    )


@with_interpret
def test_reuse(interpret):
    a0 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    a = Array(a0)
    b = array(lambda i, j: a[i, j] ** 2)
    c = array(lambda i, j: 2.0 * b[i, j] + b[i, j] ** 0.33)
    numpy.testing.assert_allclose(
        interpret(c.expr, {}), (2 * a0**2 + (a0**2) ** 0.33)
    )


@with_interpret
def test_fibonacci_fold(interpret):
    def fib(n: Array) -> Array:
        return fold(
            array(lambda i: 0, size=n),
            lambda i, acc: array(
                lambda j: Array(i == j).where(
                    Array(j == 0).where(
                        0, Array(j == 1).where(1, acc[i - 1] + acc[i - 2])
                    ),
                    acc[j],
                )
            ),
            count=n,
        )

    (n0,), fib_expr = function([scalar(int)], fib)
    numpy.testing.assert_allclose(
        interpret(fib_expr, {n0: numpy.array(8)}),
        [0, 1, 1, 2, 3, 5, 8, 13],
    )


@with_backend
def test_inline_interpret(backend):
    a = Array(numpy.random.randn(20, 30))
    b = Array(numpy.random.randn(30))
    numpy.testing.assert_allclose(
        array(lambda i: reduce_sum(lambda j: a[i, j] * b[j])).numpy(backend=backend),
        a.numpy(backend=backend) @ b.numpy(backend=backend),
    )


def _primes(n) -> list[bool]:
    return [i > 1 and all(i % d for d in range(2, i)) for i in range(n)]


@with_interpret
def test_trial_division_primes(interpret):
    def trial_division(n: Array) -> Array:
        return array(
            lambda i: (
                fold(
                    i > 1,
                    lambda d, acc: (acc & ~((i % d == 0) & (1 < d) & (d < i))),
                    count=n,
                )
            ),
            size=n,
        )

    (n0,), expr = function([scalar(int)], trial_division)
    N = 30
    numpy.testing.assert_allclose(
        interpret(expr, {n0: numpy.array(N)}),
        _primes(N),
    )


@with_interpret
def test_sieve_primes(interpret):
    def sieve(n: Array) -> Array:
        return fold(
            array(lambda i: i > 1, size=n),
            lambda d, siv: array(
                lambda i: (siv[i] & ~((d > 1) & (i >= d * d) & (i % d == 0))), size=n
            ),
            count=n,
        )

    (n0,), expr = function([scalar(int)], sieve)
    N = 30
    numpy.testing.assert_allclose(
        interpret(expr, {n0: numpy.array(N)}),
        _primes(N),
    )


@with_interpret
def test_double_transpose(interpret):
    (a0,), expr = function(
        [matrix(float)], lambda a: array(lambda i, j: array(lambda k, l: a[l, k])[j, i])
    )
    arr = numpy.random.randn(3, 3)
    numpy.testing.assert_allclose(interpret(expr, {a0: arr}), arr)


@with_interpret
def test_symmetric_sum(interpret):
    def f1(x: numpy.ndarray) -> numpy.ndarray:
        return x + x.T

    def f(x: Array) -> Array:
        return array(lambda i, j: x[i, j] + x[j, i])

    (a0,), expr = function([matrix(float)], lambda a: f(f(f(a))))
    arr = numpy.random.randn(2, 2)
    numpy.testing.assert_allclose(interpret(expr, {a0: arr}), f1(f1(f1(arr))))


@with_interpret
def test_argmin(interpret):
    def argmin_trig(n: Array, a: Array) -> Array:
        def step(i: Array, j: Array, acc: tuple[Array, ...]) -> tuple[Array, Array]:
            v = (a[i] + j.to_float()).sin()
            return (v > acc[0]).where(v, acc[0]), (v > acc[0]).where(i, acc[1])

        return array(
            lambda j: fold((-float("inf"), 0), lambda i, acc: step(i, j, acc))[1],
            size=n,
        )

    (n0, a0), expr = function([scalar(int), vector(float)], argmin_trig)
    sample_n = 5
    sample_a = numpy.random.randn(sample_n)
    numpy.testing.assert_allclose(
        interpret(expr, {n0: sample_n, a0: sample_a}),
        numpy.sin(numpy.arange(sample_n)[:, numpy.newaxis] + sample_a).argmax(axis=1),
    )


@with_interpret
def test_matrix_power_times_vector(interpret):
    def matmat(a: Array, b: Array) -> Array:
        return array(lambda i, j: reduce_sum(lambda k: a[i, k] * b[k, j]))

    def matvec(a: Array, b: Array) -> Array:
        return array(lambda i: reduce_sum(lambda k: a[i, k] * b[k]))

    def pow_mult(m: Array, n: Array, v: Array) -> Array:
        k = m.dim(0)
        id_k = array(lambda i, j: Array(i == j).where(1.0, 0.0), size=(k, k))
        mn, vn = fold(
            (id_k, v), lambda t, mv: (matmat(mv[0], m), matvec(m, mv[1])), count=n
        )
        return matvec(mn, vn)

    (m0, n0, v0), expr = function([matrix(float), scalar(int), vector(float)], pow_mult)
    sample_n, sample_k = 2, 3
    sample_m = numpy.random.randn(sample_k, sample_k)
    sample_v = numpy.random.randn(sample_k)
    numpy.testing.assert_allclose(
        interpret(expr, {m0: sample_m, n0: sample_n, v0: sample_v}),
        numpy.linalg.matrix_power(sample_m, 2 * sample_n) @ sample_v,
    )


@with_interpret
def test_mean_smoothing(interpret):
    def smooth1(a: numpy.ndarray) -> numpy.ndarray:
        (n,) = a.shape
        r = numpy.zeros(n)
        r[1:-1] = (a[2:] + a[:-2]) / 2
        r[0] = a[0]
        r[-1] = a[-1]
        return r

    def smooth(a: Array) -> Array:
        n = a.dim(0)
        return array(
            lambda i: where(
                (i > 0) & (i + 1 < n),
                (a[i - 1] + a[i + 1]) / 2.0,
                a[i],
            )
        )

    (a0,), expr = function([vector(float)], smooth)
    sample_a = numpy.random.randn(10)

    numpy.testing.assert_allclose(interpret(expr, {a0: sample_a}), smooth1(sample_a))


@with_interpret
def test_diagonal(interpret):
    (a0,), expr = function([matrix(float)], lambda a: array(lambda i: a[i, i]))
    sample_a = numpy.random.randn(10, 10)

    numpy.testing.assert_allclose(interpret(expr, {a0: sample_a}), numpy.diag(sample_a))

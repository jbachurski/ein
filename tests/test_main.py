import functools
from typing import Any

import numpy
import pytest

from ein import (
    Array,
    Scalar,
    Vec,
    array,
    fold,
    function,
    interpret_with_naive,
    interpret_with_numpy,
    interpret_with_torch,
    matrix_type,
    scalar_type,
    vector_type,
    with_varargs,
    wrap,
)
from ein.frontend import std
from ein.frontend.std import concat, reduce_max, reduce_min, reduce_sum, where

with_backend = pytest.mark.parametrize("backend", ["naive", "numpy", "torch"])
with_interpret = pytest.mark.parametrize(
    "interpret",
    [interpret_with_naive, interpret_with_numpy, interpret_with_torch],
    ids=["naive", "numpy", "torch"],
)


def test_type_checks():
    with pytest.raises(TypeError):
        _ = with_varargs([scalar_type(int)], lambda n: 1 + array(lambda i: 0, size=n))  # type: ignore


def test_function():
    @function
    def outer(x: Vec[Scalar], y: Vec[Scalar]) -> Vec[Vec[Scalar]]:
        return array(lambda i, j: x[i] * y[j])

    a, b = numpy.random.randn(3), numpy.random.rand(4)

    numpy.testing.assert_allclose(outer(a, b), outer.numpy(a, b))
    numpy.testing.assert_allclose(outer(a, b), a[:, None] * b[None, :])


@with_interpret
def test_mul_grid(interpret):
    (n0,), grid_expr = with_varargs(
        [scalar_type(int)],
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
        n, k = a.size(0), a.size(1)
        _k, m = b.size(0), b.size(1)
        array_: Any = (
            array if with_bounds_inference else functools.partial(array, size=(n, m))
        )
        sum_: Any = (
            reduce_sum
            if with_bounds_inference
            else functools.partial(reduce_sum, count=k)
        )
        return array_(lambda i, j: sum_(lambda t: a[i, t] * b[t, j]))

    (a0, b0), matmul_expr = with_varargs(
        [matrix_type(float), matrix_type(float)], matmul
    )

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
    x, u, v = wrap(x_values), wrap(u_values), wrap(v_values)

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
        loss.eval(backend=backend), ((x_values - u_values @ v_values.T) ** 2).sum()
    )


@with_interpret
def test_max_minus_min(interpret):
    (a, b), expr = with_varargs(
        [vector_type(float), vector_type(float)],
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
        m = array(lambda i: where(a[i] > b[i], a[i], b[i]))
        return array(lambda i: where(m[i] > 0.0, 1, where(a[i] == b[i], 0, -1)))

    (a0, b0), sgn_max_expr = with_varargs(
        [vector_type(float), vector_type(float)], sgn_max
    )
    a_values, b_values = numpy.random.randn(16), numpy.random.randn(16)
    numpy.testing.assert_allclose(
        interpret(sgn_max_expr, {a0: a_values, b0: b_values}),
        numpy.sign(numpy.maximum(a_values, b_values)),
    )


@with_interpret
def test_reuse(interpret):
    a0 = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    a: Vec[Vec[Scalar]] = wrap(a0)
    b = array(lambda i, j: a[i, j] ** 2)
    c = array(lambda i, j: 2.0 * b[i, j] + b[i, j] ** 0.33)
    numpy.testing.assert_allclose(
        interpret(c.expr, {}), (2 * a0**2 + (a0**2) ** 0.33)
    )


@with_interpret
def test_fibonacci_fold(interpret):
    def fib(n: Array) -> Array:
        return fold(
            array(lambda i: wrap(0), size=n),
            lambda i, acc: array(
                lambda j: where(
                    i == j,
                    where(j == 0, 0, where(j == 1, 1, acc[i - 1] + acc[i - 2])),
                    acc[j],
                )
            ),
            count=n,
        )

    (n0,), fib_expr = with_varargs([scalar_type(int)], fib)
    numpy.testing.assert_allclose(
        interpret(fib_expr, {n0: numpy.array(8)}),
        [0, 1, 1, 2, 3, 5, 8, 13],
    )


@with_backend
def test_inline_interpret(backend):
    a = wrap(numpy.random.randn(20, 30))
    b = wrap(numpy.random.randn(30))
    numpy.testing.assert_allclose(
        array(lambda i: reduce_sum(lambda j: a[i, j] * b[j])).eval(backend=backend),
        a.eval(backend=backend) @ b.eval(backend=backend),
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

    (n0,), expr = with_varargs([scalar_type(int)], trial_division)
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

    (n0,), expr = with_varargs([scalar_type(int)], sieve)
    N = 30
    numpy.testing.assert_allclose(
        interpret(expr, {n0: numpy.array(N)}),
        _primes(N),
    )


@with_interpret
def test_double_transpose(interpret):
    (a0,), expr = with_varargs(
        [matrix_type(float)],
        lambda a: array(lambda i, j: array(lambda k, l: a[l, k])[j, i]),
    )
    arr = numpy.random.randn(3, 3)
    numpy.testing.assert_allclose(interpret(expr, {a0: arr}), arr)


@with_interpret
def test_symmetric_sum(interpret):
    def f1(x: numpy.ndarray) -> numpy.ndarray:
        return x + x.T

    def f(x: Array) -> Array:
        return array(lambda i, j: x[i, j] + x[j, i])

    (a0,), expr = with_varargs([matrix_type(float)], lambda a: f(f(f(a))))
    arr = numpy.random.randn(2, 2)
    numpy.testing.assert_allclose(interpret(expr, {a0: arr}), f1(f1(f1(arr))))


@with_interpret
def test_argmin(interpret):
    def argmin_trig(n: Array, a: Array) -> Array:
        def step(i: Array, j: Array, acc: tuple[Array, Array]) -> tuple[Array, Array]:
            v = (a[i] + j.to_float()).sin()
            return where(v > acc[0], (v, i), acc)

        return array(
            lambda j: fold((-float("inf"), 0), lambda i, acc: step(i, j, acc))[1],
            size=n,
        )

    (n0, a0), expr = with_varargs([scalar_type(int), vector_type(float)], argmin_trig)
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
        k = m.size(0)
        id_k = array(lambda i, j: where(i == j, 1.0, 0.0), size=(k, k))
        mn, vn = fold(
            (id_k, v), lambda t, mv: (matmat(mv[0], m), matvec(m, mv[1])), count=n
        )
        return matvec(mn, vn)

    (m0, n0, v0), expr = with_varargs(
        [matrix_type(float), scalar_type(int), vector_type(float)], pow_mult
    )
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
        n = a.size(0)
        return array(
            lambda i: where(
                (i > 0) & (i + 1 < n),
                (a[i - 1] + a[i + 1]) / 2.0,
                a[i],
            )
        )

    (a0,), expr = with_varargs([vector_type(float)], smooth)
    sample_a = numpy.random.randn(10)

    numpy.testing.assert_allclose(interpret(expr, {a0: sample_a}), smooth1(sample_a))


@with_interpret
def test_diagonal(interpret):
    (a0,), expr = with_varargs([matrix_type(float)], lambda a: array(lambda i: a[i, i]))
    sample_a = numpy.random.randn(10, 10)

    numpy.testing.assert_allclose(interpret(expr, {a0: sample_a}), numpy.diag(sample_a))


@with_backend
def test_clipped_shift(backend):
    n = 3
    a0 = list(range(n))
    a = wrap(numpy.array(a0))

    for shift in range(-n, n + 1):
        for low in range(n):
            for high in range(n):
                for d in (-1, 1):
                    y0 = [a0[min(max(i + shift, low), high)] for i in range(n + d)]
                    y = array(
                        lambda i: a[std.min(std.max(i + shift, low), high)], size=n + d
                    )
                    assert list(y.eval(backend=backend)) == y0


@with_backend
def test_summation(backend):
    n = 4
    a0 = numpy.random.randn(n, n)
    a = wrap(a0)

    numpy.testing.assert_allclose(
        array(lambda i: reduce_sum(lambda j: a[i, j])).eval(backend=backend),
        a0.sum(axis=1),
    )
    numpy.testing.assert_allclose(
        array(lambda j: reduce_sum(lambda i: a[i, j])).eval(backend=backend),
        a0.sum(axis=0),
    )
    numpy.testing.assert_allclose(
        reduce_sum(lambda i: reduce_sum(lambda j: a[i, j])).eval(backend=backend),
        a0.sum(axis=(0, 1)),
    )


@with_backend
def test_big_permutation(backend):
    a0 = numpy.random.randn(1, 2, 3, 4, 5, 6)
    b0 = numpy.transpose(a0, (5, 1, 4, 2, 3, 0))
    a, b = wrap(a0), wrap(b0)
    # None of the following should fail.
    # - An AssertEq failing in naive indicates the expressions seem wrong
    # - A NumPy broadcast error indicates some axial permutation code is off.

    c = array(lambda p, q, r, s, t, u: a[q, s, r, p, u, t] + b[t, s, u, r, p, q])  # type: ignore
    c.eval(backend=backend)
    d = array(lambda p, q, r, s, t, u: c[u, t, s, r, q, p])  # type: ignore
    d.eval(backend=backend)
    e = array(lambda p, q, r, s, t, u: c[u, t, s, r, q, p] + d[p, q, r, s, t, u])  # type: ignore
    e.eval(backend=backend)


@with_backend
def test_concat(backend):
    a0, b0 = numpy.random.randn(5), numpy.random.randn(7)
    a, b = wrap(a0), wrap(b0)
    numpy.testing.assert_allclose(
        concat(a, b).eval(backend=backend), list(a0) + list(b0)
    )


@with_backend
def test_aligned_concat(backend):
    def single(x):
        return array(lambda _: x, size=1)

    test = array(
        lambda i: array(
            lambda j: array(
                lambda k: concat(single(i + j), single(k + j)),
                size=4,
            ),
            size=3,
        ),
        size=2,
    )

    numpy.testing.assert_allclose(
        test.eval(backend=backend),
        numpy.array(
            [[[[i + j, k + j] for k in range(4)] for j in range(3)] for i in range(2)]
        ),
    )


@with_backend
def test_record_concat(backend):
    a = array(lambda i: (i, i**2), size=5)
    b = array(lambda i: (i, i**2), size=7)
    c = array(lambda i: concat(a, b)[i][1])

    numpy.testing.assert_allclose(
        c.eval(backend=backend), [i**2 for i in range(5)] + [i**2 for i in range(7)]
    )


@with_backend
def test_reduce_sum(backend):
    a0 = numpy.random.randn(15)
    a = wrap(a0)
    numpy.testing.assert_allclose(
        a.reduce(0.0, lambda x, y: x + y).eval(backend=backend), a0.sum()
    )


@with_backend
def test_reduce_abs_sum(backend):
    a0 = numpy.random.randn(12)
    numpy.testing.assert_allclose(
        wrap(a0).reduce(0.0, lambda x, y: abs(x) + abs(y)).eval(backend=backend),
        numpy.abs(a0).sum(),
    )


@with_backend
def test_reduce_concat(backend):
    a0 = numpy.random.randn(12)
    a = wrap(a0)
    aa = array(lambda i: array(lambda _: a[i], size=1)).reduce(
        array(lambda _: 0.0, size=0), lambda x, y: concat(x, y)
    )
    numpy.testing.assert_allclose(
        aa.eval(backend=backend),
        a0,
    )

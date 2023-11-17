import numpy
import pytest

from ein import (
    Array,
    Scalar,
    array,
    fold,
    function,
    interpret_with_naive,
    interpret_with_numpy,
    matrix,
    max,
    min,
    sum,
    vector,
)

with_backend = pytest.mark.parametrize("backend", ["naive", "numpy"])
with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_naive, interpret_with_numpy], ids=["naive", "numpy"]
)


def test_type_checks():
    with pytest.raises(TypeError):
        _ = function([Scalar(int)], lambda n: 1 + array[n](lambda i: 0))


@with_interpret
def test_mul_grid(interpret):
    (n0,), grid_expr = function(
        [Scalar(int)],
        lambda n: array[n, n](lambda i, j: (i.to_float() + 1.0) / (j.to_float() + 1.0)),
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
        array_ = array if with_bounds_inference else array[n, m]
        sum_ = sum if with_bounds_inference else sum[k]
        return array_(lambda i, j: sum_(lambda t: a[i, t] * b[t, j]))

    (a0, b0), matmul_expr = function([matrix(float), matrix(float)], matmul)
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

    def square(a):
        return a * a

    inner_sum = sum if with_bounds_inference else sum[k]
    outer_sum = sum if with_bounds_inference else sum[m, n]
    loss = outer_sum(
        lambda i, j: square(x[i, j] - inner_sum(lambda t: u[i, t] * v[j, t]))
    )

    numpy.testing.assert_allclose(
        loss.numpy(backend=backend), ((x_values - u_values @ v_values.T) ** 2).sum()
    )


@with_interpret
def test_max_minus_min(interpret):
    (a, b), expr = function(
        [vector(float), vector(float)],
        lambda a, b: max(lambda i: a[i] * b[i]) - min(lambda i: a[i] * b[i]),
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
def test_fibonacci_fold(interpret):
    def fib(n: Array) -> Array:
        return fold[n](
            array[n](lambda i: 0),
            lambda i, acc: array(
                lambda j: Array(i == j).where(
                    Array(j == 0).where(
                        0, Array(j == 1).where(1, acc[i - 1] + acc[i - 2])
                    ),
                    acc[j],
                )
            ),
        )

    (n0,), fib_expr = function([Scalar(int)], fib)
    numpy.testing.assert_allclose(
        interpret(fib_expr, {n0: numpy.array(8)}),
        [0, 1, 1, 2, 3, 5, 8, 13],
    )


@with_backend
def test_inline_interpret(backend):
    a = Array(numpy.random.randn(20, 30))
    b = Array(numpy.random.randn(30))
    numpy.testing.assert_allclose(
        array(lambda i: sum(lambda j: a[i, j] * b[j])).numpy(backend=backend),
        a.numpy(backend=backend) @ b.numpy(backend=backend),
    )


def _primes(n) -> list[bool]:
    return [i > 1 and all(i % d for d in range(2, i)) for i in range(n)]


@with_interpret
def test_trial_division_primes(interpret):
    def trial_division(n: Array) -> Array:
        return array[n](
            lambda i: (
                fold[n](
                    i > 1, lambda d, acc: (acc & ~((i % d == 0) & (1 < d) & (d < i)))
                )
            )
        )

    (n0,), expr = function([Scalar(int)], trial_division)
    N = 30
    numpy.testing.assert_allclose(
        interpret(expr, {n0: numpy.array(N)}),
        _primes(N),
    )


@with_interpret
def test_sieve_primes(interpret):
    def sieve(n: Array) -> Array:
        return fold[n](
            array[n](lambda i: i > 1),
            lambda d, siv: array[n](
                lambda i: (siv[i] & ~((d > 1) & (i >= d * d) & (i % d == 0)))
            ),
        )

    (n0,), expr = function([Scalar(int)], sieve)
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

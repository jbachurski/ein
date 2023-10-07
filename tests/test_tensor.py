import numpy
import pytest

from ein.calculus import Var, Variable, VarShape
from ein.interpret import interpret as interpret_with_baseline
from ein.tensor import Tensor, array, sum
from ein.to_numpy import interpret as interpret_with_numpy

with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_baseline, interpret_with_numpy], ids=["base", "numpy"]
)


@with_interpret
def test_mul_grid(interpret):
    n0 = Variable()
    n = Var(n0)
    # FIXME: This should have proper casting behaviour.
    grid = array[n, n](lambda i, j: (i + 1.0) / (j + 1.0))
    numpy.testing.assert_allclose(
        interpret(
            grid.expr,
            {n0: numpy.array(5)},
        ),
        numpy.array([[i / j for j in range(1, 6)] for i in range(1, 6)]),
    )


@with_interpret
def test_matmul(interpret):
    a0, b0 = Variable(), Variable()
    n, k = VarShape(a0, 0), VarShape(a0, 1)
    _k, m = VarShape(b0, 0), VarShape(b0, 1)
    a, b = Tensor(Var(a0)), Tensor(Var(b0))
    matmul = array[n, m](lambda i, j: sum[k](lambda t: a[i, t] * b[t, j]))
    first = numpy.array([[1, 2, 3], [4, 5, 6]])
    second = numpy.array([[1], [0], [-1]])
    numpy.testing.assert_allclose(
        interpret(
            matmul.expr,
            {
                a0: first,
                b0: second,
            },
        ),
        first @ second,
    )


@with_interpret
def test_uv_loss(interpret):
    m, k, n = 16, 8, 12
    x_values = numpy.random.randn(m, n)
    u_values = numpy.random.randn(m, k)
    v_values = numpy.random.randn(n, k)
    x, u, v = Tensor(x_values), Tensor(u_values), Tensor(v_values)

    def square(a):
        return a * a

    loss = sum[m, n](lambda i, j: square(x[i, j] - sum[k](lambda t: u[i, t] * v[j, t])))

    numpy.testing.assert_allclose(
        interpret(loss.expr, {}), ((x_values - u_values @ v_values.T) ** 2).sum()
    )
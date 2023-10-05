import numpy

from ein.calculus import Var, Variable, VarShape
from ein.interpret import interpret
from ein.tensor import Tensor, sum, tensor


def test_mul_grid():
    n0 = Variable()
    n = Var(n0)
    # FIXME: This should have proper casting behaviour.
    grid = tensor[n, n](lambda i, j: (i + 1.0) / (j + 1.0))
    numpy.testing.assert_allclose(
        interpret(
            grid.expr,
            {n0: numpy.array(5)},
        ),
        numpy.array([[i / j for j in range(1, 6)] for i in range(1, 6)]),
    )


def test_matmul():
    a0, b0 = Variable(), Variable()
    n, k = VarShape(a0, 0), VarShape(a0, 1)
    _k, m = VarShape(b0, 0), VarShape(b0, 1)
    a, b = Tensor(Var(a0)), Tensor(Var(b0))
    matmul = tensor[n, m](lambda i, j: sum[k](lambda t: a[i, t] * b[t, j]))
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

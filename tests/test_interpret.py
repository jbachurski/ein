import numpy

from ein.calculus import (
    Add,
    At,
    Const,
    Get,
    Index,
    Multiply,
    Negate,
    Reciprocal,
    Sum,
    Value,
    Var,
    Variable,
    Vec,
)
from ein.interpret import interpret


def test_basic_arithmetic():
    two = Const(Value(numpy.array([2.0])))
    numpy.testing.assert_allclose(interpret(two, {}), 2)
    four = Add((two, two))
    numpy.testing.assert_allclose(interpret(four, {}), 4)
    minus_three = Negate((Const(Value(numpy.array([3.0]))),))
    numpy.testing.assert_allclose(interpret(minus_three, {}), -3)
    minus_twelve = Multiply((four, minus_three))
    numpy.testing.assert_allclose(interpret(minus_twelve, {}), -12)
    minus_one_twelfth = Reciprocal((minus_twelve,))
    numpy.testing.assert_allclose(interpret(minus_one_twelfth, {}), -1 / 12)


def test_basic_indices():
    i = Index()
    j = Index()
    four = Const(Value(numpy.array(4)))
    three = Const(Value(numpy.array(3)))
    table = Vec(i, four, Vec(j, three, Multiply((At(i), At(j)))))
    numpy.testing.assert_allclose(
        interpret(table, {}),
        numpy.array([[i * j for j in range(3)] for i in range(4)]),
    )


def test_basic_variables():
    x = Variable()
    y = Variable()
    x_minus_y = Add((Var(x), Negate((Var(y),))))
    numpy.testing.assert_allclose(
        interpret(x_minus_y, {x: numpy.array([4.0]), y: numpy.array([3.0])}),
        1,
    )


def test_basic_reduction_and_get():
    n = 5
    i = Index()
    a = Const(Value(numpy.arange(n)))
    the_sum = Sum(i, Const(Value(numpy.array(n))), Get(a, At(i)))
    numpy.testing.assert_allclose(interpret(the_sum, {}), numpy.arange(n).sum())


def test_matmul():
    a, b = Variable(), Variable()
    n, m, k = Variable(), Variable(), Variable()
    i, j, t = Index(), Index(), Index()
    matmul = Vec(
        i,
        Var(n),
        Vec(
            j,
            Var(m),
            Sum(
                t,
                Var(k),
                Multiply(
                    (Get(Get(Var(a), At(i)), At(t)), Get(Get(Var(b), At(t)), At(j)))
                ),
            ),
        ),
    )
    first = numpy.array([[1, 2, 3], [4, 5, 6]])
    second = numpy.array([[1], [0], [-1]])
    numpy.testing.assert_allclose(
        interpret(
            matmul,
            {
                a: first,
                b: second,
                n: first.shape[0],
                m: second.shape[1],
                k: first.shape[1],
            },
        ),
        first @ second,
    )

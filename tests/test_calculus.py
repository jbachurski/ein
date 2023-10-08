import numpy
import pytest

from ein.calculus import (
    Add,
    At,
    Const,
    Dim,
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
from ein.interpret import interpret as interpret_with_baseline
from ein.to_numpy import interpret as interpret_with_numpy

with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_baseline, interpret_with_numpy], ids=["base", "numpy"]
)


@with_interpret
def test_basic_arithmetic(interpret):
    two = Const(Value(numpy.array(2.0)))
    numpy.testing.assert_allclose(interpret(two, {}), 2)
    four = Add((two, two))
    numpy.testing.assert_allclose(interpret(four, {}), 4)
    minus_three = Negate((Const(Value(numpy.array(3.0))),))
    numpy.testing.assert_allclose(interpret(minus_three, {}), -3)
    minus_twelve = Multiply((four, minus_three))
    numpy.testing.assert_allclose(interpret(minus_twelve, {}), -12)
    minus_one_twelfth = Reciprocal((minus_twelve,))
    numpy.testing.assert_allclose(interpret(minus_one_twelfth, {}), -1 / 12)


@with_interpret
def test_basic_indices(interpret):
    i = Index()
    j = Index()
    four = Const(Value(numpy.array(4)))
    three = Const(Value(numpy.array(3)))
    table = Vec(i, four, Vec(j, three, Multiply((At(i), At(j)))))
    numpy.testing.assert_allclose(
        interpret(table, {}),
        numpy.array([[i * j for j in range(3)] for i in range(4)]),
    )


@with_interpret
def test_basic_variables(interpret):
    x = Variable()
    y = Variable()
    x_minus_y = Add((Var(x), Negate((Var(y),))))
    numpy.testing.assert_allclose(
        interpret(x_minus_y, {x: numpy.array(4.0), y: numpy.array(3.0)}),
        1,
    )


@with_interpret
def test_basic_reduction_and_get(interpret):
    n = 5
    i = Index()
    a = Const(Value(numpy.arange(n)))
    the_sum = Sum(i, Const(Value(numpy.array(n))), Get(a, At(i)))
    numpy.testing.assert_allclose(interpret(the_sum, {}), numpy.arange(n).sum())


@with_interpret
def test_matmul(interpret):
    a, b = Variable(), Variable()
    i, j, t = Index(), Index(), Index()
    matmul = Vec(
        i,
        Dim(Var(a), 0),
        Vec(
            j,
            Dim(Var(b), 1),
            Sum(
                t,
                Dim(Var(a), 1),  # == Dim(Var(b), 0)
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
            },
        ),
        first @ second,
    )

import numpy
import pytest

from ein import interpret_with_arrays, interpret_with_naive
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
from ein.type_system import Type

with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_naive, interpret_with_arrays], ids=["base", "numpy"]
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
    x0, y0 = Variable(), Variable()
    x, y = Var(x0, Type(0)), Var(y0, Type(0))
    x_minus_y = Add((x, Negate((y,))))
    numpy.testing.assert_allclose(
        interpret(x_minus_y, {x0: numpy.array(4.0), y0: numpy.array(3.0)}),
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
    a0, b0 = Variable(), Variable()
    i, j, t = Index(), Index(), Index()
    a, b = Var(a0, Type(2)), Var(b0, Type(2))
    matmul = Vec(
        i,
        Dim(a, 0),
        Vec(
            j,
            Dim(b, 1),
            Sum(
                t,
                Dim(a, 1),  # == Dim(Var(b), 0)
                Multiply(
                    (
                        Get(Get(a, At(i)), At(t)),
                        Get(Get(b, At(t)), At(j)),
                    )
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
                a0: first,
                b0: second,
            },
        ),
        first @ second,
    )

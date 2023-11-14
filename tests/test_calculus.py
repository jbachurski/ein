import numpy
import pytest

from ein import Scalar, interpret_with_naive, interpret_with_numpy, matrix, vector
from ein.calculus import (
    Add,
    At,
    CastToFloat,
    Cons,
    Const,
    Dim,
    Expr,
    First,
    Fold,
    Get,
    Index,
    Less,
    Let,
    LogicalAnd,
    LogicalNot,
    Multiply,
    Negate,
    Pair,
    Reciprocal,
    Second,
    Sin,
    Value,
    Var,
    Variable,
    Vec,
    Where,
)

with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_naive, interpret_with_numpy], ids=["naive", "numpy"]
)


def fold_sum(index: Index, size: Expr, body: Expr):
    if not isinstance(body.type, Scalar):
        raise TypeError("Can only sum over scalars")
    dtype = body.type.kind
    init = Const(Value(numpy.array(0, dtype=dtype)))
    acc = Var(Variable(), Scalar(dtype))
    return Fold(index, size, acc, init, Add((acc, body)))


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
    x, y = Var(x0, Scalar(float)), Var(y0, Scalar(float))
    x_minus_y = Add((x, Negate((y,))))
    numpy.testing.assert_allclose(
        interpret(x_minus_y, {x0: numpy.array(4.0), y0: numpy.array(3.0)}),
        1,
    )


@with_interpret
def test_basic_let_bindings(interpret):
    a0, b0, x0, y0, z0, w0 = (Variable() for _ in range(6))
    a, b, x, y, z, w = (Var(v, Scalar(float)) for v in (a0, b0, x0, y0, z0, w0))
    az = Let(((z0, a),), z)
    bw = Let(((w0, b),), w)
    expr = Let(((x0, Add((az, bw))),), Let(((y0, Multiply((x, b))),), Add((x, y))))
    numpy.testing.assert_allclose(
        interpret(expr, {a0: numpy.array(4.0), b0: numpy.array(3.0)}),
        (4 + 3) + ((4 + 3) * 3),
    )


@with_interpret
def test_basic_reduction_and_get(interpret):
    n = 5
    i = Index()
    a = Const(Value(numpy.arange(n)))
    the_sum = fold_sum(i, Const(Value(numpy.array(n))), Get(a, At(i)))
    numpy.testing.assert_allclose(interpret(the_sum, {}), numpy.arange(n).sum())


@with_interpret
def test_indexing_with_shift(interpret):
    i = Index()
    a = Var(Variable(), vector(int))
    drop_last = Vec(
        i, Add((Dim(a, 0), Negate((Const(Value(numpy.array(1))),)))), Get(a, At(i))
    )
    numpy.testing.assert_allclose(
        interpret(drop_last, {a.var: numpy.arange(5)}), numpy.arange(4)
    )


@with_interpret
def test_basic_pairs(interpret):
    if interpret == interpret_with_numpy:
        pytest.skip()
    av, bv = numpy.array([1, 2, 3]), numpy.array([-1, 1])
    a, b = Const(Value(av)), Const(Value(bv))
    p = Cons(a, Cons(Cons(a, b), b))
    numpy.testing.assert_allclose(interpret(First(p), {}), av)
    numpy.testing.assert_allclose(interpret(Second(Second(p)), {}), bv)
    numpy.testing.assert_allclose(interpret(First(First(Second(p))), {}), av)
    numpy.testing.assert_allclose(interpret(Second(First(Second(p))), {}), bv)


@with_interpret
def test_matmul(interpret):
    a0, b0 = Variable(), Variable()
    i, j, t = Index(), Index(), Index()
    a, b = Var(a0, matrix(float)), Var(b0, matrix(float))
    matmul = Vec(
        i,
        Dim(a, 0),
        Vec(
            j,
            Dim(b, 1),
            fold_sum(
                t,
                Dim(a, 1),  # == Dim(b, 0)
                Multiply(
                    (
                        Get(Get(a, At(i)), At(t)),
                        Get(Get(b, At(t)), At(j)),
                    )
                ),
            ),
        ),
    )
    first = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    second = numpy.array([[1], [0], [-1]], dtype=float)
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


@with_interpret
def test_power_fold(interpret):
    a0, n0, x0 = Variable(), Variable(), Variable()
    i = Index()
    a, n, x = Var(a0, Scalar(float)), Var(n0, Scalar(int)), Var(x0, Scalar(float))
    power_expr = Fold(i, n, x, Const(Value(numpy.array(1.0))), Multiply((x, a)))
    numpy.testing.assert_allclose(
        interpret(power_expr, {a0: numpy.array(3), n0: numpy.array(7)}), 3**7
    )


@with_interpret
def test_fibonacci_vector_fold(interpret):
    fib0, n0 = Variable(), Variable()
    i, j, j0 = Index(), Index(), Index()
    fib, n = Var(fib0, vector(int)), Var(n0, Scalar(int))
    zero = Const(Value(numpy.array(0)))
    one = Const(Value(numpy.array(1)))
    zeros = Vec(j0, n, Negate((one,)))

    def eq(x, y):
        return LogicalAnd((LogicalNot((Less((x, y)),)), LogicalNot((Less((y, x)),))))

    i1 = Add((At(i), Negate((one,))))
    i2 = Add((i1, Negate((one,))))
    fib_expr = Fold(
        i,
        n,
        fib,
        zeros,
        Vec(
            j,
            n,
            Where(
                (
                    eq(At(i), At(j)),
                    Where(
                        (
                            eq(At(i), zero),
                            zero,
                            Where(
                                (eq(At(i), one), one, Add((Get(fib, i1), Get(fib, i2))))
                            ),
                        )
                    ),
                    Get(fib, At(j)),
                )
            ),
        ),
    )
    numpy.testing.assert_allclose(
        interpret(fib_expr, {n0: numpy.array(8)}), [0, 1, 1, 2, 3, 5, 8, 13]
    )


@with_interpret
def test_argmin(interpret):
    if interpret == interpret_with_numpy:
        pytest.skip()
    i, j = Index(), Index()
    a = Var(Variable(), vector(float))
    n = Var(Variable(), Scalar(int))
    r = Var(Variable(), Pair(Scalar(float), Scalar(int)))
    a_at_i_j = Sin((Add((Get(a, At(i)), CastToFloat((At(j),)))),))
    cond_i_j = Less((First(r), a_at_i_j))
    argmin_expr = Fold(
        i,
        n,
        r,
        Cons(Const(Value(-float("inf"))), Const(Value(0))),
        Cons(
            Where((cond_i_j, a_at_i_j, First(r))), Where((cond_i_j, At(i), Second(r)))
        ),
    )
    expr = Vec(j, n, Second(argmin_expr))

    sample_n = 5
    sample_a = numpy.random.randn(sample_n)
    numpy.testing.assert_allclose(
        interpret(expr, {a.var: sample_a, n.var: sample_n}),
        numpy.sin(numpy.arange(sample_n)[:, numpy.newaxis] + sample_a).argmax(axis=1),
    )

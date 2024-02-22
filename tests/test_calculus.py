import numpy
import pytest

from ein import (
    interpret_with_naive,
    interpret_with_numpy,
    interpret_with_torch,
    matrix_type,
    scalar_type,
    vector_type,
)
from ein.calculus import (
    Add,
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
    Variable,
    Vec,
    Where,
    at,
    variable,
)
from ein.type_system import Scalar
from ein.value import Value

with_interpret = pytest.mark.parametrize(
    "interpret",
    [interpret_with_naive, interpret_with_numpy, interpret_with_torch],
    ids=["naive", "numpy", "torch"],
)


def fold_sum(counter: Variable, size: Expr, body: Expr):
    assert isinstance(body.type, Scalar)
    dtype = body.type.kind
    init = Const(Value(numpy.array(0, dtype=dtype)))
    var = Variable()
    acc = variable(var, scalar_type(dtype))
    return Fold(counter, size, var, init, Add((acc, body)))


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
    table = Vec(i, four, Vec(j, three, Multiply((at(i), at(j)))))
    numpy.testing.assert_allclose(
        interpret(table, {}),
        numpy.array([[i * j for j in range(3)] for i in range(4)]),
    )


@with_interpret
def test_basic_variables(interpret):
    x0, y0 = Variable(), Variable()
    x, y = variable(x0, scalar_type(float)), variable(y0, scalar_type(float))
    x_minus_y = Add((x, Negate((y,))))
    numpy.testing.assert_allclose(
        interpret(x_minus_y, {x0: numpy.array(4.0), y0: numpy.array(3.0)}),
        1,
    )


@with_interpret
def test_basic_let_bindings(interpret):
    a, b, x, y, z, w = (variable(Variable(), scalar_type(float)) for _ in range(6))
    az = Let(z.var, a, z)
    bw = Let(w.var, b, w)
    expr = Let(x.var, Add((az, bw)), Let(y.var, Multiply((x, b)), Add((x, y))))
    numpy.testing.assert_allclose(
        interpret(expr, {a.var: numpy.array(4.0), b.var: numpy.array(3.0)}),
        (4 + 3) + ((4 + 3) * 3),
    )


@with_interpret
def test_basic_reduction_and_get(interpret):
    n = 5
    i = variable(Variable(), scalar_type(int))
    a = Const(Value(numpy.arange(n)))
    the_sum = fold_sum(i.var, Const(Value(numpy.array(n))), Get(a, i))
    numpy.testing.assert_allclose(interpret(the_sum, {}), numpy.arange(n).sum())


@with_interpret
def test_indexing_with_shift(interpret):
    i = Index()
    a = variable(Variable(), vector_type(int))
    drop_last = Vec(
        i, Add((Dim(a, 0), Negate((Const(Value(numpy.array(1))),)))), Get(a, at(i))
    )
    numpy.testing.assert_allclose(
        interpret(drop_last, {a.var: numpy.arange(5)}), numpy.arange(4)
    )


@with_interpret
def test_basic_pairs(interpret):
    av, bv = numpy.array([1, 2, 3]), numpy.array([-1, 1])
    a, b = Const(Value(av)), Const(Value(bv))
    p = Cons(a, Cons(Cons(a, b), b))
    numpy.testing.assert_allclose(interpret(First(p), {}), av)
    numpy.testing.assert_allclose(interpret(Second(Second(p)), {}), bv)
    numpy.testing.assert_allclose(interpret(First(First(Second(p))), {}), av)
    numpy.testing.assert_allclose(interpret(Second(First(Second(p))), {}), bv)


@with_interpret
def test_repeated_squaring(interpret):
    x = variable(Variable(), scalar_type(float))
    k = 20
    init_expr = lambda: Add(  # noqa
        (
            Const(Value(numpy.array(1.0))),
            Multiply((x, Const(Value(numpy.array(1 / 2**k))))),
        )
    )
    expr = Multiply((init_expr(), init_expr()))
    for _ in range(k - 1):
        expr = Multiply((expr, expr))
    x0 = 1 + 1e-9
    numpy.testing.assert_allclose(
        interpret(expr, {x.var: x0}), (1 + x0 / 2**k) ** (2**k)
    )


@with_interpret
def test_repeated_indexing(interpret):
    init = numpy.arange(10)
    expr: Expr = Const(Value(init))
    k = 5
    for _ in range(k):
        i = Index()
        expr = Vec(i, Const(Value(init.shape[0])), Get(expr, at(i)))

    numpy.testing.assert_allclose(interpret(expr, {}), init)


@with_interpret
def test_matmul(interpret):
    a0, b0 = Variable(), Variable()
    i, j = Index(), Index()
    t = variable(Variable(), scalar_type(int))
    a, b = variable(a0, matrix_type(float)), variable(b0, matrix_type(float))
    matmul = Vec(
        i,
        Dim(a, 0),
        Vec(
            j,
            Dim(b, 1),
            fold_sum(
                t.var,
                Dim(a, 1),  # == Dim(b, 0)
                Multiply(
                    (
                        Get(Get(a, at(i)), t),
                        Get(Get(b, t), at(j)),
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
    a, n, x = (
        variable(a0, scalar_type(float)),
        variable(n0, scalar_type(int)),
        variable(x0, scalar_type(float)),
    )
    power_expr = Fold(
        Variable(), n, x.var, Const(Value(numpy.array(1.0))), Multiply((x, a))
    )
    numpy.testing.assert_allclose(
        interpret(power_expr, {a0: numpy.array(3), n0: numpy.array(7)}), 3**7
    )


@with_interpret
def test_fibonacci_vector_fold(interpret):
    fib0, n0 = Variable(), Variable()
    i = variable(Variable(), scalar_type(int))
    j, j0 = Index(), Index()
    fib, n = variable(fib0, vector_type(int)), variable(n0, scalar_type(int))
    zero = Const(Value(numpy.array(0)))
    one = Const(Value(numpy.array(1)))
    zeros = Vec(j0, n, Negate((one,)))

    def eq(x, y):
        return LogicalAnd((LogicalNot((Less((x, y)),)), LogicalNot((Less((y, x)),))))

    i1 = Add((i, Negate((one,))))
    i2 = Add((i1, Negate((one,))))
    fib_expr = Fold(
        i.var,
        n,
        fib.var,
        zeros,
        Vec(
            j,
            n,
            Where(
                (
                    eq(i, at(j)),
                    Where(
                        (
                            eq(i, zero),
                            zero,
                            Where((eq(i, one), one, Add((Get(fib, i1), Get(fib, i2))))),
                        )
                    ),
                    Get(fib, at(j)),
                )
            ),
        ),
    )
    numpy.testing.assert_allclose(
        interpret(fib_expr, {n0: numpy.array(8)}), [0, 1, 1, 2, 3, 5, 8, 13]
    )


@with_interpret
def test_argmin(interpret):
    i = variable(Variable(), scalar_type(int))
    j = Index()
    a = variable(Variable(), vector_type(float))
    n = variable(Variable(), scalar_type(int))
    r = variable(Variable(), Pair(scalar_type(float), scalar_type(int)))
    a_at_i_j = Sin((Add((Get(a, i), CastToFloat((at(j),)))),))
    cond_i_j = Less((First(r), a_at_i_j))
    argmin_expr = Fold(
        i.var,
        n,
        r.var,
        Cons(Const(Value(-float("inf"))), Const(Value(0))),
        Cons(Where((cond_i_j, a_at_i_j, First(r))), Where((cond_i_j, i, Second(r)))),
    )
    expr = Vec(j, n, Second(argmin_expr))

    sample_n = 5
    sample_a = numpy.random.randn(sample_n)
    numpy.testing.assert_allclose(
        interpret(expr, {a.var: sample_a, n.var: sample_n}),
        numpy.sin(numpy.arange(sample_n)[:, numpy.newaxis] + sample_a).argmax(axis=1),
    )

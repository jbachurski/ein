import numpy
import pytest
import scipy

from ein import (
    Tensor,
    Type,
    array,
    function,
    interpret_with_arrays,
    interpret_with_naive,
    max,
    min,
    of,
    sum,
)

with_interpret = pytest.mark.parametrize(
    "interpret", [interpret_with_naive, interpret_with_arrays], ids=["base", "numpy"]
)


@with_interpret
def test_mul_grid(interpret):
    # FIXME: This should have proper casting behaviour.
    (n0,), grid_expr = function(
        lambda n=of(Type(0)): array[n, n](lambda i, j: (i + 1.0) / (j + 1.0))
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
    def matmul(a=of(Type(2)), b=of(Type(2))):
        n, k = a.dim(0), a.dim(1)
        _k, m = b.dim(0), b.dim(1)
        array_ = array if with_bounds_inference else array[n, m]
        sum_ = sum if with_bounds_inference else sum[k]
        return array_(lambda i, j: sum_(lambda t: a[i, t] * b[t, j]))

    (a0, b0), matmul_expr = function(matmul)
    first = numpy.array([[1, 2, 3], [4, 5, 6]])
    second = numpy.array([[1], [0], [-1]])
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


@with_interpret
@pytest.mark.parametrize(
    "with_bounds_inference", [False, True], ids=["give-sizes", "infer-sizes"]
)
def test_uv_loss(interpret, with_bounds_inference):
    m, k, n = 16, 8, 12
    x_values = numpy.random.randn(m, n)
    u_values = numpy.random.randn(m, k)
    v_values = numpy.random.randn(n, k)
    x, u, v = Tensor(x_values), Tensor(u_values), Tensor(v_values)

    def square(a):
        return a * a

    inner_sum = sum if with_bounds_inference else sum[k]
    outer_sum = sum if with_bounds_inference else sum[m, n]
    loss = outer_sum(
        lambda i, j: square(x[i, j] - inner_sum(lambda t: u[i, t] * v[j, t]))
    )

    numpy.testing.assert_allclose(
        interpret(loss.expr, {}), ((x_values - u_values @ v_values.T) ** 2).sum()
    )


@with_interpret
def test_max_minus_min(interpret):
    (a, b), expr = function(
        lambda a=of(Type(1)), b=of(Type(1)): max(lambda i: a[i] * b[i])
        - min(lambda i: a[i] * b[i])
    )
    a_values, b_values = numpy.random.randn(256), numpy.random.randn(256)
    numpy.testing.assert_allclose(
        interpret(expr, {a: a_values, b: b_values}),
        (a_values * b_values).max() - (a_values * b_values).min(),
    )


@with_interpret
def test_switches(interpret):
    def sgn(a: Tensor = of(Type(1)), b: Tensor = of(Type(1))) -> Tensor:
        return array(
            lambda i: ((a[i] > b[i]).where(a[i], b[i]) > 0).where(
                1, ((a[i] == b[i]) | False).where(0, -1)
            )
        )

    (a0, b0), sgn_expr = function(sgn)
    a_values, b_values = numpy.random.randn(256), numpy.random.randn(256)
    numpy.testing.assert_allclose(
        interpret(sgn_expr, {a0: a_values, b0: b_values}),
        numpy.sign(numpy.maximum(a_values, b_values)),
    )


def test_attention():
    # Wh, ..., bM are parameters.
    # w, br, Y, ht, rt1 are arguments.
    vector, matrix = Type(rank=1), Type(rank=2)
    batched_vector, batched_matrix = matrix, Type(rank=3)

    def ein_attention_batched(
        Wh=of(matrix),
        Wr=of(matrix),
        WY=of(matrix),
        Wt=of(matrix),
        bM=of(vector),
        w=of(vector),
        br=of(vector),
        batched_Y=of(batched_matrix),
        batched_ht=of(batched_vector),
        batched_rt1=of(batched_vector),
    ) -> Tensor:
        def softmax(v):
            return array(lambda i: v[i].exp() / sum(lambda j: v[j].exp()))

        def ein_attention(Y, ht, rt1):
            Mt = array(
                lambda s, l: (
                    sum(lambda k: Y[s, k] * WY[k, l])
                    + sum(lambda k: ht[k] * Wh[k, l] + rt1[k] * Wr[k, l])
                    + bM[l]
                ).tanh()
            )
            at = softmax(array(lambda s: sum(lambda l: Mt[s, l] * w[l])))

            rt = array(
                lambda l: (
                    sum(lambda s: Y[s, l] * at[s])
                    + (sum(lambda k: rt1[k] * Wt[k, l]) + br[l]).tanh()
                )
            )

            return rt

        return array(
            lambda i: ein_attention(batched_Y[i], batched_ht[i], batched_rt1[i])
        )

    def numpy_attention(Wh, Wr, WY, Wt, bM, w, br, Y, ht, rt1):
        # -- [batch_size x hidden_dimension]
        tmp: numpy.ndarray = numpy.einsum("ik,kl->il", ht, Wh) + numpy.einsum(
            "ik,kl->il", rt1, Wr
        )

        Mt = numpy.tanh(
            numpy.einsum("ijk,kl->ijl", Y, WY) + numpy.expand_dims(tmp, axis=1) + bM
        )
        # -- [batch_size x sequence_length]
        at = scipy.special.softmax(numpy.einsum("ijk,k->ij", Mt, w), axis=-1)

        # -- [batch_size x hidden_dimension]
        rt = numpy.einsum("ijk,ij->ik", Y, at) + numpy.tanh(
            numpy.einsum("ij,jk->ik", rt1, Wt) + br
        )

        # # -- [batch_size x hidden_dimension], [batch_size x sequence_dimension]
        # return rt, at
        return rt

    arg_vars, attention_expr = function(ein_attention_batched)

    hidden = 17
    a_Wh, a_Wr, a_Wy, a_Wt = (numpy.random.randn(hidden, hidden) for _ in range(4))
    a_bM, a_w, a_br = (numpy.random.randn(hidden) for _ in range(3))
    batch, tokens = 4, 7
    a_Y = numpy.random.randn(batch, tokens, hidden)
    a_ht, a_rt1 = numpy.random.randn(batch, hidden), numpy.random.randn(batch, hidden)
    arg_arrays = (a_Wh, a_Wr, a_Wy, a_Wt, a_bM, a_w, a_br, a_Y, a_ht, a_rt1)

    numpy.testing.assert_allclose(
        interpret_with_arrays(
            attention_expr, {v: a for v, a in zip(arg_vars, arg_arrays)}
        ),
        numpy_attention(*arg_arrays),
    )

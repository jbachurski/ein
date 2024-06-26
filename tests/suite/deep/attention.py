import numpy
import scipy

from ein import Array, array, matrix_type, vector_type
from ein.frontend.std import fold_sum
from ein.phi.type_system import Vector as VectorType

from ..case import Case


class Attention(Case):
    ein_types = (
        [matrix_type(float)] * 4
        + [vector_type(float)] * 3
        + [
            VectorType(matrix_type(float)),
            VectorType(vector_type(float)),
            VectorType(vector_type(float)),
        ]
    )

    @staticmethod
    def ein_single(Wh, Wr, WY, Wt, bM, w, br, Y, ht, rt1):
        def softmax(v):
            return array(lambda i: v[i].exp() / fold_sum(lambda j: v[j].exp()))

        Mt = array(
            lambda s, l: (
                fold_sum(lambda k: Y[s, k] * WY[k, l])
                + fold_sum(lambda k: ht[k] * Wh[k, l])
                + fold_sum(lambda k: rt1[k] * Wr[k, l])
                + bM[l]
            ).tanh()
        )
        at = softmax(array(lambda s: fold_sum(lambda l: Mt[s, l] * w[l])))

        rt = array(
            lambda l: (
                fold_sum(lambda s: Y[s, l] * at[s])
                + (fold_sum(lambda k: rt1[k] * Wt[k, l]) + br[l]).tanh()
            )
        )

        return rt

    @staticmethod
    def in_ein(*args: Array) -> Array:
        Wh, Wr, WY, Wt, bM, w, br, Y, ht, rt1 = args
        return array(
            lambda i: Attention.ein_single(
                Wh, Wr, WY, Wt, bM, w, br, Y[i], ht[i], rt1[i]
            )
        )

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        Wh, Wr, WY, Wt, bM, w, br, Y, ht, rt1 = args
        # Based on https://rockt.github.io/2018/04/30/einsum
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

    @staticmethod
    def sample(
        hidden: int = 10, batch: int = 10, tokens: int = 10
    ) -> tuple[numpy.ndarray, ...]:
        Wh, Wr, Wy, Wt = (numpy.random.randn(hidden, hidden) for _ in range(4))
        bM, w, br = (numpy.random.randn(hidden) for _ in range(3))
        Y = numpy.random.randn(batch, tokens, hidden)
        ht = numpy.random.randn(batch, hidden)
        rt1 = numpy.random.randn(batch, hidden)
        return Wh, Wr, Wy, Wt, bM, w, br, Y, ht, rt1

from typing import TypeVar

import numpy
import scipy

from ein import Array, Scalar, Vec, array, ndarray_type
from ein.frontend.std import fold_max, fold_sum, where

from ..case import Case

T = TypeVar("T")


class GAT(Case):
    ein_types = list(map(lambda k: ndarray_type(k, float), (3, 4, 3, 3, 4, 2)))

    @staticmethod
    def in_ein(*args: Array) -> Array:
        adj: Vec[Vec[Vec[Scalar]]]
        vals: Vec[Vec[Vec[Vec[Scalar]]]]
        s: Vec[Vec[Vec[Scalar]]]
        t: Vec[Vec[Vec[Scalar]]]
        e: Vec[Vec[Vec[Vec[Scalar]]]]
        g: Vec[Vec[Scalar]]
        adj, vals, s, t, e, g = args

        def softmax(x: Vec[T]) -> Vec[T]:
            x_max = fold_max(lambda i: x[i])
            x1 = array(lambda i: (x[i] - x_max).exp())
            x1_sum = fold_sum(lambda i: x1[i])
            return array(lambda i: x1[i] / x1_sum)

        def leaky_relu(x: Scalar) -> Scalar:
            return where(x < 0.0, 0.01 * x, x)

        bias = array(lambda b, u, v: (adj[b, u, v] - 1.0) * 1e9)
        logits = array(
            lambda b, h, u, v: s[b, u, h] + t[b, v, h] + e[b, u, v, h] + g[b, h]
        )
        coefs = array(
            lambda b, h, u: softmax(
                array(lambda v: leaky_relu(logits[b, h, u, v]) + bias[b, u, v])
            )
        )
        return array(
            lambda b, u, h, f: fold_sum(lambda v: coefs[b, h, u, v] * vals[b, v, h, f])
        )

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        # Based on https://github.com/google-deepmind/clrs/blob/8697f51663bd77548f4b3108816c84d163883361/clrs/_src/processors.py#L99
        adj, vals, s, t, e, g = args

        def leaky_relu(x):
            return numpy.where(x < 0, 0.01 * x, x)

        batches, vertices, heads = s.shape
        bias = (adj - 1.0) * 1e9
        bias = numpy.tile(bias[..., None], (1, 1, 1, heads))  # [B, N, N, H]
        bias = numpy.transpose(bias, (0, 3, 1, 2))  # [B, H, N, N]
        vals = numpy.transpose(vals, (0, 2, 1, 3))  # [B, H, N, F]
        logits = (
            numpy.transpose(s[..., numpy.newaxis], (0, 2, 1, 3))  # + [B, H, N, 1]
            + numpy.transpose(t[..., numpy.newaxis], (0, 2, 3, 1))  # + [B, H, 1, N]
            + numpy.transpose(e, (0, 3, 1, 2))  # + [B, H, N, N]
            + g[..., numpy.newaxis, numpy.newaxis]  # + [B, H, 1, 1]
        )  # = [B, H, N, N]
        coefs = scipy.special.softmax(leaky_relu(logits) + bias, axis=-1)
        ret = numpy.matmul(coefs, vals)  # [B, H, N, F]
        ret = numpy.transpose(ret, (0, 2, 1, 3))  # [B, N, H, F]

        return ret

    @staticmethod
    def sample(
        b: int = 2, h: int = 3, f: int = 5, n: int = 7
    ) -> tuple[numpy.ndarray, ...]:
        adj = numpy.random.randint(0, 1, size=(b, n, n)).astype(float)
        vals = numpy.random.randn(b, n, h, f)
        s = numpy.random.randn(b, n, h)
        t = numpy.random.randn(b, n, h)
        e = numpy.random.randn(b, n, n, h)
        g = numpy.random.randn(b, h)
        return adj, vals, s, t, e, g

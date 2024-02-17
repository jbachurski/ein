from math import sin, tau

import numpy
from numpy import newaxis

from ein import Array, array, vector_type
from ein.frontend.std import reduce_sum

from ..case import Case


class MriQ(Case):
    ein_types = [vector_type(float)] * 8

    @staticmethod
    def in_ein(*args: Array) -> Array:
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = array(lambda i: phi_r[i] * phi_r[i] + phi_i[i] * phi_i[i])
        angles = array(
            lambda i: (
                tau * reduce_sum(lambda j: kx[i] * x[j] + ky[i] * y[j] + kz[i] * z[j])
            )
        )
        return array(lambda i: (mag[i] * angles[i]).sin())

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = phi_r * phi_r + phi_i * phi_i
        angles = tau * (
            kx[newaxis, :] * x[:, newaxis]
            + ky[newaxis, :] * y[:, newaxis]
            + kz[newaxis, :] * z[:, newaxis]
        ).sum(axis=0)
        return numpy.sin(mag * angles)

    @staticmethod
    def in_numpy_frugal(*args):
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = phi_r * phi_r + phi_i * phi_i
        (k,), (n,) = kx.shape, x.shape
        angles = numpy.zeros(k)
        for i in range(n):
            angles = angles + kx * x[i] + ky * y[i] + kz * z[i]
        angles = tau * angles
        return numpy.sin(mag * angles)

    @staticmethod
    def in_numpy_einsum(*args):
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = phi_r * phi_r + phi_i * phi_i
        angles = tau * (
            numpy.einsum("i,j->i", kx, x)
            + numpy.einsum("i,j->i", ky, y)
            + numpy.einsum("i,j->i", kz, z)
        )
        return numpy.sin(mag * angles)

    @staticmethod
    def in_numpy_smart(*args):
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = phi_r * phi_r + phi_i * phi_i
        angles = tau * (kx * x.sum() + ky * y.sum() + kz * z.sum())
        return numpy.sin(mag * angles)

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        kxs, kys, kzs, xs, ys, zs, phi_rs, phi_is = args
        mags = [phi_r * phi_r + phi_i * phi_i for phi_r, phi_i in zip(phi_rs, phi_is)]
        angles = [
            tau * sum(kx * x + ky * y + kz * z for x, y, z in zip(xs, ys, zs))
            for kx, ky, kz in zip(kxs, kys, kzs)
        ]
        return numpy.array(
            [sin(phi_mag * angle) for phi_mag, angle in zip(mags, angles)]
        )

    @staticmethod
    def sample(k: int = 5, x: int = 10) -> tuple[numpy.ndarray, ...]:
        kx, ky, kz, phi_r, phi_i = (numpy.random.randn(k) for _ in range(5))
        x, y, z = (numpy.random.randn(x) for _ in range(3))
        return kx, ky, kz, x, y, z, phi_r, phi_i

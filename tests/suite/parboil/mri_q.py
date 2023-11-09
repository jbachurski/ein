from math import sin, tau

import numpy
from numpy import newaxis

from ein import Array, array
from ein import sum as fold_sum
from ein import vector

from ..case import Case


class MriQ(Case):
    ein_types = [vector(float)] * 8

    @staticmethod
    def in_ein(*args: Array) -> Array:
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        mag = array(lambda i: phi_r[i] * phi_r[i] + phi_i[i] * phi_i[i])
        angles = array(
            lambda i: tau
            * fold_sum(lambda j: kx[i] * x[j] + ky[i] * y[j] + kz[i] * z[j])
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

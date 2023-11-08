from math import tau

import numpy
from numpy import newaxis

from ein import Array, array, sum, vector

from ..case import Case


class MriQ(Case):
    ein_types = [vector(float)] * 8

    @staticmethod
    def in_ein(*args: Array) -> Array:
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        phi_mag = array(lambda i: phi_r[i] * phi_r[i] + phi_i[i] * phi_i[i])
        angles = array(
            lambda i: sum(lambda j: tau * (kx[i] * x[j] + ky[i] * y[j] + kz[i] * z[j]))
        )
        return array(lambda i: (phi_mag[i] * angles[i]).sin())

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        kx, ky, kz, x, y, z, phi_r, phi_i = args
        phi_mag = phi_r * phi_r + phi_i * phi_i
        angles = tau * (
            kx[newaxis, :] * x[:, newaxis]
            + ky[newaxis, :] * y[:, newaxis]
            + kz[newaxis, :] * z[:, newaxis]
        ).sum(axis=0)
        return numpy.sin(phi_mag * angles)

    @staticmethod
    def sample(k: int = 5, x: int = 10) -> tuple[numpy.ndarray, ...]:
        kx, ky, kz, phi_r, phi_i = (numpy.random.randn(k) for _ in range(5))
        x, y, z = (numpy.random.randn(x) for _ in range(3))
        return kx, ky, kz, x, y, z, phi_r, phi_i

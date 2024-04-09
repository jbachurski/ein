from typing import Any

import numpy

from ein import Array, array, fold, matrix_type, scalar_type
from ein.frontend.std import max_monoid, min_monoid

from ..case import Case

MAX_PD = 3000000.0
PRECISION = 0.001
SPEC_HEAT_SI = 1750000.0
K_SI = 100.0

FACTOR_CHIP = 0.5

T_CHIP = 0.0005
CHIP_HEIGHT = 0.016
CHIP_WIDTH = 0.016

T_AMB = 80.0


def params(r, c):
    grid_height = CHIP_HEIGHT / r
    grid_width = CHIP_WIDTH / c
    cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * grid_width * grid_height
    Rx = grid_width / (2.0 * K_SI * T_CHIP * grid_height)
    Ry = grid_height / (2.0 * K_SI * T_CHIP * grid_width)
    Rz = T_CHIP / (K_SI * grid_height * grid_width)
    max_slope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI)
    step = PRECISION / max_slope
    return Rx, Ry, Rz, step, cap


class Hotspot(Case):
    ein_types = [scalar_type(int), matrix_type(float), matrix_type(float)]

    @staticmethod
    def in_ein(*args: Array) -> Array:
        it, init_ts, p = args
        r, c = init_ts.size(0), init_ts.size(1)
        Rx, Ry, Rz, step, cap = params(r.float(), c.float())
        min: Any = min_monoid.concat
        max: Any = max_monoid.concat

        def step_t(ts, y, x):
            dx = (ts[y, min(x + 1, c - 1)] + ts[y, max(x - 1, 0)] - 2.0 * ts[y, x]) / Rx
            dy = (ts[min(y + 1, r - 1), x] + ts[max(y - 1, 0), x] - 2.0 * ts[y, x]) / Ry
            d = (step / cap) * (p[y, x] + dx + dy + (T_AMB - ts[y, x]) / Rz)
            return ts[y, x] + d

        return fold(
            init_ts, lambda _, ts: array(lambda y, x: step_t(ts, y, x)), count=it
        )

    @staticmethod
    def in_numpy(*args: numpy.ndarray) -> numpy.ndarray:
        it_, ts, p = args
        it = int(it_)
        assert ts.shape == p.shape
        r, c = ts.shape
        Rx, Ry, Rz, step, cap = params(r, c)

        def pad(arr, pads):
            return numpy.pad(arr, pads, mode="edge")

        ts = ts.copy()
        for _ in range(it):
            ts_y_up = pad(ts, ((1, 0), (0, 0)))[:-1, :]
            ts_y_down = pad(ts, ((0, 1), (0, 0)))[1:, :]
            ts_x_up = pad(ts, ((0, 0), (1, 0)))[:, :-1]
            ts_x_down = pad(ts, ((0, 0), (0, 1)))[:, 1:]
            dx = (ts_x_up + ts_x_down - 2 * ts) / Rx
            dy = (ts_y_up + ts_y_down - 2 * ts) / Ry
            d = (step / cap) * (p + dx + dy + (T_AMB - ts) / Rz)
            ts += d
        return ts

    @staticmethod
    def in_python(*args: numpy.ndarray) -> numpy.ndarray:
        it_, ts, p = args
        it = int(it_)
        assert ts.shape == p.shape
        r, c = ts.shape
        Rx, Ry, Rz, step, cap = params(r, c)
        for _ in range(it):
            ts1 = ts.copy()
            for y in range(r):
                for x in range(c):
                    dx = (
                        ts[y, min(x + 1, c - 1)] + ts[y, max(x - 1, 0)] - 2 * ts[y, x]
                    ) / Rx
                    dy = (
                        ts[min(y + 1, r - 1), x] + ts[max(y - 1, 0), x] - 2 * ts[y, x]
                    ) / Ry
                    d = (step / cap) * (p[y, x] + dx + dy + (T_AMB - ts[y, x]) / Rz)
                    ts1[y, x] += d
            ts = ts1
        return ts

    @staticmethod
    def sample(it: int = 3, n: int = 7, m: int = 11) -> tuple[numpy.ndarray, ...]:
        ts = numpy.random.randn(n, m)
        p = numpy.random.randn(n, m)
        return numpy.array(it), ts, p

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Self

from ein import Float, array, function, scalar_type, wrap
from ein.debug import pretty_phi, pretty_yarr_of_phi


@dataclass(frozen=True)
class MultiDual:
    real: Float
    imag: dict[object, Float]

    @classmethod
    def const(cls, x: Float) -> MultiDual:
        return MultiDual(x, {})

    @classmethod
    def diff(cls, f: Callable[[Self], Self]) -> Callable[[Float], Float]:
        def go(x: Float) -> Float:
            who = object()
            return f(cls(x, {who: wrap(1.0)})).imag[who]

        return go

    def __add__(self, other: MultiDual | Float) -> MultiDual:
        if not isinstance(other, MultiDual):
            other = MultiDual.const(other)
        zero = wrap(0.0)
        return MultiDual(
            -self.real,
            {
                o: self.imag.get(o, zero) + other.imag.get(o, zero)
                for o in self.imag | other.imag
            },
        )

    def __neg__(self) -> MultiDual:
        return MultiDual(-self.real, {o: -i for o, i in self.imag.items()})

    def __mul__(self, other: MultiDual | Float) -> MultiDual:
        if not isinstance(other, MultiDual):
            other = MultiDual.const(other)
        zero = wrap(0.0)
        return MultiDual(
            self.real * other.real,
            {
                o: self.real * other.imag.get(o, zero)
                + other.real * self.imag.get(o, zero)
                for o in self.imag | other.imag
            },
        )

    def __radd__(self, other: MultiDual | Float) -> MultiDual:
        return self + other

    def __rmul__(self, other: MultiDual | Float) -> MultiDual:
        return self * other


if __name__ == "__main__":

    @function
    def h(x0, y0):
        return MultiDual.diff(lambda x: x * MultiDual.diff(lambda y: x + y)(y0))(x0)

    @function
    def hh():
        return array(
            lambda i, j: MultiDual.diff(
                lambda x: x * MultiDual.diff(lambda y: x + y)(i.float() / 10.0)
            )(j.float() / 10.0),
            size=(11, 11),
        )

    print(pretty_phi(h.phi(scalar_type(float), scalar_type(float))[1]), "\n")
    print(pretty_phi(hh.phi()[1]), "\n")
    print(pretty_yarr_of_phi(hh.phi()[1]), "\n")

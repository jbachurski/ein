from functools import cache
from typing import cast

import numpy.testing

from ein.calculus import Cons, Expr, First, Get, Second, Vec, at
from ein.midend.substitution import substitute
from ein.symbols import Index
from ein.type_system import Pair, Vector


def struct_of_arrays_transform(program: Expr):
    # FIXME: Common subexpressions for both pair elements are duplicated
    #  across the term graph. The vec-let to let-vec transformation should help.

    @cache
    def go(expr: Expr) -> Expr:
        match expr:
            case Get(arr, it):

                def get_map(sub: Expr) -> Expr:
                    match sub.type:
                        case Vector():
                            return Get(sub, go(it))
                        case Pair():
                            return Cons(get_map(First(sub)), get_map(Second(sub)))
                    assert False, f"Unexpected type in this context: {sub.type}"

                return get_map(go(arr))
            case Vec(index, size, Cons(first, second)):
                i, j = Index(), Index()
                return Cons(
                    go(Vec(i, size, cast(Expr, substitute(first, {index: at(i)})))),
                    go(Vec(j, size, cast(Expr, substitute(second, {index: at(j)})))),
                )
        return expr.map(go)

    @cache
    def elim(expr: Expr) -> Expr:
        expr = expr.map(elim)
        match expr:
            case First(Cons(first, _second)):
                return elim(first)
            case Second(Cons(_first, second)):
                return elim(second)
            case Cons(First(p), Second(p_)) if p == p_:
                return elim(p)
        return expr

    return elim(go(program))


def main():
    from dataclasses import dataclass

    import numpy as np

    from ein import Array, array, debug, structs

    s = structs(lambda i: (i, i**2, {"+": i**3, "-": -(i**3)}), size=10)
    a = array(lambda j: s[j, 2, "-"])
    expr = struct_of_arrays_transform(a.expr)
    print(expr)
    debug.plot_phi_graph(expr)
    print(Array(expr).numpy())
    numpy.testing.assert_allclose(Array(expr).numpy(), [-(i**3) for i in range(10)])

    @dataclass
    class C:
        real: Array  # scalar
        imag: Array  # scalar, uhhh

        def __add__(self, other) -> "C":
            return C(self.real + other.real, self.imag + other.imag)

        def __mul__(self, other) -> "C":
            return C(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )

    c = structs(
        lambda i: C(
            (3.14 * i.to_float() / 10.0).cos(),
            (3.14 * i.to_float() / 10.0).sin(),
        ),
        size=11,
    )

    cc = array(lambda i: (c[i] * c[i] + c[i]).real)

    def cis(x):
        return np.cos(x) + np.sin(x) * 1j

    print(cc.numpy())
    c1 = cis(np.linspace(0, 3.14, 11))
    cc1 = np.real(c1 * (c1 + 1))
    numpy.testing.assert_allclose(cc.numpy(), cc1)


if __name__ == "__main__":
    main()

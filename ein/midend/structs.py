from functools import cache
from typing import cast

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
    from ein import Array, array, debug, structs

    s = structs(lambda i: (i, i**2, {"+": i**3, "-": -(i**3)}), size=10)
    a = array(lambda j: s[j, 2, "+"])
    expr = struct_of_arrays_transform(a.expr)
    print(expr)
    debug.plot_phi_graph(expr)
    print(Array(expr).numpy())


if __name__ == "__main__":
    main()

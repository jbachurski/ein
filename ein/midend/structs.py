import functools
from typing import TypeVar, assert_never, cast

from ein.calculus import Cons, Expr, First, Get, Second, Vec, at
from ein.midend.substitution import substitute
from ein.symbols import Index
from ein.type_system import Pair, Scalar, Type, Vector

F = TypeVar("F")


def _soa_transform_type(type_: Type) -> Type:
    match type_:
        case Vector(Pair(first, second)):
            return Pair(
                _soa_transform_type(Vector(first)), _soa_transform_type(Vector(second))
            )
        case Vector(elem):
            return Vector(_soa_transform_type(elem))
        case Pair(first, second):
            return Pair(_soa_transform_type(first), _soa_transform_type(second))
        case Scalar():
            return type_
        case _:
            assert_never(type_)


def struct_of_arrays_transform(program: Expr):
    # FIXME: Common subexpressions for both pair elements are duplicated
    #  across the term graph. The vec-let to let-vec transformation should help.

    def validate(f):
        @functools.wraps(f)
        def fun(x: Expr) -> Expr:
            y = f(x)
            assert y.type == _soa_transform_type(x.type)
            return y

        return fun

    @validate
    @functools.cache
    def go(expr: Expr) -> Expr:
        match expr:
            case Get(arr, it):
                arr, it = go(arr), go(it)

                def rec(sub: Expr):
                    match sub.type:
                        case Pair():
                            return Cons(rec(First(sub)), rec(Second(sub)))
                    return Get(sub, it)

                return rec(arr)
            case Vec(index, size, body):
                size, body = go(size), go(body)
                if isinstance(body.type, Pair):
                    i, j = Index(), Index()
                    body1 = cast(Expr, substitute(First(body), {index: at(i)}))
                    body2 = cast(Expr, substitute(Second(body), {index: at(j)}))
                    return Cons(go(Vec(i, size, body1)), go(Vec(j, size, body2)))
        return expr.map(go)

    @functools.cache
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

    original_type = program.type
    program = elim(go(program))
    assert (
        original_type == program.type
    ), "Transformation changed resulting program type"
    return program

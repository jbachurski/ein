import functools
from typing import TypeVar, assert_never, cast

from ein import calculus
from ein.calculus import (
    Concat,
    Cons,
    Dim,
    Expr,
    First,
    Get,
    Reduce,
    Second,
    Store,
    Vec,
    at,
)
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
            case Store(symbol, type_):
                return Store(symbol, _soa_transform_type(type_))

            case Dim(arr, axis):
                arr = go(arr)

                def tuple_dim(sub: Expr) -> tuple[Expr, ...]:
                    match sub.type:
                        case Pair():
                            return tuple_dim(First(sub)) + tuple_dim(Second(sub))
                    return (Dim(sub, axis),)

                dims = tuple_dim(arr)
                return (
                    calculus.AssertEq(dims[0], dims[1:]) if len(dims) > 1 else dims[0]
                )

            case Get(arr, it):
                arr, it = go(arr), go(it)

                def tuple_get(sub: Expr) -> Expr:
                    match sub.type:
                        case Pair():
                            return Cons(tuple_get(First(sub)), tuple_get(Second(sub)))
                    return Get(sub, it)

                return tuple_get(arr)

            case Concat(first, second):
                first, second = go(first), go(second)

                def tuple_cat(sub_first: Expr, sub_second: Expr) -> Expr:
                    assert sub_first.type == sub_second.type
                    match sub_first.type:
                        case Pair():
                            return Cons(
                                tuple_cat(First(sub_first), First(sub_second)),
                                tuple_cat(Second(sub_first), Second(sub_second)),
                            )
                    return Concat(sub_first, sub_second)

                return tuple_cat(first, second)

            case Vec(index, size, body):
                size, body = go(size), go(body)
                if isinstance(body.type, Pair):
                    i, j = Index(), Index()
                    body1 = cast(Expr, substitute(First(body), {index: at(i)}))
                    body2 = cast(Expr, substitute(Second(body), {index: at(j)}))
                    return Cons(go(Vec(i, size, body1)), go(Vec(j, size, body2)))

            case Reduce(init, x, y, xy, vecs):
                init, xy = go(init), go(xy)
                vecs = tuple(go(vec) for vec in vecs)

                def in_tuple(sub: calculus.Expr) -> tuple[calculus.Expr, ...]:
                    match sub.type:
                        case Pair():
                            return in_tuple(First(sub)) + in_tuple(Second(sub))
                    return (sub,)

                def into_tuple(*args: calculus.Expr) -> calculus.Expr:
                    return functools.reduce(Cons, args)

                prim_vecs = sum((in_tuple(vec) for vec in vecs), ())
                xy = into_tuple(*in_tuple(xy))
                # FIXME: The encoding for x and y might also change, inverse to xy.
                #  A substitution may be necessary.
                #  This only works now because usually the encoding is the trivial list.
                return Reduce(init, x, y, xy, prim_vecs)

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

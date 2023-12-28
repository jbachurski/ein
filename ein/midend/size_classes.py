from typing import Callable, TypeAlias, TypeVar, Union

import numpy

from ein import calculus
from ein.symbols import Index

from .equiv import Equivalence

T = TypeVar("T")
S = TypeVar("S")
SizeEquivalence: TypeAlias = Equivalence[Union[Index, calculus.Expr]]


def _dim_of(expr: calculus.Expr, axis: int = 0) -> calculus.Expr:
    match expr:
        case calculus.Get(target, _item):
            return _dim_of(target, axis + 1)
        case calculus.Vec():
            if axis == 0:
                return expr.size
            # return _dim_of(expr.body, axis - 1)
    return calculus.Dim(expr, axis)


def seq(a: Callable[[T], None], f: Callable[[T], S]) -> Callable[[T], S]:
    def inner(x: T) -> S:
        a(x)
        return f(x)

    return inner


def id(x: T) -> T:
    return x


def _update_size_classes(
    program: calculus.Expr,
    sizes: SizeEquivalence,
    skip: Callable[[calculus.Expr], bool],
    trigger: list[tuple[set[calculus.Expr], calculus.Expr]],
) -> None:
    vis = set()

    def go(expr: calculus.Expr) -> None:
        if expr in vis or skip(expr):
            return
        vis.add(expr)
        match expr:
            case calculus.Const(value):
                if isinstance(value.value, numpy.ndarray):
                    shape = value.array.shape
                    for axis, dim in enumerate(shape):
                        sizes.unite(
                            calculus.Dim(expr, axis),
                            calculus.Const(calculus.Value(dim)),
                        )
            case calculus.AssertEq(_target, operands):
                for sub in operands:
                    sizes.unite(expr, sub)
            case calculus.Let(var, bind, _body) if len(
                bind.type.primitive_type.elems
            ) == 1:
                rank = bind.type.primitive_type.single.rank
                sizes.unite(calculus.Store(var, bind.type), bind)
                # TODO: This should really be an e-graph instead.
                for axis in range(rank):
                    sizes.unite(
                        calculus.Dim(calculus.Store(var, bind.type), axis),
                        calculus.Dim(bind, axis),
                    )
            case calculus.Vec(index, size, body):
                sizes.unite(index, size)
                sizes.unite(index, calculus.Dim(expr, 0))
                rank = expr.type.primitive_type.single.rank
                for axis in range(1, rank):
                    sizes.unite(calculus.Dim(expr, rank), calculus.Dim(body, rank - 1))
            case calculus.Get(target, _item):
                rank = expr.type.primitive_type.single.rank
                for axis in range(rank):
                    sizes.unite(
                        calculus.Dim(expr, axis),
                        calculus.Dim(target, axis + 1),
                    )
            case calculus.Fold(_counter, _count, acc, init, body) if len(
                init.type.primitive_type.elems
            ) == 1:
                rank = expr.type.primitive_type.single.rank
                for axis in range(rank):
                    triangle = [
                        calculus.Dim(calculus.Store(acc, init.type), axis),
                        calculus.Dim(body, axis),
                        calculus.Dim(init, axis),
                    ]
                    for i in range(3):
                        trigger.append(
                            (
                                {triangle[i], triangle[(i + 1) % 3]},
                                triangle[(i + 2) % 3],
                            )
                        )
                    trigger.append((set(triangle), calculus.Dim(expr, axis)))
        expr.map(seq(go, id))
        return None

    go(program)


def update_size_classes(
    program: calculus.Expr,
    sizes: SizeEquivalence,
    skip: Callable[[calculus.Expr], bool],
) -> None:
    trigger: list[tuple[set[calculus.Expr], calculus.Expr]] = []
    _update_size_classes(program, sizes, skip, trigger)
    while True:
        got = set()
        for i, (required, obtained) in enumerate(trigger):
            if sizes.equiv(*required):
                sizes.unite(next(iter(required)), obtained)
                got.add(i)
        if not got:
            break
        trigger = [x for i, x in enumerate(trigger) if i not in got]


def find_size_classes(program: calculus.Expr) -> SizeEquivalence:
    sizes = SizeEquivalence()
    update_size_classes(program, sizes, lambda _: False)
    return sizes

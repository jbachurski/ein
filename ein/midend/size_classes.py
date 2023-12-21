from typing import Callable, TypeAlias, TypeVar, Union

from ein import calculus
from ein.symbols import Index

from .equiv import Equivalence

T = TypeVar("T")
S = TypeVar("S")
SizeEquivalence: TypeAlias = Equivalence[Union[Index, calculus.Expr]]


def seq(a: Callable[[T], None], f: Callable[[T], S]) -> Callable[[T], S]:
    def inner(x: T) -> S:
        a(x)
        return f(x)

    return inner


def id(x: T) -> T:
    return x


def update_size_classes(
    program: calculus.Expr,
    sizes: SizeEquivalence,
    skip: Callable[[calculus.Expr], bool],
) -> None:
    vis = set()

    def go(expr: calculus.Expr) -> None:
        if expr in vis or skip(expr):
            return
        vis.add(expr)
        match expr:
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
        expr.map(seq(go, id))
        return None

    go(program)


def find_size_classes(program: calculus.Expr) -> SizeEquivalence:
    sizes = SizeEquivalence()
    update_size_classes(program, sizes, lambda _: False)
    return sizes

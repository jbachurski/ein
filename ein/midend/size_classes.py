from typing import Callable, TypeVar, Union, cast

import numpy

from ein.phi import phi
from ein.symbols import Index, Symbol

from .equiv import Equivalence
from .substitution import substitute

T = TypeVar("T")
S = TypeVar("S")


def _with_indices_at_zero(expr: phi.Expr) -> phi.Expr:
    subs: dict[Symbol, phi.Expr] = {
        index: phi.Const(phi.Value(0)) for index in expr.free_indices
    }
    return cast(phi.Expr, substitute(expr, subs))


def _dim_of(expr: phi.Expr, axis: int = 0) -> phi.Expr:
    match expr:
        case phi.Get(target, _item):
            return _dim_of(target, axis + 1)
        case phi.Vec():
            if axis == 0:
                return expr.size
            # FIXME: This causes naive to run really slow on test_big_permutation
            # return _dim_of(expr.body, axis - 1)
    return phi.Dim(expr, axis)


def seq(a: Callable[[T], None], f: Callable[[T], S]) -> Callable[[T], S]:
    def inner(x: T) -> S:
        a(x)
        return f(x)

    return inner


def id(x: T) -> T:
    return x


class SizeEquivalence(Equivalence[Union[Index, phi.Expr]]):
    def _get_parent(self, u: Union[Index, phi.Expr]) -> Union[Index, phi.Expr]:
        # We use the rectangular property of comprehensions,
        # supposing all indexes into arrays are at 0.
        if isinstance(u, phi.AbstractExpr):
            u = _with_indices_at_zero(cast(phi.Expr, u))
        return super()._get_parent(u)


def _update_size_classes(
    program: phi.Expr,
    sizes: SizeEquivalence,
    trigger: list[tuple[set[phi.Expr], phi.Expr]],
) -> None:
    vis = set()

    def go(expr: phi.Expr) -> None:
        if expr in vis:
            return
        vis.add(expr)
        match expr:
            case phi.Const(value):
                if isinstance(value.value, numpy.ndarray):
                    shape = value.array.shape
                    for axis, dim in enumerate(shape):
                        sizes.unite(
                            phi.Dim(expr, axis),
                            phi.Const(phi.Value(dim)),
                        )
            case phi.AssertEq(target, operands):
                sizes.unite(expr, target)
                for sub in operands:
                    sizes.unite(operands[0], sub)
            case phi.Let(var, bind, _body) if len(bind.type.primitive_type.elems) == 1:
                rank = bind.type.primitive_type.single.rank
                sizes.unite(phi.Store(var, bind.type), bind)
                # TODO: This should really be an e-graph instead.
                for axis in range(rank):
                    sizes.unite(
                        phi.Dim(phi.Store(var, bind.type), axis),
                        phi.Dim(bind, axis),
                    )
            case phi.Vec(index, size, body):
                sizes.unite(index, size)
                sizes.unite(index, phi.Dim(expr, 0))
                rank = expr.type.primitive_type.single.rank
                for axis in range(1, rank):
                    sizes.unite(phi.Dim(expr, rank), phi.Dim(body, rank - 1))
            case phi.Get(target, _item):
                rank = expr.type.primitive_type.single.rank
                for axis in range(rank):
                    # We can use the fact arrays themselves are rectangular,
                    # and suppose the index is anything else (representative is 0).
                    sizes.unite(
                        phi.Dim(expr, axis),
                        phi.Dim(
                            phi.Get(target, phi.Const(phi.Value(0))),
                            axis,
                        ),
                    )
                    sizes.unite(
                        phi.Dim(expr, axis),
                        phi.Dim(target, axis + 1),
                    )
            case phi.Fold(_counter, _count, acc, init, body) if len(
                init.type.primitive_type.elems
            ) == 1:
                rank = expr.type.primitive_type.single.rank
                for axis in range(rank):
                    triangle = [
                        phi.Dim(phi.Store(acc, init.type), axis),
                        phi.Dim(body, axis),
                        phi.Dim(init, axis),
                    ]
                    for i in range(3):
                        trigger.append(
                            (
                                {triangle[i], triangle[(i + 1) % 3]},
                                triangle[(i + 2) % 3],
                            )
                        )
                    trigger.append((set(triangle), phi.Dim(expr, axis)))
        expr.map(seq(go, id))
        return None

    go(program)


def update_size_classes(
    program: phi.Expr,
    sizes: SizeEquivalence,
) -> None:
    trigger: list[tuple[set[phi.Expr], phi.Expr]] = []
    _update_size_classes(program, sizes, trigger)
    while True:
        got = set()
        for i, (required, obtained) in enumerate(trigger):
            if sizes.equiv(*required):
                sizes.unite(next(iter(required)), obtained)
                got.add(i)
        if not got:
            break
        trigger = [x for i, x in enumerate(trigger) if i not in got]


def find_size_classes(program: phi.Expr) -> SizeEquivalence:
    sizes = SizeEquivalence()
    update_size_classes(program, sizes)
    return sizes

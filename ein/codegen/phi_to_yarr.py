import functools
import itertools
from dataclasses import dataclass
from functools import cache
from string import ascii_lowercase
from typing import Callable, Iterable, assert_never, cast

from term import Term

from ein.codegen import yarr
from ein.midend.size_classes import (
    SizeEquivalence,
    find_size_classes,
    update_size_classes,
)
from ein.midend.structs import struct_of_arrays_transform
from ein.midend.substitution import substitute
from ein.phi import calculus
from ein.phi.type_system import Pair, Scalar, to_float
from ein.symbols import Index, Symbol, Variable

from . import array_indexing, axial
from .axial import Axial


def transform(
    program: calculus.Expr,
    *,
    use_takes: bool = True,
    use_slice_pads: bool = True,
    use_slice_elision: bool = True,
    use_reductions: bool = True,
    use_einsum: bool = True,
    do_shape_cancellations: bool = True,
    do_tuple_cancellations: bool = True,
) -> yarr.Expr:
    transformed: dict[calculus.Expr, Axial] = {}

    program = struct_of_arrays_transform(program)
    size_class = find_size_classes(program)

    def _go(
        expr: calculus.Expr,
        index_sizes: dict[Index, yarr.Expr],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        match expr:
            case calculus.Const(value):
                return Axial([], yarr.Const(value))
            case calculus.Store(index, _) if isinstance(index, Index):
                return Axial(
                    [index],
                    yarr.Range(index_sizes[index]),
                )
            case calculus.Store(var, inner_type) if isinstance(var, Variable):
                axes = var_axes.get(var, ())
                return Axial(
                    axes,
                    yarr.Var(
                        var, inner_type.primitive_type.with_rank_delta(+len(axes))
                    ),
                )
            case calculus.Store(symbol, _inner_type):
                raise NotImplementedError(f"Unhandled symbol of type {type(symbol)}")
            case calculus.Let(var, bind_, body_):
                bind = go(bind_, index_sizes, var_axes)
                return go(
                    body_,
                    index_sizes,
                    var_axes | {var: bind._axes},
                ).within((var, bind.expr))
            case calculus.AssertEq(target_, _):
                return go(target_, index_sizes, var_axes)
            case calculus.Dim(target_, pos):
                target = go(target_, index_sizes, var_axes)
                return Axial(
                    [],
                    yarr.Dim(target.positional_axis(pos), target.expr),
                )
            case calculus.Fold(counter, size_, acc, init_, body_):
                if use_einsum:
                    dot = Dot.lift(expr)
                    if dot.contract and any(len(prod) > 1 for prod in dot.products):
                        realised_counter = {
                            counter: Index() for counter in dot.contract
                        }
                        subs: dict[Symbol, Term] = {
                            counter: calculus.at(index)
                            for counter, index in realised_counter.items()
                        }
                        sum_axes = {
                            realised_counter[counter]: go(
                                size, index_sizes, var_axes
                            ).normal
                            for counter, size in dot.contract.items()
                        }

                        def go_product(*args: calculus.Expr) -> Axial | None:
                            for counter, size in dot.contract.items():
                                for operand in args:
                                    if not would_be_memory_considerate_axis(
                                        operand, counter, size, size_class
                                    ):
                                        return None
                            with_realised = [
                                cast(calculus.Expr, substitute(operand, subs))
                                for operand in args
                            ]
                            for counter, size in dot.contract.items():
                                size_class.unite(realised_counter[counter], size)
                            for expr_with_realised in with_realised:
                                update_size_classes(
                                    expr_with_realised,
                                    size_class,
                                )
                            axial_operands = [
                                go(operand, index_sizes | sum_axes, var_axes)
                                for operand in with_realised
                            ]
                            operand_axes = [operand._axes for operand in axial_operands]
                            result_axes = tuple(
                                index
                                for index in axial._alignment(*operand_axes)
                                if index not in sum_axes
                            )
                            subscripts = to_einsum_subs(operand_axes, result_axes)
                            return Axial(
                                result_axes,
                                yarr.Einsum(
                                    subscripts,
                                    tuple(operand.expr for operand in axial_operands),
                                ),
                            )

                        summands: list[Axial] = []
                        for curr in dot.products:
                            in_array = go_product(*curr)
                            if in_array is None:
                                break
                            summands.append(in_array)
                        else:
                            used_axes = axial._alignment(*(op._axes for op in summands))
                            return Axial(
                                used_axes,
                                functools.reduce(
                                    lambda a, b: yarr.BinaryElementwise(
                                        yarr.BinaryElementwise.Kind.add, a, b
                                    ),
                                    (
                                        op.aligned(used_axes, leftpad=False)
                                        for op in summands
                                    ),
                                ),
                            )
                if use_reductions:
                    reduced = match_reduction(
                        *(counter, size_, acc, init_, body_, size_class),  #
                        lambda sub: go(sub, index_sizes, var_axes),
                    )
                    if reduced is not None:
                        return reduced
                size = go(size_, index_sizes, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile fold with index-dependent size"
                init = go(init_, index_sizes, var_axes)
                # FIXME: We need to establish an alignment that includes free indices from both init and body.
                #  Doing this by constructing the body twice is a really bad idea.
                #  But just using body.free_indices isn't deterministic enough.
                pre_transformed = transformed.copy()
                acc_axes = go(
                    body_,
                    index_sizes,
                    var_axes | {acc: init._axes},
                )._axes
                transformed.clear()
                transformed.update(pre_transformed)
                body = go(
                    body_,
                    index_sizes,
                    var_axes | {acc: acc_axes},
                )
                return Axial(
                    acc_axes,
                    yarr.Fold(
                        counter,
                        size.normal,
                        acc,
                        init.aligned(acc_axes),
                        body.aligned(acc_axes),
                    ),
                )
            case calculus.Reduce(init_, x, y, xy_, vecs_):
                init = go(init_, index_sizes, var_axes)
                vecs = [go(vec_, index_sizes, var_axes) for vec_ in vecs_]

                # FIXME: Like for Fold, we trace the body twice to get a deterministic choice of axes.
                some_axes = axial._alignment(init._axes, *(vec._axes for vec in vecs))
                pre_transformed = transformed.copy()
                axes = go(
                    xy_, index_sizes, var_axes | {x: some_axes, y: some_axes}
                )._axes
                transformed.clear()
                transformed.update(pre_transformed)

                # Insert the batch axis to be the first one before ones reduced
                #   this is a convenient choice, as [vecs] also follow it
                #   (their outermost axis is reduced, so this inserted axis corresponds to it)
                xy_axes = (*axes, Index())
                xy = go(xy_, index_sizes, var_axes | {x: xy_axes, y: xy_axes})

                # FIXME: There's probably something wrong here.
                init_expr = init.aligned(axes, repeats=index_sizes)
                vecs_expr = (
                    yarr.Tuple(
                        tuple(vec.aligned(axes, repeats=index_sizes) for vec in vecs)
                    )
                    if len(vecs) > 1
                    else vecs[0].aligned(axes, repeats=index_sizes)
                )
                return Axial(
                    axes,
                    yarr.Reduce(init_expr, x, y, xy.expr, vecs_expr, len(axes)),
                )

            case calculus.Get(target_, item_):
                return array_indexing.transform_get(
                    target_,
                    item_,
                    lambda expr_: go(expr_, index_sizes, var_axes),
                    size_class,
                    index_sizes,
                    use_takes=use_takes,
                    use_slice_pads=use_slice_pads,
                    use_slice_elision=use_slice_elision,
                )
            case calculus.Concat(first_, second_):
                first = go(first_, index_sizes, var_axes)
                second = go(second_, index_sizes, var_axes)
                used_axes = axial._alignment(first._axes, second._axes)
                return Axial(
                    used_axes,
                    yarr.Concat(
                        tuple(
                            op.aligned(used_axes, repeats=index_sizes)
                            for op in (first, second)
                        ),
                        len(used_axes),
                    ),
                )
            case calculus.Vec(index, size_, target_):
                size = go(size_, index_sizes, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile index comprehension with index-dependent size"
                return go(target_, index_sizes | {index: size.normal}, var_axes).along(
                    index, size.expr
                )
            case calculus.AbstractScalarOperator(operands_):
                ops = [go(op, index_sizes, var_axes) for op in operands_]
                if expr.ufunc == to_float:
                    (target,) = ops
                    return Axial(
                        target._axes,
                        yarr.Cast(float, target.expr),
                    )
                else:
                    used_axes = axial._alignment(*(op._axes for op in ops))
                    return Axial(
                        used_axes,
                        yarr.ELEMENTWISE_UFUNCS[expr.ufunc](
                            *(op.aligned(used_axes, leftpad=False) for op in ops)
                        ),
                    )
            case calculus.Cons(first_, second_):
                first = go(first_, index_sizes, var_axes)
                second = go(second_, index_sizes, var_axes)
                return first.cons(second)
            case calculus.First(target_):
                assert isinstance(target_.type, Pair)
                k = len(target_.type.first.primitive_type.elems)
                return go(target_, index_sizes, var_axes).slice_tuple(0, k)
            case calculus.Second(target_):
                assert isinstance(target_.type, Pair)
                k = len(target_.type.first.primitive_type.elems)
                n = len(target_.type.primitive_type.elems)
                return go(target_, index_sizes, var_axes).slice_tuple(k, n)
            case calculus.Extrinsic(_type, fun, operands_):
                ops = [go(op, index_sizes, var_axes) for op in operands_]
                used_axes = axial._alignment(*(op._axes for op in ops))
                ops_expr = tuple(op.aligned(used_axes, leftpad=False) for op in ops)
                prim_type = _type.primitive_type.with_rank_delta(len(used_axes))
                return Axial(used_axes, yarr.Extrinsic(prim_type, fun, ops_expr))
            case _:
                assert_never(expr)
        assert False  # noqa

    def go(
        expr: calculus.Expr,
        index_sizes: dict[Index, yarr.Expr],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        if expr not in transformed:
            transformed[expr] = _go(expr, index_sizes, var_axes)
        return transformed[expr]

    # FIXME: Let-bound variables get different treatment here than free variables.
    #  Free variables are always assumed to depend on no indices (empty free-index set).
    array_program = go(program, {}, {}).normal
    if do_shape_cancellations:
        array_program = cancel_shape_ops(array_program)
    if do_tuple_cancellations:
        array_program = cancel_tuple_ops(array_program)
    array_indexing.match_index_clipped_shift.cache_clear()

    return array_program


def cancel_shape_ops(program: yarr.Expr) -> yarr.Expr:
    @cache
    def go(expr: yarr.Expr) -> yarr.Expr:
        match expr:
            case yarr.Transpose(permutation, target):
                if permutation == tuple(range(len(permutation))):
                    return go(target)
            case yarr.Unsqueeze(axes, target):
                if not axes:
                    return go(target)
            case yarr.Squeeze(axes, target):
                if not axes:
                    return go(target)
        return expr.map(go)

    return go(program)


def cancel_tuple_ops(program: yarr.Expr) -> yarr.Expr:
    @cache
    def go(expr: yarr.Expr) -> yarr.Expr:
        match expr:
            case yarr.Untuple(at, _arity, yarr.Tuple(elems)):
                return go(elems[at])
        return expr.map(go)

    return go(program)


def would_be_memory_considerate_axis(
    body: calculus.Expr,
    counter: Variable,
    size: calculus.Expr,
    size_class: SizeEquivalence,
) -> bool:
    ok: bool = True

    @cache
    def go(expr: calculus.Expr) -> calculus.Expr:
        nonlocal ok
        if counter not in expr.free_symbols:
            return expr
        if sum(counter in sub.free_symbols for sub in expr.subterms) > 1:
            ok = False
        match expr:
            case calculus.Get(target, item):
                size_matches = size_class.equiv(size, calculus.Dim(target, 0))
                if (
                    counter in item.free_symbols
                    and not item.free_indices
                    and size_matches
                ):
                    go(target)
                    return expr
            case calculus.Store(symbol, _inner_type):
                if symbol == counter:
                    ok = False
        return expr.map(go)

    go(body)
    return ok


def match_reduction_by_body(
    body: calculus.Expr, acc: Variable
) -> tuple[calculus.Expr, yarr.ReduceAxis.Kind] | None:
    red = {
        calculus.Add: yarr.ReduceAxis.Kind.add,
        calculus.Min: yarr.ReduceAxis.Kind.minimum,
        calculus.Max: yarr.ReduceAxis.Kind.maximum,
    }
    for typ, kind in red.items():
        if isinstance(body, typ):
            first, second = body.operands
            if isinstance(first, calculus.Store) and first.symbol == acc:
                return second, kind
            elif isinstance(second, calculus.Store) and second.symbol == acc:
                return first, kind
    return None


def match_reduction(
    counter: Variable,
    size: calculus.Expr,
    acc: Variable,
    init: calculus.Expr,
    body: calculus.Expr,
    size_class: SizeEquivalence,
    go: Callable[[calculus.Expr], Axial],
) -> Axial | None:
    if init.type != Scalar(float) or init.free_indices:
        return None
    red = match_reduction_by_body(body, acc)
    if red is None:
        return None
    elem, kind = red
    if not would_be_memory_considerate_axis(elem, counter, size, size_class):
        return None
    index = Index()
    with_counter_axis = cast(
        calculus.Expr, substitute(elem, {counter: calculus.at(index)})
    )
    vec_expr = calculus.Vec(index, size, with_counter_axis)
    update_size_classes(
        vec_expr,
        size_class,
    )
    vec = go(vec_expr)
    reduced = yarr.ReduceAxis(kind, vec.positional_axis(0), vec.expr)
    reduced_with_init = yarr.BinaryElementwise(
        yarr.REDUCE_UNDERLYING[kind], go(init).normal, reduced
    )
    return Axial(vec._axes, reduced_with_init)


def match_summation(expr: calculus.Fold) -> tuple[Variable, calculus.Expr] | None:
    if not isinstance(expr, calculus.Fold):
        return None
    red = match_reduction_by_body(expr.body, expr.acc)
    if red is None:
        return None
    body, kind = red
    if kind != yarr.ReduceAxis.Kind.add:
        return None
    if expr.init != calculus.Const(calculus.Value(0.0)):
        return None
    return expr.counter, body


@dataclass(frozen=True)
class Dot:
    contract: dict[Variable, calculus.Expr]
    products: tuple[tuple[calculus.Expr, ...], ...]

    @staticmethod
    def _values_agree(first: dict, second: dict) -> bool:
        return all(
            first[counter] is second[counter] for counter in set(first) & set(second)
        )

    @classmethod
    def pure(cls, expr: calculus.Expr) -> "Dot":
        return cls({}, ((expr,),))

    def sum(self, counter: Variable, size: calculus.Expr) -> "Dot":
        assert counter not in self.contract
        return Dot(self.contract | {counter: size}, self.products)

    def __add__(self, other: "Dot") -> "Dot":
        assert self._values_agree(self.contract, other.contract)
        return Dot(self.contract | other.contract, self.products + other.products)

    def __mul__(self, other: "Dot") -> "Dot":
        if set(self.contract) & set(other.contract):
            raise ValueError(
                "Multiplying dot products with common contractions is non-linear"
            )
        return Dot(
            self.contract | other.contract,
            tuple(p + q for p in self.products for q in other.products),
        )

    def __neg__(self) -> "Dot":
        return Dot(
            self.contract,
            tuple(
                tuple(e if i else calculus.Negate((e,)) for i, e in enumerate(p))
                for p in self.products
            ),
        )

    def __sub__(self, other: "Dot") -> "Dot":
        return self + (-other)

    @classmethod
    def lift(cls, expr: calculus.Expr) -> "Dot":
        match expr:
            case calculus.Fold() if (summation := match_summation(expr)) is not None:
                counter, body = summation
                return cls.lift(body).sum(counter, expr.size)
            case calculus.Add((first, second)):
                return cls.lift(first) + cls.lift(second)
            case calculus.Negate((target,)):
                return -cls.lift(target)
            case calculus.Subtract((first, second)):
                return cls.lift(first) - cls.lift(second)
            case calculus.Multiply((first, second)):
                fst, snd = cls.lift(first), cls.lift(second)
                if not (set(fst.contract) & set(snd.contract)):
                    return fst * snd
        return cls.pure(expr)


def to_einsum_subs(
    operand_axes: Iterable[tuple[Index, ...]], result_axes: tuple[Index, ...]
) -> str:
    alphabet: dict[Index, str] = {}
    for index in itertools.chain(*operand_axes, result_axes):
        if index not in alphabet:
            alphabet[index] = ascii_lowercase[len(alphabet)]
    return (
        ",".join("".join(alphabet[index] for index in op) for op in operand_axes)
        + "->"
        + "".join(alphabet[index] for index in result_axes)
    )


def apply_inplace_on_temporaries(program: yarr.Expr) -> yarr.Expr:
    # Assumes that an elementwise operation used as an operand to another one
    # will not be broadcast (already has the same shape as the result).
    # Additionally, we assume that the result will not be reused anywhere else (needs to be let-bound).
    # This will also interact with any implicit-promotion optimisations, as here we assume dtypes are consistent.
    # This is why we only do this for BinaryElementwise and assume we have already done an explicit Cast.
    TEMPORARIES = (
        yarr.UnaryElementwise,
        yarr.BinaryElementwise,
        yarr.TernaryElementwise,
        yarr.Range,
        yarr.Cast,
        yarr.Pad,
        yarr.Repeat,
        yarr.ReduceAxis,
    )

    @cache
    def go(expr: yarr.Expr) -> yarr.Expr:
        first: yarr.Expr
        second: yarr.Expr
        expr = expr.map(go)
        match expr:
            case yarr.BinaryElementwise(kind, first, second, None):
                # This logic is really shaky and interacts with optimisations to the number of axis manipulation calls.
                # Should have more in-depth analysis on what broadcasting might occur
                rank = expr.type.single.rank
                rank1, rank2 = first.type.single.rank, second.type.single.rank
                if rank:
                    if rank == rank1 and isinstance(first, TEMPORARIES):
                        return yarr.BinaryElementwise(kind, first, second, 0)
                    if rank == rank2 and isinstance(second, TEMPORARIES):
                        return yarr.BinaryElementwise(kind, first, second, 1)
        return expr

    return go(program)

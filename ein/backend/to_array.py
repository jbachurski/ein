import itertools
from functools import cache
from string import ascii_lowercase
from typing import Callable, Iterable, assert_never, cast

from ein import calculus
from ein.midend.size_classes import (
    SizeEquivalence,
    find_size_classes,
    update_size_classes,
)
from ein.midend.substitution import substitute
from ein.symbols import Index, Variable
from ein.type_system import Pair, Scalar, to_float

from . import array_calculus, array_indexing, axial
from .axial import Axial


def transform(
    program: calculus.Expr,
    *,
    use_takes: bool = True,
    use_slice_pads: bool = True,
    use_slice_elision: bool = True,
    use_reductions: bool = True,
    use_einsum: bool = False,
    do_shape_cancellations: bool = True,
    do_tuple_cancellations: bool = True,
) -> array_calculus.Expr:
    transformed: dict[calculus.Expr, Axial] = {}

    size_class = find_size_classes(program)

    def _go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        if use_einsum:
            realised_axes, summed_axes_, expr_operands = match_einsum(expr)
            if summed_axes_:
                realised_sum_axis = {index: Index() for index in summed_axes_}
                summed_axes = {
                    realised_sum_axis[index]: go(size, index_sizes, var_axes).normal
                    for index, size in summed_axes_.items()
                }

                def with_realised_summed_axes(e: calculus.Expr) -> calculus.Expr:
                    return cast(
                        calculus.Expr,
                        substitute(
                            e,
                            {
                                index: calculus.at(realised_sum_axis[index])
                                for index in summed_axes_
                            },
                        ),
                    )

                # FIXME: Here, we realise axes along which we were summing (folding) prior.
                #  This might be a bad idea if those ranges are big - needs a way of establishing intent
                #  whether elements summed over can be put in memory. To prevent accidentally compiling
                #  inefficient code for some programs, we need a heuristic that an operand does not
                #  materialise significantly more than what was already in-memory.
                #  Note: we associate array(...) constructors to put contents into memory.
                operands = tuple(
                    go(
                        with_realised_summed_axes(expr_operand),
                        index_sizes | summed_axes,
                        var_axes,
                    )
                    for expr_operand in expr_operands
                )
                if any(
                    set(op1._axes) & set(op2._axes)
                    for i, op1 in enumerate(operands)
                    for op2 in operands[:i]
                ):
                    axes = tuple(
                        axis
                        for axis in axial._alignment(
                            *(operand._axes for operand in operands)
                        )
                        if axis in realised_axes
                    )
                    subs = to_einsum_subs(
                        [operand._axes for operand in operands], tuple(axes)
                    )
                    return Axial(
                        axes,
                        array_calculus.Einsum(
                            subs, tuple(operand.expr for operand in operands)
                        ),
                    )

        match expr:
            case calculus.Const(value):
                return Axial([], array_calculus.Const(value))
            case calculus.Store(index, _) if isinstance(index, Index):
                return Axial(
                    [index],
                    array_calculus.Range(index_sizes[index]),
                )
            case calculus.Store(var, inner_type) if isinstance(var, Variable):
                axes = var_axes.get(var, ())
                return Axial(
                    axes,
                    array_calculus.Var(
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
                    array_calculus.Dim(target.positional_axis(pos), target.expr),
                )
            case calculus.Fold(counter, size_, acc, init_, body_):
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
                    array_calculus.Fold(
                        counter,
                        size.normal,
                        acc,
                        init.aligned(acc_axes),
                        body.aligned(acc_axes),
                    ),
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
                        array_calculus.Cast(float, target.expr),
                    )
                else:
                    used_axes = axial._alignment(*(op._axes for op in ops))
                    return Axial(
                        used_axes,
                        array_calculus.ELEMENTWISE_UFUNCS[expr.ufunc](
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
            case _:
                assert_never(expr)
        assert False  # noqa

    def go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
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
    match_sum.cache_clear()
    match_einsum.cache_clear()
    array_indexing.match_index_clipped_shift.cache_clear()

    return array_program


def cancel_shape_ops(program: array_calculus.Expr) -> array_calculus.Expr:
    @cache
    def go(expr: array_calculus.Expr) -> array_calculus.Expr:
        match expr:
            case array_calculus.Transpose(permutation, target):
                if permutation == tuple(range(len(permutation))):
                    return go(target)
            case array_calculus.Unsqueeze(axes, target):
                if not axes:
                    return go(target)
            case array_calculus.Squeeze(axes, target):
                if not axes:
                    return go(target)
        return expr.map(go)

    return go(program)


def cancel_tuple_ops(program: array_calculus.Expr) -> array_calculus.Expr:
    @cache
    def go(expr: array_calculus.Expr) -> array_calculus.Expr:
        match expr:
            case array_calculus.Untuple(at, _arity, array_calculus.Tuple(elems)):
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
                if not item.free_indices and size_matches:
                    go(target)
                    return expr
            case calculus.Store(symbol, _inner_type):
                if symbol == counter:
                    ok = False
        return expr.map(go)

    go(body)
    return ok


@cache
def match_sum(
    expr: calculus.Expr,
) -> tuple[Variable, calculus.Expr, calculus.Expr] | None:
    match expr:
        case calculus.Fold(
            counter,
            size,
            acc,
            calculus.Const(init),
            calculus.Add((calculus.Store(acc_, _), body)),
        ):
            if (
                not init.array.ndim
                and float(init.array) == 0.0
                and acc == acc_
                and acc not in body.free_symbols
            ):
                return counter, size, body
    return None


def match_reduction_by_body(
    body: calculus.Expr, acc: Variable
) -> tuple[calculus.Expr, array_calculus.Reduce.Kind] | None:
    red = {
        calculus.Add: array_calculus.Reduce.Kind.add,
        calculus.Min: array_calculus.Reduce.Kind.minimum,
        calculus.Max: array_calculus.Reduce.Kind.maximum,
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
    update_size_classes(vec_expr, size_class, lambda e: counter in e.free_symbols)
    vec = go(vec_expr)
    reduced = array_calculus.Reduce(kind, vec.positional_axis(0), vec.expr)
    reduced_with_init = array_calculus.BinaryElementwise(
        array_calculus.REDUCE_UNDERLYING[kind], go(init).expr, reduced
    )
    return Axial(vec._axes, reduced_with_init)


@cache
def match_einsum(
    expr: calculus.Expr,
) -> tuple[set[Index], dict[Variable, calculus.Expr], tuple[calculus.Expr, ...]]:
    match expr:
        case calculus.Multiply((first, second)):
            first_axes, first_sums, first_operands = match_einsum(first)
            second_axes, second_sums, second_operands = match_einsum(second)
            return (
                first_axes | second_axes,
                first_sums | second_sums,
                first_operands + second_operands,
            )
    sum_pattern = match_sum(expr)
    if sum_pattern is not None:
        counter, size, body = sum_pattern
        axes, sizes, operands = match_einsum(body)
        return axes, {counter: size, **sizes}, operands
    return expr.free_indices, {}, (expr,)


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

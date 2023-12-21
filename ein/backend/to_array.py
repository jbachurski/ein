import itertools
from dataclasses import dataclass
from functools import cache
from string import ascii_lowercase
from typing import Callable, Iterable, Self, TypeAlias, TypeVar, assert_never, cast

from ein import calculus
from ein.midend.size_classes import SizeEquivalence, find_size_classes
from ein.midend.substitution import substitute
from ein.symbols import Index, Symbol, Variable
from ein.type_system import Pair, PrimitiveArrayType, to_float

from . import array_calculus, axial
from .axial import Axial

T = TypeVar("T")


def _first_opt(*args: T | None) -> T | None:
    for x in args:
        if x is not None:
            return x
    return None


@dataclass(frozen=True)
class Builder:
    expr: array_calculus.Expr

    @classmethod
    def const(cls, value) -> Self:
        return cls(array_calculus.Const(calculus.Value(value)))

    def __neg__(self) -> "Builder":
        return Builder(
            array_calculus.UnaryElementwise(
                array_calculus.UnaryElementwise.Kind.negative, self.expr
            )
        )

    def __add__(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.add, self.expr, other.expr
            )
        )

    def __sub__(self, other: "Builder") -> "Builder":
        return self + (-other)

    def min(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.minimum, self.expr, other.expr
            )
        )

    def max(self, other: "Builder") -> "Builder":
        return Builder(
            array_calculus.BinaryElementwise(
                array_calculus.BinaryElementwise.Kind.maximum, self.expr, other.expr
            )
        )


def transform(
    program: calculus.Expr,
    *,
    use_takes: bool = True,
    use_slice_pads: bool = True,
    use_slice_elision: bool = True,
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
                return transform_get(
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
    match_index_clipped_shift.cache_clear()

    return array_program


def transform_get(
    target_: calculus.Expr,
    item_: calculus.Expr,
    go: Callable[[calculus.Expr], Axial],
    size_class: SizeEquivalence,
    index_sizes: dict[Index, array_calculus.Expr],
    *,
    use_takes: bool,
    use_slice_pads: bool,
    use_slice_elision: bool,
) -> Axial:
    target = go(target_)
    rank = target.expr.type.single.rank
    ts = match_index_clipped_shift(item_)
    if use_slice_pads and ts is not None:
        index, (shift, low, high) = ts
        axes = target._axes + (index,)
        if index not in target_.free_indices:
            is_trivial = {shift, low, high} == {None}
            if (
                use_slice_elision
                and is_trivial
                and size_class.equiv(index, calculus.Dim(target_, 0))
            ):
                return Axial(axes, target.expr)
            if use_slice_pads:
                zero = Builder.const(0).expr
                size = index_sizes[index]
                size1 = (Builder(size) + Builder.const(-1)).expr
                return Axial(
                    axes,
                    pad_slice_get(
                        target.expr,
                        target.positional_axis(0),
                        go(shift).normal if shift is not None else zero,
                        go(low).normal if low is not None else zero,
                        go(high).normal if high is not None else size1,
                        size,
                    ),
                )
    item = go(item_)
    if use_takes and not (target.type.free_indices & item.type.free_indices):
        axes = target._axes + item._axes
        take_axes: list[array_calculus.Expr | None] = [None] * rank
        take_axes[target.positional_axis(0)] = item.expr
        return Axial(axes, array_calculus.Take(target.expr, tuple(take_axes)))
    return general_get(target, item)


def _put_at(n: int, p: int, x: T) -> tuple[T | None, ...]:
    return p * (None,) + (x,) + (n - p - 1) * (None,)


def pad_slice_get(
    target: array_calculus.Expr,
    axis: int,
    shift_: array_calculus.Expr,
    low_: array_calculus.Expr,
    high_: array_calculus.Expr,
    size_: array_calculus.Expr,
) -> array_calculus.Expr:
    rank = target.type.single.rank
    assert 0 <= axis < rank
    assert all(
        e.type.single == PrimitiveArrayType(0, int)
        for e in (shift_, low_, high_, size_)
    )
    dim = Builder(array_calculus.Dim(axis, target))

    shift = Builder(shift_)
    low = Builder(low_).max(Builder.const(0))
    high = Builder(high_).min(dim - Builder.const(1))
    size = Builder(size_)

    # We want to compute an array b such that
    #  b[i] = a[min(max(i + shift, low), high)]
    # using the operation
    #  b = pad(a[start:end], (left, right))
    # (where indexing is along the selected axis)
    start = shift.max(low).min(high)
    stop = (shift + size - Builder.const(1)).max(low).min(high) + Builder.const(1)
    left = low.max(-shift)
    right = size - (stop - start) - left

    return array_calculus.Pad(
        array_calculus.Slice(
            target,
            _put_at(rank, axis, start.expr),
            _put_at(rank, axis, stop.expr),
        ),
        _put_at(rank, axis, left.expr),
        _put_at(rank, axis, right.expr),
    )


def general_get(target: Axial, item: Axial) -> Axial:
    axes = axial._alignment(item._axes, target._axes)
    k = len(axes)
    expr = array_calculus.Squeeze(
        (k,),
        array_calculus.Gather(
            k,
            target.aligned(axes),
            array_calculus.Unsqueeze(
                tuple(
                    range(
                        len(axes),
                        len(axes) + target.type.type.single.rank,
                    )
                ),
                item.aligned(axes),
            ),
        ),
    )
    return Axial(axes, expr)


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


def safe_realise(body: calculus.Expr, counter: Variable) -> bool:
    ok: bool = True

    @cache
    def go(expr: calculus.Expr) -> calculus.Expr:
        nonlocal ok
        if expr not in expr.free_symbols:
            return expr
        match expr:
            case calculus.Get(target, item):
                if isinstance(item, calculus.Store):
                    return target
            case calculus.Store(symbol):
                if symbol == counter:
                    return expr
        return expr.map(go)

    go(body)
    return ok


ClippedShift: TypeAlias = tuple[
    calculus.Expr | None, calculus.Expr | None, calculus.Expr | None
]


@cache
def match_index_clipped_shift(expr: calculus.Expr) -> tuple[Index, ClippedShift] | None:
    if len(expr.free_indices) == 1:
        (index,) = expr.free_indices
        t = match_clipped_shift(expr, index)
        if t is not None:
            return index, t
    return None


@cache
def match_clipped_shift(expr: calculus.Expr, symbol: Symbol) -> ClippedShift | None:
    # expr is equivalent to the form min(max(symbol + shift, low), high)
    # where symbol occurs in no subterm of shift/low/high
    if not (expr.free_symbols <= {symbol}):
        return None
    if symbol not in expr.free_symbols:
        return None, None, None
    match expr:
        case calculus.Store(symbol_, _inner_type):
            if symbol == symbol_:
                return None, None, None
        case calculus.Add((first, second)):
            return _first_opt(
                _clip_shift_add(first, second, symbol),
                _clip_shift_add(second, first, symbol),
            )
        case calculus.Min((first, second)):
            return _first_opt(
                _clip_shift_min(first, second, symbol),
                _clip_shift_min(second, first, symbol),
            )
        case calculus.Max((first, second)):
            return _first_opt(
                _clip_shift_max(first, second, symbol),
                _clip_shift_max(second, first, symbol),
            )
    return None


def _clip_shift_add(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        if lo is None and hi is None:
            return calculus.Add((d, other)) if d is not None else other, None, None
    return None


def _clip_shift_min(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        hi = calculus.Min((hi, other)) if hi is not None else other
        return d, lo, hi
    return None


def _clip_shift_max(shifted: calculus.Expr, other: calculus.Expr, symbol: Symbol):
    t = match_clipped_shift(shifted, symbol)
    if t is not None:
        d, lo, hi = t
        lo = calculus.Max((lo, other)) if lo is not None else other
        hi = calculus.Max((hi, other)) if hi is not None else None
        return d, lo, hi
    return None


def match_shift(expr: calculus.Expr, symbol: Symbol) -> calculus.Expr | None:
    t = match_clipped_shift(expr, symbol)
    if t is not None:
        d, lo, hi = t
        return d if lo is None and hi is None else None
    return None


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

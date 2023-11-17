from functools import cache
from typing import assert_never

from ein import calculus
from ein.midend.size_classes import find_size_classes
from ein.symbols import Index, Variable
from ein.type_system import Pair, PrimitiveType, to_float

from . import array_calculus, axial
from .axial import Axial


def transform(
    program: calculus.Expr,
    *,
    use_slices: bool = True,
    slice_elision: bool = True,
    use_takes: bool = True,
    do_shape_cancellations: bool = True,
    do_tuple_cancellations: bool = True,
    do_inplace_on_temporaries: bool = True,
) -> array_calculus.Expr:
    transformed: dict[calculus.Expr, Axial] = {}

    size_class = find_size_classes(program)

    def _go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_vars: dict[Index, Variable],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        result: array_calculus.Expr
        match expr:
            case calculus.Const(value):
                return Axial([], array_calculus.Const(value.array))
            case calculus.At(index):
                if index in index_sizes:  # vector index
                    return Axial(
                        [index],
                        array_calculus.Range(index_sizes[index]),
                    )
                else:  # fold index
                    return Axial(
                        [],
                        array_calculus.Var(
                            index_vars[index],
                            PrimitiveType.of_array(0, int),
                        ),
                    )
            case calculus.Var(var, var_type):
                axes = var_axes.get(var, ())
                return Axial(
                    axes,
                    array_calculus.Var(
                        var, var_type.primitive_type.with_rank_delta(+len(axes))
                    ),
                )
            case calculus.Let(bindings_, body_):
                bindings = tuple(
                    (var, go(binding, index_sizes, index_vars, var_axes))
                    for var, binding in bindings_
                )
                return go(
                    body_,
                    index_sizes,
                    index_vars,
                    var_axes | {var: binding._axes for var, binding in bindings},
                ).within(*((var, binding.expr) for var, binding in bindings))
            case calculus.AssertEq(target_, _):
                return go(target_, index_sizes, index_vars, var_axes)
            case calculus.Dim(target_, pos):
                target = go(target_, index_sizes, index_vars, var_axes)
                return Axial(
                    [],
                    array_calculus.Dim(target.positional_axis(pos), target.expr),
                )
            case calculus.Fold(index, size_, acc, init_, body_):
                size = go(size_, index_sizes, index_vars, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile fold with index-dependent size"
                init = go(init_, index_sizes, index_vars, var_axes)
                index_var = Variable()
                # FIXME: We need to establish an alignment that includes free indices from both init and body.
                #  Doing this by constructing the body twice is a really bad idea.
                #  But just using body.free_indices isn't deterministic enough.
                pre_transformed = transformed.copy()
                acc_axes = go(
                    body_,
                    index_sizes,
                    index_vars | {index: index_var},
                    var_axes | {acc.var: init._axes},
                )._axes
                transformed.clear()
                transformed.update(pre_transformed)
                body = go(
                    body_,
                    index_sizes,
                    index_vars | {index: index_var},
                    var_axes | {acc.var: acc_axes},
                )
                return Axial(
                    acc_axes,
                    array_calculus.Fold(
                        index_var,
                        size.normal,
                        acc.var,
                        init.aligned(acc_axes),
                        body.aligned(acc_axes),
                    ),
                )
            case calculus.Get(
                target_, calculus.At(index)
            ) if index in index_sizes and use_slices:
                target = go(target_, index_sizes, index_vars, var_axes)
                if slice_elision and size_class.equiv(index, calculus.Dim(target_, 0)):
                    result = target.expr
                else:
                    rank = target.expr.type.single.rank
                    slice_axes: list[array_calculus.Expr | None] = [None] * rank
                    slice_axes[target.positional_axis(0)] = index_sizes[index]
                    result = array_calculus.Slice(target.expr, tuple(slice_axes))

                return Axial(target._axes + (index,), result)
            case calculus.Get(target_, item_):
                target = go(target_, index_sizes, index_vars, var_axes)
                item = go(item_, index_sizes, index_vars, var_axes)
                used_axes = axial._alignment(target._axes, item._axes)
                rank = target.expr.type.single.rank
                if use_takes and not item.expr.type.single.rank:
                    take_axes: list[array_calculus.Expr | None] = [None] * rank
                    take_axes[target.positional_axis(0)] = item.expr
                    result = array_calculus.Take(target.expr, tuple(take_axes))
                else:
                    k = len(used_axes)
                    result = array_calculus.Squeeze(
                        (k,),
                        array_calculus.Gather(
                            k,
                            target.aligned(used_axes),
                            array_calculus.Unsqueeze(
                                tuple(
                                    range(
                                        len(used_axes),
                                        len(used_axes) + target.type.type.single.rank,
                                    )
                                ),
                                item.aligned(used_axes),
                            ),
                        ),
                    )
                return Axial(used_axes, result)
            case calculus.Vec(index, size_, target_):
                size = go(size_, index_sizes, index_vars, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile index comprehension with index-dependent size"
                return go(
                    target_, index_sizes | {index: size.normal}, index_vars, var_axes
                ).along(index, size.expr)
            case calculus.AbstractScalarOperator(operands):
                ops = [go(op, index_sizes, index_vars, var_axes) for op in operands]
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
                first = go(first_, index_sizes, index_vars, var_axes)
                second = go(second_, index_sizes, index_vars, var_axes)
                return first.cons(second)
            case calculus.First(target_):
                assert isinstance(target_.type, Pair)
                k = len(target_.type.first.primitive_type.elems)
                return go(target_, index_sizes, index_vars, var_axes).slice_tuple(0, k)
            case calculus.Second(target_):
                assert isinstance(target_.type, Pair)
                k = len(target_.type.first.primitive_type.elems)
                n = len(target_.type.primitive_type.elems)
                return go(target_, index_sizes, index_vars, var_axes).slice_tuple(k, n)
            case _:
                assert_never(expr)
        assert False  # noqa

    def go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_vars: dict[Index, Variable],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        if expr not in transformed:
            transformed[expr] = _go(expr, index_sizes, index_vars, var_axes)
        return transformed[expr]

    # FIXME: Let-bound variables get different treatment here than free variables.
    #  Free variables are always assumed to depend on no indices (empty free-index set).
    array_program = go(program, {}, {}, {}).normal
    if do_shape_cancellations:
        array_program = cancel_shape_ops(array_program)
    if do_tuple_cancellations:
        array_program = cancel_tuple_ops(array_program)
    if do_inplace_on_temporaries:
        array_program = inplace_on_temporaries(array_program)

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


def inplace_on_temporaries(program: array_calculus.Expr) -> array_calculus.Expr:
    # Assumes that an elementwise operation used as an operand to another one
    # will not be broadcast (already has the same shape as the result).
    # Additionally, we assume that the result will not be reused anywhere else (needs to be let-bound).
    # This will also interact with any implicit-promotion optimisations, as here we assume dtypes are consistent.
    # This is why we only do this for BinaryElementwise and assume we have already done an explicit Cast.
    TEMPORARIES = (
        array_calculus.UnaryElementwise,
        array_calculus.BinaryElementwise,
        array_calculus.TernaryElementwise,
        array_calculus.Range,
        array_calculus.Cast,
    )

    @cache
    def go(expr: array_calculus.Expr) -> array_calculus.Expr:
        first: array_calculus.Expr
        second: array_calculus.Expr
        expr = expr.map(go)
        match expr:
            case array_calculus.BinaryElementwise(kind, first, second, None):
                # This logic is really shaky and interacts with optimisations to the number of axis manipulation calls.
                # Should have more in-depth analysis on what broadcasting might occur
                rank1, rank2 = first.type.single.rank, second.type.single.rank
                if rank1 == rank2:
                    if rank1 and isinstance(first, TEMPORARIES):
                        return array_calculus.BinaryElementwise(kind, first, second, 0)
                    if rank2 and isinstance(second, TEMPORARIES):
                        return array_calculus.BinaryElementwise(kind, first, second, 1)
        return expr

    return go(program)

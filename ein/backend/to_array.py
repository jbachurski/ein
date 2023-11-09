from functools import cache
from typing import assert_never

import numpy

from ein import calculus
from ein.symbols import Index, Variable
from ein.type_system import to_float

from . import array_calculus, axial
from .axial import Axial


def transform(
    program: calculus.Expr,
    *,
    use_slices: bool = True,
    slice_elision: bool = True,
    use_takes: bool = True,
    do_cancellations: bool = True,
) -> array_calculus.Expr:
    transformed: dict[calculus.Expr, Axial] = {}

    def _go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_class: dict[Index, set[calculus.Dim]],
        index_vars: dict[Index, Variable],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        result: array_calculus.Expr
        match expr:
            case calculus.Const(value):
                return Axial(
                    reversed(range(value.array.ndim)),
                    array_calculus.Const(value.array),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.At(index):
                if index in index_sizes:  # vector index
                    return Axial(
                        [index],
                        array_calculus.Range(index_sizes[index]),
                        expr.type.primitive_type.single.kind,
                    )
                else:  # fold index
                    return Axial(
                        [],
                        array_calculus.Var(index_vars[index], 0),
                        expr.type.primitive_type.single.kind,
                    )
            case calculus.Var(var, var_type):
                rank = var_type.primitive_type.single.rank
                axes = var_axes.get(var, list(reversed(range(rank))))
                return Axial(
                    axes,
                    array_calculus.Var(var, len(axes)),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Let(bindings_, body_):
                bindings = tuple(
                    (var, go(binding, index_sizes, index_class, index_vars, var_axes))
                    for var, binding in bindings_
                )
                return go(
                    body_,
                    index_sizes,
                    index_class,
                    index_vars,
                    var_axes | {var: binding.axes for var, binding in bindings},
                ).within(*((var, binding.normal) for var, binding in bindings))
            case calculus.AssertEq(target_, _):
                return go(target_, index_sizes, index_class, index_vars, var_axes)
            case calculus.Dim(target_, pos):
                target = go(target_, index_sizes, index_class, index_vars, var_axes)
                pos = target.type.array_type.rank - pos - 1
                return Axial(
                    [],
                    array_calculus.Dim(target.axes.index(pos), target.array),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Fold(index, size_, body_, init_, acc):
                size = go(size_, index_sizes, index_class, index_vars, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile fold with vector-index-dependent size"
                init = go(init_, index_sizes, index_class, index_vars, var_axes)
                index_var = Variable()
                # FIXME: We need to establish an alignment that includes free indices from both init and body.
                #  Doing this by constructing the body twice is a really bad idea.
                #  But just using body.free_indices isn't deterministic enough.
                pre_transformed = transformed.copy()
                acc_axes = go(
                    body_,
                    index_sizes,
                    index_class,
                    index_vars | {index: index_var},
                    var_axes | {acc.var: init.axes},
                ).axes
                transformed.clear()
                transformed.update(pre_transformed)
                body = go(
                    body_,
                    index_sizes,
                    index_class,
                    index_vars | {index: index_var},
                    var_axes | {acc.var: acc_axes},
                )
                return Axial(
                    acc_axes,
                    array_calculus.Fold(
                        index_var,
                        acc.var,
                        axial.align(init, acc_axes),
                        size.normal,
                        axial.align(body, acc_axes),
                    ),
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Get(
                target_, calculus.At(index)
            ) if index in index_sizes and use_slices:
                target = go(target_, index_sizes, index_class, index_vars, var_axes)
                at_axis = target.type.array_type.rank - 1
                if not slice_elision or not any(
                    dim.operand == target_ for dim in index_class[index]
                ):
                    result = array_calculus.Slice(
                        target.array,
                        tuple(
                            index_sizes[index] if axis == at_axis else None
                            for axis in target.axes
                        ),
                    )
                else:
                    result = target.array
                return Axial(
                    [axis if axis != at_axis else index for axis in target.axes],
                    result,
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Get(target_, item_):
                target = go(target_, index_sizes, index_class, index_vars, var_axes)
                item = go(item_, index_sizes, index_class, index_vars, var_axes)
                used_axes = axial.alignment(target.axes, item.axes)
                at_axis = target.type.array_type.rank - 1
                k = used_axes.index(at_axis)
                if use_takes and not item.axes:
                    result = array_calculus.Take(
                        target.array,
                        tuple(
                            item.array if axis == at_axis else None
                            for axis in used_axes
                        ),
                    )
                else:
                    result = array_calculus.Squeeze(
                        (k,),
                        array_calculus.Gather(
                            k,
                            axial.align(target, used_axes),
                            axial.align(item, used_axes),
                        ),
                    )
                return Axial(
                    used_axes[:k] + used_axes[k + 1 :],
                    result,
                    expr.type.primitive_type.single.kind,
                )
            case calculus.Vec(index, size_, target_):
                size = go(size_, index_sizes, index_class, index_vars, var_axes)
                assert (
                    not size.type.free_indices
                ), "Cannot compile index comprehension with vector-index-dependent size"
                size_class = set()
                if isinstance(size_, calculus.AssertEq):
                    size_class = {
                        e for e in size_.operands if isinstance(e, calculus.Dim)
                    }
                target = go(
                    target_,
                    index_sizes | {index: size.normal},
                    index_class | {index: size_class},
                    index_vars,
                    var_axes,
                )
                if index in target.axes:
                    return Axial(
                        (
                            target.type.array_type.rank if axis == index else axis
                            for axis in target.axes
                        ),
                        target.array,
                        expr.type.primitive_type.single.kind,
                    )
                else:
                    return Axial(
                        (target.type.array_type.rank, *target.axes),
                        array_calculus.Repeat(
                            0, size.array, array_calculus.Unsqueeze((0,), target.array)
                        ),
                        expr.type.primitive_type.single.kind,
                    )
            case calculus.AbstractScalarOperator(operands):
                ops = [
                    go(op, index_sizes, index_class, index_vars, var_axes)
                    for op in operands
                ]
                used_axes = axial.alignment(*(op.axes for op in ops))
                if expr.ufunc == to_float:
                    (target,) = ops
                    return Axial(
                        used_axes,
                        array_calculus.Cast(numpy.dtype(float), target.array),
                        expr.type.primitive_type.single.kind,
                    )
                else:
                    return Axial(
                        used_axes,
                        array_calculus.ELEMENTWISE[expr.ufunc](
                            *(axial.align(op, used_axes) for op in ops)
                        ),
                        expr.type.primitive_type.single.kind,
                    )
            case calculus.Cons() | calculus.First() | calculus.Second():
                raise NotImplementedError(
                    "Pairs are not yet implemented in the NumPy backend"
                )
            case _:
                assert_never(expr)
        assert False  # noqa

    def go(
        expr: calculus.Expr,
        index_sizes: dict[Index, array_calculus.Expr],
        index_class: dict[Index, set[calculus.Dim]],
        index_vars: dict[Index, Variable],
        var_axes: dict[Variable, axial.Axes],
    ) -> Axial:
        if expr not in transformed:
            transformed[expr] = _go(
                expr, index_sizes, index_class, index_vars, var_axes
            )
        return transformed[expr]

    # FIXME: Let-bound variables get different treatment here than free variables.
    #  Free variables are always assumed to depend on no indices (empty free-index set).
    array_program = go(program, {}, {}, {}, {}).normal
    if do_cancellations:
        array_program = cancel_ops(array_program)

    return array_program


def cancel_ops(program: array_calculus.Expr) -> array_calculus.Expr:
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

# TODO: Unify the axial calculus with the base one by having both positional and named axes.

from functools import cache
from typing import assert_never

from . import axial_calculus, calculus
from .axial_calculus import ValueAxis, VariableAxis
from .calculus import Index, Variable


def transform(
    program: calculus.Expr, ranks: dict[Variable, int]
) -> tuple[axial_calculus.Expr, tuple[Index, ...]]:
    @cache
    def implied_axes(expr: calculus.Expr) -> tuple[Index, ...]:
        match expr:
            case calculus.Vec(index, size, body):
                assert not implied_axes(size)
                return index, *implied_axes(body)
            case calculus.Sum(_index, size, body):
                assert not implied_axes(size)
                assert implied_axes(body) == ()
                return ()
            case calculus.Get(target, item):
                assert not implied_axes(item)
                return implied_axes(target)[1:]
            case calculus.Const(value):
                return tuple(ValueAxis(value, i) for i in range(value.array.ndim))
            case calculus.At(_index):
                return ()
            case calculus.Var(var):
                return tuple(VariableAxis(var, i) for i in range(ranks[var]))
            case calculus.VarShape(var, axis):
                assert 0 <= axis < ranks[var]
                return ()
            case calculus.Negate(operands):
                (target,) = operands
                assert implied_axes(target) == ()
                return ()
            case calculus.Reciprocal(operands):
                (target,) = operands
                assert implied_axes(target) == ()
                return ()
            case calculus.Add(operands):
                first, second = operands
                assert implied_axes(first) == implied_axes(second) == ()
                return ()
            case calculus.Multiply(operands):
                first, second = operands
                assert implied_axes(first) == implied_axes(second) == ()
                return ()
            case _:
                assert_never(expr)

    transformed: dict[calculus.Expr, axial_calculus.Expr] = {}

    def go(
        expr: calculus.Expr, sizes: dict[Index, axial_calculus.Expr]
    ) -> axial_calculus.Expr:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    def _go(
        expr: calculus.Expr, sizes: dict[Index, axial_calculus.Expr]
    ) -> axial_calculus.Expr:
        match expr:
            case calculus.Vec(index, size, body):
                return go(body, {index: go(size, sizes)} | sizes)
            case calculus.Sum(index, size, body):
                return axial_calculus.Sum(
                    index, go(body, {index: go(size, sizes)} | sizes)
                )
            case calculus.Get(target, item):
                axis, *_ = implied_axes(target)
                return axial_calculus.Get(go(target, sizes), go(item, sizes), axis)
            case calculus.Const(value):
                return axial_calculus.Const(value)
            case calculus.At(index):
                assert index in sizes
                return axial_calculus.Range(index, sizes[index])
            case calculus.Var(var):
                return axial_calculus.Var(var)
            case calculus.VarShape(var, axis):
                return axial_calculus.VarShape(var, axis)
            case calculus.Negate(operands):
                (target,) = operands
                return axial_calculus.Negate((go(target, sizes),))
            case calculus.Reciprocal(operands):
                (target,) = operands
                return axial_calculus.Reciprocal((go(target, sizes),))
            case calculus.Add(operands):
                first, second = operands
                return axial_calculus.Add((go(first, sizes), (go(second, sizes))))
            case calculus.Multiply(operands):
                first, second = operands
                return axial_calculus.Multiply((go(first, sizes), (go(second, sizes))))
            case _:
                assert_never(program)

    return go(program, {}), implied_axes(program)

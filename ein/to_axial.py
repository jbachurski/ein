# TODO: Unify the axial calculus with the base one by having both positional and named axes.

from functools import cache
from typing import assert_never

import numpy

from . import axial_calculus, calculus
from .calculus import Index, Value, Variable


def transform(
    program: calculus.Expr, ranks: dict[Variable, int]
) -> axial_calculus.Expr:
    @cache
    def implied_axes(expr: calculus.Expr) -> tuple[tuple[Index, calculus.Expr], ...]:
        match expr:
            case calculus.Vec(index, size, body):
                assert not implied_axes(size)
                return (index, size), *implied_axes(body)
            case calculus.Sum(_index, size, body):
                assert not implied_axes(size)
                assert implied_axes(body) == ()
                return ()
            case calculus.Get(target, item):
                assert not implied_axes(item)
                return implied_axes(target)[1:]
            case calculus.Const(value):
                return tuple(
                    (Index(), calculus.Const(Value(numpy.array(d))))
                    for d in value.array.shape
                )
            case calculus.At(_index):
                return ()
            case calculus.Var(var):
                return tuple(
                    (Index(), calculus.VarShape(var, i)) for i in range(ranks[var])
                )
            case calculus.VarShape(var, axis):
                assert 0 <= axis < len(implied_axes(var))
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
                assert_never(program)

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
                (axis, _), *_ = implied_axes(target)
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
                return axial_calculus.Add((go(first, sizes), (go(second, sizes))))
            case _:
                assert_never(program)

    return go(program, {})

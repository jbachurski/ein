# TODO: Unify the axial calculus with the base one by having both positional and named axes.

from functools import cache
from typing import assert_never

from . import calculus
from .calculus import Index, ValueAxis, Variable, VariableAxis


def transform(
    program: calculus.Expr, ranks: dict[Variable, int]
) -> tuple[calculus.Expr, tuple[Index, ...]]:
    @cache
    def implied_positional_axes(expr: calculus.Expr) -> tuple[Index, ...]:
        match expr:
            case calculus.Vec(index, size, body):
                assert not implied_positional_axes(size)
                return index, *implied_positional_axes(body)
            case calculus.AbstractScalarReduction(_index, size, body):
                assert not implied_positional_axes(size)
                assert implied_positional_axes(body) == ()
                return ()
            case calculus.AbstractScalarAxisReduction(_index, body):
                assert implied_positional_axes(body) == ()
                return ()
            case calculus.Get(operand, item):
                assert not implied_positional_axes(item)
                return implied_positional_axes(operand)[1:]
            case calculus.Const(value):
                return tuple(ValueAxis(value, i) for i in range(value.array.ndim))
            case calculus.At(_index):
                return ()
            case calculus.Range(_index):
                return ()
            case calculus.Var(var):
                return tuple(VariableAxis(var, i) for i in range(ranks[var]))
            case calculus.Dim(target, axis):
                if isinstance(axis, int):
                    assert 0 <= axis < len(implied_positional_axes(target))
                return ()
            case calculus.Where(cond, false, true):
                assert (
                    implied_positional_axes(cond)
                    == implied_positional_axes(false)
                    == implied_positional_axes(true)
                    == ()
                )
                return ()
            case calculus.AbstractScalarOperator(operands):
                assert all(
                    implied_positional_axes(operand) == () for operand in operands
                )
                return ()
            case _:
                assert_never(expr)

    transformed: dict[calculus.Expr, calculus.Expr] = {}

    def go(expr: calculus.Expr, sizes: dict[Index, calculus.Expr]) -> calculus.Expr:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    def _go(expr: calculus.Expr, sizes: dict[Index, calculus.Expr]) -> calculus.Expr:
        match expr:
            case calculus.Vec(index, size, body):
                return go(body, {index: go(size, sizes)} | sizes)
            case calculus.Sum(index, size, body):
                return calculus.AxisSum(
                    index, go(body, {index: go(size, sizes)} | sizes)
                )
            case calculus.Maximum(index, size, body):
                return calculus.AxisMaximum(
                    index, go(body, {index: go(size, sizes)} | sizes)
                )
            case calculus.AbstractScalarAxisReduction(index, body):
                return type(expr)(index, go(body, sizes))
            case calculus.Get(operand, item, None):
                axis, *_ = implied_positional_axes(operand)
                return calculus.Get(go(operand, sizes), go(item, sizes), axis)
            case calculus.Get(operand, item, axis):
                return calculus.Get(go(operand, sizes), go(item, sizes), axis)
            case calculus.Const(value):
                return calculus.Const(value)
            case calculus.At(index):
                return calculus.Range(index, sizes[index])
            case calculus.Range(axis, size):
                return calculus.Range(axis, go(size, sizes))
            case calculus.Var(var):
                return calculus.Var(var)
            case calculus.Dim(operand, int(positional_axis)):
                return calculus.Dim(
                    go(operand, sizes),
                    implied_positional_axes(operand)[positional_axis],
                )
            case calculus.Dim(operand, axis):
                return calculus.Dim(go(operand, sizes), axis)
            case calculus.Where(cond, true, false):
                return calculus.Where(
                    go(cond, sizes), go(true, sizes), go(false, sizes)
                )
            case calculus.Negate(operands):
                (target,) = operands
                return calculus.Negate((go(target, sizes),))
            case calculus.Reciprocal(operands):
                (target,) = operands
                return calculus.Reciprocal((go(target, sizes),))
            case calculus.LogicalNot(operands):
                (target,) = operands
                return calculus.LogicalNot((go(target, sizes),))
            case calculus.Add(operands):
                first, second = operands
                return calculus.Add((go(first, sizes), (go(second, sizes))))
            case calculus.Multiply(operands):
                first, second = operands
                return calculus.Multiply((go(first, sizes), (go(second, sizes))))
            case calculus.Less(operands):
                first, second = operands
                return calculus.Less((go(first, sizes), (go(second, sizes))))
            case calculus.LogicalAnd(operands):
                first, second = operands
                return calculus.LogicalAnd((go(first, sizes), (go(second, sizes))))
            case _:
                assert_never(expr)

    return go(program, {}), implied_positional_axes(program)

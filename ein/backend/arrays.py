from typing import assert_never

import numpy.typing

from ein import calculus
from ein.backend import axial
from ein.backend.axial import Axial, StagedAxial
from ein.calculus import Expr
from ein.symbols import Index, Variable


def to_axial(program: Expr) -> StagedAxial:
    transformed: dict[Expr, StagedAxial] = {}

    def _go(expr: Expr, sizes: dict[Index, Expr]) -> StagedAxial:
        match expr:
            case calculus.Const(value):
                return axial.Const(value.array).stage()
            case calculus.At(index):
                return axial.Range(index).stage(go(sizes[index], {}))
            case calculus.Var(var, type_):
                return axial.Var(var, type_).stage()
            case calculus.Dim(operand, axis):
                return axial.Dim(axis).stage(go(operand, sizes))
            case calculus.Get(operand, item):
                return axial.Gather().stage(go(operand, sizes), go(item, sizes))
            case calculus.Vec(index, size, body):
                return axial.Vector(index).stage(
                    go(size, {}), go(body, {index: size, **sizes})
                )
            case calculus.AbstractScalarReduction(index, size, body):
                return axial.Reduce(expr.ufunc, index).stage(
                    go(size, {}), go(body, {index: size, **sizes})
                )
            case calculus.AbstractScalarOperator(operands):
                return axial.Elementwise(expr.ufunc).stage(
                    *(go(operand, sizes) for operand in operands)
                )
            case _:
                assert_never(expr)

    def go(expr: Expr, sizes: dict[Index, Expr]) -> StagedAxial:
        if expr not in transformed:
            transformed[expr] = _go(expr, sizes)
        return transformed[expr]

    return go(program, {})


def interpret(program: Expr, env: dict[Variable, numpy.ndarray]) -> numpy.ndarray:
    staged_program = to_axial(program)
    results: dict[StagedAxial, Axial] = {}

    def go(staged: StagedAxial) -> Axial:
        if staged not in results:
            args = [go(operand) for operand in staged.operands]
            results[staged] = staged.operation.apply(*args, env=env)
        return results[staged]

    result = go(staged_program)
    assert not result.type.free_indices
    rank = result.type.type.rank
    inv: list[int | None] = [None for _ in range(rank)]
    for i, p in enumerate(result.axes):
        assert isinstance(p, int)
        # Axes are numbered in reverse order for axials
        inv[rank - p - 1] = i

    return numpy.transpose(result.value, inv)


__all__ = ["interpret"]

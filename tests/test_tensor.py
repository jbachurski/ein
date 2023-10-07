from typing import Any

import numpy

from ein.calculus import Var, Variable, VarShape
from ein.interpret import interpret
from ein.tensor import Tensor, array, sum


def repr_node(nodes, i):
    node = nodes[i]
    args, kwargs = node.mapped_args(
        lambda var: str(var), lambda arg: f"%{nodes.index(arg)}", lambda x: str(x)
    )
    return f"%{i} = {node.fun.__name__}({', '.join(map(str, args))}, {', '.join(f'{k}={v}' for k, v in kwargs.items())})"


def interpret_with_numpy(program, env: dict[Variable, numpy.ndarray]):
    from ein import to_axial, to_numpy

    ranks = {var: array.ndim for var, array in env.items()}
    axial_program, axial_program_axes = to_axial.transform(program, ranks)
    numpy_program = to_numpy.transform(axial_program, axial_program_axes, ranks)
    nodes = list(numpy_program.linearize())
    results: list[Any] = []
    for i, node in enumerate(nodes):
        print(repr_node(nodes, i))
        args, kwargs = node.mapped_args(
            lambda var: env[var], lambda arg: results[nodes.index(arg)], lambda x: x
        )
        results.append(node.fun(*args, **kwargs))
        print(results[-1])
    return results[-1]


def test_mul_grid():
    n0 = Variable()
    n = Var(n0)
    # FIXME: This should have proper casting behaviour.
    grid = array[n, n](lambda i, j: (i + 1.0) / (j + 1.0))
    numpy.testing.assert_allclose(
        interpret(
            grid.expr,
            {n0: numpy.array(5)},
        ),
        numpy.array([[i / j for j in range(1, 6)] for i in range(1, 6)]),
    )


def test_matmul():
    a0, b0 = Variable(), Variable()
    n, k = VarShape(a0, 0), VarShape(a0, 1)
    _k, m = VarShape(b0, 0), VarShape(b0, 1)
    a, b = Tensor(Var(a0)), Tensor(Var(b0))
    matmul = array[n, m](lambda i, j: sum[k](lambda t: a[i, t] * b[t, j]))
    first = numpy.array([[1, 2, 3], [4, 5, 6]])
    second = numpy.array([[1], [0], [-1]])
    numpy.testing.assert_allclose(
        interpret(
            matmul.expr,
            {
                a0: first,
                b0: second,
            },
        ),
        first @ second,
    )
    numpy.testing.assert_allclose(
        interpret_with_numpy(
            matmul.expr,
            {
                a0: first,
                b0: second,
            },
        ),
        first @ second,
    )

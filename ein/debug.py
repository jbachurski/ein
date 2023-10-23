from typing import Any

from . import calculus

try:
    import pydot
except ImportError:
    pydot = None


def graph(program):
    dot = pydot.Dot(graph_type="digraph")
    vis: dict[Any, str] = {}
    ids: dict[type[Any], int] = {}

    def go(expr) -> str:
        if expr in vis:
            return vis[expr]
        i = ids.get(type(expr), 0)
        name = vis[expr] = f"{type(expr).__name__}_{i}"
        ids[type(expr)] = i + 1
        meta, preds = expr.debug
        label = (
            "<"
            + f"<b>{type(expr).__name__}</b><br/>"
            + (
                f"<i>{getattr(expr, 'type').pretty}</i><br/>"
                if hasattr(expr, "type")
                else ""
            )
            + "<br/>".join(
                f"{key}={value if not hasattr(value, '__call__') else value.__name__}"
                for key, value in meta.items()
            )
            + ">"
        )
        dot.add_node(pydot.Node(name, label=label, shape="box", fontname="Courier New"))
        for pred_name in (go(pred) for pred in preds):
            dot.add_edge(pydot.Edge(pred_name, name))
        return name

    go(program)
    return dot


def transform_graphs(
    program: calculus.Expr,
) -> "tuple[pydot.Dot, pydot.Dot, pydot.Dot]":
    from backend import to_array, to_axial

    axial_program = to_axial.transform(program)
    array_program = to_array.transform(axial_program)
    return graph(program), graph(axial_program), graph(array_program)

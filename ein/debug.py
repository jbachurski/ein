import html
import os.path
from typing import Any, Iterable, assert_never, cast

from ein.codegen import phi_to_yarr, yarr
from ein.midend import lining
from ein.phi import calculus

try:
    import pydot
except ImportError:
    pydot = None


def _snip_limit(s: str, n: int) -> str:
    while "  " in s:
        s = s.replace("  ", " ")
    return s[: n // 2 - 1] + "..." + s[-n // 2 + 1 :] if len(s) > n else s


def _meta_value_repr(value: Any) -> str:
    return _snip_limit(
        str(
            html.escape(repr(value))
            if not hasattr(value, "__call__") or not hasattr(value, "__name__")
            else value.__name__
        ),
        30,
    )


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
                f"{key}={_meta_value_repr(value)}" for key, value in meta.items()
            )
            + ">"
        )
        dot.add_node(pydot.Node(name, label=label, shape="box", fontname="Courier New"))
        for pred_name in (go(pred) for pred in preds):
            dot.add_edge(pydot.Edge(pred_name, name))
        return name

    go(program)
    return dot


def array_from_phi_program(program: calculus.Expr) -> yarr.Expr:
    return phi_to_yarr.transform(program)


def plot_graph(dot: pydot.Dot) -> None:
    import io

    import matplotlib.image as img
    import matplotlib.pyplot as plt

    arr = img.imread(io.BytesIO(bytes(dot.create(format="png"))))
    dpi = 100
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(arr)
    plt.show()


def plot_phi_graph(program: calculus.Expr) -> None:
    plot_graph(graph(program))


def plot_array_graph(program: calculus.Expr) -> None:
    plot_graph(graph(array_from_phi_program(program)))


def pretty_print(program: calculus.Expr) -> str:
    def go(expr: calculus.Expr) -> Iterable[str]:
        match expr:
            case calculus.Const(value):
                yield _meta_value_repr(value)
            case calculus.Store(symbol, _inner_type):
                yield str(symbol)
            case calculus.Let(var, bind, body):
                yield f"let {var} ="
                yield from ("  " + line for line in go(bind))
                yield "in"
                yield from ("  " + line for line in go(body))
            case calculus.AssertEq(target, operands):
                yield f"assert {' = '.join(' '.join(go(op)) for op in operands)} in"
                yield from ("  " + line for line in go(target))
            case calculus.Dim(target, pos):
                yield f"({' '.join(go(target))}).|{pos}|"
            case calculus.Fold(counter, size, acc, init, body):
                yield f"fold {counter} count"
                yield from ("  " + line for line in go(size))
                yield f"from {acc} = "
                yield from ("  " + line for line in go(init))
                yield "by"
                yield from ("  " + line for line in go(body))
            case calculus.Reduce(init, x, y, xy, vecs):
                yield "reduce"
                for vec in vecs:
                    yield from ("  " + line for line in go(vec))
                yield "from"
                yield from ("  " + line for line in go(init))
                yield f"where {x} * {y} = "
                yield from ("  " + line for line in go(xy))
            case calculus.Get(target, item):
                yield "get"
                yield from ("  " + line for line in go(target))
                yield "at"
                yield from ("  " + line for line in go(item))
            case calculus.Concat(first, second):
                yield "concat"
                yield from ("  " + line for line in go(first))
                yield from ("  " + line for line in go(second))
            case calculus.Vec(index, size, target):
                yield f"array {index} size"
                yield from ("  " + line for line in go(size))
                yield "of"
                yield from ("  " + line for line in go(target))
            case calculus.AbstractScalarOperator(operands):
                yield getattr(
                    expr.ufunc,
                    "name",
                    getattr(expr.ufunc, "__name__", repr(expr.ufunc)),
                )
                for op in operands:
                    yield from ("  " + line for line in go(op))
            case calculus.Cons(first, second):
                yield "("
                yield from ("  " + line for line in go(first))
                yield from ("  " + line for line in go(second))
                yield ")"
            case calculus.First(target):
                yield "fst"
                yield from ("  " + line for line in go(target))
            case calculus.Second(target):
                yield "snd"
                yield from ("  " + line for line in go(target))
            case calculus.Extrinsic(_type, fun, operands):
                yield f"extrinsic<{fun.__name__}>"
                for op in operands:
                    yield from ("  " + line for line in go(op))
            case _:
                assert_never(expr)

    return "\n".join(go(cast(calculus.Expr, lining.outline(lining.inline(program)))))


def save_graph(dot: pydot.Dot, path: str, fmt: str | None = None) -> None:
    if fmt is None:
        path_no_ext, ext = os.path.splitext(path)
        if ext:
            fmt = ext
            path = path_no_ext
        else:
            fmt = "png"
    dot.write(os.path.expanduser(path) + f".{fmt}", format=fmt)

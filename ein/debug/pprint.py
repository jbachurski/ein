import abc
from dataclasses import dataclass

import numpy

from ein.codegen import phi_to_yarr, yarr
from ein.midend.lining import outline
from ein.midend.structs import struct_of_arrays_transform
from ein.phi import phi


class Repr(abc.ABC):
    @abc.abstractmethod
    def repr(self, width: int, indent: int, after: int) -> str:
        ...


@dataclass(frozen=True)
class Line(Repr):
    content: str

    def repr(
        self,
        width: int | None = None,
        indent: int | None = None,
        after: int | None = None,
    ) -> str:
        return self.content


@dataclass(frozen=True)
class Concat(Repr):
    parts: tuple[Repr, ...]

    def repr(self, width: int, indent: int, after: int) -> str:
        result: list[str] = []
        for part in self.parts:
            curr_line = ("".join(result).splitlines() or ["?" * after])[-1]
            result += part.repr(width, indent, len(curr_line))
        return "".join(result)


@dataclass(frozen=True)
class Nest(Repr):
    blocks: tuple[tuple[str, Repr, str], ...]

    def repr(self, width: int, indent: int, after: int) -> str:
        result: list[str] = []
        for prefix, block, suffix in self.blocks:
            curr_line = (("".join(result) + prefix).splitlines() or ["?" * after])[-1]
            sub = block.repr(width, indent, len(curr_line))
            badness = max(
                (len(curr_line) if not i else 0) + len(ln)
                for i, ln in enumerate(sub.splitlines())
            )
            if badness <= width and len(sub.splitlines()) <= 1:
                result += [prefix, sub, suffix]
            else:
                sub = block.repr(width, indent, len(curr_line) + indent)
                result += [
                    prefix,
                    "\n",
                    "\n".join(" " * indent + ln for ln in sub.splitlines()),
                    "\n",
                    suffix.lstrip(" "),
                ]
        return "".join(result)


OPS = {
    numpy.add: "+",
    numpy.multiply: "*",
    numpy.subtract: "-",
    numpy.negative: "-",
}


def pars(r: Repr, par: bool) -> Repr:
    return Concat((Line("("), r, Line(")"))) if par else r


def _pretty_call(name: str | Repr, *args: Repr):
    return Concat(
        (
            Line(name) if isinstance(name, str) else name,
            Nest(
                tuple(
                    [
                        ("(" if not i else ", ", arg, "" if i + 1 < len(args) else ")")
                        for i, arg in enumerate(args)
                    ]
                )
            ),
        )
    )


def _pretty_let(symbol: str, bound_type: str, bound: Repr, body: Repr) -> Repr:
    return Concat(
        (
            Nest(
                (
                    (
                        f"let {symbol}: {bound_type} = ",
                        bound,
                        " ",
                    ),
                )
            ),
            Line("in\n"),
            body,
        ),
    )


def _pretty_fold(
    counter: str, acc_type: str, size: Repr, acc: str, init: Repr, body: Repr
) -> Repr:
    return Nest(
        (
            (f"fold {counter}[", size, "]"),
            (f" init {acc}: {acc_type} = ", init, ""),
            (f" by {acc} => ", body, ""),
        )
    )


def _pretty_reduce(
    vecs: list[Repr],
    init_type: str,
    init: Repr,
    axis: str | None,
    x: str,
    y: str,
    xy: Repr,
) -> Repr:
    return Concat(
        (
            Nest(
                (
                    (f"reduce[id: {init_type} = ", init, ", "),
                    (f"({x}, {y}) => ", xy, f", axis={axis}]" if axis else "]"),
                )
            ),
            _pretty_call("", *vecs),
        )
    )


def _pretty_phi(expr: phi.Expr, par: bool) -> Repr:
    match expr:
        case phi.Const(value):
            value = numpy.array(value.array).tolist()
            return Line(str(value))
        case phi.Store(symbol, _):
            return Line(str(symbol))
        case phi.Let(var, bound, body):
            return pars(
                _pretty_let(
                    str(var),
                    str(bound.type),
                    _pretty_phi(bound, False),
                    _pretty_phi(body, False),
                ),
                par,
            )
        case phi.Vec(index, size, body):
            return pars(
                Nest(
                    (
                        (f"for {index}[", _pretty_phi(size, False), "]. "),
                        ("", _pretty_phi(body, False), ""),
                    )
                ),
                par,
            )
        case phi.Get(vec, item):
            return Concat(
                (_pretty_phi(vec, True), Line("["), _pretty_phi(item, False), Line("]"))
            )
        case phi.Dim(target_, axis):
            return _pretty_call(f"size[{axis}]", _pretty_phi(target_, False))
        case phi.AssertEq(target_, targets_):
            targets_ = (target_, *targets_)
            targets = [_pretty_phi(t, False) for t in targets_]
            ret: list[Repr] = [Line("{")]
            for target in targets:
                ret.append(target)
                ret.append(Line(", "))
            if targets:
                ret.pop()
            ret.append(Line("}"))
            return Concat(tuple(ret))
        case phi.Fold(counter, size, acc, init, body):
            return pars(
                _pretty_fold(
                    str(counter),
                    str(init.type),
                    _pretty_phi(size, False),
                    str(acc),
                    _pretty_phi(init, False),
                    _pretty_phi(body, False),
                ),
                par,
            )
        case phi.Reduce(init, x, y, xy, vecs):
            return pars(
                _pretty_reduce(
                    [_pretty_phi(vec, False) for vec in vecs],
                    str(init.type),
                    _pretty_phi(init, False),
                    None,
                    str(x),
                    str(y),
                    _pretty_phi(xy, False),
                ),
                par,
            )
        case phi.First(target):
            return _pretty_call("fst", _pretty_phi(target, False))
        case phi.Second(target):
            return _pretty_call("snd", _pretty_phi(target, False))
        case phi.Cons(first, second):
            return _pretty_call(
                "", _pretty_phi(first, False), _pretty_phi(second, False)
            )
        case phi.Where((cond, true, false)):
            return Concat(
                (
                    Line("if "),
                    _pretty_phi(cond, False),
                    Line(" then "),
                    _pretty_phi(true, False),
                    Line(" else "),
                    _pretty_phi(false, False),
                )
            )
        case phi.AbstractScalarOperator(ops_):
            ops = [_pretty_phi(op, True) for op in ops_]
            sign = OPS.get(expr.ufunc, type(expr).__name__.lower())
            match ops:
                case (op0,):
                    return pars(Concat((Line(sign), op0)), par)
                case (op1, op2):
                    return pars(Concat((op1, Line(f" {sign} "), op2)), par)
            return _pretty_call(sign, *ops)
    return Line(str(expr))


def _pretty_yarr(expr: yarr.Expr, par: bool) -> Repr:
    match expr:
        case yarr.Const(value):
            value = numpy.array(value.array).tolist()
            return Line(str(value))
        case yarr.Var(var, _):
            return Line(str(var))
        case yarr.Let(var, bound, body):
            return pars(
                _pretty_let(
                    str(var),
                    str(bound.type),
                    _pretty_yarr(bound, False),
                    _pretty_yarr(body, False),
                ),
                par,
            )
        case yarr.Fold(counter, size, acc, init, body):
            return pars(
                _pretty_fold(
                    str(counter),
                    str(init.type),
                    _pretty_yarr(size, False),
                    str(acc),
                    _pretty_yarr(init, False),
                    _pretty_yarr(body, False),
                ),
                par,
            )
        case yarr.Reduce(init, x, y, xy, vecs, axis):
            return pars(
                _pretty_reduce(
                    [_pretty_yarr(vecs, False)],
                    str(init.type),
                    _pretty_yarr(init, False),
                    str(axis),
                    str(x),
                    str(y),
                    _pretty_yarr(xy, False),
                ),
                par,
            )
        case yarr.Dim(axis, target_):
            return _pretty_call(f"size[{axis}]", _pretty_yarr(target_, False))
        case yarr.Untuple(at, _, target):
            return Concat((_pretty_yarr(target, True), Line(f".{at}")))
        case yarr.Tuple(operands_):
            operands = [_pretty_yarr(o_, False) for o_ in operands_]
            return _pretty_call("", *operands)
        case yarr.AbstractElementwise():
            operands = [_pretty_yarr(o_, False) for o_ in expr.operands]
            return _pretty_call(
                expr.kind.name if hasattr(expr, "kind") else "?", *operands
            )
        case yarr.Range(size):
            return _pretty_call("range", _pretty_yarr(size, False))
        case yarr.Transpose(pi, target):
            return _pretty_call("transpose", Line(str(pi)), _pretty_yarr(target, False))
        case yarr.Squeeze(axes, target):
            return _pretty_call("squeeze", Line(str(axes)), _pretty_yarr(target, False))
        case yarr.Unsqueeze(axes, target):
            return _pretty_call(
                "unsqueeze", Line(str(axes)), _pretty_yarr(target, False)
            )
        case yarr.Repeat(axis, count, target):
            return _pretty_call(
                "repeat",
                Line(str(axis)),
                _pretty_yarr(count, False),
                _pretty_yarr(target, False),
            )
        case yarr.Cast(dtype, target):
            return _pretty_call(f"cast[{dtype.__name__}]", _pretty_yarr(target, False))
        case yarr.ReduceAxis(kind, axis, target_):
            return _pretty_call(
                f"reduce[{kind.name}, axis={axis}]", _pretty_yarr(target_, False)
            )

    return Line(str(expr))


def pretty_phi(expr: phi.Expr, width: int = 80, indent: int = 2) -> str:
    return _pretty_phi(outline(struct_of_arrays_transform(expr)), False).repr(
        width, indent, 0
    )


def pretty_yarr(expr: yarr.Expr, width: int = 60, indent: int = 2) -> str:
    return _pretty_yarr(outline(expr), False).repr(width, indent, 0)


def pretty_yarr_of_phi(expr: phi.Expr, width: int = 60, indent: int = 2) -> str:
    return pretty_yarr(phi_to_yarr.transform(expr), width, indent)


if __name__ == "__main__":
    import numpy

    from ein import Float, Vec, array, wrap
    from ein.frontend.std import fold_sum

    def mean(xs: Vec[Float]) -> Float:
        return fold_sum(lambda i: xs[i]) / xs.size().float()

    a0, b0 = numpy.random.randn(5), numpy.random.randn(7)
    a: Vec[Float] = wrap(a0)
    b: Vec[Float] = wrap(b0)
    cov = array(lambda i, j: (a[i] - mean(a)) * (b[j] - mean(b)))
    print(pretty_phi(cov.expr))
    print(pretty_yarr_of_phi(cov.expr))

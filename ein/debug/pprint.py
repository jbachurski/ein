import abc
from dataclasses import dataclass

import numpy

from ein.codegen import yarr
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
            if badness <= width:
                result += [prefix, sub, suffix]
            else:
                sub = block.repr(width, indent, len(curr_line) + indent)
                result += [
                    prefix,
                    "\n",
                    "\n".join(" " * indent + ln for ln in sub.splitlines()),
                    "\n",
                    suffix,
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


def _pretty_phi(expr: phi.Expr, par: bool) -> Repr:
    match expr:
        case phi.Const(value):
            value = numpy.array(value.array).tolist()
            return Line(str(value))
        case phi.Store(symbol, _):
            return Line(str(symbol))
        case phi.Let(var, bound, body):
            return pars(
                Concat(
                    (
                        Nest(((f"let {var} = ", _pretty_phi(bound, False), " "),)),
                        Line("in\n"),
                        _pretty_phi(body, False),
                    ),
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
            return Concat(
                (Line(f"size[{axis}]("), _pretty_phi(target_, False), Line(")"))
            )
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
                Nest(
                    (
                        (f"fold {counter}[", _pretty_phi(size, False), "]"),
                        (f" with {acc} = ", _pretty_phi(init, False), ""),
                        (f" by {acc} => ", _pretty_phi(body, False), ""),
                    )
                ),
                par,
            )
        case phi.First(target):
            return pars(Concat((Line("fst "), _pretty_phi(target, True))), par)
        case phi.Second(target):
            return pars(Concat((Line("snd "), _pretty_phi(target, True))), par)
        case phi.Cons(first, second):
            return Concat(
                (
                    Line("("),
                    _pretty_phi(first, False),
                    Line(", "),
                    _pretty_phi(second, False),
                    Line(")"),
                )
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
    return Line(str(expr))


def _pretty_yarr(expr: yarr.Expr, par: bool) -> Repr:
    return Line(str(expr))  # TODO


def pretty_phi(expr: phi.Expr, width: int = 60, indent: int = 2) -> str:
    return _pretty_phi(outline(struct_of_arrays_transform(expr)), False).repr(
        width, indent, 0
    )


def pretty_yarr(expr: yarr.Expr, width: int = 60, indent: int = 2) -> str:
    return _pretty_yarr(outline(expr), False).repr(width, indent, 0)


if __name__ == "__main__":
    from ein import array

    x = array(lambda i: i, size=5)
    y = array(lambda i: 2 * x[i] - i)
    print(pretty_phi(y.expr))

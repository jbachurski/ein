import abc
from dataclasses import dataclass
from typing import TypeAlias, assert_never

from ein.type_system import Pair, Scalar, Type, Vector

Layout: TypeAlias = "AtomLayout | VecLayout | PositionalLayout"


@dataclass(frozen=True)
class AbstractLayout(abc.ABC):
    @abc.abstractmethod
    def match(self, type_: Type) -> bool:
        ...


@dataclass(frozen=True)
class AtomLayout(AbstractLayout):
    def match(self, type_: Type) -> bool:
        match type_:
            case Scalar():
                return True
        return False


@dataclass(frozen=True)
class VecLayout(AbstractLayout):
    sub: Layout

    def match(self, type_: Type) -> bool:
        match type_:
            case Vector():
                return True
        return False


@dataclass(frozen=True)
class PositionalLayout(AbstractLayout):
    subs: tuple[Layout, ...]

    def match(self, type_: Type) -> bool:
        assert self.subs
        if len(self.subs) == 1:
            (sub,) = self.subs
            return sub.match(type_)
        match type_:
            case Pair(first, second):
                rest = PositionalLayout(self.subs[1:])
                return self.subs[0].match(first) and rest.match(second)
        return False


def build_layout(obj, f) -> Layout:
    if isinstance(obj, (tuple, list)):
        if not obj:
            raise ValueError("Constructed layout cannot contain empty nodes")
        return PositionalLayout(tuple(build_layout(o, f) for o in obj))
    return f(obj)


def fold_layout(obj, f, merge):
    def reduce(args):
        return args[0] if len(args) == 1 else merge(args[0], reduce(args[1:]))

    if isinstance(obj, (tuple, list)):
        return reduce([fold_layout(o, f, merge) for o in obj])
    return f(obj)


def unambiguous_layout(type_: Type) -> Layout:
    match type_:
        case Scalar():
            return AtomLayout()
        case Vector(elem):
            return VecLayout(unambiguous_layout(elem))
        case Pair():
            raise ValueError(
                "Expression type contains pairs, which cannot form an unambiguous array layout"
            )
    assert_never(type_)


if __name__ == "__main__":
    layout = build_layout(
        [
            AtomLayout(),
            [[AtomLayout(), AtomLayout()], PositionalLayout((AtomLayout(),))],
        ],
        lambda x: x,
    )
    print(layout)

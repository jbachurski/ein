import abc
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeAlias, assert_never

from ein.type_system import Pair, Scalar, Type, Vector

Layout: TypeAlias = "AtomLayout | VecLayout | PositionalLayout | LabelledLayout"


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
    tag: type | None = None

    def values(self) -> tuple[Layout, ...]:
        return self.subs

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


@dataclass(frozen=True)
class LabelledLayout(AbstractLayout):
    subs: tuple[tuple[str, Layout], ...]
    tag: type | None = None

    def values(self) -> tuple[Layout, ...]:
        return tuple(sub for _, sub in self.subs)

    def match(self, type_: Type) -> bool:
        assert self.subs
        if len(self.subs) == 1:
            ((_, sub),) = self.subs
            return sub.match(type_)
        match type_:
            case Pair(first, second):
                rest = LabelledLayout(self.subs[1:])
                return self.subs[0][1].match(first) and rest.match(second)
        return False


def build_layout(obj, f) -> Layout:
    if isinstance(obj, (tuple, list, dict)):
        if not obj:
            raise ValueError("Constructed layout cannot contain empty nodes")
    if isinstance(obj, (tuple, list)):
        return PositionalLayout(tuple(build_layout(o, f) for o in obj), type(obj))
    elif isinstance(obj, dict):
        return LabelledLayout(
            tuple((n, build_layout(o, f)) for n, o in obj.items()), type(obj)
        )
    elif is_dataclass(obj):
        return LabelledLayout(
            tuple(
                (field.name, build_layout(getattr(obj, field.name), f))
                for field in fields(obj)
            ),
            type(obj),
        )
    return f(obj)


def fold_layout(layout, obj, f, merge):
    def reduce(args):
        return args[0] if len(args) == 1 else merge(args[0], reduce(args[1:]))

    match layout:
        case PositionalLayout(subs):
            return reduce(
                [
                    fold_layout(sub, o, f, merge)
                    for sub, o in zip(subs, obj, strict=True)
                ]
            )
        case LabelledLayout(subs):
            if is_dataclass(obj):
                return reduce(
                    [
                        fold_layout(sub, getattr(obj, name), f, merge)
                        for name, sub in subs
                    ]
                )
            assert len(subs) == len(obj)
            return reduce([fold_layout(sub, obj[name], f, merge) for name, sub in subs])
    return f(obj)


def unambiguous_layout(type_: Type) -> Layout:
    match type_:
        case Scalar():
            return AtomLayout()
        case Vector(elem):
            return VecLayout(unambiguous_layout(elem))
        case Pair():
            raise ValueError(
                "Expression type contains pairs, which cannot form an unambiguous array-struct layout"
            )
    assert_never(type_)

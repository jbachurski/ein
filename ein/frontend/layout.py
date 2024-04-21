import abc
from dataclasses import dataclass, fields, is_dataclass
from typing import TypeAlias, Union, assert_never

from ein.phi.type_system import Pair, Scalar, Type, Vector

Layout: TypeAlias = Union[
    "AtomLayout", "VecLayout", "PositionalLayout", "LabelledLayout"
]


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
        return PositionalLayout(tuple(build_layout(o, f) for o in obj))
    elif isinstance(obj, dict):
        return LabelledLayout(tuple((n, build_layout(o, f)) for n, o in obj.items()))
    elif is_dataclass(obj):
        return LabelledLayout(
            tuple(
                (field.name, build_layout(getattr(obj, field.name), f))
                for field in fields(obj)
            ),
            type(obj),
        )
    return f(obj)


def fold_layout(layout, args, atom, vec, merge):
    def get(name):
        def do(x):
            return (
                getattr(x, name)
                if isinstance(name, str) and hasattr(x, name)
                else x[name]
            )

        return do

    def reduce(xs):
        return xs[0] if len(xs) == 1 else merge(xs[0], reduce(xs[1:]))

    print(layout, args)
    match layout:
        case AtomLayout():
            return atom(*args)
        case VecLayout(_sub):
            return vec(*args)
        case PositionalLayout(subs):
            return reduce(
                [
                    fold_layout(sub, sub_args, atom, vec, merge)
                    for sub, sub_args in zip(subs, zip(*args), strict=True)
                ]
            )
        case LabelledLayout(subs):
            return reduce(
                [
                    fold_layout(sub, list(map(get(name), args)), atom, vec, merge)
                    for name, sub in subs
                ]
            )
        case _:
            assert_never(layout)


def map_layout(layout, args, atom, vec):
    def get(name):
        def do(x):
            return getattr(x, name) if hasattr(x, name) else x[name]

        return do

    match layout:
        case AtomLayout():
            return atom(*args)
        case VecLayout(_sub):
            return vec(*args)
        case PositionalLayout(subs, tag):
            ret_args = tuple(
                map_layout(sub, sub_args, atom, vec)
                for sub, sub_args in zip(subs, zip(*args), strict=True)
            )
            return tag(*ret_args) if tag is not None else ret_args
        case LabelledLayout(subs, tag):
            ret_kwargs = {
                name: map_layout(sub, list(map(get(name), args)), atom, vec)
                for name, sub in subs
            }
            return tag(**ret_kwargs) if tag is not None else ret_kwargs
        case _:
            assert_never(layout)


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

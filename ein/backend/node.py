from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeAlias, TypeVar

import numpy
from calculus import Variable

T = TypeVar("T")
NodeArg: TypeAlias = "Variable | Node | int | numpy.array | tuple[NodeArg, ...]"


def identity(x: T) -> T:
    return x


@dataclass(frozen=True, eq=False)
class Node:
    class Function(Protocol):
        def __call__(self, *args: NodeArg, **kwargs: NodeArg) -> "Node":
            ...

    fun: Callable[..., Any]
    args: tuple[NodeArg, ...]
    kwargs: dict[str, NodeArg]

    @staticmethod
    def map_args(
        arg: NodeArg,
        for_var: Callable[[Variable], Any],
        for_node: Callable[["Node"], Any],
        for_other: Callable[[Any], Any],
    ) -> Any:
        if isinstance(arg, Variable):
            return for_var(arg)
        elif isinstance(arg, Node):
            return for_node(arg)
        elif isinstance(arg, tuple):
            return tuple(
                Node.map_args(elem, for_var, for_node, for_other) for elem in arg
            )
        return for_other(arg)

    def mapped_args(
        self,
        for_var: Callable[[Variable], Any],
        for_node: Callable[["Node"], Any],
        for_other: Callable[[Any], Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return tuple(
            self.map_args(arg, for_var, for_node, for_other) for arg in self.args
        ), {
            k: self.map_args(arg, for_var, for_node, for_other)
            for k, arg in self.kwargs.items()
        }

    def linearize(self) -> Generator["Node", None, None]:
        seen: set[Node] = set()
        written: set[
            Node
        ] = set()  # This is just for a sanity check there are no cycles

        def go(node: Node) -> Generator[Node, None, None]:
            if node in seen:
                assert node in written
            seen.add(node)
            dependencies = []

            node.mapped_args(
                lambda var: None,
                lambda sub_node: dependencies.append(sub_node),
                lambda x: None,
            )

            for dep in dependencies:
                yield from go(dep)
            yield node

            written.add(node)

        return go(self)


def node(fun: Callable[..., Any]) -> Node.Function:
    def inner(*args, **kwargs):
        return Node(fun, args, kwargs)

    return inner

from typing import Generic, Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


# Disjoint Set / Find & Union
class Equivalence(Generic[T]):
    _parent: dict[T, T]
    _classes: dict[T, set[T]]

    def __init__(self):
        self._parent = {}
        self._classes = {}

    def _get_parent(self, u: T) -> T:
        return self._parent.get(u, u)

    def find(self, u: T) -> set[T]:
        return self._classes.get(u, {u})

    def equiv(self, *args: T) -> bool:
        if len(args) <= 1:
            return True
        u = self._get_parent(args[0])
        for v in args[1:]:
            if u != self._get_parent(v):
                return False
        return True

    def unite(self, u: T, v: T) -> bool:
        u, v = self._get_parent(u), self._get_parent(v)
        if u == v:
            return False
        if len(self.find(u)) > len(self.find(v)):
            u, v = v, u
        self._classes.setdefault(v, {v})
        for w in self._classes.pop(u, {u}):
            self._parent[w] = v
            self._classes[v].add(w)
        return True

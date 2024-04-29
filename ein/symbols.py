from typing import Any, ClassVar


class Symbol:
    _pref: ClassVar[str] = "?"
    _ids: ClassVar[dict[Any, int]] = {}

    def __repr__(self) -> str:
        if self not in self._ids:
            self._ids[self] = len(self._ids)
        return f"{self._pref}{self._ids[self]}"


class Variable(Symbol):
    _pref = "&"
    _ids = {}


class Index(Symbol):
    _pref = "@"
    _ids = {}

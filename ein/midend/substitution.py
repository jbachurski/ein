from functools import cache
from typing import Mapping

from ein.symbols import Symbol
from ein.term import Term


def substitute(term: Term, subs: Mapping[Symbol, Term]) -> Term:
    relevant_symbols = set(subs)

    @cache
    def go(t: Term) -> Term:
        if not t.free_symbols & relevant_symbols:
            return t
        symbol = t.unwrap_symbol()
        if symbol is not None and symbol in subs:
            return subs[symbol]
        return t.map(go)

    return go(term)

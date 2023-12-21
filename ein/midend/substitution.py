from functools import cache

from ein.symbols import Symbol
from ein.term import Term


def substitute(term: Term, subs: dict[Symbol, Term]) -> Term:
    @cache
    def go(t: Term) -> Term:
        symbol = t.unwrap_symbol()
        if symbol is not None and symbol in subs:
            return subs[symbol]
        return t.map(go)

    return go(term)

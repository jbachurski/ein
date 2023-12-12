from functools import cache

from ein.symbols import Index, Variable
from ein.term import Term


def substitute(term: Term, subs: dict[Index | Variable, Term]) -> Term:
    @cache
    def go(t: Term) -> Term:
        if t.unwrap_var() in subs:
            return subs[t.unwrap_var()]  # type: ignore
        elif t.unwrap_index() in subs:
            return subs[t.unwrap_index()]  # type: ignore
        return t.map(go)

    return go(term)

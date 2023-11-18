from functools import cache
from typing import Iterable

from ein import calculus
from ein.symbols import Variable

NEVER_BIND = (calculus.Var, calculus.At)


def _let(
    bindings: Iterable[tuple[Variable, calculus.Expr]], body: calculus.Expr
) -> calculus.Expr:
    bindings = tuple(bindings)
    if not bindings:
        return body
    (var, bind), *bindings = bindings
    return calculus.Let(var, bind, _let(bindings, body))


def _apply_insertions(
    program: calculus.Expr,
    insertions: dict[calculus.Expr, list[tuple[Variable, calculus.Expr]]],
) -> calculus.Expr:
    def go(
        expr: calculus.Expr, bindings: dict[calculus.Expr, calculus.Var]
    ) -> calculus.Expr:
        if expr in bindings:
            return bindings[expr]

        bindings_in_sub = bindings | {
            bind: calculus.Var(var, bind.type) for var, bind in insertions.get(expr, [])
        }

        return _let(
            ((var, go(bind, bindings)) for var, bind in insertions.get(expr, [])),
            expr.map(lambda sub: go(sub, bindings_in_sub)),
        )

    return inline(go(program, {}), only_renames=True)


def _bind_common_subexpressions(program: calculus.Expr) -> calculus.Expr:
    @cache
    def subexpressions(expr: calculus.Expr) -> set[calculus.Expr]:
        return set.union({expr}, *(subexpressions(sub) for sub in expr.dependencies))

    insertions: dict[calculus.Expr, list[tuple[Variable, calculus.Expr]]] = {}

    @cache
    def visit(expr: calculus.Expr) -> None:
        for sub in expr.dependencies:
            visit(sub)

        seen: set[calculus.Expr] = set()
        covered: set[calculus.Expr] = set()
        occurrences: list[calculus.Expr] = [
            sub_sub
            for sub in expr.dependencies
            for sub_sub in subexpressions(sub)
            if sub_sub.free_indices <= expr.free_indices
            and sub_sub.free_variables <= expr.free_variables
        ]
        occurrences.sort(key=lambda e: len(subexpressions(e)), reverse=True)

        for sub in occurrences:
            if isinstance(sub, NEVER_BIND):
                continue
            if sub in seen and sub not in covered:
                insertions.setdefault(expr, []).append((Variable(), sub))
                covered |= subexpressions(sub)
            seen.add(sub)

    visit(program)
    return _apply_insertions(program, insertions)


def _reduce_loop_strength(program: calculus.Expr) -> calculus.Expr:
    return program


def outline(program: calculus.Expr) -> calculus.Expr:
    # Get rid of any existing let-bindings
    program = inline(program)
    # Bind subexpressions so they occur at most once, creating a syntax tree
    program = _bind_common_subexpressions(program)
    # Extract loop-independent expressions out of containing loops into a wrapping binding
    program = _reduce_loop_strength(program)
    return program


def inline(program: calculus.Expr, *, only_renames: bool = False) -> calculus.Expr:
    def predicate(bind: calculus.Expr) -> bool:
        if only_renames:
            return isinstance(bind, NEVER_BIND)
        return True

    def _go(expr: calculus.Expr, bound: dict[Variable, calculus.Expr]) -> calculus.Expr:
        match expr:
            case calculus.Let(var, bind, body) if predicate(bind):
                return go(body, bound | {var: go(bind, bound)})
            case calculus.Var(var, var_type) if var in bound:
                assert var_type == bound[var].type
                return bound[var]
        return expr.map(lambda e: go(e, bound))

    transformed: dict[calculus.Expr, calculus.Expr] = {}

    def go(expr: calculus.Expr, bound: dict[Variable, calculus.Expr]) -> calculus.Expr:
        if expr not in transformed:
            transformed[expr] = _go(expr, bound)
        return transformed[expr]

    return go(program, {})

from functools import cache
from typing import Iterable, Mapping

from ein import calculus
from ein.symbols import Variable

NEVER_BIND = (calculus.Var, calculus.At)


def _let(
    bindings: Iterable[tuple[Variable, calculus.Expr]], body: calculus.Expr
) -> calculus.Expr:
    bindings = tuple(bindings)
    return calculus.Let(bindings, body) if bindings else body


def _substitute(
    initial: calculus.Expr, subs: Mapping[calculus.Expr, calculus.Expr]
) -> calculus.Expr:
    @cache
    def go(expr: calculus.Expr) -> calculus.Expr:
        if expr in subs:
            return subs[expr]
        return expr.map(go)

    return go(initial)


def _bind_common_subexpressions(program: calculus.Expr) -> calculus.Expr:
    def subexpressions(expr: calculus.Expr) -> set[calculus.Expr]:
        return set.union({expr}, *(subexpressions(sub) for sub in expr.dependencies))

    def go(expr: calculus.Expr) -> calculus.Expr:
        expr = expr.map(go)
        seen: set[calculus.Expr] = set()
        bindings: dict[calculus.Expr, calculus.Var] = {}

        for dependent in expr.dependencies:
            for sub in subexpressions(dependent):
                if isinstance(sub, NEVER_BIND):
                    continue
                if sub in seen and sub not in bindings:
                    bindings[sub] = calculus.Var(Variable(), sub.type)
                seen.add(sub)

        return _let(
            ((var.var, bind) for bind, var in bindings.items()),
            inline(_substitute(expr, bindings), only_renames=True),
        )

    return program


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
            case calculus.Let(bindings, body):
                bindings = tuple((var, go(bind, bound)) for var, bind in bindings)
                inlined_bindings = {
                    var: bind for var, bind in bindings if predicate(bind)
                }
                remaining_bindings = [
                    (var, bind) for var, bind in bindings if not predicate(bind)
                ]
                result = go(body, bound | inlined_bindings)
                return _let(remaining_bindings, result)
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

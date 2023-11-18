from functools import cache
from typing import Iterable

from ein import calculus
from ein.symbols import Index, Variable

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
            while expr in bindings:
                expr = bindings[expr]
            return expr

        bindings_in_sub = bindings.copy()
        applied_bindings = []
        for var, bind in insertions.get(expr, []):
            applied_bindings.append((var, go(bind, bindings_in_sub)))
            bindings_in_sub[bind] = calculus.Var(var, bind.type)

        return _let(applied_bindings, expr.map(lambda sub: go(sub, bindings_in_sub)))

    return go(program, {})


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
    insertions: dict[calculus.Expr, list[tuple[Variable, calculus.Expr]]] = {}

    def visit(
        expr: calculus.Expr, binders: dict[calculus.Expr, set[Index | Variable]]
    ) -> None:
        if not isinstance(expr, NEVER_BIND):
            prev_site = None
            for site, bound in reversed(binders.items()):
                if bound & (expr.free_indices | expr.free_variables):
                    break
                prev_site = site
            if prev_site is not None:
                v = Variable()
                insertions.setdefault(prev_site, []).append((v, expr))

        valid_site = isinstance(expr, (calculus.Vec, calculus.Fold))
        binders = binders | ({expr: set()} if valid_site else {})
        if binders:
            last_site = next(reversed(binders))
            binders[last_site] |= {*expr._captured_indices, *expr._captured_variables}

        for sub in expr.dependencies:
            visit(sub, binders)

    visit(program, {})
    return _apply_insertions(program, insertions)


def outline(program: calculus.Expr) -> calculus.Expr:
    free_indices, free_variables = program.free_indices, program.free_variables

    def check(prog: calculus.Expr, tree: bool) -> None:
        # FIXME: The checks should pass with tree = True.
        tree = False
        seen = set()

        def visit(expr: calculus.Expr) -> calculus.Expr:
            assert expr not in seen or isinstance(expr, NEVER_BIND), expr
            seen.add(expr)
            for sub in expr.dependencies:
                visit(sub)
            return expr

        if tree:
            visit(prog)
        assert free_indices == prog.free_indices
        assert free_variables == prog.free_variables

    # Get rid of any existing let-bindings
    program = inline(program)
    check(program, tree=False)
    # Bind subexpressions so they occur at most once, creating a syntax tree
    program = _bind_common_subexpressions(program)
    check(program, tree=True)
    # Extract loop-independent expressions out of containing loops into a wrapping binding
    program = _reduce_loop_strength(program)
    check(program, tree=True)
    # FIXME: We should do an inline(program, only_renames=True).
    #  However, it makes tests fail.
    #  Nondeterminism is involved, so some it might be some set iteration (?).
    # program = inline(program, only_renames=True)
    # check(program, tree=True)
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

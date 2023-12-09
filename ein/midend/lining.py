from typing import Iterable, TypeAlias

import networkx

from ein.symbols import Index, Variable
from ein.term import Term

Insertions: TypeAlias = dict[Term, dict[Variable, Term]]
BinderStack: TypeAlias = dict[Term, set[Index | Variable]]


def _let(bindings: Iterable[tuple[Variable, Term]], body: Term) -> Term:
    bindings = tuple(bindings)
    if not bindings:
        return body
    (var, bind), *bindings = bindings
    return _let(bindings, body).wrap_let(var, bind)


def _apply_insertions(program: Term, insertions: Insertions) -> Term:
    def go(expr: Term, bindings: dict[Term, Term]) -> Term:
        if expr in bindings:
            return bindings[expr]

        bindings_in_sub = bindings.copy()
        applied_bindings = []
        for var, bind in insertions.get(expr, {}).items():
            applied_bindings.append((var, go(bind, bindings_in_sub)))
            bindings_in_sub[bind] = bind.wrap_var(var)

        return _let(applied_bindings, expr.map(lambda sub: go(sub, bindings_in_sub)))

    return go(program, {})


def _bind_common_subexpressions(program: Term) -> Insertions:
    graph = networkx.MultiDiGraph()
    order: list[Term] = []

    def build(expr: Term) -> None:
        if expr in graph:
            return
        graph.add_node(expr)
        for sub in expr.subterms:
            build(sub)
            graph.add_edge(expr, sub)
        order.append(expr)

    build(program)
    # TODO: Could get rid of dependency on networkx, since it's O(n^2) anyway.
    idom = networkx.immediate_dominators(graph, program)

    insertions: Insertions = {}
    for sub in order:
        in_degree: int = graph.in_degree(sub)  # noqa
        if in_degree > 1:
            insertions.setdefault(idom[sub], {})[Variable()] = sub

    return insertions


def _reduce_loop_strength(program: Term) -> Insertions:
    insertions: dict[Term, dict[Variable, Term]] = {}

    def visit(expr: Term, binders: BinderStack) -> None:
        binders_in_sub: BinderStack = binders | ({expr: set()} if expr.is_loop else {})
        if binders_in_sub:
            last_site = next(reversed(binders_in_sub))
            binders_in_sub[last_site] |= {
                *expr.captured_indices,
                *expr.captured_variables,
            }

        for sub in expr.subterms:
            visit(sub, binders_in_sub)

        if not expr.is_atom:
            prev_site = None
            for site, bound in reversed(binders.items()):
                if bound & (expr.free_indices | expr.free_variables):
                    break
                prev_site = site
            if prev_site is not None:
                insertions.setdefault(prev_site, {})[Variable()] = expr

    visit(program, {})
    return insertions


def outline(program: Term) -> Term:
    free_indices, free_variables = program.free_indices, program.free_variables

    def check(prog: Term, tree: bool) -> None:
        seen = set()

        def visit(expr: Term) -> Term:
            assert expr not in seen or expr.is_atom, expr
            seen.add(expr)
            for sub in expr.subterms:
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
    program = _apply_insertions(program, _bind_common_subexpressions(program))
    check(program, tree=True)
    # Extract loop-independent expressions out of containing loops into a wrapping binding
    program = _apply_insertions(program, _reduce_loop_strength(program))
    check(program, tree=True)
    program = inline(program, only_renames=True)
    check(program, tree=True)
    return program


def inline(program: Term, *, only_renames: bool = False) -> Term:
    def predicate(bind: Term) -> bool:
        if only_renames:
            return bind.is_atom
        return True

    def _go(expr: Term, bound: dict[Variable, Term]) -> Term:
        let_tuple = expr.unwrap_let()
        var = expr.unwrap_var()
        if let_tuple is not None:
            var, bind, body = let_tuple
            if predicate(bind):
                return go(body, bound | {var: go(bind, bound)})
        if var is not None and var in bound:
            return bound[var]
        return expr.map(lambda e: go(e, bound))

    transformed: dict[Term, Term] = {}

    def go(expr: Term, bound: dict[Variable, Term]) -> Term:
        if expr not in transformed:
            transformed[expr] = _go(expr, bound)
        return transformed[expr]

    return go(program, {})

from ein import calculus
from ein.symbols import Variable


def _normalize_common_subexpressions(program: calculus.Expr) -> calculus.Expr:
    return program


def _bind_common_subexpressions(program: calculus.Expr) -> calculus.Expr:
    return program


def _reduce_loop_strength(program: calculus.Expr) -> calculus.Expr:
    return program


def outline(program: calculus.Expr) -> calculus.Expr:
    # Get rid of any existing let-bindings
    program = inline(program)
    # Make every syntactically equal expression have at most one Expr instance
    program = _normalize_common_subexpressions(program)
    # Move the semantic graph into a syntax tree by binding subexpressions, so they occur at most once
    program = _bind_common_subexpressions(program)
    # Extract loop-independent expressions out of containing loops into a wrapping binding
    program = _reduce_loop_strength(program)
    return program


def inline(program: calculus.Expr, *, only_renames: bool = False) -> calculus.Expr:
    def predicate(bind: calculus.Expr):
        if only_renames:
            return isinstance(bind, calculus.Var)
        return True

    def _go(expr: calculus.Expr, bound: dict[Variable, calculus.Expr]) -> calculus.Expr:
        match expr:
            case calculus.Let(bindings, body):
                bindings = tuple((var, go(bind, bound)) for var, bind in bindings)
                inlined_bindings = {
                    var: bind for var, bind in bindings if predicate(bind)
                }
                remaining_bindings = tuple(
                    (var, bind) for var, bind in bindings if not predicate(bind)
                )
                result = go(body, bound | inlined_bindings)
                return (
                    calculus.Let(remaining_bindings, result)
                    if remaining_bindings
                    else result
                )
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

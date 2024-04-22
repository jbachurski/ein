import abc
from typing import (
    Any,
    Callable,
    Generic,
    Self,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy

from ein.backend import BACKENDS, DEFAULT_BACKEND, Backend
from ein.phi import phi
from ein.phi.phi import AbstractExpr, Expr
from ein.phi.type_system import AbstractType
from ein.phi.type_system import Scalar as ScalarType
from ein.phi.type_system import Type
from ein.phi.type_system import Vector as VectorType
from ein.symbols import Variable
from ein.value import Value, _TorchTensor

from .layout import (
    AtomLayout,
    LabelledLayout,
    Layout,
    PositionalLayout,
    VecLayout,
    build_layout,
    fold_layout,
    unambiguous_layout,
)

T = TypeVar("T")
S = TypeVar("S")
Array: TypeAlias = Any
ScalarLike = int | float | bool | Union["Scalar"]
ArrayLike: TypeAlias = ScalarLike | numpy.ndarray | _TorchTensor | Union["Vec"]


def _project_tuple(expr: phi.Expr, i: int, n: int) -> phi.Expr:
    for _ in range(i):
        expr = phi.Second(expr)
    return phi.First(expr) if i + 1 < n else expr


def _layout_struct_to_expr(layout: Layout, struct) -> Expr:
    return fold_layout(
        layout,
        [struct],
        lambda a: wrap(a).expr,
        lambda a: a.expr,
        lambda a, b: phi.Cons(a, b),
    )


def _to_array(expr: Expr, layout: Layout | None = None):
    if layout is None:
        layout = unambiguous_layout(expr.type)
    match layout:
        case AtomLayout():
            return Scalar(expr)
        case VecLayout(_sub):
            return Vec(expr, layout)
        case PositionalLayout(subs, tag):
            args = tuple(
                _to_array(_project_tuple(expr, i, len(subs)), sub)
                for i, sub in enumerate(subs)
            )
            return tag(*args) if tag is not None else args
        case LabelledLayout(subs, tag):
            kwargs = {
                name: _to_array(_project_tuple(expr, i, len(subs)), sub)
                for i, (name, sub) in enumerate(subs)
            }
            return tag(**kwargs) if tag is not None else kwargs
    assert False, "Expected a tag for layout: {layout}"


class _Array:
    expr: Expr

    def __bool__(self):
        raise TypeError(
            "Ein arrays don't have a boolean value and cannot be used in conditions - "
            "did you accidentally include it in an if or while statement?"
        )

    def __iter__(self):
        raise TypeError(
            "Ein arrays cannot be iterated over, as their computation is staged lazily - "
            "did you accidentally include it in a for-loop?"
        )

    @abc.abstractmethod
    def assume(self, other) -> Self:
        ...

    def _assume_expr(self, other) -> phi.Expr:
        return phi.First(
            phi.Cons(
                self.expr,
                _layout_struct_to_expr(
                    build_layout(other, lambda a: wrap(a).layout), other
                ),
            )
        )

    def eval(
        self,
        *,
        env: dict[Variable, numpy.ndarray | _TorchTensor] | None = None,
        backend: Backend | str = DEFAULT_BACKEND,
    ) -> numpy.ndarray:
        env = env if env is not None else {}
        if not self.expr.free_symbols <= set(env):
            raise ValueError(
                f"Cannot evaluate array, as it depends on free variables: {self.expr.free_symbols}"
            )
        return BACKENDS[backend](self.expr, env)

    def torch(self, *, env: dict[Variable, numpy.ndarray | _TorchTensor] | None = None):
        return self.eval(env=env, backend="torch")

    def numpy(self, *, env: dict[Variable, numpy.ndarray] | None = None):
        return self.eval(env=env, backend="numpy")


class Vec(_Array, Generic[T]):
    expr: Expr
    _layout: VecLayout

    def __init__(self, expr: Expr, layout: "Layout | None" = None):
        assert isinstance(expr, AbstractExpr)
        self.expr = cast(Expr, expr)
        layout = unambiguous_layout(self.expr.type) if layout is None else layout
        assert isinstance(layout, VecLayout)
        self._layout = layout

    def assume(self, other) -> Self:
        return type(self)(self._assume_expr(other), self.layout)

    @property
    def layout(self):
        return self._layout

    @overload
    def __getitem__(
        self: "Vec[Vec[Vec[Vec[Vec[S]]]]]", item_like: tuple[ScalarLike, ...]
    ) -> Array:
        ...

    @overload
    def __getitem__(
        self: "Vec[Vec[Vec[Vec[S]]]]",
        item_like: tuple[ScalarLike, ScalarLike, ScalarLike, ScalarLike],
    ) -> S:
        ...

    @overload
    def __getitem__(
        self: "Vec[Vec[Vec[S]]]", item_like: tuple[ScalarLike, ScalarLike, ScalarLike]
    ) -> S:
        ...

    @overload
    def __getitem__(self: "Vec[Vec[S]]", item_like: tuple[ScalarLike, ScalarLike]) -> S:
        ...

    @overload
    def __getitem__(self, item_like: ScalarLike) -> T:
        ...

    def __getitem__(self, item_like):
        item: tuple = (item_like,) if not isinstance(item_like, tuple) else item_like
        if not item:
            return self
        curr, *rest = item

        def maybe_rest(x):
            return x[tuple(rest)] if rest else x

        sub_expr: phi.Expr
        sub_layout: Layout
        match curr:
            case slice(start=start, stop=stop, step=step):
                # FIXME: Handle other slices for indexing into Ein arrays.
                if {start, stop, step} != {None}:
                    raise ValueError(
                        f"Unhandled slice for Ein indexing: {curr}, only empty slices are allowed."
                    )
                from .comprehension import array

                # This is not an infinite recursion, as [i] is not a slice
                return array(lambda i: maybe_rest(self[i]))
            case _:
                sub_expr = phi.Get(self.expr, wrap(curr).expr)
                sub_layout = self.layout.sub
                sub = _to_array(sub_expr, sub_layout)
        return maybe_rest(sub)

    def concat(self, other: "Vec[T]") -> "Vec[T]":
        assert self.layout == other.layout
        return _to_array(phi.Concat(self.expr, other.expr), self.layout)

    def size(self, axis: int = 0) -> "Scalar":
        return Scalar(phi.Dim(self.expr, axis))

    def or_empty(self, item, empty: T) -> T:
        from ein.frontend.std import where

        return where(self.size(0) > 0, self[item], empty)

    def reduce(self, init: T, f: Callable[[T, T], T]) -> T:
        x, y = Variable(), Variable()
        type_ = self.expr.type
        assert isinstance(type_, VectorType)
        x_, y_ = (
            _to_array(phi.variable(v, type_.elem), self.layout.sub) for v in (x, y)
        )
        xy_ = f(x_, y_)
        layout = build_layout(xy_, lambda a: wrap(a).layout)
        if self.layout.sub != layout:
            raise TypeError(
                f"Monoid must preserve the same structure, but reduces {self.layout.sub} into {layout}"
            )
        layout_init = build_layout(init, lambda a: wrap(a).layout)
        if layout_init != layout:
            raise TypeError(
                f"Monoid identity must have the same structure as its reductions, "
                f"but is {layout_init} != {layout}"
            )
        init_expr = _layout_struct_to_expr(layout, init)
        xy = _layout_struct_to_expr(layout, xy_)
        expr = phi.Reduce(init_expr, x, y, xy, (self.expr,))
        return _to_array(expr, self.layout.sub)


class Scalar(_Array):
    expr: Expr

    def __init__(self, expr: Expr):
        assert isinstance(expr, AbstractExpr)
        self.expr = cast(Expr, expr)

    @property
    def layout(self) -> Layout:
        return AtomLayout()

    def assume(self, other) -> Self:
        return type(self)(self._assume_expr(other))

    def float(self) -> "Scalar":
        return Scalar(phi.CastToFloat((self.expr,)))

    def __add__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Add((self.expr, wrap(other).expr)))

    def __sub__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Subtract((self.expr, wrap(other).expr)))

    def __mul__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Multiply((self.expr, wrap(other).expr)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> "Scalar":
        return Scalar(phi.Negate((self.expr,)))

    def __rsub__(self, other: ScalarLike) -> "Scalar":
        return (-self) + other

    def __truediv__(self, other: ScalarLike) -> "Scalar":
        return self * Scalar(phi.Reciprocal((wrap(other).expr,)))

    def __rtruediv__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other) / self)

    def __mod__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Modulo((self.expr, wrap(other).expr)))

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise NotImplementedError("Power modulo is not supported")
        if isinstance(power, int):
            if power < 0:
                return (1.0 / self) ** (-power)

            def go(k: int) -> Scalar:
                if k == 1:
                    return self
                elif k % 2:
                    return go(k - 1) * self
                sub = go(k // 2)
                return sub * sub

            return go(power)

        return Scalar(phi.Power((self.expr, wrap(power).expr)))

    def __invert__(self) -> "Scalar":
        return Scalar(phi.LogicalNot((self.expr,)))

    def __and__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.LogicalAnd((self.expr, wrap(other).expr)))

    def __or__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.LogicalOr((self.expr, wrap(other).expr)))

    def __lt__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Less((self.expr, wrap(other).expr)))

    def __ne__(self, other: ScalarLike) -> "Scalar":  # type: ignore
        return Scalar(phi.NotEqual((self.expr, wrap(other).expr)))

    def __eq__(self, other: ScalarLike) -> "Scalar":  # type: ignore
        return Scalar(phi.Equal((self.expr, wrap(other).expr)))

    def __gt__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other).__lt__(self))

    def __le__(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.LessEqual((self.expr, wrap(other).expr)))

    def __ge__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other).__le__(self))

    def where(self, true: ScalarLike, false: ScalarLike) -> "Scalar":
        return Scalar(phi.Where((self.expr, wrap(true).expr, wrap(false).expr)))

    def min(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Min((self.expr, wrap(other).expr)))

    def max(self, other: ScalarLike) -> "Scalar":
        return Scalar(phi.Max((self.expr, wrap(other).expr)))

    def abs(self) -> "Scalar":
        zero = 0 if self.expr.type == ScalarType(int) else 0.0
        return (self > zero).where(self, -self)

    __abs__ = abs

    def exp(self) -> "Scalar":
        return Scalar(phi.Exp((self.expr,)))

    def sin(self) -> "Scalar":
        return Scalar(phi.Sin((self.expr,)))

    def cos(self) -> "Scalar":
        return Scalar(phi.Cos((self.expr,)))

    def tanh(self) -> "Scalar":
        a, b = self.exp(), (-self).exp()
        return (a - b) / (a + b)


def ext(
    fun: Callable, signature: "Type | tuple[Sequence[Type], Type]"
) -> Callable[..., Array]:
    input_signature: Sequence[Type] | None
    if isinstance(signature, AbstractType):
        input_signature = None
        output_signature = signature
    else:
        input_signature, output_signature = signature
        input_signature = tuple(input_signature)

    def extrinsic(*args: ArrayLike) -> Array:
        operands = tuple(wrap(a).expr for a in args)
        if input_signature is not None:
            for i, (op, exp) in enumerate(zip(operands, input_signature, strict=True)):
                assert unambiguous_layout(op.type)
                if op.type != exp:
                    raise TypeError(
                        f"Expected {exp} in argument {i} of {extrinsic.__name__}, got {op.type}"
                    )
        return _to_array(phi.Extrinsic(output_signature, fun, operands))

    if hasattr(fun, "__name__"):
        extrinsic.__name__ = f"{extrinsic}_{fun.__name__}"
    return extrinsic


def wrap(array_like: ArrayLike) -> Array:
    if isinstance(array_like, (Scalar, Vec)):
        return cast(Array, array_like)
    if not isinstance(array_like, (int, float, bool, numpy.ndarray, _TorchTensor)):
        raise TypeError(f"Invalid type for an ein Array: {type(array_like).__name__}")
    expr = phi.Const(Value(array_like))
    return _to_array(expr)

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

from ein import calculus
from ein.backend import BACKENDS, DEFAULT_BACKEND, Backend
from ein.calculus import AbstractExpr, Expr
from ein.frontend.layout import (
    AbstractLayout,
    AtomLayout,
    LabelledLayout,
    Layout,
    PositionalLayout,
    VecLayout,
    build_layout,
    fold_layout,
    unambiguous_layout,
)
from ein.symbols import Variable
from ein.type_system import AbstractType
from ein.type_system import Scalar as ScalarType
from ein.type_system import Type
from ein.value import Value, _TorchTensor

T = TypeVar("T")
S = TypeVar("S")
Array: TypeAlias = Any
ScalarLike = int | float | bool | Union["Scalar"]
ArrayLike: TypeAlias = ScalarLike | numpy.ndarray | _TorchTensor | Union["Vec"]


def _project_tuple(expr: calculus.Expr, i: int, n: int) -> calculus.Expr:
    for _ in range(i):
        expr = calculus.Second(expr)
    return calculus.First(expr) if i + 1 < n else expr


def _layout_struct_to_expr(layout: Layout, struct) -> Expr:
    return fold_layout(
        layout,
        [struct],
        lambda a: wrap(a).expr,
        lambda a: a.expr,
        lambda a, b: calculus.Cons(a, b),
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

    def _assume_expr(self, other) -> calculus.Expr:
        return calculus.First(
            calculus.Cons(
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
    _layout: Layout

    def __init__(self, expr: Expr, layout: "Layout | None" = None):
        assert isinstance(expr, AbstractExpr)
        self.expr = cast(Expr, expr)
        self._layout = unambiguous_layout(self.expr.type) if layout is None else layout
        assert isinstance(self._layout, AbstractLayout)
        assert (
            getattr(self._layout, "tag", None) is None
        ), "Unexpected tagged layout in this context"

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
        item = (item_like,) if not isinstance(item_like, tuple) else item_like
        expr = self.expr
        layout = self.layout
        for axis_item in item:
            match layout:
                case VecLayout(sub):
                    expr = calculus.Get(expr, wrap(axis_item).expr)
                    layout = sub
                case AtomLayout():
                    raise ValueError("Cannot index into a scalar array")
                case _:
                    assert False, f"Unexpected layout in indexing: {layout}"
        return _to_array(expr, layout)

    def concat(self, other: "Vec[T]") -> "Vec[T]":
        assert self.layout == other.layout
        return _to_array(calculus.Concat(self.expr, other.expr), self.layout)

    def size(self, axis: int) -> "Scalar":
        return Scalar(calculus.Dim(self.expr, axis))


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

    def to_float(self) -> "Scalar":
        return Scalar(calculus.CastToFloat((self.expr,)))

    def __add__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Add((self.expr, wrap(other).expr)))

    def __sub__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Subtract((self.expr, wrap(other).expr)))

    def __mul__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Multiply((self.expr, wrap(other).expr)))

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self) -> "Scalar":
        return Scalar(calculus.Negate((self.expr,)))

    def __rsub__(self, other: ScalarLike) -> "Scalar":
        return (-self) + other

    def __truediv__(self, other: ScalarLike) -> "Scalar":
        return self * Scalar(calculus.Reciprocal((wrap(other).expr,)))

    def __rtruediv__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other) / self)

    def __mod__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Modulo((self.expr, wrap(other).expr)))

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

        return Scalar(calculus.Power((self.expr, wrap(power).expr)))

    def __invert__(self) -> "Scalar":
        return Scalar(calculus.LogicalNot((self.expr,)))

    def __and__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.LogicalAnd((self.expr, wrap(other).expr)))

    def __or__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.LogicalOr((self.expr, wrap(other).expr)))

    def __lt__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Less((self.expr, wrap(other).expr)))

    def __ne__(self, other: ScalarLike) -> "Scalar":  # type: ignore
        return Scalar(calculus.NotEqual((self.expr, wrap(other).expr)))

    def __eq__(self, other: ScalarLike) -> "Scalar":  # type: ignore
        return Scalar(calculus.Equal((self.expr, wrap(other).expr)))

    def __gt__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other).__lt__(self))

    def __le__(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.LessEqual((self.expr, wrap(other).expr)))

    def __ge__(self, other: ScalarLike) -> "Scalar":
        return cast(Scalar, wrap(other).__le__(self))

    def where(self, true: ScalarLike, false: ScalarLike) -> "Scalar":
        return Scalar(calculus.Where((self.expr, wrap(true).expr, wrap(false).expr)))

    def min(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Min((self.expr, wrap(other).expr)))

    def max(self, other: ScalarLike) -> "Scalar":
        return Scalar(calculus.Max((self.expr, wrap(other).expr)))

    def abs(self) -> "Scalar":
        zero = 0 if self.expr.type == ScalarType(int) else 0.0
        return (self > zero).where(self, -self)

    __abs__ = abs

    def exp(self) -> "Scalar":
        return Scalar(calculus.Exp((self.expr,)))

    def sin(self) -> "Scalar":
        return Scalar(calculus.Sin((self.expr,)))

    def cos(self) -> "Scalar":
        return Scalar(calculus.Cos((self.expr,)))

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
        return _to_array(calculus.Extrinsic(output_signature, fun, operands))

    if hasattr(fun, "__name__"):
        extrinsic.__name__ = f"{extrinsic}_{fun.__name__}"
    return extrinsic


def wrap(array_like: ArrayLike) -> Array:
    if isinstance(array_like, (Scalar, Vec)):
        return cast(Array, array_like)
    if not isinstance(array_like, (int, float, bool, numpy.ndarray, _TorchTensor)):
        raise TypeError(f"Invalid type for an ein Array: {type(array_like).__name__}")
    expr = calculus.Const(Value(array_like))
    return _to_array(expr)

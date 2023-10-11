import inspect
from dataclasses import dataclass
from typing import Callable, Self, TypeVar, assert_never

from ein import calculus
from ein.calculus import Expr, Index, Var, Variable

from .tensor import Tensor, TensorLike

T = TypeVar("T")


def identity(x: T) -> T:
    return x


@dataclass
class TensorComprehension:
    application: Callable[[Index, Expr, Expr], Expr]
    pre: Callable[[Expr], Expr]
    post: Callable[[Expr], Expr]
    sizes: tuple[TensorLike, ...] | None

    def __init__(self, *, application, pre=identity, post=identity, sizes=None):
        self.application = application
        self.pre = pre
        self.post = post
        self.sizes = sizes

    def __getitem__(self, sizes) -> Self:
        if self.sizes is not None:
            raise TypeError("Already specified the sizes for the tensor comprehension.")
        if not isinstance(sizes, tuple):
            sizes = (sizes,)
        return type(self)(
            application=self.application, pre=self.pre, post=self.post, sizes=sizes
        )

    @staticmethod
    def _size_of(expr: calculus.Expr) -> calculus.Dim:
        match expr:
            case calculus.Get(operand):
                sub = TensorComprehension._size_of(operand)
                return calculus.Dim(
                    sub.operand, sub.axis + 1 if isinstance(sub.axis, int) else sub.axis
                )
            case calculus.Vec() | calculus.Const() | calculus.Var():
                return calculus.Dim(expr, 0)
            case (
                calculus.At()
                | calculus.Dim()
                | calculus.Where()
                | calculus.AbstractScalarReduction()
                | calculus.AbstractScalarOperator()
            ):
                assert False, f"Cannot take size of scalar {expr}"
            case _:
                assert_never(expr)

    def __call__(self, constructor: Callable[..., TensorLike]) -> Tensor:
        n = (
            len(inspect.signature(constructor).parameters)
            if self.sizes is None
            else len(self.sizes)
        )
        indices = [Index() for _ in range(n)]
        wrapped_indices = [Tensor(calculus.At(index)) for index in indices]
        base_body = constructor(*wrapped_indices)  # type: ignore
        body = self.pre(Tensor(base_body).expr)
        if self.sizes is None:
            size_of: dict[Index, Expr] = {}
            for rank, index in enumerate(indices):
                candidates = (
                    self._size_of(expr)
                    for expr in body.direct_indices.get(index, set())
                )
                candidates = (
                    shape_expr
                    for shape_expr in candidates
                    if not shape_expr.free_indices
                )
                if not candidates:
                    raise ValueError(
                        f"Cannot infer bounds for index without direct gets: {index}"
                    )
                else:
                    # TODO: Handle the ignored cases here by requiring equivalence.
                    shape_expr, *_ = candidates
                    size_of[index] = shape_expr
        else:
            size_of = {
                index: Tensor(size).expr
                for index, size in zip(indices, self.sizes, strict=True)
            }
        for index in reversed(indices):
            body = self.application(index, size_of[index], body)
        return Tensor(self.post(Tensor(body).expr))


array = TensorComprehension(application=calculus.Vec)
sum = TensorComprehension(application=calculus.Sum)
max = TensorComprehension(application=calculus.Maximum)
min = TensorComprehension(
    application=calculus.Maximum,
    pre=lambda expr: calculus.Negate((expr,)),
    post=lambda expr: calculus.Negate((expr,)),
)


class VariableTensor(Tensor):
    expr: Var

    def __init__(self):
        super().__init__(Var(Variable()))

    @property
    def var(self) -> Variable:
        return self.expr.var


def function(fun: Callable[..., TensorLike]) -> tuple[tuple[Variable, ...], Expr]:
    args = [VariableTensor() for _ in range(len(inspect.signature(fun).parameters))]
    return tuple(arg.var for arg in args), Tensor(fun(*args)).expr

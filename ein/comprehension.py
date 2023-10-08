import inspect
from dataclasses import dataclass
from typing import Callable, Protocol, Self, assert_never

from . import calculus
from .calculus import Expr, Index
from .tensor import Tensor, TensorLike


class _TensorConstructor(Protocol):
    def __call__(self, *args: Index) -> TensorLike:
        ...


@dataclass
class TensorComprehension:
    application: Callable[[Index, Expr, Expr], Expr]
    sizes: tuple[TensorLike, ...] | None

    def __init__(self, *, application, sizes=None):
        self.application = application
        self.sizes = sizes

    def __getitem__(self, sizes) -> Self:
        if self.sizes is not None:
            raise TypeError("Already specified the sizes for the tensor comprehension.")
        if not isinstance(sizes, tuple):
            sizes = (sizes,)
        return type(self)(application=self.application, sizes=sizes)

    @staticmethod
    def _size_of(expr: calculus.Expr) -> calculus.Dim:
        match expr:
            case calculus.Get(operand):
                sub = TensorComprehension._size_of(operand)
                return calculus.Dim(sub.operand, sub.axis + 1)
            case calculus.Vec() | calculus.Const() | calculus.Var():
                return calculus.Dim(expr, 0)
            case (
                calculus.At()
                | calculus.Sum()
                | calculus.Dim()
                | calculus.Negate()
                | calculus.Reciprocal()
                | calculus.Add()
                | calculus.Multiply()
            ):
                assert False, f"Cannot take size of type-wise scalar {expr}"
            case _:
                assert_never(expr)

    def __call__(self, constructor: _TensorConstructor) -> Tensor:
        n = (
            len(inspect.signature(constructor).parameters)
            if self.sizes is None
            else len(self.sizes)
        )
        indices = [Index() for _ in range(n)]
        wrapped_indices = [Tensor(calculus.At(index)) for index in indices]
        base_body = constructor(*wrapped_indices)  # type: ignore
        body = Tensor(base_body).expr
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
        return Tensor(body)


array = TensorComprehension(application=calculus.Vec)
sum = TensorComprehension(application=calculus.Sum)

# from typing import TYPE_CHECKING, Protocol, Self, TypeAlias, Union
#
# import numpy
#
# from ein.backend import DEFAULT_BACKEND, Backend
# from ein.calculus import Expr
# from ein.frontend.layout import Layout
# from ein.symbols import Variable
#
# class Array(Scalar, Vec):
#     @property
#     def expr(self) -> Expr:
#         return NotImplemented
#
#     @property
#     def layout(self) -> Layout:
#         return NotImplemented
#
#     def numpy(
#         self,
#         *,
#         env: dict[Variable, numpy.ndarray] | None = None,
#         backend: Backend | str = DEFAULT_BACKEND,
#     ) -> numpy.ndarray:
#         ...
#
#     def __getitem__(
#         self, item_like: ArrayLike | str | tuple[ArrayLike | str, ...]
#     ) -> Self:
#         ...
#
#     def size(self, axis: int) -> Self:
#         ...
#
#     def to_float(self) -> Self:
#         ...
#
#     def __add__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __radd__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __neg__(self) -> Self:
#         ...
#
#     def __sub__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __rsub__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __mul__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __rmul__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __truediv__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __rtruediv__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __mod__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __pow__(self, power, modulo=None):
#         ...
#
#     def __invert__(self) -> Self:
#         ...
#
#     def __and__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __or__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __lt__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __ne__(self, other: ArrayLike) -> Self:  # type: ignore
#         ...
#
#     def __eq__(self, other: ArrayLike) -> Self:  # type: ignore
#         ...
#
#     def __gt__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __le__(self, other: ArrayLike) -> Self:
#         ...
#
#     def __ge__(self, other: ArrayLike) -> Self:
#         ...
#
#     def where(self, true: ArrayLike, false: ArrayLike) -> Self:
#         ...
#
#     def min(self, other: ArrayLike) -> Self:
#         ...
#
#     def max(self, other: ArrayLike) -> Self:
#         ...
#
#     def exp(self) -> Self:
#         ...
#
#     def sin(self) -> Self:
#         ...
#
#     def cos(self) -> Self:
#         ...
#
#     def tanh(self) -> Self:
#         ...

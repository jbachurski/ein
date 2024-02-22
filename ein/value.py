from typing import cast

import numpy

from ein.type_system import Pair, Scalar, Type, ndarray_type

BIG_DATA_SIZE: int = 1024

try:
    from torch import Tensor as _TorchTensor
except ImportError:
    _TorchTensor = type("_TorchTensor", (object,), {})  # noqa


ArrayType = numpy.ndarray | _TorchTensor


class Value:
    value: numpy.ndarray | tuple["Value", "Value"]

    def __init__(
        self,
        value: "Value | tuple[Value, Value] | numpy.typing.ArrayLike",
    ):
        if isinstance(value, Value):
            self.value = value.value
        elif isinstance(value, tuple):
            first, second = value
            self.value = (Value(first), Value(second))
        elif isinstance(value, numpy.ndarray):
            self.value = value
            self.value.flags.writeable = False
        elif numpy.isscalar(value):
            self.value = numpy.array(value)
            self.value.flags.writeable = False
        elif isinstance(value, _TorchTensor):
            self.value = value
        else:
            raise ValueError(f"Invalid type for Ein value: {type(value).__name__}")

    def __eq__(self, other) -> bool:
        if not isinstance(other, Value):
            return False
        if isinstance(self.value, numpy.ndarray):
            if not isinstance(other.value, numpy.ndarray):
                return False
            if len(self.value.data) != len(other.value.data):
                return False
            if self.value.dtype != other.value.dtype:
                return False
            if len(self.value.data) < BIG_DATA_SIZE:
                return self.value.data == other.value.data
            return self is other
        return self.value == other.value

    def __hash__(self) -> int:
        if isinstance(self.value, numpy.ndarray):
            if len(self.value.data) < BIG_DATA_SIZE:
                return hash((self.value.dtype, self.value.data.tobytes()))
            return hash(id(self.value))
        return hash(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    @property
    def array(self) -> numpy.ndarray:
        if not isinstance(self.value, numpy.ndarray):
            raise TypeError(f"Value is not an array but one was expected: {self.value}")
        return self.value

    @property
    def pair(self) -> tuple["Value", "Value"]:
        if not isinstance(self.value, tuple):
            raise TypeError(f"Value is not a pair but one was expected: {self.value}")
        _, _ = self.value
        return cast(tuple[Value, Value], self.value)

    @property
    def type(self) -> Type:
        if isinstance(self.value, numpy.ndarray):
            return ndarray_type(
                self.array.ndim, Scalar.from_dtype(self.array.dtype).kind
            )
        elif isinstance(self.value, tuple):
            first, second = self.value
            return Pair(first.type, second.type)
        assert False

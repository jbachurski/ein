from typing import TypeAlias, cast

import numpy

from ein.type_system import Pair, Type, type_from_ndarray

BIG_DATA_SIZE: int = 1024

try:
    from torch import Tensor as _TorchTensor
except ImportError:
    _TorchTensor = type("_TorchTensor", (object,), {})  # noqa


_ArrayType: TypeAlias = numpy.ndarray | _TorchTensor


class Value:
    value: _ArrayType | tuple["Value", "Value"]

    def __init__(
        self,
        value: "Value | tuple[Value, Value] | _ArrayType | int | float | bool",
    ):
        if isinstance(value, Value):
            self.value = value.value
        elif isinstance(value, tuple):
            first, second = value
            self.value = (Value(first), Value(second))
        elif isinstance(value, numpy.ndarray):
            self.value = value
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
        elif isinstance(self.value, _TorchTensor):
            return hash(id(self.value))
        return hash(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    @property
    def array(self) -> numpy.ndarray:
        if not isinstance(self.value, numpy.ndarray):
            raise TypeError(
                f"Value is not a NumPy array but one was expected: {self.value}"
            )
        return self.value

    @property
    def any_array(self) -> _ArrayType:
        if not isinstance(self.value, (numpy.ndarray, _TorchTensor)):
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
        if isinstance(self.value, (numpy.ndarray, _TorchTensor)):
            return type_from_ndarray(self.any_array)
        elif isinstance(self.value, tuple):
            first, second = self.value
            return Pair(first.type, second.type)
        assert False

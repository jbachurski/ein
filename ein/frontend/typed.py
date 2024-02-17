from typing import Callable, TypeVar

from ein.frontend.comprehension import Idx, array
from ein.frontend.ndarray import Vec

T = TypeVar("T")


def array1(constructor: Callable[[Idx], T]) -> Vec[T]:
    return array(constructor)


def array2(constructor: Callable[[Idx, Idx], T]) -> Vec[Vec[T]]:
    return array(constructor)


def array3(constructor: Callable[[Idx, Idx, Idx], T]) -> Vec[Vec[Vec[T]]]:
    return array(constructor)


def array4(constructor: Callable[[Idx, Idx, Idx, Idx], T]) -> Vec[Vec[Vec[Vec[T]]]]:
    return array(constructor)

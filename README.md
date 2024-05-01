# Ein

Ein is a _pointful array language_ embedded in Python. It is compiled into efficient _point-free_ whole-array operations as present in libraries like NumPy.

The `ein` library forms Ein's frontend in Python, with a lean API focused on array comprehensions constructed via the `array` function. It has NumPy, PyTorch, and JAX backends, and can thus be integrated with existing code.

## Installation

To simply install the `ein` package into `pip` (which should work without poetry), use `pip install .` (where `.` is assumed to be the repository's root directory).
Alternatively, use `pip install git+https://github.com/jbachurski/ein.git` (once the repository is public).
Ein might also be published on PyPI in the future.

To check the basics work, in your Python REPL of choice try:

## Quick start

In Ein, you can use array comprehensions, defined via the `array` function. Ein defers computations -- when you're done, use the `.eval()` method to evaluate. Here's how you can compute a triangular mask of given size:
```py
from ein import array, where

print(array(
    lambda i, j: where(i <= j, 1.0, 0.0),
    size=(3, 4)
).eval())
# -> [[1. 1. 1. 1.]
#     [0. 1. 1. 1.]
#     [0. 0. 1. 1.]]
```

If you would like to involve existing `numpy.ndarray`s in computations, use `wrap` and then evaluate as usual. Here we compute the outer product of a pair of vectors:

```py
import numpy
from ein import array, wrap
u, v = wrap(numpy.array([1., 2.])), wrap(numpy.array([-0.5, 0.0, 0.5]))
print(array(lambda i, j: u[i] * v[j]).eval())
# [[-0.5  0.   0.5]
#  [-1.   0.   1. ]]
```

Ein is strongly typed and in contrast in NumPy it does not perform type promotion, which can cause type errors unwillingly.

```py
from ein import array
array(lambda i: i, size=5).eval()
# -> array([0, 1, 2, 3, 4])
```

## Development

**Environment.** Ein uses [Poetry](https://python-poetry.org) for project configuration and managing environments. To set up your development environment, use `poetry install`, at which point you can use `poetry shell` to activate the virtual environment. For reproducibility, a `poetry.lock` file was used to keep track of dependencies - but this can be adapted accordingly.

**Tests.** The repository uses the [pytest](https://pytest.org) test framework. In a Poetry environment (spawned by `poetry shell`), you may use `pytest tests` to run the test suite.

**Pre-commit hooks.** If you wish to use the same suite of linters, configure the [pre-commit hooks](https://pre-commit.com), as configured by `.pre-commit-config.yaml`. This can be done by executing `pre-commit install`, which sets up a dedicated environment (this might take a minute). Then use `pre-commit run --all-files` to run them manually.

## Documentation

There is currently no full documentation, so this section briefly explains the main functions available in `ein`.

Types `T` allowed and returned by Ein functions are given by:

```
T ::= Scalar                                        (scalars)
    | Vec[T]                                        (vectors)
    | tuple[T, ...] | dict[str, T] | DataclassT     (records)
```
Scalars have operator overloading and various numeric methods. For instance, `a + b` is addition, and `a.float()` converts and integer to a float. Python `int`s and `float`s are automatically promoted to an Ein `Scalar`.

Vectors `Vec[T]` can be indexed, returning an instance of (exactly) `T`. For instance, indexing into a vector of tuples (and records generally) `Vec[tuple[Scalar, Scalar]]` actually returns a Python `tuple`.

For record types, `DataclassT` is constrained to just frozen dataclasses of other Ein values of `T`. However, these may have custom methods implemented, and will be reconstructed accordingly.

### `array`

Array comprehensions, introduced with `array`, are used to construct vectors (type `Vec`).

```
def array(f: (Int) -> T, size: Int | None = None) -> Vec[T]: ...
```

`array` takes a function `f`, such that the element at index `i` is given by `f(i)`. For convenience, `f` can take multiple arguments, in which case the array defines multiple dimensions.

```py
from ein import array

array(lambda i: i, size=5).eval()
# -> array([0, 1, 2, 3, 4])

array(lambda i, j: 3*i + j, size=(2, 3)).eval()
# -> array([[0, 1, 2],
#           [3, 4, 5]])
```

If `size is None`, Ein attempts to infer it from sites where the introduced index is used to directly index into an array. If the defined array has multiple dimensions, a tuple of sizes may be provided. Ein supports dynamically-sized arrays (as long as sizes do not depend on an index, which would lead to jagged arrays), so besides usual `int`s Ein integers can also be passed to `size`.

### `fold`

(Indexed) folds are available via a simple `for` loop-like primitive. Folds don't iterate over an existing vector, and instead over some range `[0, 1, ..., count - 1]`.

```
def fold(init: T, step: (Int, T) -> T, count: Int | None = None) -> T: ...
```

Similarly to `array`, folds can also perform size inference (in this case to assign a value to `count`).

```py
from ein import array, fold

print(fold(1, lambda i, acc: acc * (i + 1), count=5).eval())
# -> 120 == 5!

print(fold(1., lambda i, acc: (i + 1).float() ** acc, count=3).eval())
# -> 9.0 == 3 ** (2 ** 1)

a = array(lambda i: i, size=5)
print(fold(0, lambda i, acc: acc + a[i]).eval())
# -> 10 == 0 + 1 + 2 + 3 + 4
```

### `wrap`

Wrapping introduces existing arrays into the Ein program. This can also be used with `torch.Tensor`s and `jax.Array`s when using the appropriate backends.

```
def wrap(x: numpy.ndarray | ...) -> T: ...
```

### `eval`

```
def eval() -> T: ...
```

By default, Ein evaluates using its NumPy backend. There are also specialised methods - `.numpy()`, `.torch()`, and `.jax()` - for calling specific backends.

### `@function`

A function can be decorated with `@function`, in which case it automatically `wrap`s its arguments and `eval`s the returned value.

### `Vec.reduce`


Vectors can be `reduce`d with custom reductions, which can be much faster than a `fold`.

```
def reduce(vec: Vec[T], ident: T, cat: (T, T) -> T) -> T: ...
```

The arguments `(ident, cat)` must form a _monoid_ (`cat` must be associative, with an identity `ident`).

For instance:

```py
import numpy
from ein import array, wrap

a = wrap(numpy.array([3, 6, -1, 5]))
print(a.reduce(0, lambda x, y: x + y).eval())
# -> 13 == 3 + 6 - 1 + 5
```

This also works with records:

```py
import numpy
from ein import array, wrap, where

a = wrap(numpy.array([3.0, 6.0, -1.0, 5.0]))
value, index = array(
    lambda i: {"value": a[i], "index": i}
).reduce({"value": float('inf'), "index": 0}, lambda x, y: where(
    x["value"] < y["value"],
    x, y
)).values()
print(value.eval(), index.eval())
# -> -1.0 2  as min(a) == -1.0 / argmin(a) == 2
```

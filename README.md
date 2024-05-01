# Ein

Ein is a _pointful array language_ embedded in Python. It is compiled into efficient _point-free_ whole-array operations as present in libraries like NumPy.

The `ein` library forms Ein's frontend in Python, with a lean API focused on array comprehensions constructed via the `array` function. It has NumPy, PyTorch, and JAX backends, and can thus be integrated with existing code.

## Setup

**Installation**. To simply install the `ein` package into `pip` (which should work without poetry), use `pip install .` (where `.` is assumed to be the repository's root directory).

**Environment.** Ein uses [Poetry](https://python-poetry.org) for project configuration and managing environments. To set up your development environment, use `poetry install`, at which point you can use `poetry shell` to activate the virtual environment. For reproducibility, a `poetry.lock` file was used to keep track of dependencies - but this can be adapted accordingly.

**Tests.** The repository uses the [pytest](https://pytest.org) test framework. In a Poetry environment (spawned by `poetry shell`), you may use `pytest tests` to run the test suite.

**Pre-commit hooks.** If you wish to use the same suite of linters, configure the [pre-commit hooks](https://pre-commit.com), as configured by `.pre-commit-config.yaml`. This can be done by executing `pre-commit install`, which sets up a dedicated environment (this might take a minute). Then use `pre-commit run --all-files` to run them manually.

import jax
import jax.numpy as jnp
import numpy

from ein import array, function, wrap
from ein.frontend.std import fold_sum


def test_backend_eval():
    assert jnp.allclose(
        array(lambda i, j: i * j, size=(3, 4)).jax(),
        jnp.asarray([[i * j for j in range(4)] for i in range(3)]),
    )
    assert jnp.allclose(
        array(lambda i, j: wrap(jnp.asarray(2)) * i * j, size=(3, 4)).jax(),
        jnp.asarray([[2 * i * j for j in range(4)] for i in range(3)]),
    )


def test_function_calls_backend():
    @function
    def fun(a):
        return array(lambda i, j: a * i * j, size=(3, 4))

    assert jnp.allclose(
        fun.jax(jnp.asarray(2)),
        jnp.asarray([[2 * i * j for j in range(4)] for i in range(3)]),
    )


def test_jax_grad():
    @function
    def dot(u, v):
        return fold_sum(lambda i: u[i] * v[i])

    a, b = numpy.random.randn(4), numpy.random.randn(4)
    assert jnp.allclose(jax.grad(dot.jax)(a, b), b)

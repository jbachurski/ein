import numpy
import torch

from ein import Scalar, Vec, array, ext, function, scalar_type, wrap


def test_backend_call():
    torch.testing.assert_allclose(
        array(lambda i, j: i * j, size=(3, 4)).torch(),
        torch.from_numpy(numpy.array([[i * j for j in range(4)] for i in range(3)])),
    )


def test_inline_wrap():
    a0, b0 = torch.randn(3), torch.randn(4)
    a, b = wrap(a0), wrap(b0)
    torch.testing.assert_allclose(
        array(lambda i, j: a[i] * b[j]).torch(), a0[:, None] * b0[None, :]
    )


def test_basic_extrinsic():
    def relu(x: Scalar) -> Scalar:
        return ext(torch.relu, scalar_type(float))(x)

    a0 = torch.randn(3, 5)
    a = wrap(a0)

    torch.testing.assert_allclose(
        array(lambda i, j: relu(a[i, j])).torch(), torch.relu(a0)
    )


def test_function():
    @function
    def outer(x: Vec[Scalar], y: Vec[Scalar]) -> Vec[Vec[Scalar]]:
        return array(lambda i, j: x[i] * y[j])

    a, b = torch.randn(3), torch.randn(4)
    torch.testing.assert_allclose(outer(a, b), outer.torch(a, b))
    torch.testing.assert_allclose(outer(a, b), a[:, None] * b[None, :])
    torch.testing.assert_allclose(outer(a, b), outer(a, b.numpy()))

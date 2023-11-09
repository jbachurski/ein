import numpy

from ein import interpret_with_numpy


def test_attention():
    from .suite.deep.attention import Attention

    args = Attention.sample()

    numpy.testing.assert_allclose(
        Attention.in_ein_function(interpret_with_numpy, *args),
        Attention.in_numpy(*args),
    )


def test_mri_q():
    from .suite.parboil.mri_q import MriQ

    args = MriQ.sample()

    ref = MriQ.in_numpy(*args)

    numpy.testing.assert_allclose(
        MriQ.in_ein_function(interpret_with_numpy, *args), ref
    )

    numpy.testing.assert_allclose(MriQ.in_numpy_frugal(*args), ref)

    numpy.testing.assert_allclose(MriQ.in_numpy_einsum(*args), ref)

    numpy.testing.assert_allclose(MriQ.in_python(*args), ref)

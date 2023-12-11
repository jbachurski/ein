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


def test_stencil():
    from .suite.parboil.stencil import Stencil

    args = Stencil.sample()

    ref = Stencil.in_numpy(*args)

    numpy.testing.assert_allclose(
        Stencil.in_ein_function(interpret_with_numpy, *args), ref
    )
    numpy.testing.assert_allclose(Stencil.in_python(*args), ref)


def test_kmeans():
    from .suite.rodinia.kmeans import KMeans

    args = KMeans.sample(12, 7, 3, 50)

    ref = KMeans.in_numpy(*args)
    numpy.testing.assert_allclose(
        KMeans.in_ein_function(interpret_with_numpy, *args), ref
    )
    numpy.testing.assert_allclose(KMeans.in_python(*args), ref)


def test_nn():
    from .suite.rodinia.nn import NN

    args = NN.sample()

    ref = NN.in_numpy(*args)
    numpy.testing.assert_allclose(NN.in_ein_function(interpret_with_numpy, *args), ref)
    numpy.testing.assert_allclose(NN.in_python(*args), ref)


def test_pathfinder():
    from .suite.rodinia.pathfinder import Pathfinder

    args = Pathfinder.sample()

    ref = Pathfinder.in_numpy(*args)
    numpy.testing.assert_allclose(
        Pathfinder.in_ein_function(interpret_with_numpy, *args), ref
    )
    numpy.testing.assert_allclose(Pathfinder.in_python(*args), ref)

import numpy
import pytest

from ein import interpret_with_naive, interpret_with_numpy

from . import with_interpret, with_interpret_for_dynamic_sizes


@with_interpret
def test_attention(interpret):
    if interpret == interpret_with_naive:
        # FIXME: Why does CSE fail??
        return pytest.mark.skip()
    from .suite.deep.attention import Attention

    args = Attention.sample()

    numpy.testing.assert_allclose(
        Attention.in_ein_function(interpret, *args),
        Attention.in_numpy(*args),
    )


@with_interpret
def test_gat(interpret):
    from .suite.deep.gat import GAT

    args = GAT.sample()

    numpy.testing.assert_allclose(
        GAT.in_ein_function(interpret, *args), GAT.in_numpy(*args)
    )


@with_interpret
def test_semiring(interpret):
    from .suite.misc.semiring import FunWithSemirings

    args = FunWithSemirings.sample()
    numpy.testing.assert_allclose(
        FunWithSemirings.in_ein_function(interpret, *args),
        FunWithSemirings.in_numpy(*args),
    )

    numpy.testing.assert_allclose(
        FunWithSemirings.in_ein_function(interpret, *args),
        FunWithSemirings.in_python(*args),
    )


@with_interpret_for_dynamic_sizes
def test_mandelbrot(interpret):
    from .suite.misc.mandelbrot import Mandelbrot

    args = Mandelbrot.sample()

    numpy.testing.assert_allclose(
        Mandelbrot.in_ein_function(interpret, *args),
        Mandelbrot.in_python(*args),
    )

    numpy.testing.assert_allclose(
        Mandelbrot.in_ein_function(interpret, *args),
        Mandelbrot.in_numpy(*args),
    )


@with_interpret
def test_mri_q(interpret):
    from .suite.parboil.mri_q import MriQ

    args = MriQ.sample()

    ref = MriQ.in_numpy(*args)

    numpy.testing.assert_allclose(MriQ.in_ein_function(interpret, *args), ref)
    numpy.testing.assert_allclose(MriQ.in_numpy_frugal(*args), ref)
    numpy.testing.assert_allclose(MriQ.in_numpy_einsum(*args), ref)
    numpy.testing.assert_allclose(MriQ.in_numpy_smart(*args), ref)
    numpy.testing.assert_allclose(MriQ.in_python(*args), ref)


def test_stencil():
    from .suite.parboil.stencil import Stencil

    args = Stencil.sample()

    ref = Stencil.in_numpy(*args)

    numpy.testing.assert_allclose(
        Stencil.in_ein_function(interpret_with_numpy, *args), ref
    )
    numpy.testing.assert_allclose(Stencil.in_python(*args), ref)


def test_hotspot():
    from .suite.rodinia.hotspot import Hotspot

    args = Hotspot.sample()

    ref = Hotspot.in_numpy(*args)
    numpy.testing.assert_allclose(
        Hotspot.in_ein_function(interpret_with_numpy, *args), ref
    )
    numpy.testing.assert_allclose(Hotspot.in_python(*args), ref)


@with_interpret_for_dynamic_sizes
def test_kmeans(interpret):
    from .suite.rodinia.kmeans import KMeans

    args = KMeans.sample()

    ref = KMeans.in_numpy(*args)
    numpy.testing.assert_allclose(KMeans.in_ein_function(interpret, *args), ref)
    numpy.testing.assert_allclose(KMeans.in_python(*args), ref)


@with_interpret_for_dynamic_sizes
def test_nn(interpret):
    from .suite.rodinia.nn import NN

    args = NN.sample()

    ref = NN.in_numpy(*args)
    numpy.testing.assert_allclose(NN.in_ein_function(interpret, *args), ref)
    numpy.testing.assert_allclose(NN.in_python(*args), ref)


@with_interpret_for_dynamic_sizes
def test_pathfinder(interpret):
    from .suite.rodinia.pathfinder import Pathfinder

    args = Pathfinder.sample()

    ref = Pathfinder.in_numpy(*args)
    numpy.testing.assert_allclose(Pathfinder.in_ein_function(interpret, *args), ref)
    numpy.testing.assert_allclose(Pathfinder.in_python(*args), ref)

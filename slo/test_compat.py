import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from itertools import product

import slo
from slo.backends import available_backends
BACKENDS = available_backends()


@pytest.mark.parametrize("backend,N,K",
    product( BACKENDS, [23,45], [1,2,3] ) )
def test_compat_Multiply(backend, N, K):
    pymr = pytest.importorskip('pymr')
    b = backend()
    x = slo.util.rand64c(N,K)
    d = slo.util.rand64c(N,1)

    D0 = pymr.linop.Multiply( (N,K), (N,K), d, dtype=d.dtype )
    D1 = b.Diag(d)

    y_exp = pymr.util.reshape(D0 * pymr.util.vec(x), (N,K))
    y_act = D1 * x

    npt.assert_allclose(y_act, y_exp, rtol=1e-5)


@pytest.mark.parametrize("backend,X,Y,Z,P,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3], [1,2,3] ) )
def test_compat_Zpad(backend, X,Y,Z, P, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    i_shape = (X, Y, Z, K)
    o_shape = (X+2*P, Y+2*P, Z+2*P, K)

    x = slo.util.rand64c( *i_shape )

    D0 = pymr.linop.Zpad( o_shape, i_shape, dtype=x.dtype )
    D1 = b.Zpad(o_shape[:3], i_shape[:3], dtype=x.dtype)

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(y_act, y_exp, rtol=1e-5)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_UnscaledFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), normalize=False, center=False, dtype=np.dtype('complex64'))
    D1 = b.UnscaledFFT(shape[:3], dtype=np.dtype('complex64'))

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(y_act, y_exp, rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_UnitaryFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), center=False, dtype=np.dtype('complex64'))
    S, F = b.FFT(shape[:3], dtype=np.dtype('complex64'))
    D1 = S * F

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(y_act, y_exp, rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_CenteredFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), dtype=np.dtype('complex64'))
    Mk, S, F, Mx = b.FFTc(shape[:3], dtype=np.dtype('complex64'))
    D1 = Mk * S * F * Mx

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(abs(y_act), abs(y_exp), rtol=1e-2)

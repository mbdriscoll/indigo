import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse as spp
from itertools import product

import indigo
from indigo.backends import available_backends

BACKENDS = available_backends()

@pytest.mark.parametrize("backend,L,M,N,K,density",
    list(product( BACKENDS, [3,4], [5,6], [7,8], [1,8,9,17], [1,0.01,0.1,0.5] ))
)
def test_Realize_Product(backend, L, M, N, K, density):
    b = backend()
    A0_h = indigo.util.randM(L, M, density)
    A1_h = indigo.util.randM(M, N, density)
    A0 = b.SpMatrix(A0_h, name='A0')
    A1 = b.SpMatrix(A1_h, name='A1')
    A = A0 * A1
    A = A.realize()

    # forward
    x = b.rand_array((N,K))
    y = b.rand_array((L,K))
    A.eval(y, x)

    x_h = x.to_host()
    y_act = y.to_host()
    y_exp = A0_h @ (A1_h @ x_h)
    npt.assert_allclose(y_act, y_exp, rtol=1e-3)

    # adjoint
    x = b.rand_array((L,K))
    y = b.rand_array((N,K))
    A.H.eval(y, x)

    x_h = x.to_host()
    y_act = y.to_host()
    y_exp = A1_h.H @ (A0_h.H @ x_h)
    npt.assert_allclose(y_act, y_exp, rtol=1e-3)

    # shape
    assert A.shape == (L,N)
    assert A.H.shape == (N,L)

    # dtype
    assert A.dtype == np.dtype('complex64')

@pytest.mark.parametrize("backend", BACKENDS )
def test_Realize_HStack(backend):
    from indigo.operators import SpMatrix
    b = backend()
    x = b.Eye(4)
    y = b.Eye(4)
    z = b.HStack((x,y))
    zr = z.realize()
    assert isinstance(zr, SpMatrix)

@pytest.mark.parametrize("backend", BACKENDS )
def test_Realize_BlockDiag(backend):
    from indigo.operators import SpMatrix
    b = backend()
    x = b.Eye(4)
    y = b.Eye(4)
    z = b.BlockDiag((x,y))
    zr = z.realize()
    assert isinstance(zr, SpMatrix)

@pytest.mark.parametrize("backend", BACKENDS )
def test_DistributeKroniOverProd(backend):
    from indigo.operators import Product, Kron
    from indigo.transforms import DistributeKroniOverProd
    b = backend()
    x = b.Eye(4)
    y = b.Eye(4)
    z = b.KronI(2, x*y)
    z2 = DistributeKroniOverProd().visit(z)
    assert isinstance(z2, Product)
    assert isinstance(z2.left, Kron)
    assert isinstance(z2.right, Kron)


@pytest.mark.parametrize("backend", BACKENDS )
def test_DistributeAdjointOverProd(backend):
    from indigo.operators import Product, Adjoint
    from indigo.transforms import DistributeAdjointOverProd
    b = backend()
    x = b.Eye(4)
    y = b.Eye(4)
    z = b.Adjoint(x*y)
    z2 = DistributeAdjointOverProd().visit(z)
    assert isinstance(z2, Product)
    assert isinstance(z2.left, Adjoint)
    assert isinstance(z2.right, Adjoint)


@pytest.mark.parametrize("backend", BACKENDS )
def test_LiftUnscaledFFTs(backend):
    from indigo.operators import Product, Adjoint
    from indigo.transforms import LiftUnscaledFFTs
    b = backend()
    s = b.Eye(4)
    f = b.UnscaledFFT((2,2), dtype=s.dtype)
    z = (f*s)*(s*f)
    z2 = LiftUnscaledFFTs().visit(z)


@pytest.mark.parametrize("backend,M",
    list(product( BACKENDS, [3,4]))
)
def test_Realize_Eye(backend, M):
    from indigo.operators import SpMatrix
    b = backend()
    A = b.Eye(M, dtype=np.complex64)
    A = A.realize()
    assert isinstance(A, SpMatrix)


@pytest.mark.parametrize("backend,M",
    list(product( BACKENDS, [3,4]))
)
def test_Realize_Scale(backend, M):
    from indigo.operators import SpMatrix
    b = backend()
    A = 3 * b.Eye(M, dtype=np.complex64)
    A = A.realize()
    assert isinstance(A, SpMatrix)


@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_Realize_One(backend, M, N):
    from indigo.operators import SpMatrix
    b = backend()
    A = b.One((M,N))
    A = A.realize()
    assert isinstance(A, SpMatrix)
    assert np.all(A._matrix.data == 1)

@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_SpyOut(backend, M, N):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from indigo.operators import SpMatrix
    from indigo.transforms import SpyOut
    b = backend()
    A = b.One((M,N)).realize()
    SpyOut().visit(A)

@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_GroupRLP(backend, M, N):
    from indigo.operators import SpMatrix
    from indigo.transforms import GroupRightLeaningProducts
    b = backend()
    A = b.One((M,N)).realize()
    B = A * A.H * A
    GroupRightLeaningProducts().visit(B)

@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_MakeRL(backend, M, N):
    from indigo.operators import SpMatrix
    from indigo.transforms import MakeRightLeaning
    b = backend()
    A = b.One((M,N)).realize()
    B = A * A.H * A
    MakeRightLeaning().visit(B)

@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_LiftUFFTS(backend, M, N):
    from indigo.transforms import LiftUnscaledFFTs
    b = backend()
    A = b.One((M,N)).realize()
    B = b.KronI(3, A)
    LiftUnscaledFFTs().visit(B)

@pytest.mark.parametrize("backend,M,N",
    list(product( BACKENDS, [3,4], [5,6]))
)
def test_LiftUFFTS2(backend, M, N):
    from indigo.operators import SpMatrix
    from indigo.transforms import LiftUnscaledFFTs
    b = backend()
    A = b.UnscaledFFT((M,N), dtype=np.complex64).realize().H
    LiftUnscaledFFTs().visit(A)

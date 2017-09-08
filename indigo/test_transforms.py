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
    from indigo.operators import Product, KronI
    from indigo.transforms import DistributeKroniOverProd
    b = backend()
    x = b.Eye(4)
    y = b.Eye(4)
    z = b.KronI(2, x*y)
    z2 = DistributeKroniOverProd().visit(z)
    assert isinstance(z2, Product)
    assert isinstance(z2.left_child, KronI)
    assert isinstance(z2.right_child, KronI)


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
    assert isinstance(z2.left_child, Adjoint)
    assert isinstance(z2.right_child, Adjoint)


@pytest.mark.parametrize("backend", BACKENDS )
def test_LiftUnscaledFFTs(backend):
    from indigo.operators import Product, Adjoint
    from indigo.transforms import LiftUnscaledFFTs
    b = backend()
    s = b.Eye(4)
    f = b.UnscaledFFT((2,2), dtype=s.dtype)
    z = (f*s)*(s*f)
    z2 = LiftUnscaledFFTs().visit(z)

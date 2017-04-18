import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from itertools import product

import slo
from slo.backends import available_backends

BACKENDS = available_backends()

@pytest.mark.parametrize("backend,L,M,N,K,density",
    list(product( BACKENDS, [3,4], [5,6], [7,8], [1,8,9,17], [1,0.01,0.1,0.5] ))
)
def test_Realize_Product(backend, L, M, N, K, density):
    b = backend()
    A0_h = slo.util.randM(L, M, density)
    A1_h = slo.util.randM(M, N, density)
    A0 = b.SpMatrix(A0_h, name='A0')
    A1 = b.SpMatrix(A1_h, name='A1')
    A = A0 * A1
    A = A.optimize()

    # forward
    x = b.rand_array((N,K))
    y = b.rand_array((L,K))
    A.eval(y, x)

    x_h = x.to_host()
    y_act = y.to_host()
    y_exp = A0_h @ (A1_h @ x_h)
    npt.assert_allclose(y_act, y_exp, rtol=1e-5)

    # adjoint
    x = b.rand_array((L,K))
    y = b.rand_array((N,K))
    A.H.eval(y, x)

    x_h = x.to_host()
    y_act = y.to_host()
    y_exp = A1_h.H @ (A0_h.H @ x_h)
    npt.assert_allclose(y_act, y_exp, rtol=1e-5)

    # shape
    assert A.shape == (L,N)
    assert A.H.shape == (N,L)

    # dtype
    assert A.dtype == np.dtype('complex64')

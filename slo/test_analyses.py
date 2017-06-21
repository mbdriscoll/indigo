import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from itertools import product

import slo
from slo.operators import Product
from slo.backends import available_backends
BACKENDS = available_backends()

@pytest.mark.parametrize("backend,L,M,N,K,density,batch",
    product( BACKENDS, [3,4], [5,6], [7,8], [1,8,9,17], [0.01,0.1,0.5,1], [None, 1, 2, 10] ))
def test_Memusage_Product(backend, L, M, N, K, density, batch):
    b = backend()
    A0_h = slo.util.randM(L, M, density)
    A1_h = slo.util.randM(M, N, density)
    A0 = b.SpMatrix(A0_h, name='A0', batch=batch)
    A1 = b.SpMatrix(A1_h, name='A1', batch=batch)
    A = Product(b, A0, A1, batch=batch)

    # forward
    alpha = beta = 1.0
    x = b.rand_array((N,K))
    y = b.rand_array((L,K))
    y_exp = beta * y.to_host() + alpha * A0_h @ (A1_h @ x.to_host())
    A.eval(y, x, alpha=alpha, beta=beta)
    npt.assert_allclose(y.to_host(), y_exp, rtol=1e-5)

    bs = min(batch, x.shape[1]) if batch else x.shape[1]

    a0_mbytes = A0.memusage(x.shape, x.dtype)[0]
    a1_mbytes = A1.memusage(x.shape, x.dtype)[0]
    tmp_mbytes = (A0.shape[1] * bs * x.dtype.itemsize) / 1024 / 1024
    cg_tmp_mbytes = x.nbytes*4 / 1024 / 1024

    mbytes = a0_mbytes + a1_mbytes + tmp_mbytes + cg_tmp_mbytes

    memusage_mbytes = sum(A.memusage(x.shape, x.dtype))

    assert(memusage_mbytes == mbytes)

    # adjoint
    x = b.rand_array((L,K))
    y = b.rand_array((N,K))
    bs = min(batch, x.shape[1]) if batch else x.shape[1]
    tmp_mbytes = (A0.H.shape[0] * bs * x.dtype.itemsize) / 1024 / 1024
    cg_tmp_mbytes = x.nbytes*4 / 1024 / 1024


    mbytes = a0_mbytes + a1_mbytes + tmp_mbytes + cg_tmp_mbytes

    memusage_mbytes = sum(A.H.memusage(x.shape, x.dtype))

    assert(memusage_mbytes == mbytes)

    # shape
    assert A.shape == (L,N)
    assert A.H.shape == (N,L)

    # dtype
    assert A.dtype == np.dtype('complex64')


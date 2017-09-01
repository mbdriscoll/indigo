import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from itertools import product

import indigo
from indigo.operators import Product
from indigo.backends import available_backends
BACKENDS = available_backends()

@pytest.mark.parametrize("backend,L,M,N,K,density,b0,b1,b2",
    product( BACKENDS, [3,4], [5,6], [7,8], [1,8,9,17], [0.01,0.1,0.5,1],
        [None, 1, 2, 10], [None, 1, 2], [None, 1, 2]
))
def test_Memusage_Product(backend, L, M, N, K, density, b0, b1, b2):
    b = backend()
    A0_h = indigo.util.randM(L, M, density)
    A1_h = indigo.util.randM(M, N, density)
    A0 = b.SpMatrix(A0_h, name='A0', batch=b0)
    A1 = b.SpMatrix(A1_h, name='A1', batch=b1)
    A = Product(b, A0, A1, batch=b2)

    # x = b.rand_array((N,K))
    # y = b.rand_array((L,K))

    a0_nbytes = A0_h.data.nbytes + A0_h.indices.nbytes + A0_h.indptr.nbytes
    a1_nbytes = A1_h.data.nbytes + A1_h.indices.nbytes + A1_h.indptr.nbytes

    # forward
    x_nbytes  = N * K * A.dtype.itemsize
    bs = min(K, b2 or K)
    tmp_nbytes = A0.shape[1] * bs * A0.dtype.itemsize
    nbytes_exp = a0_nbytes + a1_nbytes + tmp_nbytes
    nbytes_act = A.memusage(ncols=K)
    assert nbytes_exp == nbytes_act

    # adjoint
    x_nbytes  = L * K * A.dtype.itemsize
    tmp_nbytes = A0.H.shape[0] * bs * A.dtype.itemsize
    nbytes_exp = a0_nbytes + a1_nbytes + tmp_nbytes
    nbytes_act = A.H.memusage(ncols=K)
    assert nbytes_exp == nbytes_act

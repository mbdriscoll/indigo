import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from itertools import product

import pymr
from pymr.backends import available_backends

BACKENDS = available_backends()

@pytest.mark.parametrize("backend,N",
    product( BACKENDS, [23,45] ))
def test_compat_Product(backend, N):
    b = backend()
    d = (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex64)
    D0 = pymr.linop.Multiply(d.shape, d.shape, d)
    D1 = b.SpMatrix( spp.diags([d], offsets=[0]) )

    x = (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex64)
    y_exp = D0 * pymr.util.vec(x)

    # forward
    x_d = b.copy_array(x)
    y_d = b.zero_array(y_exp.shape, x.dtype)
    D1.eval( y_d, x_d )
    y_act = y_d.to_host()
    npt.assert_allclose(y_act, y_exp, rtol=1e-5)

    # adjoint
    x_exp = D0.H * y_exp
    D1.H.eval( x_d, y_d )
    x_act = x_d.to_host()
    npt.assert_allclose(x_act, x_exp, rtol=1e-5)

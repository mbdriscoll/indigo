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
def test_compat_Multiply(backend, N,K):
    pymr = pytest.importorskip('pymr')
    b = backend()
    x = slo.util.rand64c(N,K)
    d = slo.util.rand64c(N,1)

    D0 = pymr.linop.Multiply( (N,K), (N,K), d, dtype=d.dtype )
    D1 = b.Diag(d)

    y_exp = pymr.util.reshape(D0 * pymr.util.vec(x), (N,K))
    y_act = D1 * x

    npt.assert_allclose(y_act, y_exp, rtol=1e-5)

import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from itertools import product

import slo
from slo.backends import available_backends
BACKENDS = available_backends()


@pytest.mark.parametrize("backend,N",
    product( BACKENDS, [23,45] ) )
def test_compat_Multiply(backend, N):
    pymr = pytest.importorskip('pymr')
    b = backend()
    x = slo.util.rand64c(N,1)
    d = slo.util.rand64c(N,1)

    y_exp = x * d

    D0 = pymr.linop.Multiply( (N,1), (N,1), d, dtype=d.dtype )
    y_act0 = D0 * x
    npt.assert_allclose(y_act0, y_exp, rtol=1e-5)

    D1 = b.Diag(d)
    y_act1 = D1 * x
    npt.assert_allclose(y_act1, y_exp, rtol=1e-5)

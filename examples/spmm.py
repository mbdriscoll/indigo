"""
spmm.py

This script performs sparse-times-dense matrix multiplication on
a CPU or GPU as specified.
"""

import numpy as np
import scipy.sparse as spp

import slo.backends.cuda

# problem dimensions
M, N, K = 100, 200, 4

# create data
x = (np.random.rand(N,K) + 1j*np.random.rand(N,K)).astype(np.complex64)
m = spp.rand(M, N, density=0.23).astype(np.complex64)

backend = slo.backends.cuda.CudaBackend()
M = backend.SpMatrix(m)

# try automatic interface
y = M * x

# try manual interface
x_d = backend.copy_array(x)
y_d = backend.zero_array(y.shape, y.dtype)
M.eval(y_d, x_d)
y = y_d.to_host()

# check answer
np.testing.assert_allclose( y, m @ x, rtol=1e-4 )
print('ok')

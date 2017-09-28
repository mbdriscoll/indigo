import numpy as np
from ctypes import cdll
import scipy.sparse as spp

from indigo.backends.cuda import CudaBackend
from indigo.backends import _customgpu

class CustomGpuBackend(CudaBackend):

    def ccsrmm(self, Y, A_shape, A_indx, A_ptr, A_vals, X, alpha, beta, adjoint=False, exwrite=False):
        if adjoint and exwrite:
            (M, K), N = A_shape, X.shape[1]
            ldx = X._leading_dims[0]
            ldy = Y._leading_dims[0]
            _customgpu.exw_csrmm(M, N, K, alpha, A_vals._arr.value, A_indx._arr.value, A_ptr._arr.value,
                X._arr.value, ldx, beta, Y._arr.value, ldy)
        else:
            super(CustomGpuBackend, self).ccsrmm(
                Y, A_shape, A_indx, A_ptr, A_vals, X, alpha, beta, adjoint, exwrite=exwrite
            )

    def diamm(self, y, shape, offsets, data, x, alpha=1.0, beta=0.0, adjoint=True):
        ldx = x._leading_dims[0]
        ldy = y._leading_dims[0]
        (K, N), M = x.shape, y.shape[0]
        noffsets = offsets.size
        _customgpu.diamm(M, N, K, noffsets, offsets._arr.value, data._arr.value,
            alpha, x._arr.value, ldx, beta, y._arr.value, ldy, adjoint)

    def onemm(self, y, x, alpha, beta):
        ldx = x._leading_dims[0]
        ldy = y._leading_dims[0]
        (K, N), M = x.shape, y.shape[0]
        _customgpu.onemm(M, N, K,
            alpha, x._arr.value, ldx,
            beta,  y._arr.value, ldy)

    def max(self, val, arr):
        _customgpu.max(arr.size*2, val, arr._arr.value)


    class dia_matrix(object):
        """
        A device-resident sparse matrix in DIA format.
        """
        def __init__(self, backend, A, name='mat'):
            """
            Create a matrix from the given `scipy.sparse.sppmatrix`.
            """
            assert isinstance(A, spp.dia_matrix)
            A = A.astype(np.complex64)
            self._backend = backend
            self.data = backend.copy_array(A.data.T, name=name+".data")
            self.offsets = backend.copy_array(A.offsets, name=name+".data")
            self.shape = A.shape
            self.dtype = A.dtype

        def forward(self, y, x, alpha=1, beta=0):
            """ y[:] = A * x """
            self._backend.diamm(y, self.shape, self.offsets, self.data,
                x, alpha=alpha, beta=beta, adjoint=False)

        def adjoint(self, y, x, alpha=1, beta=0):
            """ y[:] = A.H * x """
            self._backend.diamm(y, self.shape, self.offsets, self.data,
                x, alpha=alpha, beta=beta, adjoint=True)

        @property
        def nbytes(self):
            return self.offsets.nbytes + self.data.nbytes

        @property
        def nnz(self):
            return self.data.size

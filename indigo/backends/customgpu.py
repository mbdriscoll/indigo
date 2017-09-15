import numpy as np
from ctypes import cdll

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

    def onemm(self, y, x, alpha, beta):
        ldx = x._leading_dims[0]
        ldy = y._leading_dims[0]
        (K, N), M = x.shape, y.shape[0]
        _customgpu.onemm(M, N, K,
            alpha, x._arr.value, ldx,
            beta,  y._arr.value, ldy)

    def max(self, val, arr):
        _customgpu.max(arr.size*2, val, arr._arr.value)

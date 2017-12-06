import numpy as np
from ctypes import cdll

from indigo.backends.mkl import MklBackend
from indigo.backends import _customcpu

class CustomCpuBackend(MklBackend):

    class csr_matrix(MklBackend.csr_matrix):
        _index_base = 0

        def _type_correct(self, A):
            return A

    def ccsrmm(self, Y, A_shape, A_indx, A_ptr, A_vals, X, alpha, beta, adjoint=False, exwrite=False):
        ldx = X._leading_dim
        ldy = Y._leading_dim
        (M, K), N = A_shape, X.shape[1]
        _customcpu.csrmm(adjoint, M, N, K, alpha,
            A_vals._arr, A_indx._arr, A_ptr._arr,
            X._arr, ldx, beta, Y._arr, ldy, exwrite)

    def onemm(self, y, x, alpha, beta):
        ldx = x._leading_dim
        ldy = y._leading_dim
        (K, N), M = x.shape, y.shape[0]
        _customcpu.onemm(M, N, K, alpha, x._arr, ldx, beta, y._arr, ldy)

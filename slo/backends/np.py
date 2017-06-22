from ctypes import *

import numpy as np
import scipy.sparse as spp
from numpy.ctypeslib import ndpointer

from slo.backends.backend import Backend

class NumpyBackend(Backend):

    def __init__(self, device_id=0):
        super(NumpyBackend, self).__init__()

    # -----------------------------------------------------------------------
    # Arrays
    # -----------------------------------------------------------------------
    class dndarray(Backend.dndarray):
        def _copy_from(self, arr):
            np.copyto(self._arr, arr)

        def _copy_to(self, arr):
            np.copyto(arr, self._arr)

        def _copy(self, d_arr):
            np.copyto(self._arr, d_arr._arr)

        def _malloc(self, shape, dtype):
            return np.ndarray(shape, dtype, order='F')

        def _free(self):
            del self._arr

        def _zero(self):
            self._arr[:] = 0

        def __getitem__(self, slc):
            d = self._arr.reshape(self.shape, order='F')[slc]
            ld = self._leading_dims
            return self._backend.dndarray( self._backend, d.shape, d.dtype,
                ld=ld, own=False, data=d)

        @staticmethod
        def from_param(obj):
            if not isinstance(obj, NUMPY.dndarray):
                raise ArgumentError('{} is not a dndarray'.format( type(obj) ))
            return obj._arr.ctypes.get_as_parameter()

    # -----------------------------------------------------------------------
    # BLAS Routines
    # -----------------------------------------------------------------------
    def axpy(self, y, alpha, x):
        """ y += alpha * x """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        y._arr += alpha * x._arr.reshape( y.shape, order='F' )

    def dot(self, x, y):
        """ returns x^T * y """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        return np.vdot( x._arr, y._arr ).real

    def norm2(self, x):
        """ returns ||x||_2"""
        assert isinstance(x, self.dndarray)
        return np.linalg.norm(x._arr)**2

    def scale(self, x, alpha):
        """ x *= alpha """
        assert isinstance(x, self.dndarray)
        x._arr *= alpha

    def cgemm(self, y, M, x, alpha, beta, forward):
        x, y, M = x._arr, y._arr, M._arr
        if forward:
            y[:] = alpha * (M @ x) + beta * y
        else:
            MH = np.conj(M.T)
            y[:] = alpha * (MH @ x) + beta * y

    # -----------------------------------------------------------------------
    # FFT Routines
    # -----------------------------------------------------------------------
    def fftn(self, y, x):
        X = x._arr.reshape( x.shape, order='F' )
        Y = y._arr.reshape( y.shape, order='F' )
        Y[:] = np.fft.fftn(X, axes=(0,1,2))

    def ifftn(self, y, x):
        scale = np.prod(x.shape[:3])
        X = x._arr.reshape( x.shape, order='F' )
        Y = y._arr.reshape( y.shape, order='F' )
        Y[:] = np.fft.ifftn(X, axes=(0,1,2)) * scale

    # -----------------------------------------------------------------------
    # CSRMM Routine
    # -----------------------------------------------------------------------
    def ccsrmm(self, y, A_shape, A_indx, A_ptr, A_vals, x, alpha, beta, adjoint=False):
        A = spp.csr_matrix((A_vals._arr, A_indx._arr, A_ptr._arr), shape=A_shape)
        X = x._arr.reshape( x.shape, order='F')
        Y = y._arr.reshape( y.shape, order='F')
        if adjoint:
            Y[:] = alpha * (A.H @ X) + beta * Y
        else:
            Y[:] = alpha * (A @ X) + beta * Y

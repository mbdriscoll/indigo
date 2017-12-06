from ctypes import *

import numpy as np
import scipy.sparse as spp
from numpy.ctypeslib import ndpointer

from indigo.backends.backend import Backend

class NumpyBackend(Backend):

    def __init__(self, device_id=0):
        super(NumpyBackend, self).__init__()

    # -----------------------------------------------------------------------
    # Arrays
    # -----------------------------------------------------------------------
    class dndarray(Backend.dndarray):
        def _copy_from(self, arr):
            self._arr.flat[:] = arr.flat

        def _copy_to(self, arr):
            arr.flat[:] = self._arr.flat

        def _copy(self, d_arr):
            dst = self._arr.reshape(-1, order='F')
            src = d_arr._arr.reshape(-1, order='F')
            dst.flat[:] = src.flat

        def _malloc(self, shape, dtype):
            return np.ndarray(shape, dtype, order='F')

        def _free(self):
            del self._arr

        def _zero(self):
            self._arr[:] = 0

        def __getitem__(self, slc):
            d = self._arr.reshape(self.shape, order='F')[slc]
            ld = self._leading_dim
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
    def axpby(self, beta, y, alpha, x):
        """ y = beta*y + alpha*x """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        x = x._arr.reshape(y._arr.shape, order='F')
        y._arr[:] = beta * y._arr + alpha * x

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

    def cgemm(self, y, M, x, alpha=1, beta=0, forward=True, left=True):
        y, M, x = y._arr, M._arr, x._arr
        if not forward:
            M = np.conj(M.T)
        if left:
            x = x.reshape((M.shape[1],-1), order='F')
            y = y.reshape((M.shape[0],-1), order='F')
            y[:] = alpha * (M @ x) + beta * y
        else:
            x = x.reshape((-1,M.shape[0]), order='F')
            y = y.reshape((-1,M.shape[1]), order='F')
            y[:] = alpha * (x @ M) + beta * y

    def csymm(self, y, M, x, alpha, beta, left=True):
        return self.cgemm(y, M, x, alpha, beta, forward=True, left=left)

    # -----------------------------------------------------------------------
    # OneMM Routines
    # -----------------------------------------------------------------------
    def onemm(self, y, x, alpha, beta):
        y._arr[:] = beta * y._arr + alpha * \
            np.broadcast_to(x._arr.sum(axis=0, keepdims=True), y.shape)

    # -----------------------------------------------------------------------
    # FFT Routines
    # -----------------------------------------------------------------------
    def fftn(self, y, x):
        X = x._arr.reshape( x.shape, order='F' )
        Y = y._arr.reshape( y.shape, order='F' )
        ndim = X.ndim-1
        axes = tuple(range(ndim))
        Y[:] = np.fft.fftn(X, axes=axes)

    def ifftn(self, y, x):
        X = x._arr.reshape( x.shape, order='F' )
        Y = y._arr.reshape( y.shape, order='F' )
        ndim = X.ndim-1
        axes = tuple(range(ndim))
        scale = np.prod( X.shape[:ndim] )
        Y[:] = np.fft.ifftn(X, axes=axes) * scale

    # -----------------------------------------------------------------------
    # CSRMM Routine
    # -----------------------------------------------------------------------
    def ccsrmm(self, y, A_shape, A_indx, A_ptr, A_vals, x, alpha, beta, adjoint=False, exwrite=False):
        A = spp.csr_matrix((A_vals._arr, A_indx._arr, A_ptr._arr), shape=A_shape)
        X = x._arr.reshape( x.shape, order='F')
        Y = y._arr.reshape( y.shape, order='F')
        if adjoint:
            Y[:] = alpha * (A.H @ X) + beta * Y
        else:
            Y[:] = alpha * (A @ X) + beta * Y

    def cdiamm(self, y, shape, offsets, data, x, alpha=1.0, beta=0.0, adjoint=True):
        A = spp.dia_matrix((data._arr.T, offsets._arr), shape=shape)
        X = x._arr.reshape( x.shape, order='F' )
        Y = y._arr.reshape( y.shape, order='F' )
        if adjoint:
            Y[:] = alpha * (A.H @ X) + beta * Y
        else:
            Y[:] = alpha * (A @ X) + beta * Y

    # -----------------------------------------------------------------------
    # Misc Routines
    # -----------------------------------------------------------------------
    @staticmethod
    def max(val, arr):
        mr = np.maximum(arr._arr.real, val)
        mi = np.maximum(arr._arr.imag, val)
        arr._arr[:] = mr + 1j * mi

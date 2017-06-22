import logging
import os, sys, time
from ctypes import *

import numpy as np
from numpy.ctypeslib import ndpointer

from slo.util import c_complex
from slo.backends.backend import Backend

log = logging.getLogger(__name__)

# Find MKL library
dll_ext = '.dylib' if sys.platform == 'darwin' else '.so'
libmkl_rt = cdll.LoadLibrary('libmkl_rt' + dll_ext)

class sparse_matrix_t(c_void_p):
    pass
    
class sparse_status_t(c_int):
    SUCCESS = 0
    NOT_INITIALIZED = 1
    ALLOC_FAILED = 2
    INVALID_VALUE = 3
    EXECUTION_FAILED = 4
    INTERNAL_ERROR = 5
    NOT_SUPPORTED = 6

class sparse_index_base_t(c_int):
    BASE_ZERO = 0
    BASE_ONE  = 1

class sparse_operation_t(c_int):
    NON_TRANSPOSE = 10
    TRANSPOSE = 11
    CONJUGATE_TRANSPOSE = 12

class sparse_matrix_type_t(c_int):
    GENERAL = 20
    SYMMETRIC = 21
    HERMITIAN = 22
    TRIANGULAR = 23
    DIAGONAL = 24
    BLOCK_TRIANGULAR = 25
    BLOCK_DIAGONAL = 26

class sparse_fill_mode_t(c_int):
    NONE = 0
    LOWER = 40
    UPPER = 41

class sparse_diag_type_t(c_int):
    NON_UNIT = 50
    UNIT = 51

class matrix_descr(Structure):
    _fields_ = [
        ("type", sparse_matrix_type_t),
        ("mode", sparse_fill_mode_t),
        ("diag", sparse_diag_type_t),
    ]

    def __init__(self):
        self.type = sparse_matrix_type_t.GENERAL

class sparse_layout_t(c_int):
    ROW_MAJOR = 60
    COLUMN_MAJOR = 61


class MklBackend(Backend):

    def __init__(self, device_id=0):
        super(MklBackend, self).__init__()
        log.info('mkl_get_version() reports: %s', self.get_version())
        self._fft_descs = dict()

    def wrap(fn):
        libfn = getattr(libmkl_rt, fn.__name__)
        libfn.argtypes = [fn.__annotations__[arg] for arg in fn.__code__.co_varnames]
        libfn.restype  =  fn.__annotations__['return']
        def wrapped(self, *args, **kwargs):
            res = libfn(*args, **kwargs)
            if isinstance(res, c_long) and res.value != 0:
                s = self.DftiErrorMessage( res )
                raise RuntimeError( s.decode('ascii') )
            return res
        return wrapped

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
            ld = self._leading_dims
            d = self._arr.reshape(self.shape, order='F')[slc]
            return self._backend.dndarray( self._backend, d.shape, d.dtype,
                ld=ld, own=False, data=d )

        def to_host(self):
            return self._arr

        @staticmethod
        def from_param(obj):
            if not isinstance(obj, MklBackend.dndarray):
                raise ArgumentError('{} is not a dndarray'.format( type(obj) ))
            return obj._arr.ctypes.get_as_parameter()

        def as_nparray(self):
            return self._arr
    
    @wrap
    def mkl_get_max_threads() -> c_int:
        pass

    def get_max_threads(self):
        return self.mkl_get_max_threads()

    @wrap
    def mkl_get_version_string( buf : c_char_p, length : c_int ) -> c_void_p:
        pass

    def get_version(self):
        buf = create_string_buffer(128)
        self.mkl_get_version_string(buf, len(buf))
        return buf.value.decode('ascii')

    # -----------------------------------------------------------------------
    # BLAS Routines
    # -----------------------------------------------------------------------
    def axpy(self, y, alpha, x):
        """ y += alpha * x """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        alpha = np.array(alpha, dtype=np.complex64)
        self.cblas_caxpy( y.size, alpha, x._arr, 1, y._arr, 1 )

    def dot(self, x, y):
        """ returns x^T * y """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        dotc = np.array(0, dtype=np.complex64)
        self.cblas_cdotc_sub( x.size, x._arr, 1, y._arr, 1, dotc )
        return dotc.real

    def norm2(self, x):
        """ returns ||x||_2"""
        assert isinstance(x, self.dndarray)
        res = self.cblas_scnrm2( x.size, x._arr, 1 )
        return res**2

    def scale(self, x, alpha):
        """ x *= alpha """
        assert isinstance(x, self.dndarray)
        a = np.array(alpha, dtype=np.complex64)
        self.cblas_cscal( x.size, a, x._arr, 1 )

    def cgemm(self, y, M, x, alpha, beta, forward):
        layout = MklBackend.CBlasLayout.ColMajor
        if forward:
            transa = MklBackend.CBlasTranspose.NoTrans
        else:
            transa = MklBackend.CBlasTranspose.ConjTrans
        transb = MklBackend.CBlasTranspose.NoTrans
        (m, n), k = y.shape, x.shape[0]
        alpha = np.array(alpha, dtype=np.complex64)
        beta  = np.array( beta, dtype=np.complex64)
        lda = M.shape[0]
        ldb = x.shape[0]
        ldc = y.shape[0]
        self.cblas_cgemm(
            layout, transa, transb, m, n, k,
            alpha, M, lda, x, ldb, beta, y, ldc
        )

    class CBlasLayout(c_uint):
        RowMajor = 101
        ColMajor = 102

    class CBlasTranspose(c_uint):
        NoTrans   = 111
        Trans     = 112
        ConjTrans = 113

    @wrap
    def cblas_cgemm(
        layout : CBlasLayout,
        transa : CBlasTranspose,
        transb : CBlasTranspose,
        m     : c_int,
        n     : c_int,
        k     : c_int,
        alpha : ndpointer(dtype=np.complex64, ndim=0),
        a     : dndarray,
        lda   : c_int,
        b     : dndarray,
        ldb   : c_int,
        beta  : ndpointer(dtype=np.complex64, ndim=0),
        c     : dndarray,
        ldc   : c_int,
    ) -> c_void_p:
        pass

    @wrap
    def cblas_caxpy(
        n : c_int,
        a : ndpointer(dtype=np.complex64, ndim=0),
        x : ndpointer(dtype=np.complex64),
        incx : c_int,
        y : ndpointer(dtype=np.complex64),
        incy : c_int,
    ) -> c_void_p:
        pass

    @wrap
    def cblas_cdotc_sub(
        n    : c_int,
        x    : ndpointer(dtype=np.complex64),
        incx : c_int,
        y    : ndpointer(dtype=np.complex64),
        incy : c_int,
        dotc : ndpointer(dtype=np.complex64, ndim=0),
    ) -> c_void_p:
        pass

    @wrap
    def cblas_scnrm2(
        n : c_int,
        x    : ndpointer(dtype=np.complex64),
        incx : c_int,
    ) -> c_float:
        pass

    @wrap
    def cblas_cscal(
        n    : c_int,
        a    : ndpointer(dtype=np.complex64, ndim=0),
        x    : ndpointer(dtype=np.complex64),
        incx : c_int,
    ) -> c_void_p:
        pass



    # -----------------------------------------------------------------------
    # FFT Routines
    # -----------------------------------------------------------------------
    class DFTI_DESCRIPTOR_HANDLE(c_void_p):
        pass

    class status_t(c_long):
        pass

    # read these values out of `gcc -E mkl.h | grep DFTI_WHATEVER`
    # they could change so we're living on the edge here
    DFTI_SINGLE = 35
    DFTI_COMPLEX = 32
    DFTI_INPUT_DISTANCE = 14
    DFTI_OUTPUT_DISTANCE = 15
    DFTI_NUMBER_OF_TRANSFORMS = 7
    DFTI_PLACEMENT = 11
    DFTI_INPLACE = 43
    DFTI_NOT_INPLACE = 44

    def _get_or_create_fft_desc(self, x, axes):
        assert axes == (0,1,2)
        key = (x.shape, x.dtype, axes)
        if key not in self._fft_descs:
            N = x.shape[:3][::-1]
            batch = int(x.size / np.prod(N))
            desc = self.DFTI_DESCRIPTOR_HANDLE()
            lengths = (c_long*3)(*N)
            self.DftiCreateDescriptor( byref(desc),
                self.DFTI_SINGLE, self.DFTI_COMPLEX, 3, lengths)
            self.DftiSetValue( desc, self.DFTI_NUMBER_OF_TRANSFORMS, batch )
            self.DftiSetValue( desc, self.DFTI_PLACEMENT, self.DFTI_NOT_INPLACE )
            self.DftiSetValue( desc, self.DFTI_INPUT_DISTANCE, np.prod(N) )
            self.DftiSetValue( desc, self.DFTI_OUTPUT_DISTANCE, np.prod(N) )
            self.DftiCommitDescriptor( desc )
            self._fft_descs[key] = desc
        return self._fft_descs[key]

    def fftn(self, y, x):
        desc = self._get_or_create_fft_desc( x, axes=(0,1,2) )
        self.DftiComputeForward( desc, x, y )

    def ifftn(self, y, x):
        desc = self._get_or_create_fft_desc( x, axes=(0,1,2) )
        self.DftiComputeBackward( desc, x, y )

    def __del__(self):
        for desc in self._fft_descs.values():
            self.DftiFreeDescriptor( byref(desc) )

    @wrap
    def DftiErrorMessage(
        status : status_t,
    ) -> c_char_p:
        pass

    @wrap
    def DftiComputeForward(
        desc_handle : DFTI_DESCRIPTOR_HANDLE,
        x_in        : dndarray,
        y_out       : dndarray,
    ) -> status_t:
        pass

    @wrap
    def DftiComputeBackward(
        desc_handle : DFTI_DESCRIPTOR_HANDLE,
        x_in        : dndarray,
        y_out       : dndarray,
    ) -> status_t:
        pass

    @wrap
    def DftiCommitDescriptor(
        desc_handle : DFTI_DESCRIPTOR_HANDLE,
    ) -> status_t:
        pass

    @wrap
    def DftiSetValue(
        desc_handle : DFTI_DESCRIPTOR_HANDLE,
        param       : c_uint,
        value       : c_uint,
    ) -> status_t:
        pass

    @wrap
    def DftiCreateDescriptor(
        desc_handle : POINTER(DFTI_DESCRIPTOR_HANDLE),
        precision   : c_uint,
        domain      : c_uint,
        dimension   : c_long,
        length      : c_long*3,
    ) -> status_t:
        pass

    @wrap
    def DftiFreeDescriptor(
        desc_handle : POINTER(DFTI_DESCRIPTOR_HANDLE),
    ) -> status_t:
        pass

    # -----------------------------------------------------------------------
    # CSRMM Routines
    # -----------------------------------------------------------------------
    class csr_matrix(Backend.csr_matrix):
        def __init__(self, backend, A, name='mat'):
            self._name = name
            self._backend = backend
            self._mat_h = A = A.tocsr() # store this, since MKL doesn't make a copy
            self._mat_d = sparse_matrix_t()
            status = backend.mkl_sparse_c_create_csr( byref(self._mat_d),
                sparse_index_base_t.BASE_ZERO, A.shape[0], A.shape[1],
                A.indptr[:-1], A.indptr[1:], A.indices, A.data,
            )
            assert status.value == sparse_status_t.SUCCESS, status.value

            # guess how much space MKL will use:
            self._nbytes = A.indices.nbytes + A.data.nbytes + A.indptr.nbytes
            self._nnz = A.nnz
            self._hints = set()

        def __del__(self):
            self._backend.mkl_sparse_destroy( self._mat_d )

        def forward(self, y, x, alpha=1, beta=0):
            """ y[:] = A * x """
            self._eval(y, x, alpha, beta, op=sparse_operation_t.NON_TRANSPOSE)
                
        def adjoint(self, y, x, alpha=1, beta=0):
            """ y[:] = A.H * x """
            self._eval(y, x, alpha, beta, op=sparse_operation_t.CONJUGATE_TRANSPOSE)

        def _eval(self, y, x, alpha, beta, op):
            columns = x.shape[1]
            descr = matrix_descr()
            self.optimize(op, descr, columns)
            ldx = x._leading_dims[0]
            ldy = y._leading_dims[0]
            status = self._backend.mkl_sparse_c_mm(op,
                c_complex(alpha),
                self._mat_d, descr,
                sparse_layout_t.COLUMN_MAJOR,
                x._arr, columns, ldx,
                c_complex(beta),
                y._arr, ldy
            )
            assert status.value == sparse_status_t.SUCCESS, status.value

        def optimize(self, op, descr, columns):
            key = (op, columns)
            if key not in self._hints:
                ncalls = 200
                trans = (op == sparse_operation_t.CONJUGATE_TRANSPOSE)
                log.debug('updating hints for %s: trans=%s, columns=%d, ncalls=%d',
                          self._name, trans, columns, ncalls)
                self._hints.add(key)
                # set hint
                #status = self._backend.mkl_sparse_set_mm_hint( self._mat_d, op,
                #    descr, sparse_layout_t.COLUMN_MAJOR, columns, ncalls)
                #assert status.value == sparse_status_t.SUCCESS, status.value
                # optimize
                status = self._backend.mkl_sparse_optimize( self._mat_d )
                assert status.value == sparse_status_t.SUCCESS, status.value
                
        @property
        def nbytes(self):
            return self._nbytes

        @property
        def nnz(self):
            return self._nnz

    @wrap
    def mkl_sparse_c_create_csr(
        A      : POINTER(sparse_matrix_t),
        indexing : sparse_index_base_t,
        rows   : c_int,
        cols   : c_int,
        rows_s : ndpointer(dtype=np.int32,     ndim=1),
        rows_e : ndpointer(dtype=np.int32,     ndim=1),
        colidx : ndpointer(dtype=np.int32,     ndim=1),
        values : ndpointer(dtype=np.complex64, ndim=1),
    ) -> sparse_status_t:
         pass

    @wrap
    def mkl_sparse_destroy(
        A : sparse_matrix_t,
    ) -> sparse_status_t:
        pass

    @wrap
    def mkl_sparse_c_mm(
        oper  : sparse_operation_t,
        alpha : c_complex,
        A     : sparse_matrix_t,
        descr : matrix_descr,
        layout: sparse_layout_t,
        x     : ndpointer(dtype=np.complex64),
        columns: c_int,
        ldx   : c_int,
        beta  : c_complex,
        y     : ndpointer(dtype=np.complex64),
        ldy   : c_int,
    ) -> sparse_status_t:
        pass

    @wrap
    def mkl_sparse_set_mm_hint(
        A     : sparse_matrix_t,
        oper  : sparse_operation_t,
        descr : matrix_descr,
        layout: sparse_layout_t,
        dense_matrix_size : c_int,
        expected_calls : c_int,
    ) -> sparse_status_t:
        pass

    @wrap
    def mkl_sparse_optimize(
        A : sparse_matrix_t,
    ) -> sparse_status_t:
        pass

import gc
import sys
import time
import logging
from ctypes import *

import numpy as np
from numpy.ctypeslib import ndpointer

from .backend import Backend

log = logging.getLogger(__name__)

cudart   = cdll.LoadLibrary("libcudart.so")
cusparse = cdll.LoadLibrary("libcusparse.so")
cufft    = cdll.LoadLibrary("libcufft.so")
cublas   = cdll.LoadLibrary("libcublas.so")
nvtx     = cdll.LoadLibrary("libnvToolsExt.so")

class c_complex(c_float * 2):
    def __init__(self, a):
        super().__init__()
        self[0] = a.real
        self[1] = a.imag

class CudaBackend(Backend):

    def __init__(self, device_id=0):
        super(CudaBackend, self).__init__()

        self._cublas_handle = self.cublasHandle_t(self)
        self._cusparse_handle = self.cusparseHandle_t(self)
        self._mat_descr = self.cusparseMatDescr_t(self)

        self.cudaSetDevice( device_id )
        cu_device = c_int()
        self.cudaGetDevice( byref(cu_device) )
        log.info("using CUDA device #%d", cu_device.value)

        self._plans = dict()

    class cudaError_t(c_long):
        def _check(self, backend, fn):
            if self.value != 0:
                name = backend.cudaGetErrorName( self ).decode('ascii')
                desc = backend.cudaGetErrorString( self ).decode('ascii')
                log.critical("%s returned exit code %d: %s (%s)",
                    fn.__name__, self.value, name, desc)
                raise RuntimeError

    def wrap(lib):
        def wrapper(fn):
            libfn = getattr(lib, fn.__name__)
            libfn.argtypes = [fn.__annotations__[arg] for arg in fn.__code__.co_varnames]
            libfn.restype  =  fn.__annotations__['return']
            def wrapped(backend, *args, **kwargs):
                res = libfn(*args, **kwargs)
                if hasattr(res, '_check'):
                    res._check(backend, fn)
                return res
            return wrapped
        return wrapper

    @wrap(cudart)
    def cudaSetDevice(device : c_int) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaGetDevice(device : POINTER(c_int)) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaGetErrorName( err : cudaError_t ) -> c_char_p:
        pass

    @wrap(cudart)
    def cudaGetErrorString( err : cudaError_t ) -> c_char_p:
        pass

    @wrap(cudart)
    def cudaGetDeviceCount(count: POINTER(c_int)) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaDeviceSynchronize() -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaMalloc(devPtr : POINTER(c_ulong), size : c_size_t) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaFree(devPtr : c_ulong) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaMemcpy( dst:c_ulong, src:c_ulong, count:c_size_t, kind:c_int) -> cudaError_t:
        pass

    @wrap(cudart)
    def cudaMemset(
        devPtr : c_ulong,
        size   : c_int,
        count  : c_size_t
    ) -> cudaError_t:
        pass

    cudaMemcpy.HostToHost     =   0
    cudaMemcpy.HostToDevice   =   1
    cudaMemcpy.DeviceToHost   =   2
    cudaMemcpy.DeviceToDevice =   3
    cudaMemcpy.Default        =   4

    def barrier(self):
        self.cudaDeviceSynchronize()

    # -----------------------------------------------------------------------
    # Arrays
    # -----------------------------------------------------------------------
    class dndarray(Backend.dndarray):
        def _copy_from(self, arr):
            src, dst = arr.ctypes.data, self._arr
            size, kind = arr.nbytes, CudaBackend.cudaMemcpy.HostToDevice
            self._backend.cudaMemcpy(dst, src, size, kind)

        def _copy_to(self, arr):
            assert arr.flags['F_CONTIGUOUS']
            src, dst = self._arr, arr.ctypes.data
            size, kind = arr.nbytes, CudaBackend.cudaMemcpy.DeviceToHost
            self._backend.cudaMemcpy(dst, src, size, kind)

        def _copy(self, d_arr):
            src, dst = d_arr._arr, self._arr
            size, kind = self.nbytes, CudaBackend.cudaMemcpy.DeviceToDevice
            self._backend.cudaMemcpy(dst, src, size, kind)

        def _malloc(self, shape, dtype):
            align = 256
            self._fullarr = c_ulong()
            self._backend.cudaMalloc( byref(self._fullarr), self.nbytes + align)
            _arr = self._fullarr.value
            while _arr % align != 0:
                _arr += 1
            return c_ulong(_arr)

        def _free(self):
            self._backend.cudaFree( self._fullarr )

        def _zero(self):
            self._backend.cudaMemset( self._arr, 0, self.nbytes )

        def __getitem__(self, slc):
            if isinstance(slc, slice):
                slc = [slc]
            start, shape = [], []
            for s, n in zip(slc, self.shape):
                if isinstance(s, int):
                    s = slice(s, s+1)
                b, e = s.start or 0, s.stop or n
                if b < 0: b = n+b # remove neg begining indices
                if e < 0: e = n+e # remove neg ending indices
                if e < b: e = b   # disallow negative sizes
                if e > n: e = n   # fix over-slices
                start.append(b)
                shape.append(e-b)
            idx = np.ravel_multi_index(start, self.shape, order='F')
            ptr = self._arr.value + idx * np.dtype(self.dtype).itemsize
            ptr = c_ulong(ptr)
            ld = self._leading_dims
            return self._backend.dndarray(self._backend, tuple(shape),
                self.dtype, ld=ld, own=False, data=ptr)

        @staticmethod
        def from_param(obj):
            if not isinstance(obj, CudaBackend.dndarray):
                raise ArgumentError('{} is not a dndarray'.format( type(obj) ))
            return obj._arr


    # -----------------------------------------------------------------------
    # BLAS Routines
    # -----------------------------------------------------------------------

    class cublasHandle_t(c_void_p):
        def __init__(self, backend):
            super().__init__()
            self._backend = backend
            self._backend.cublasCreate_v2( byref(self) )

        def __del__(self):
            self._backend.cublasDestroy_v2( self )


    class cublasStatus_t(c_int):
        def _check(self, backend, fn):
            if self.value != 0:
                log.critical("%s returned exit code %d", fn.__name__, self.value)
                raise RuntimeError

    @wrap(cublas)
    def cublasCreate_v2( handle : POINTER(cublasHandle_t) ) -> cublasStatus_t:
        pass

    @wrap(cublas)
    def cublasDestroy_v2( handle : cublasHandle_t ) -> cublasStatus_t:
        pass

    def axpy(self, y, alpha, x):
        """ y += alpha * x """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        alpha = np.array(alpha, dtype=np.complex64)
        self.cublasCaxpy_v2( self._cublas_handle,
            y.size, alpha, x._arr, 1, y._arr, 1 )

    @wrap(cublas)
    def cublasCaxpy_v2(
        handle : cublasHandle_t,
        n      : c_int,
        alpha  : ndpointer(dtype=np.complex64, ndim=0),
        x      : c_ulong, incx : c_int,
        y      : c_ulong, incy : c_int,
    ) -> cublasStatus_t:
        pass

    def dot(self, x, y):
        """ returns x^T * y """
        assert isinstance(x, self.dndarray)
        assert isinstance(y, self.dndarray)
        dotc = np.array(0, dtype=np.complex64)
        self.cublasCdotc_v2( self._cublas_handle, x.size,
            x._arr, 1, y._arr, 1, dotc)
        return dotc.real

    @wrap(cublas)
    def cublasCdotc_v2(
        handle : cublasHandle_t,
        n      : c_int,
        x      : c_ulong,
        incx   : c_int,
        y      : c_ulong,
        incy   : c_int,
        result : ndpointer(dtype=np.complex64, ndim=0),
    ) -> cublasStatus_t:
        pass

    def norm2(self, x):
        """ returns ||x||_2"""
        assert isinstance(x, self.dndarray)
        result = np.array(0, dtype=np.complex64)
        self.cublasScnrm2_v2( self._cublas_handle, x.size, x._arr, 1, result )
        return result**2

    @wrap(cublas)
    def cublasScnrm2_v2(
        handle : cublasHandle_t,
        n      : c_int,
        x      : c_ulong,
        incx   : c_int,
        result : ndpointer(dtype=np.complex64, ndim=0),
    ) -> cublasStatus_t:
        pass

    def scale(self, x, alpha):
        """ x *= alpha """
        assert isinstance(x, self.dndarray)
        alpha = np.array(alpha, dtype=np.complex64)
        self.cublasCscal_v2( self._cublas_handle, x.size, alpha, x._arr, 1 )

    @wrap(cublas)
    def cublasCscal_v2(
        handle : cublasHandle_t,
        n      : c_int,
        alpha  : ndpointer(dtype=np.complex64, ndim=0),
        x      : c_ulong,
        incx   : c_int,
    ) -> cublasStatus_t:
        pass

    def cgemm(self, y, M, x, alpha, beta, forward):
        """ y = alpha * M * X + beta * Y """
        assert isinstance(x, self.dndarray)
        alpha = np.array(alpha, dtype=np.complex64)
        beta  = np.array( beta, dtype=np.complex64)
        (m, n), k = y.shape, x.shape[0]
        lda = M.shape[0]
        ldb = x.shape[0]
        ldc = y.shape[0]
        if forward:
            transa = CudaBackend.cublasOperator_t.CUBLAS_OP_N
        else:
            transa = CudaBackend.cublasOperator_t.CUBLAS_OP_C
        transb = CudaBackend.cublasOperator_t.CUBLAS_OP_N
        self.cublasCgemm_v2( self._cublas_handle, transa, transb,
            m, n, k, alpha, M, lda, x, ldb, beta, y, ldc )

    class cublasOperator_t(c_uint):
        CUBLAS_OP_N = 0
        CUBLAS_OP_T = 1
        CUBLAS_OP_C = 2

    @wrap(cublas)
    def cublasCgemm_v2(
        handle : cublasHandle_t,
        transa : cublasOperator_t,
        transb : cublasOperator_t,
        m      : c_int,
        n      : c_int,
        k      : c_int,
        alpha  : ndpointer(dtype=np.complex64, ndim=0),
        M      : dndarray,
        lda    : c_int,
        x      : dndarray,
        ldb    : c_int,
        beta   : ndpointer(dtype=np.complex64, ndim=0),
        y      : dndarray,
        ldc    : c_int,
    ) -> cublasStatus_t:
        pass

    # -----------------------------------------------------------------------
    # FFT Routines
    # -----------------------------------------------------------------------

    class cufftHandle_t(c_int):
        def __init__(self, backend):
            super(CudaBackend.cufftHandle_t, self).__init__()
            self._backend = backend
            self._backend.cufftCreate( byref(self) )

        def __del__(self):
            self._backend.cufftDestroy(self)

            
    class cufftType_t(c_int):
        pass

    class cufftResult_t(c_int):
        def _check(self, backend, fn):
            if self.value != 0:
                log.critical("cufft function %s returned error code %d",
                    fn.__name__, self.value)
                raise RuntimeError

    CUFFT_C2C = 0x29
    CUFFT_FORWARD = -1
    CUFFT_INVERSE =  1

    @wrap(cufft)
    def cufftSetAutoAllocation(
        plan : cufftHandle_t,
        auto : c_int,
    ) -> cufftResult_t:
        pass

    @wrap(cufft)
    def cufftSetWorkArea(
        plan : cufftHandle_t,
        area : dndarray,
    ) -> cufftResult_t:
        pass

    @wrap(cufft)
    def cufftMakePlanMany(
        plan    : cufftHandle_t,
        rank    : c_int,
        n       : POINTER(c_int),
        inembed : POINTER(c_int), istride: c_int, idist: c_int,
        onembed : POINTER(c_int), ostride: c_int, odist: c_int,
        typ     : cufftType_t,
        batch   : c_int,
        workSize: POINTER(c_size_t),
    ) -> cufftResult_t:
        pass

    @wrap(cufft)
    def cufftExecC2C(
        plan    : cufftHandle_t,
        idata   : dndarray,
        odata   : dndarray,
        direction : c_int,
    ) -> cufftResult_t:
        pass

    @wrap(cufft)
    def cufftCreate(
        plan : POINTER(cufftHandle_t),
    ) -> cufftResult_t:
        pass

    @wrap(cufft)
    def cufftDestroy(
        plan : cufftHandle_t,
    ) -> cufftResult_t:
        pass

    def _get_or_create_plan(self, x_shape):
        if x_shape not in self._plans:
            N = x_shape[:-1][::-1]
            batch = x_shape[-1]
            dims = (c_int*len(N))(*N)
            plan = CudaBackend.cufftHandle_t(self)
            ws = c_size_t()
            self.cufftSetAutoAllocation(plan, 0)
            self.cufftMakePlanMany(plan, len(dims), dims,
                None, 0, 0, None, 0, 0, CudaBackend.CUFFT_C2C,
                batch, byref(ws))
            self._plans[x_shape] = (plan, ws.value)
        return self._plans[x_shape]

    def _fft_workspace_size(self, x_shape):
        plan, workSize = self._get_or_create_plan(x_shape)
        return workSize

    def fftn(self, y, x):
        plan, workSize = self._get_or_create_plan(x.shape)
        with self.scratch(nbytes=workSize) as tmp:
            self.cufftSetWorkArea(plan, tmp)
            self.cufftExecC2C(plan, x, y, CudaBackend.CUFFT_FORWARD)

    def ifftn(self, y, x):
        plan, workSize = self._get_or_create_plan(x.shape)
        with self.scratch(nbytes=workSize) as tmp:
            self.cufftSetWorkArea(plan, tmp)
            self.cufftExecC2C(plan, x, y, CudaBackend.CUFFT_INVERSE)

    # -----------------------------------------------------------------------
    # Cusparse
    # -----------------------------------------------------------------------
    class cusparseHandle_t(c_void_p):
        def __init__(self, backend):
            super().__init__()
            self._backend = backend
            self._backend.cusparseCreate( byref(self) )

        def __del__(self):
            self._backend.cusparseDestroy(self)


    class cusparseOperation_t(c_int): pass

    class cusparseMatDescr_t(c_void_p):
        def __init__(self, backend):
            super().__init__()
            self._backend = backend
            self._backend.cusparseCreateMatDescr( byref(self) )

        def __del__(self):
            self._backend.cusparseDestroyMatDescr(self)


    class cusparseStatus_t(c_int):
        def _check(self, backend, fn):
            if self.value != 0:
                log.critical("cusparse function %s returned error code %d",
                    fn.__name__, self.value)
                raise RuntimeError


    CUSPARSE_OPERATION_NON_TRANSPOSE       = 0
    CUSPARSE_OPERATION_TRANSPOSE           = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

    @wrap(cusparse)
    def cusparseCreate(
        handle : POINTER(cusparseHandle_t)
    ) -> cusparseStatus_t:
        pass

    @wrap(cusparse)
    def cusparseDestroy(
        handle : cusparseHandle_t
    ) -> cusparseStatus_t:
        pass

    @wrap(cusparse)
    def cusparseCreateMatDescr(
        descr : POINTER(cusparseMatDescr_t)
    ) -> cusparseStatus_t:
        pass

    @wrap(cusparse)
    def cusparseDestroyMatDescr(
        descr : cusparseMatDescr_t
    ) -> cusparseStatus_t:
        pass

    @wrap(cusparse)
    def cusparseCcsrmm(
        handle     : cusparseHandle_t,
        transA     : cusparseOperation_t,
        m          : c_int,
        n          : c_int,
        k          : c_int,
        nnz        : c_int,
        alpha      : c_complex,
        descrA     : cusparseMatDescr_t,
        csrValA    : dndarray,
        csrRowPtrA : dndarray,
        csrColIndA : dndarray,
        B          : dndarray,
        ldb        : c_int,
        beta       : c_complex,
        C          : dndarray,
        ldc        : c_int,
    ) -> cusparseStatus_t:
        pass

    def ccsrmm(self, y, A_shape, A_indx, A_ptr, A_vals, x, alpha, beta, adjoint=False, exwrite=False):
        m, k = A_shape
        n = x.shape[1]
        ldx = x._leading_dims[0]
        ldy = y._leading_dims[0]
        if adjoint:
            trans = self.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
        else:
            trans = self.CUSPARSE_OPERATION_NON_TRANSPOSE
        alpha = c_complex(alpha)
        beta  = c_complex(beta)
        self.cusparseCcsrmm( self._cusparse_handle, trans, m, n, k,
            A_vals.size, (alpha), self._mat_descr,
            A_vals, A_ptr, A_indx, x, ldx, (beta), y, ldy
        )

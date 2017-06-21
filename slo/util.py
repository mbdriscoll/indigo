import time
import logging
import numpy as np
import scipy.sparse as spp
from ctypes import cdll, c_int

log = logging.getLogger(__name__)

def rand64c(*shape, order='F'):
    """
    Constructs a `np.ndarray` of the requested shape and
    populates it with random np.complex64 values.
    """
    r = np.random.rand(*shape).astype(np.float32)
    i = np.random.rand(*shape).astype(np.float32)
    arr = (r + 1j*i).astype(np.complex64)
    if order == 'F':
        arr = np.asfortranarray(arr)
    return arr


def randM(M, N, density):
    """
    Constructs a `scipy.sparse.spmatrix'  of the requested shape and
    density and populates it with random np.complex64 values.
    """
    A_r = spp.random(M, N, density=density, format='csr', dtype=np.float32).astype(np.complex64)
    A_i = spp.random(M, N, density=density, format='csr', dtype=np.float32)
    A = A_r + 1j * A_i
    return A


class profile(object):
    extra = dict()

    def __init__(self, event, **kwargs):
        self._event = event
        self._kwargs = kwargs

    def __enter__(self):
        if log.level > logging.DEBUG:
            return

        self._backend.barrier()
        self._start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        if log.level > logging.DEBUG:
            return

        self._backend.barrier()
        data = dict(
            duration = time.time() - self._start,
            event    = self._event,
        )
        data.update(self.extra)
        data.update(self._kwargs)

        if 'nflops' in data:
            data['gflop_rate'] = data['nflops'] / data['duration'] * 1e-9
            del data['nflops']

        if 'nbytes' in data:
            data['membw_rate'] = data['nbytes'] / data['duration'] * 1e-9
            del data['nbytes']

        kvs = sorted(data.items(), key=lambda kv: kv[0])

        def fmt(k, v):
            if isinstance(v, float):
                return k, "%2.5g" % v
            else:
                return k, repr(v)

        msg = "PROFILE(%s)" % ", ".join("%s=%s" % fmt(*kv) for kv in kvs)
        log.debug(msg)


class Timer(object):
    def __init__(self):
        self._times = []

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self._times.append( time.time() - self._start )

    @property
    def median(self):
        return np.median( self._times )

    @property
    def mean(self):
        return np.mean( self._times )

    @property
    def max(self):
        return np.amax( self._times )

    @property
    def min(self):
        return np.amin( self._times )

class Memusage(object):
    """
    Estimates maximum memory needed for evaluating a tree
    during CG.
    """
    def __init__(self):
        self.dataItems = {}

    def estimate_nbytes(self, root, x_shape, x_dtype):
        self.dataItems = {}
        self.dataCount = 0
        self.visit(root)
        data_nbytes = sum([v[1] for k, v in self.dataItems.items()])
        intermediate_nbytes = self.intermediate_nbytes(root, x_shape, x_dtype)
        tmp_nbytes = np.prod(x_shape) * x_dtype.itemsize
        return tuple(nb / 1024 / 1024 for nb in [data_nbytes, intermediate_nbytes, tmp_nbytes*4])

    def visit(self, node):
        method_name = "visit_%s" % type(node).__name__
        visitor_method = getattr(self, method_name, None)
        if visitor_method:
            visitor_method(node)
        else:
            self.generic_visit(node)

    def generic_visit(self, node):
        if hasattr(node, '_children'):
            for c in node._children:
                self.visit(c)

    def visit_DenseMatrix(self, node):
        self.dataItems[id(node)] = (node._name, node._matrix.nbytes)

    def visit_SpMatrix(self, node):
        if node._matrix is not None:
            self.estimate_spm_nbytes(node._matrix, node=node, name=node._name)

    def estimate_spm_nbytes(self, x, node=None, name=''):
        # Currently, all matrices are actually CSR.
        x = x.tocsr()
        if isinstance(x, spp.csr_matrix) or isinstance(x, spp.csc_matrix):
            self.dataItems[id(node)] = (node._name, x.data.nbytes + x.indptr.nbytes + x.indices.nbytes)
        elif isinstance(x, spp.coo_matrix):
            self.dataItems[id(node)] = (node._name, x.col.nbytes + x.row.nbytes + x.data.nbytes)
        elif isinstance(x, spp.dia_matrix):
            self.dataItems[id(node)] = (node._name, x.data.nbytes + x.offsets.nbytes)
        else:
            log.warn('Matrix type %s unsupported by Memusage' % (type(x)))

    def intermediate_nbytes(self, node, x_shape, x_dtype):
        from slo.operators import Product,UnscaledFFT,KronI
        nbytes = 0
        x_shape = (x_shape[0], min(x_shape[1], node._batch) if node._batch is not None else x_shape[1])
        if isinstance(node, UnscaledFFT):
            nbytes += node._mem_usage(x_shape)
        elif isinstance(node, Product):
            intermediate_shape = node._intermediate_shape(x_shape)
            nbytes += node._mem_usage(x_shape, x_dtype)
            max_child = max(self.intermediate_nbytes(node._children[0], intermediate_shape, x_dtype), \
                            self.intermediate_nbytes(node._children[1], x_shape, x_dtype))
            return nbytes + max_child
        elif isinstance(node, KronI):
            cb = node._c * x_shape[1]
            x_shape = (np.prod(x_shape) // cb, cb)
        else:
            pass
        if hasattr(node, '_children'):
            max_child = max([self.intermediate_nbytes(c, x_shape, x_dtype) for c in node._children])
            nbytes += max_child
        return nbytes


import logging
import numpy as np
import scipy.sparse as spp

from slo.transforms import Visitor

log = logging.getLogger(__name__)

class Memusage(Visitor):
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

    def visit_DenseMatrix(self, node):
        self.dataItems[id(node)] = (node._name, node._matrix.nbytes)

    def visit_SpMatrix(self, node):
        if node._matrix is not None:
            self.estimate_spm_nbytes(node._matrix, node=node, name=node._name)

    def estimate_spm_nbytes(self, x, node=None, name=''):
        data = x.nnz * x.dtype.itemsize
        rowptr = x.shape[0] * np.dtype('int32').itemsize
        colind = x.nnz * np.dtype('int32').itemsize
        self.dataItems[id(node)] = (node._name, data + rowptr + colind)

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


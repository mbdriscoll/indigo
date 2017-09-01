import logging
import numpy as np
import scipy.sparse as spp
from contextlib import contextmanager

from indigo.transforms import Visitor

log = logging.getLogger(__name__)

class Memusage(Visitor):
    """
    Estimates maximum memory needed for evaluating a tree
    during CG.
    """
    def measure(self, node, ncols=1):
        self._seen = set()
        self._current_mem  = [0]
        self._current_cols = [ncols]
        self._max_mem = 0
        self.visit(node)
        return self._max_mem

    @contextmanager
    def _push_mem(self, nbytes):
        self._current_mem.append(nbytes)
        yield
        self._current_mem.pop()

    @contextmanager
    def _push_cols(self, ncols):
        self._current_cols.append(ncols)
        yield
        self._current_cols.pop()
    
    def generic_visit(self, node):
        self._max_mem = max( self._max_mem, sum(self._current_mem) )
        super(Memusage, self).generic_visit(node)

    def visit_Product(self, node):
        ncols = np.prod(self._current_cols)
        nbytes = node._mem_usage(ncols)
        with self._push_mem(nbytes):
            self.generic_visit(node)

    def visit_KronI(self, node):
        with self._push_cols( node._c ):
            self.generic_visit(node)

    def visit_UnscaledFFT(self, node):
        ncols = np.prod(self._current_cols)
        nbytes = node._mem_usage(ncols)
        with self._push_mem(nbytes):
            self.generic_visit(node)

    def visit_DenseMatrix(self, node):
        if id(node) in self._seen:
            return
        self._seen.add(id(node))
        nbytes = node._matrix.nbytes
        self._current_mem[0] += nbytes

    def visit_SpMatrix(self, node):
        if id(node) in self._seen:
            return
        self._seen.add(id(node))
        nrows = node.shape[0]
        rowptr = (nrows+1) * np.dtype('int32').itemsize
        colind =  node.nnz * np.dtype('int32').itemsize
        data   =  node.nnz * node.dtype.itemsize
        nbytes = data + rowptr + colind
        self._current_mem[0] += nbytes

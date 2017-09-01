import sys
import time
import logging
import io, copy
import itertools
import numpy as np
import scipy.sparse as spp
from ctypes import c_ulong

from indigo.util import profile

log = logging.getLogger(__name__)

class Operator(object):
    def __init__(self, backend, name='', batch=None):
        self._backend = backend
        self._batch = batch
        self._name = name

    def eval(self, y, x, alpha=1, beta=0, forward=True):
        """ y = A * x """
        if x.ndim == 1: x = x.reshape( (x.shape[0], 1) )
        if y.ndim == 1: y = y.reshape( (y.shape[0], 1) )
        M, N = self.shape if forward else tuple(reversed(self.shape))
        if x.shape[0] != N or \
           y.shape[0] != M or \
           x.shape[1] != y.shape[1]:
            raise ValueError("Dimension mismatch: attemping {} = {} * {} ({}, {})".format(
                y.shape, (M,N), x.shape, forward, type(self)))

        batch_size = self._batch or x.shape[1]
        for b in range(0, x.shape[1], batch_size):
          x_slc = x[:,b:b+batch_size]
          y_slc = y[:,b:b+batch_size]
          self._eval(y_slc, x_slc, alpha=alpha, beta=beta, forward=forward)

    @property
    def shape(self):
        raise NotImplemented()

    @property
    def dtype(self):
        raise NotImplemented()

    def __mul__(self, other):
        if isinstance(other, Operator):
            return Product(self._backend, self, other)
        elif isinstance(other, np.ndarray):
            log.warn("using indigow evaluation interface")
            x = other.reshape( (self.shape[1], -1), order='F' )
            x_d = self._backend.copy_array(x)
            y_d = self._backend.zero_array( (self.shape[0],x.shape[1]), dtype=other.dtype )
            self.eval(y_d, x_d)
            return y_d.to_host()
        else:
            raise ValueError("Cannot multiply Operator by %s" % type(other))
            

    @property
    def H(self):
        return Adjoint(self._backend, self, name=self._name+".H")

    def dump(self):
        """
        Returns a textual representation of the operator tree.
        """
        with io.StringIO('w') as f:
            self._dump(file=f, indent=0)
            s = f.getvalue()
        return s

    def _dump(self, file, indent=0):
        name = self._name or 'noname'
        print('{name}, {type}, {shape}, {dtype}'.format(
            name='|   ' * indent + name, type=type(self).__name__,
            shape=self.shape, dtype=self.dtype), file=file)

    def optimize(self, recipe=None):
        from indigo.transforms import Optimize
        return Optimize(recipe).visit(self)

    def memusage(self, ncols=1):
        from indigo.analyses import Memusage
        return Memusage().measure(self, ncols)


class CompositeOperator(Operator):
    def __init__(self, backend, *children, **kwargs):
        super().__init__(backend, **kwargs)
        self._adopt(children)

    @property
    def dtype(self):
        return self._children[0].dtype

    @property
    def child(self):
        assert len(self._children) == 1
        return self._children[0]

    @property
    def children(self):
        return self._children

    @property
    def left_child(self):
        assert len(self._children) == 2
        return self._children[0]

    @property
    def right_child(self):
        assert len(self._children) == 2
        return self._children[1]

    def _adopt(self, children):
        self._children = children

    def _dump(self, file, indent=0):
        s = super()._dump(file, indent)
        for c in self._children:
            c._dump(file, indent+1)

    def realize(self):
        from indigo.transforms import RealizeMatrices
        return RealizeMatrices().visit(self)


class Adjoint(CompositeOperator):
    def __init__(self, backend, children, *args, **kwargs):
        super().__init__(backend, children, *args, **kwargs)

    @property
    def shape(self):
        return tuple(reversed(self.child.shape))

    @property
    def dtype(self):
        return self.child.dtype

    @property
    def H(self):
        return self.child

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        self.child.eval( y, x, alpha, beta, forward=not forward)


class SpMatrix(Operator):
    def __init__(self, backend, M, **kwargs):
        """
        Create a new Sparse Matrix Operator from a concrete sparse matrix.
        """
        super().__init__(backend, **kwargs)
        assert isinstance(M, spp.spmatrix)
        self._matrix = M
        self._matrix_d = None

        self._allow_exwrite = False

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def nnz(self):
        return self._matrix.nnz

    def _get_or_create_device_matrix(self):
        if self._matrix_d is None:
            M = self._matrix.tocsr()
            M.sort_indices() # cuda requires sorted indictes
            self._matrix_d = self._backend.csr_matrix(self._backend, M, self._name)
            if not self._allow_exwrite:
                log.debug("disallowing exwrite for %s" % self._name)
                self._matrix_d._exwrite = False
            else:
                log.debug("allowing exwrite for %s" % self._name)
        return self._matrix_d

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        M = self._get_or_create_device_matrix()
        if forward:
            nbytes = M.nbytes + x.nbytes*M._col_frac + y.nbytes*2
        else:
            beta_part = 1 if beta == 0 else 2
            col_part = 1 if M._exwrite else 2
            nbytes = M.nbytes + x.nbytes*M._row_frac + y.nbytes*(beta_part+col_part*M._col_frac)
        nflops = 5 * len(self._matrix.data) * x.shape[1]
        with profile("csrmm", nbytes=nbytes, shape=x.shape, forward=forward, nflops=nflops) as p:
            if forward:
                M.forward(y, x, alpha=alpha, beta=beta)
            else:
                M.adjoint(y, x, alpha=alpha, beta=beta)
        profile.ktime += p.duration


class DenseMatrix(Operator):
    def __init__(self, backend, M, **kwargs):
        """
        Create a new Sparse Matrix Operator from a concrete sparse matrix.
        """
        super().__init__(backend, **kwargs)
        assert isinstance(M, np.ndarray)
        assert M.dtype == np.dtype('complex64')
        assert M.flags['F_CONTIGUOUS']
        assert M.ndim == 2
        self._matrix = M
        self._matrix_d = None

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def shape(self):
        return self._matrix.shape

    def _get_or_create_device_matrix(self):
        if self._matrix_d is None:
            self._matrix_d = self._backend.copy_array( self._matrix )
        return self._matrix_d

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        M_d = self._get_or_create_device_matrix()
        (m, n), k = M_d.shape, x.shape[1]
        nflops = m * n * k * 5
        with profile("cgemm", nflops=nflops):
            self._backend.cgemm(y, M_d, x, alpha, beta, forward=forward)


class UnscaledFFT(Operator):
    def __init__(self, backend, ft_shape, dtype=np.dtype('complex64'), forward=True, **kwargs):
        super().__init__(backend, **kwargs)
        self._ft_shape = ft_shape
        self._dtype = dtype

    @property
    def shape(self):
        n = np.prod(self._ft_shape)
        return (n,n)

    @property
    def dtype(self):
        return self._dtype

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        assert alpha == 1 and beta == 0
        X = x.reshape( self._ft_shape + (x.shape[1],) )
        Y = y.reshape( self._ft_shape + (x.shape[1],) )

        lens, batch = np.prod(X.shape[:-1]), X.shape[-1]
        nflops = batch * 5 * lens * np.log2(lens)

        if isinstance(X._arr, np.ndarray):
            ptr = X._arr.ctypes.get_data()
        elif isinstance(X._arr, c_ulong):
            ptr = X._arr.value
        else:
            ptr = 0

        align = 1
        while ptr % align == 0:
            align *= 2
        align //= 2

        with profile("fft", nflops=nflops, shape=X.shape, aligned=align) as p:
            if forward:
                self._backend.fftn(Y, X)
            else:
                self._backend.ifftn(Y, X)
        profile.ktime += p.duration

    def _mem_usage(self, ncols):
        ncols = min(ncols, self._batch or ncols)
        ft_shape = self._ft_shape + (ncols,)
        return self._backend._fft_workspace_size(ft_shape)


class KronI(CompositeOperator):
    def __init__(self, backend, c, *args, **kwargs):
        super().__init__(backend, *args, **kwargs)
        self._c = c

    @property
    def shape(self):
        h, w = self.child.shape
        return (self._c * h, self._c * w)

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        cb = self._c * x.shape[1]
        X = x.reshape( (x.size // cb, cb) )
        Y = y.reshape( (y.size // cb, cb) )
        self.child.eval(Y, X, alpha=alpha, beta=beta, forward=forward)


class BlockDiag(CompositeOperator):
    @property
    def shape(self):
        h, w = 0, 0
        for child in self._children:
            h += child.shape[0]
            w += child.shape[1]
        return h, w

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        h_offset, w_offset = 0, 0
        for C in self._children:
            h, w = C.shape if forward else reversed(C.shape)
            slc_x = slice( w_offset, w_offset+w )
            slc_y = slice( h_offset, h_offset+h )
            C.eval( y[slc_y,:], x[slc_x,:], alpha=alpha, beta=beta, forward=forward)
            h_offset += h
            w_offset += w


class VStack(CompositeOperator):
    @property
    def shape(self):
        h, w = 0, 0
        for child in self._children:
            h += child.shape[0]
            w  = child.shape[1]
        return h, w

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        if forward:
            return self._eval_forward(y, x, alpha, beta)
        else:
            return self._eval_adjoint(y, x, alpha, beta)

    def _eval_forward(self, y, x, alpha=1, beta=0):
        h_offset = 0
        for C in self._children:
            h = C.shape[0]
            slc = slice( h_offset, h_offset+h )
            C.eval( y[slc,:], x, alpha=alpha, beta=beta, forward=True)
            h_offset += h

    def _eval_adjoint(self, y, x, alpha=1, beta=0):
        self._backend.scale(y, beta)
        w_offset = 0
        for C in self._children:
            w = C.shape[0]
            slc = slice( w_offset, w_offset+w )
            C.eval( y, x[slc,:], alpha=alpha, beta=1, forward=False)
            w_offset += w

    def _adopt(self, children):
        widths = [child.shape[1] for child in children]
        names = [child._name for child in children]
        if len(set(widths)) > 1:
            raise ValueError("Mismatched widths in VStack: attempting to stack {}".format(
                list(zip(widths, names))))
        super()._adopt(children)


class HStack(CompositeOperator):
    @property
    def shape(self):
        h, w = 0, 0
        for child in self._children:
            h  = child.shape[0]
            w += child.shape[1]
        return h, w

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        if forward:
            return self._eval_forward(y, x, alpha, beta)
        else:
            return self._eval_adjoint(y, x, alpha, beta)

    def _eval_forward(self, y, x, alpha=1, beta=0):
        self._backend.scale(y, beta)
        w_offset = 0
        for C in self._children:
            w = C.shape[1]
            slc = slice( w_offset, w_offset+w )
            C.eval( y, x[slc,:], alpha=alpha, beta=1, forward=True)
            w_offset += w

    def _eval_adjoint(self, y, x, alpha=1, beta=0):
        w_offset = 0
        for C in self._children:
            w = C.shape[1]
            slc = slice( w_offset, w_offset+w )
            C.eval( y[slc,:], x, alpha=alpha, beta=beta, forward=False)
            w_offset += w

    def _adopt(self, children):
        heights = [child.shape[0] for child in children]
        names = [child._name for child in children]
        if len(set(heights)) > 1:
            raise ValueError("Mismatched heights in HStack: attempting to stack {}".format(
                list(zip(heights, names))))
        super()._adopt(children)


class Product(CompositeOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._intermediate = None
        self._name = "{}*{}".format(self.left._name, self.right._name)

    @property
    def left(self):
        return self._children[0]

    @property
    def right(self):
        return self._children[1]

    @property
    def shape(self):
        h = self.left.shape[0]
        w = self.right.shape[1]
        return h, w

    def _adopt(self, children):
        L, R = children
        if L.shape[1] != R.shape[0]:
            raise ValueError("Mismatched shapes in Product: attempting {} x {} ({} x {})".format(
                L.shape, R.shape, L._name, R._name))
        super()._adopt(children)

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        L, R = self._children
        with self._backend.scratch(shape=(R.shape[0],x.shape[1])) as tmp:
            if forward:
                R.eval(tmp, x, alpha=alpha, beta=0, forward=True)
                L.eval(y, tmp, alpha=1,  beta=beta, forward=True)
            else:
                L.eval(tmp, x, alpha=alpha, beta=0, forward=False)
                R.eval(y, tmp, alpha=1,  beta=beta, forward=False)

    def _mem_usage(self, ncols):
        ncols = min(ncols, self._batch or ncols)
        nrows = self._children[1].shape[0]
        return nrows * ncols * self.dtype.itemsize

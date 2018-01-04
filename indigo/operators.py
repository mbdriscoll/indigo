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
    def __init__(self, backend, name='', alpha=1, batch=None):
        self._backend = backend
        self._batch = batch
        self._name = name

    def eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        """
        if left:
            y = A * x
        else:
            y = x * A
        """
        M, N = self.shape if forward else tuple(reversed(self.shape))
        if left: # left-multiply
            x = x.reshape( (N,-1) )
            y = y.reshape( (M,-1) )
            assert x.shape[1] == y.shape[1], "Dimension mismatch"
        else: # right-multiply
            x = x.reshape( (-1,M) )
            y = y.reshape( (-1,N) )
            assert x.shape[0] == y.shape[0], "Dimension mismatch"
        self._eval(y, x, alpha=alpha, beta=beta, forward=forward, left=left)

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
            x = other.reshape( (self.shape[1], -1), order='F' )
            x_d = self._backend.copy_array(x)
            y_d = self._backend.zero_array( (self.shape[0],x.shape[1]), dtype=other.dtype )
            self.eval(y_d, x_d)
            return y_d.to_host()
        elif isinstance(other, (int, float, complex)):
            return Scale(self._backend, other, self)
        else:
            raise ValueError("Cannot multiply Operator by %s" % type(other))

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self * other
        else:
            raise ValueError("Cannot right-multiply Operator by %s" % type(other))

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = other * self._backend.Eye(self.shape[1])
        if isinstance(other, Operator):
            return Sum(self._backend, self, other)
        else:
            raise ValueError("Cannot right-add Operator by %s" % type(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + Scale(self._backend, -1, other)

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
        size = self._mem_usage(ncols=1) / 1e6
        print('{name}, {type}, {shape}, {size} MB, {dtype}'.format(
            name='|   ' * indent + name, type=type(self).__name__,
            size=size, shape=self.shape, dtype=self.dtype), file=file)

    def optimize(self, recipe=None):
        from indigo.transforms import Optimize
        return Optimize(recipe).visit(self)

    def memusage(self, ncols=1):
        from indigo.analyses import Memusage
        return Memusage().measure(self, ncols)

    def _mem_usage(self, ncols):
        return 0

    def has(self, *op_classes):
        """ True if this operator or any of its children are of the given type(s). """
        from indigo.analyses import TreeHasOp
        return TreeHasOp(op_classes).search(self)


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

    def _adopt(self, children):
        self._children = children

    def _dump(self, file, indent=0):
        s = super()._dump(file, indent)
        for c in self._children:
            c._dump(file, indent+1)

    def realize(self):
        from indigo.transforms import RealizeMatrices
        return RealizeMatrices().visit(self)


class BinaryOperator(CompositeOperator):
    @property
    def left(self):
        return self._children[0]

    @property
    def right(self):
        return self._children[1]


class MatrixFreeOperator(CompositeOperator):
    def __init__(self, backend, shape, *args, dtype=np.dtype('complex64'), **kwargs):
        super().__init__(backend, *args, **kwargs)
        self._dtype = dtype
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype


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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        self.child.eval( y, x, alpha, beta, forward=not forward, left=left)


class SpMatrix(Operator):
    def __init__(self, backend, M, **kwargs):
        """
        Create a new Sparse Matrix Operator from a concrete sparse matrix.
        """
        super().__init__(backend, **kwargs)
        assert isinstance(M, spp.spmatrix)
        self._matrix = M
        self._matrix_d = None

        self._allow_exwrite = True
        self._use_dia = False

    @property
    def dtype(self):
        return self._matrix.dtype

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def nnz(self):
        return self._matrix.nnz

    def _mem_usage(self, ncols=1):
        # FIXME device matrix hasn't been realized so actually not very accurate
        return self._matrix.data.nbytes

    def _get_or_create_device_matrix(self):
        if self._matrix_d is None:
            self._matrix = self._matrix.astype(np.complex64)
            assert self._matrix.dtype == np.dtype('complex64'), 'Indigo only supports single precision complex numbers for now.'
            if self._use_dia:
                log.debug("storing in DIA format: %s", self._name)
                M = self._matrix.todia()
                self._matrix_d = self._backend.dia_matrix(self._backend, M, self._name)
            else:
                log.debug("storing in CSR format: %s", self._name)
                M = self._matrix.tocsr()
                M.sort_indices() # cuda requires sorted indictes
                self._matrix_d = self._backend.csr_matrix(self._backend, M, self._name)
                if not self._allow_exwrite:
                    log.debug("disallowing exwrite for %s" % self._name)
                    self._matrix_d._exwrite = False
                else:
                    log.debug("allowing exwrite for %s" % self._name)
        return self._matrix_d

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        M = self._get_or_create_device_matrix()
        if forward:
            read_frac, write_frac = M._col_frac, M._row_frac
        else:
            read_frac, write_frac = M._row_frac, M._col_frac
        if beta == 0:
            y_part = 1
        elif beta == 1:
            y_part = write_frac * (1 if M._exwrite else 2)
        else:
            y_part = 2
        nbytes = M.nbytes + x.nbytes*read_frac + y.nbytes*y_part
        nflops = 5 * len(self._matrix.data) * x.shape[1]
        event = 'csrmm' if 'csr' in type(M).__name__ else 'diamm'
        with profile(event, xval=read_frac, yval=y_part, nbytes=nbytes, shape=x.shape, forward=forward, nflops=nflops) as p:
            if forward:
                M.forward(y, x, alpha=alpha, beta=beta)
            else:
                M.adjoint(y, x, alpha=alpha, beta=beta)


class DenseMatrix(Operator):
    def __init__(self, backend, M, **kwargs):
        super().__init__(backend, **kwargs)
        assert isinstance(M, np.ndarray)
        M = np.require(M, requirements='F')
        assert M.dtype == np.dtype('complex64')
        assert M.ndim == 2
        self._matrix = M
        self._matrix_d = None
        self._real_symmetric = M.shape[0] == M.shape[1] and \
            np.allclose(M.imag, 0) and np.allclose(M, M.T)

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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left and not self._real_symmetric:
            raise NotImplementedError("Right-multiplication not implemented for non-real-symmetric {}.".format(self.__class__.__name__))
        M_d = self._get_or_create_device_matrix()
        (m, n), k = M_d.shape, x.shape[1]
        nflops = m * n * k * 5
        if self._real_symmetric:
            with profile("csymm", nflops=nflops/2):
                self._backend.csymm(y, M_d, x, alpha=alpha, beta=beta, left=left)
        else:
            with profile("cgemm", nflops=nflops):
                self._backend.cgemm(y, M_d, x, alpha=alpha, beta=beta, forward=forward)


class UnscaledFFT(MatrixFreeOperator):
    def __init__(self, backend, ft_shape, forward=True, **kwargs):
        self._ft_shape = ft_shape
        n = np.prod(self._ft_shape)
        super().__init__(backend, shape=(n,n), **kwargs)

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        assert alpha == 1, "FFT expected alpha == 1, got %s" % alpha
        assert  beta == 0, "FFT expected beta == 0, got %s" % beta
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

    def _mem_usage(self, ncols):
        ncols = min(ncols, self._batch or ncols)
        ft_shape = self._ft_shape + (ncols,)
        return self._backend._fft_workspace_size(ft_shape)


class Eye(MatrixFreeOperator):
    def __init__(self, backend, n, **kwargs):
        super().__init__(backend, shape=(n,n), **kwargs)

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        nbytes = (0 if alpha == 0 else x.nbytes) + \
                 (0 if  beta == 0 else y.nbytes)
        with profile("axpby", nbytes=nbytes) as p:
            self._backend.axpby(beta, y, alpha, x)


class Kron(BinaryOperator):
    """ op := A \kron B """
    @property
    def shape(self):
        h = int(np.prod([c.shape[0] for c in self._children]))
        w = int(np.prod([c.shape[1] for c in self._children]))
        return (h,w)

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        L, R = self.children

        # transpose-corrected shapes
        R_shape = R.shape if forward else R.shape[::-1]
        L_shape = L.shape if forward else L.shape[::-1]

        if isinstance(L, Eye):
            R.eval(y, x, alpha=alpha, beta=beta, forward=forward, left=left)
        elif isinstance(R, Eye):
            L.eval(y, x, alpha=alpha, beta=beta, forward=forward, left=not left)
        else:
            x = x.reshape( (-1, L_shape[0]) )
            y = y.reshape( (-1, L_shape[1]) )
            tmp_shape = (x.shape[0], L_shape[1])
            with self._backend.scratch(shape=tmp_shape) as tmp:
                if forward:
                    L.eval(tmp, x, alpha=alpha, beta=0,    forward=not forward, left=not left)
                    tmp = tmp.reshape( (R_shape[1], -1) )
                    R.eval(y, tmp, alpha=1,     beta=beta, forward=forward,     left=left)
                else:
                    L.eval(tmp, x, alpha=alpha, beta=0,    forward=forward,     left=not left)
                    tmp = tmp.reshape( (R_shape[1], -1) )
                    R.eval(y, tmp, alpha=1,     beta=beta, forward=forward, left=left)


class BlockDiag(CompositeOperator):
    @property
    def shape(self):
        h, w = 0, 0
        for child in self._children:
            h += child.shape[0]
            w += child.shape[1]
        return h, w

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        h_offset, w_offset = 0, 0
        for C in self._children:
            h, w = C.shape if forward else reversed(C.shape)
            slc_x = slice( w_offset, w_offset+w )
            slc_y = slice( h_offset, h_offset+h )
            C.eval( y[slc_y,:], x[slc_x,:], alpha=alpha, beta=beta, forward=forward, left=left)
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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        if forward:
            return self._eval_forward(y, x, alpha, beta, left=left)
        else:
            return self._eval_adjoint(y, x, alpha, beta, left=left)

    def _eval_forward(self, y, x, alpha=1, beta=0, left=True):
        h_offset = 0
        for C in self._children:
            h = C.shape[0]
            slc = slice( h_offset, h_offset+h )
            C.eval( y[slc,:], x, alpha=alpha, beta=beta, forward=True, left=left)
            h_offset += h

    def _eval_adjoint(self, y, x, alpha=1, beta=0, left=True):
        self._backend.scale(y, beta)
        w_offset = 0
        for C in self._children:
            w = C.shape[0]
            slc = slice( w_offset, w_offset+w )
            C.eval( y, x[slc,:], alpha=alpha, beta=1, forward=False, left=left)
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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        if forward:
            return self._eval_forward(y, x, alpha, beta, left=left)
        else:
            return self._eval_adjoint(y, x, alpha, beta, left=left)

    def _eval_forward(self, y, x, alpha=1, beta=0, left=True):
        self._backend.scale(y, beta)
        w_offset = 0
        for C in self._children:
            w = C.shape[1]
            slc = slice( w_offset, w_offset+w )
            C.eval( y, x[slc,:], alpha=alpha, beta=1, forward=True, left=left)
            w_offset += w

    def _eval_adjoint(self, y, x, alpha=1, beta=0, left=True):
        w_offset = 0
        for C in self._children:
            w = C.shape[1]
            slc = slice( w_offset, w_offset+w )
            C.eval( y[slc,:], x, alpha=alpha, beta=beta, forward=False, left=left)
            w_offset += w

    def _adopt(self, children):
        heights = [child.shape[0] for child in children]
        names = [child._name for child in children]
        if len(set(heights)) > 1:
            raise ValueError("Mismatched heights in HStack: attempting to stack {}".format(
                list(zip(heights, names))))
        super()._adopt(children)


class Product(BinaryOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "{}*{}".format(self.left._name, self.right._name)

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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        L, R = self._children
        with self._backend.scratch(shape=(R.shape[0],x.shape[1])) as tmp:
            if forward:
                R.eval(tmp, x, alpha=alpha, beta=0, forward=True, left=left)
                L.eval(y, tmp, alpha=1,  beta=beta, forward=True, left=left)
            else:
                L.eval(tmp, x, alpha=alpha, beta=0, forward=False, left=left)
                R.eval(y, tmp, alpha=1,  beta=beta, forward=False, left=left)

    def _mem_usage(self, ncols):
        ncols = min(ncols, self._batch or ncols)
        nrows = self._children[1].shape[0]
        return nrows * ncols * self.dtype.itemsize


class Sum(CompositeOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "{}+{}".format(self.left._name, self.right._name)

    @property
    def left(self):
        return self._children[0]

    @property
    def right(self):
        return self._children[1]

    @property
    def shape(self):
        return self.left.shape

    def _adopt(self, children):
        L, R = children
        if L.shape != R.shape:
            raise ValueError("Mismatched shapes in Sum: attempting {} + {} ({} + {})".format(
                L.shape, R.shape, L._name, R._name))
        super()._adopt(children)

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        L, R = self._children
        R.eval(y, x, alpha=alpha, beta=beta, forward=forward, left=left)
        L.eval(y, x, alpha=alpha, beta=1.0,  forward=forward, left=left)

    def _mem_usage(self, ncols):
        return 0


class Scale(CompositeOperator):
    def __init__(self, backend, v, child, **kwargs):
        super().__init__(backend, child, **kwargs)
        self._name = "%s*{}".format(child._name)
        self._val = v

    @property
    def shape(self):
        return self.child.shape

    @property
    def dtype(self):
        return self.child.dtype

    def _eval(self, y, x, alpha=1, beta=0, forward=True, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        a = alpha * (self._val if forward else np.conj(self._val))
        self.child.eval(y, x, alpha=a, beta=beta, forward=forward, left=left)


class One(MatrixFreeOperator):
    def _eval(self, y, x, alpha=1, beta=0, forward=None, left=True):
        if not left:
            raise NotImplementedError("Right-multiplication not implemented for {}.".format(self.__class__.__name__))
        nbytes = (0 if alpha == 0 else x.nbytes) + \
                 (0 if beta == 0 else y.nbytes)
        with profile("onemm", nbytes=nbytes) as p:
            self._backend.onemm(y, x, alpha, beta)

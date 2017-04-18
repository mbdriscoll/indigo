import time
import logging
import io, copy
import itertools
import numpy as np
import scipy.sparse as spp

log = logging.getLogger(__name__)

class Operator(object):
    def __init__(self, backend, name=''):
        self._backend = backend
        self._stream = None
        self._name = name

    def eval(self, y, x, alpha=1, beta=0, forward=True, wait_for=None):
        """ y = A * x """
        if x.ndim == 1: x = x.reshape( (x.shape[0], 1) )
        if y.ndim == 1: y = y.reshape( (y.shape[0], 1) )
        M, N = self.shape if forward else tuple(reversed(self.shape))
        if x.shape[0] != N or \
           y.shape[0] != M or \
           x.shape[1] != y.shape[1]:
            raise ValueError("Dimension mismatch: attemping {} = {} * {} ({}, {})".format(
                y.shape, (M,N), x.shape, forward, type(self)))
        if not( x.dtype == self.dtype == y.dtype ):
            raise ValueError("Dtype mismatch: attemping {} = {} * {}".format(
                y.dtype, self.dtype, x.dtype))
        self.signal(after=wait_for)
        self._eval(y, x, alpha=alpha, beta=beta, forward=forward)
        return self.signal()

    @property
    def shape(self):
        raise NotImplemented()

    @property
    def dtype(self):
        raise NotImplemented()

    @property
    def stream(self):
        if self._stream is None:
            self._stream = self._backend.Stream(self._name)
        return self._stream

    def __mul__(self, other):
        if isinstance(other, Operator):
            return Product(self._backend, self, other)
        elif isinstance(other, np.ndarray):
            x_d = self._backend.copy_array(other)
            y_d = self._backend.zero_array( (self.shape[0],1), dtype=other.dtype )
            self.eval(y_d, x_d)
            return y_d.to_host()
        else:
            raise ValueError("Cannot multiply Operator by %s" % type(other))
            

    @property
    def H(self):
        return Adjoint(self._backend, self, name=self._name+".H")

    def signal(self, after=None):
        if after:
            self.stream.wait_for(after)
        return self.stream.signal()

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

    def optimize(self):
        from slo.transforms import Optimize
        return Optimize().visit(self)


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

    def _adopt(self, children):
        dtypes = [child.dtype for child in children]
        names  = [child._name for child in children]
        if len(set(dtypes)) > 1:
            raise ValueError("Operators have inconsistent dtypes: {}".format(
                set(dtypes)))
        self._children = children

    def _dump(self, file, indent=0):
        s = super()._dump(file, indent)
        for c in self._children:
            c._dump(file, indent+1)


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

    def _eval(self, y, x, alpha=1, beta=0, forward=True, wait_for=None):
        return self.child.eval( y, x, alpha, beta, forward=not forward, wait_for=wait_for )



class SpMatrix(Operator):
    def __init__(self, backend, M, **kwargs):
        """
        Create a new Sparse Matrix Operator from a concrete sparse matrix.

        Parameters
        ----------
        backend : pymr.backends.Backend
            Backend instance.
        M : scipy.sparse.csr_matrix
            Sparse Matrix in forward configuration.
        """
        super().__init__(backend, **kwargs)
        assert isinstance(M, spp.spmatrix)
        assert M.dtype == np.dtype('complex64')
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
            M = self._matrix.tocsr()
            M.sort_indices() # cuda requires sorted indictes
            self._matrix_d = self._backend.csr_matrix(self._backend, M, self._name)
        return self._matrix_d

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        M_d = self._get_or_create_device_matrix()
        if forward:
            M_d.forward(y, x, alpha=alpha, beta=beta, stream=self.stream)
        else:
            M_d.adjoint(y, x, alpha=alpha, beta=beta, stream=self.stream)


class UnscaledFFT(Operator):
    def __init__(self, backend, ft_shape, dtype=np.dtype('complex64'), forward=True, **kwargs):
        super().__init__(backend, **kwargs)
        self._ft_shape = ft_shape
        self._default_batch = 1
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
        batch = self._default_batch or x.shape[1]
        assert x.shape[1] % batch == 0
        for b in range( x.shape[1] // batch ):
            batch_shape = self._ft_shape + (batch,)
            vslc = slice( b*batch, (b+1)*batch )
            X = x[:,vslc].reshape( batch_shape )
            Y = y[:,vslc].reshape( batch_shape )
            if forward:
                self._backend.fftn(Y, X, self.stream)
            else:
                self._backend.ifftn(Y, X, self.stream)

    def _mem_usage(self):
        return 0


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
        ready = self.signal()
        done = self.child.eval(Y, X, alpha=alpha, beta=beta, forward=forward, wait_for=ready)
        self.signal(after=[done])


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
        child_signals = []
        ready = self.signal()
        for C in self._children:
            h, w = C.shape if forward else reversed(C.shape)
            slc_x = slice( w_offset, w_offset+w )
            slc_y = slice( h_offset, h_offset+h )
            child_signal = C.eval( y[slc_y,:], x[slc_x,:], alpha=alpha, beta=beta, forward=forward, wait_for=ready)
            h_offset += h
            w_offset += w
            child_signals.append(child_signal)
        self.signal(after=child_signals)


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
        children_done = []
        ready = self.signal()
        for C in self._children:
            h = C.shape[0]
            slc = slice( h_offset, h_offset+h )
            child_done = C.eval( y[slc,:], x, alpha=alpha, beta=beta, forward=True, wait_for=ready )
            h_offset += h
            children_done.append(child_done)
        self.signal(after=children_done)

    def _eval_adjoint(self, y, x, alpha=1, beta=0):
        assert beta in (0,1)
        if beta == 0:
            y._zero()
        w_offset = 0
        accum = self._backend.zero_array( y.shape, y.dtype ) # TODO cache dynamic malloc
        last_signal = self.signal()
        for C in self._children:
            w = C.shape[0]
            slc = slice( w_offset, w_offset+w )
            last_signal = C.eval( accum, x[slc,:], alpha=alpha, beta=1, forward=False, wait_for=last_signal )
            w_offset += w
        self.signal(after=[last_signal])
        y.copy(accum, self.stream)
        del accum

    def _adopt(self, children):
        widths = [child.shape[1] for child in children]
        names = [child._name for child in children]
        if len(set(widths)) > 1:
            raise ValueError("Mismatched widths in VStack: attempting to stack {}".format(
                list(zip(widths, names))))
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

    def _get_or_create_intermediate(self, batch, dtype):
        if self._intermediate is None:
            intermediate_shape = (self._children[0].shape[1], batch)
            arr_name = "%s intermediate" % self._name
            arr = self._backend.zero_array( intermediate_shape, dtype, name=arr_name )
            self._intermediate = arr
        return self._intermediate

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        batch = x.shape[1]
        tmp = self._get_or_create_intermediate( batch, x.dtype )
        last_signal = self.signal()
        L, R = self._children
        if forward:
            r_sig = R.eval(tmp, x, alpha=alpha, beta=0, forward=True, wait_for=last_signal)
            done  = L.eval(y, tmp, alpha=1,  beta=beta, forward=True, wait_for=r_sig)
        else:
            l_sig = L.eval(tmp, x, alpha=alpha, beta=0, forward=False, wait_for=last_signal)
            done  = R.eval(y, tmp, alpha=1,  beta=beta, forward=False, wait_for=l_sig)
        self.signal(after=done)

    def _mem_usage(self):
        return getattr(self._intermediate, 'nbytes', 0)


class Allreduce(Operator):
    def __init__(self, backend, n, team, dtype=np.dtype('complex64'), forward=True, **kwargs):
        super().__init__(backend, **kwargs)
        self._dtype = dtype
        self._team = team
        self._n = n

    @property
    def shape(self):
        return (self._n, self._n)

    @property
    def dtype(self):
        return self._dtype

    def _eval(self, y, x, alpha=1, beta=0, forward=True):
        assert alpha == 1 and beta == 0
        if forward:
            y.copy(x)
        else:
            if self._team.size > 1:
                x_h = x.to_host()
                self._team.Allreduce( MPI.IN_PLACE, x_h )
                y.copy_from(x_h)
            else:
                y.copy(x)

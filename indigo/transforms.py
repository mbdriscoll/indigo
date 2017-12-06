import copy
import random
import logging
import numpy as np
import scipy.sparse as spp
from itertools import repeat, combinations
from collections import defaultdict

from indigo.operators import (
    CompositeOperator, Product,
    Eye, BlockDiag, Kron,
    VStack, SpMatrix,
    Adjoint, UnscaledFFT,
)

log = logging.getLogger(__name__)


class Transform(object):
    """
    Visitor class for manipulating operator trees.

    See Also
    --------
    `ast.NodeTransformer`
    """
    def visit(self, node):
        method_name = "visit_%s" % type(node).__name__
        visitor_method = getattr(self, method_name, None)
        if visitor_method:
            node = visitor_method(node)
        else:
            node = self.generic_visit(node)
        return node

    def generic_visit(self, node):
        if isinstance(node, CompositeOperator):
            node._adopt( [self.visit(c) for c in node._children] )
        return node

class Visitor(object):
    """
    Visitor class for traversing operator trees.

    See Also
    --------
    `ast.NodeVisitor`
    """
    def visit(self, node):
        self.generic_visit(node)
        method_name = "visit_%s" % type(node).__name__
        visitor_method = getattr(self, method_name, None)
        if visitor_method:
            visitor_method(node)

    def generic_visit(self, node):
        if isinstance(node, CompositeOperator):
            for child in node._children:
                self.visit(child)


class Optimize(Transform):
    def __init__(self, recipe):
        super(Transform, self).__init__()
        self._recipe = recipe or []

    def visit(self, node):
        for Step in self._recipe:
            log.info("running optimization step: %s" % Step.__name__)
            node = Step().visit(node)

        # reserve scratch space
        shape = (node.memusage() // node.dtype.itemsize,)
        b = node._backend
        b._scratch = b.empty_array(shape, node.dtype)
        b._scratch_pos = 0

        return node


class RealizeMatrices(Transform):
    """
    Converts CompositeOps into SpMatrix ops if all
    children of the CompositeOp are SpMatrices.
    """
    def visit_Product(self, node):
        """ Product( SpMatrices+ ) => SpMatrix """
        node = self.generic_visit(node)
        left, right = node._children
        if isinstance(left, SpMatrix) and isinstance(right, SpMatrix):
            name = "{}*{}".format(left._name, right._name)
            log.debug('realizing product %s * %s', left._name, right._name)
            m = left._matrix @ right._matrix
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_VStack(self, node):
        """ VStack( SpMatrices ) => SpMatrix """
        node = self.generic_visit(node)
        if all(isinstance(c, SpMatrix) for c in node._children):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            log.debug('realizing vstack %s', ', '.join(c._name for c in node._children))
            m = spp.vstack( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node
    
    def visit_HStack(self, node):
        """ HStack( SpMatrices ) => SpMatrix """
        node = self.generic_visit(node)
        if all(isinstance(c, SpMatrix) for c in node._children):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            log.debug('realizing hstack %s', ', '.join(c._name for c in node._children))
            m = spp.hstack( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_BlockDiag(self, node):
        """ BlockDiag( SpMatrices ) => SpMatrix """
        node = self.generic_visit(node)
        if all(isinstance(c, SpMatrix) for c in node.children):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            log.debug('realizing block_diag %s', ', '.join(c._name for c in node._children))
            m = spp.block_diag( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_Kron(self, node):
        """ Kron(I, SpMatrix) => SpMatrix """
        node = self.generic_visit(node)
        L, R = node.children
        if isinstance(L, Eye):
            L = L.realize()
        if isinstance(L, SpMatrix) and isinstance(R, SpMatrix):
            name = "({}(x){})".format(L._name, R._name)
            log.debug('realizing kron %s x %s', L._name, R._name)
            K = spp.kron(L._matrix, R._matrix)
            return SpMatrix( node._backend, K, name=name )
        else:
            return node

    def visit_Adjoint(self, node):
        """ Adjoint(M) ==> M.H """
        node = self.generic_visit(node)
        child = node.child
        if isinstance(child, SpMatrix):
            log.debug('realizing adjoint %s', child._name)
            m = child._matrix.getH()
            name = "{}.H".format(child._name)
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_Eye(self, node):
        node = self.generic_visit(node)
        eye = spp.eye(node.shape[0], dtype=node.dtype)
        return SpMatrix( node._backend, eye, name=node._name )

    def visit_Scale(self, node):
        node = self.generic_visit(node)
        if isinstance(node.child, SpMatrix):
            mat = node.child._matrix * node._val
            return SpMatrix( node._backend, mat, name=node._name )
        else:
            return node

    def visit_One(self, node):
        one = spp.csr_matrix( np.ones(node.shape, dtype=node.dtype) )
        return SpMatrix( node._backend, one, name=node._name)


class DistributeKroniOverProd(Transform):
    """ Kron(I, A*B) ==> Kron(I, A) * Kron(I, B) """
    def visit_Kron(self, node):
        node = self.generic_visit(node)
        L, R = node.children
        if isinstance(L, Eye) and isinstance(R, Product):
            kl = node._backend.Kron( L, R.left )
            kr = node._backend.Kron( L, R.right )
            return self.visit(kl * kr)
        else:
            return node


class DistributeAdjointOverProd(Transform):
    """ Adjoint(A*B) ==> Adjoint(B) * Adjoint(A) """
    def visit_Adjoint(self, node):
        node = self.generic_visit(node)
        if isinstance(node.child, Product):
            l, r = node.child.children
            return r.H * l.H
        else:
            return node


class LiftUnscaledFFTs(Transform):
    def visit_Product(self, node):
        node = self.generic_visit(node)
        if isinstance(node, Product):
            l, r = node.children
            if isinstance(l, Product) and l.left.has(UnscaledFFT):
                ll, lr = l.children
                node = ll * self.visit(lr*r)
            elif isinstance(r, Product) and r.right.has(UnscaledFFT):
                rl, rr = r.children
                node = self.visit(l*rl) * rr
        return self.generic_visit(node)

    def visit_Adjoint(self, node):
        node = self.generic_visit(node)
        if isinstance(node.child, Product):
            node = node.child.right.H * node.child.left.H
        return self.generic_visit(node)

    def visit_Kron(self, node):
        node = self.generic_visit(node)
        L, R = node.children
        if isinstance(L, Eye) and isinstance(R, Product):
            node = node._backend.Kron(L, node.child.left) * \
                   node._backend.Kron(L, node.child.right)
        return self.generic_visit(node)

class MakeRightLeaning(Transform):
    def visit_Product(self, node):
        node = self.generic_visit(node)
        if isinstance(node, Product) and isinstance(node.left, Product):
            l, r = node.children
            ll, lr = l.children
            return ll * self.visit(lr*r)
        else:
            return node


class GroupRightLeaningProducts(Transform):
    def visit_Product(self, node):
        node = self.generic_visit(node)
        if isinstance(node, Product):
            l, r = node.children
            if isinstance(r, Product):
                rl, rr = r.children
                if isinstance(rl, SpMatrix):
                    node = (l*rl) * rr
        return node


class SpyOut(Visitor):
    def visit_SpMatrix(self, node):
        from matplotlib import pyplot as plt

        m = node._matrix
        fig, ax = plt.subplots(1, figsize=(16,16))
        ax.spy(m, markersize=1)
        fig.savefig('mat.%s.png' % node._name)

        from scipy.io import mmwrite
        mmwrite('mat.%s.mtx' % node._name, m)

import copy
import random
import logging
import numpy as np
import scipy.sparse as spp
from itertools import repeat, combinations
from collections import defaultdict

from indigo.operators import (
    CompositeOperator, Product,
    KronI, BlockDiag,
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

    def visit_KronI(self, node):
        """ KronI(c, SpMatrix) => SpMatrix """
        node = self.generic_visit(node)
        child = node.child
        if isinstance(child, SpMatrix):
            name = "{}+".format(child._name)
            I = spp.identity(node._c, dtype=child.dtype)
            log.debug('realizing kroni %s', child._name)
            K = spp.kron(I, child._matrix)
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


class DistributeKroniOverProd(Transform):
    """ KronI(A*B) ==> KronI(A) * KronI(B) """
    def visit_KronI(self, node):
        node = self.generic_visit(node)
        if isinstance(node.child, Product):
            l, r = node.child.children
            kl = l._backend.KronI( node._c, l )
            kr = r._backend.KronI( node._c, r )
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
        l = self.visit(node.left_child)
        r = self.visit(node.right_child)
        if isinstance(l, Product):
            ll = l.left_child
            lr = l.right_child
            if ll.has(UnscaledFFT):
                return ll * self.visit(lr*r)
        if isinstance(r, Product):
            rl = r.left_child
            rr = r.right_child
            if rr.has(UnscaledFFT):
                return self.visit(l*rl) * rr
        return l*r

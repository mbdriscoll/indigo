import copy
import random
import logging
import numpy as np
import scipy.sparse as spp
from itertools import repeat, combinations
from collections import defaultdict

from slo.operators import (
    CompositeOperator, Product,
    KronI, BlockDiag,
    VStack, SpMatrix
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
    def visit(self, node):
        steps = [
            #Normalize,
            #TreeTransformations,
            RealizeMatrices,
            #CoalesceAdjoints,
            StoreMatricesInAdjointOrder,
        ]

        for Step in steps:
            log.info("running optimization step: %s" % Step.__name__)
            node = Step().visit(node)

        return node


class Normalize(Transform):
    def visit_Product(self, node):
        """
        Convert nested binary products into a single n-ary product.
        """
        super().generic_visit(node)
        new_kids = []
        for child in node._children:
            if isinstance(child, Product):
                new_kids.extend(child._children)
            else:
                new_kids.append(child)
        node._adopt(new_kids)
        return node


class TreeTransformations(Transform):
    """
    Manipulates CompositeOperators.
    """
    def visit_Product(self, node):
        node = self.generic_visit(node)
        return node

        if False and self.ask("avoid distributing %s" % node._name):
            return node
            
        new_kids = []
        left = node._children[0]
        for right in node._children[1:]:
            M = self._distribute(left, right)
            if M is not None:
                left = M
            else:
                new_kids.append(left)
                left = right
        new_kids.append(left)
        if len(new_kids) == 1:
            return new_kids[0]
        else:
            node._adopt(new_kids)
            return node

    def visit_BlockDiag(self, node):
        """ BDiag(A*D,B*E,C*F) => BDiag(A,B,C)*BDiag(D,E,F) """
        node = self.generic_visit(node)
        new_kids = []
        if all( isinstance(kid,Product) for kid in node._children ) and \
                self.ask("distribute <%s>" % node._name) and False:
            for grandkids in zip(*[child._children for child in node._children]):
                new_kids.append( BlockDiag(node._backend, *grandkids) )
            return Product( node._backend, *new_kids, name=node._name )
        else:
            return node

    def visit_KronI(self, node):
        """ KronI(A*B) => KronI(A)*KronI(B) """
        node = self.generic_visit(node)
        child = node._children[0]
        if isinstance(child, Product) and \
                self.ask("distribute <%s>" % node._name) and False:
            grandkids = child._children
            new_kids = [KronI(node._backend, node._c, grandkid, name=child._name) for grandkid in grandkids]
            return Product( node._backend, *new_kids, name=node._name )
        else:
            return node

    def visit_VStack(self, node):
        """ VStack( A*D; B*E; C*F ) => BlockDiag(A,B,C) * VStack(D,E,F) """
        node = self.generic_visit(node)
        new_kids = []
        if all( isinstance(kid,Product) for kid in node._children ) and \
                self.ask("distribute <%s>" % node._name) and False:
            for grandkids in zip(*[child._children for child in node._children]):
                new_kids.append( VStack(node._backend, *grandkids) )
            return Product( node._backend, *new_kids, name=node._name )
        else:
            return node

    def _distribute(self, left, right):
        """
        Symbolically multiplies and returns LEFT and RIGHT if possible.
        Otherwise returns None. Generally performs conversions of the form:
            OP(A,B,C) * OP(D,E,F) => OP(A*D) * OP(D*E) * OP(C*F)
        """
        b = left._backend

        if isinstance(left, KronI) and isinstance(right, KronI):
            return b.KronI( left._c, left._child * right._child, name=left._name )

        signatures = {
            (b.BlockDiag, b.BlockDiag) : b.BlockDiag,
            (b.BlockDiag, b.KronI    ) : b.BlockDiag,
            (b.BlockDiag, b.VStack   ) : b.VStack,

            (b.KronI, b.BlockDiag) : b.BlockDiag,
            (b.KronI, b.VStack   ) : b.VStack,
          # (b.KronI, b.KronI    ) : b.KronI, # special case above

            (b.VStack, b.BlockDiag) : b.BlockDiag,
            (b.VStack, b.KronI    ) : b.BlockDiag,
        }

        input_type = ( left.__class__, right.__class__ )
        if input_type in signatures:
            operator = signatures[input_type]
            lc = repeat( left._child) if isinstance( left, KronI) else  left._children
            rc = repeat(right._child) if isinstance(right, KronI) else right._children
            return operator([L*R for L,R in zip(lc,rc)], name=left._name)


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
            m = spp.vstack( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def dont_visit_BlockDiag(self, node):
        """ BlockDiag( SpMatrices ) => SpMatrix """
        if all(isinstance(c, SpMatrix) for c in node._children):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            m = spp.block_diag( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def dont_visit_KronI(self, node):
        """ KronI(c, SpMatrix) => SpMatrix """
        C = node._children[0]
        if isinstance(C, SpMatrix):
            name = "{}+".format(C._name)
            I = spp.eye(node._c, dtype=C.dtype)
            K = spp.kron(I, C._matrix)
            return SpMatrix( node._backend, K, name=name )
        else:
            return node

class StoreMatricesInAdjointOrder(Transform):
    def visit_SpMatrix(self, node):
        """ SpMatrix => Adjoint(SpMatrix.H) """
        M = node._matrix
        return SpMatrix( node._backend, M.getH(), name=node._name ).H

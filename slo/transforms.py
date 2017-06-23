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
    VStack, HStack,
    SpMatrix, Adjoint
)

log = logging.getLogger(__name__)


class Transform(object):
    """
    Visitor class for manipulating operator trees.

    See Also
    --------
    `ast.NodeTransformer`
    """
    def __init__(self, instructions):
        self.instructions = instructions

    def ask_bool(self, op):
        answer = bool(np.random.randint(2))
        self.instructions.append((op, answer))
        return answer

    def ask_int(self, op):
        answer = np.random.randint(1, 16)
        self.instructions.append((op, answer))
        return answer

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
    def __init__(self, passes, instructions):
        self.passes = passes
        self.instructions = instructions

    def visit(self, node):
        steps = [
            #Normalize,
            TreeTransformations,
            RealizeMatrices,
            OperatorTransformations,
            #CoalesceAdjoints,
            #StoreMatricesInAdjointOrder,
        ]


        for p in range(self.passes['tree']):
            log.info('running optimization pass: %s' % TreeTransformations.__name__)
            node = TreeTransformations(self.instructions).visit(node)
        for p in range(self.passes['realize']):
            log.info('running optimization pass: %s' % RealizeMatrices.__name__)
            node = RealizeMatrices(self.instructions).visit(node)
        for p in range(self.passes['operator']):
            log.info('running optimization pass: %s' % OperatorTransformations.__name__)
            node = OperatorTransformations(self.instructions).visit(node)

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


class OperatorTransformations(Transform):
    def visit_Product(self, node):
        batch = self.ask_int("batch_size <%s>" % node._name)
        node._batch = batch
        return node

    def visit_UnscaledFFT(self, node):
        batch = self.ask_int("batch_size <%s>" % node._name)
        node._batch = batch
        return node

class TreeTransformations(Transform):
    """
    Manipulates CompositeOperators.
    """
    def visit_Product(self, node):
        node = self.generic_visit(node)

        if not self.ask_bool("distribute %s" % node._name):
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
                self.ask_bool("distribute <%s>" % node._name):
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
                self.ask_bool("distribute <%s>" % node._name):
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
                self.ask_bool("distribute <%s>" % node._name):
            leftKids = [child._children[0] for child in node._children]
            rightKids = [child._children[1] for child in node._children]
            left = node._backend.BlockDiag(leftKids, name=node._name+'.BlockDiag')
            right = node._backend.VStack(rightKids, name=node._name+'.VStack')
            return left*right
        else:
            return node

    def visit_Adjoint(self, node):
        """ Adjoint(CompositeOperator(A, B)) => CompositeOperator(Adjoint(B), Adjoint(A)) """
        node = self.generic_visit(node)
        child = node._children[0]
        if isinstance(child, Product) and self.ask_bool("distribute <%s>" % node._name):
            return Product(node._backend, \
                Adjoint(node._backend, child.right, name=child.right._name), \
                Adjoint(node._backend, child.left, name=child.left._name), \
                name=node._name)
        elif isinstance(child, BlockDiag) and self.ask_bool("distribute <%s>" % node._name):
            return node._backend.BlockDiag([Adjoint(node._backend, c, name=c._name) for c in child._children], name=node._name)
        elif isinstance(child, KronI) and self.ask_bool("distribute <%s>" % node._name):
            return node._backend.KronI(child._c, *[Adjoint(node._backend, c, name=c._name) for c in child._children], name=node._name)
        elif isinstance(child, VStack) and self.ask_bool("distribute <%s>" % node._name):
            return node._backend.HStack([Adjoint(node._backend, c, name=c._name) for c in child._children], name=node._name)
        elif isinstance(child, HStack) and self.ask_bool("distribute <%s>" % node._name):
            return node._backend.VStack([Adjoint(node._backend, c, name=c._name) for c in child._children], name=node._name)
        elif isinstance(child, Adjoint):
            return child._children[0]
        return node

    def _distribute(self, left, right):
        """
        Symbolically multiplies and returns LEFT and RIGHT if possible.
        Otherwise returns None. Generally performs conversions of the form:
            OP(A,B,C) * OP(D,E,F) => OP(A*D) * OP(D*E) * OP(C*F)
        """
        b = left._backend

        if isinstance(left, KronI) and isinstance(right, KronI):
            return b.KronI( left._c, left.child * right.child, name=left._name )

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
            lc = repeat( left.child) if isinstance( left, KronI) else  left._children
            rc = repeat(right.child) if isinstance(right, KronI) else right._children
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
        if isinstance(left, SpMatrix) and isinstance(right, SpMatrix) and \
            self.ask_bool('realize <%s>' % node._name):
            name = "{}*{}".format(left._name, right._name)
            log.debug('computing %s * %s', left._name, right._name)
            m = left._matrix @ right._matrix
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_VStack(self, node):
        """ VStack( SpMatrices ) => SpMatrix """
        node = self.generic_visit(node)
        if all(isinstance(c, SpMatrix) for c in node._children) and \
            self.ask_bool('realize <%s>' % node._name):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            log.debug('stacking %s', ', '.join(c._name for c in node._children))
            m = spp.vstack( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_BlockDiag(self, node):
        """ BlockDiag( SpMatrices ) => SpMatrix """
        node = self.generic_visit(node)
        if all(isinstance(c, SpMatrix) for c in node._children) and \
            self.ask_bool('realize <%s>' % node._name):
            name = "{}+".format(node._children[0]._name)
            dtype = node._children[0].dtype
            m = spp.block_diag( [c._matrix for c in node._children], dtype=dtype )
            return SpMatrix( node._backend, m, name=name )
        else:
            return node

    def visit_KronI(self, node):
        """ KronI(c, SpMatrix) => SpMatrix """
        node = self.generic_visit(node)
        C = node._children[0]
        if isinstance(C, SpMatrix) and \
            self.ask_bool('realize <%s>' % node._name):
            name = "{}+".format(C._name)
            I = spp.eye(node._c, dtype=C.dtype)
            K = spp.kron(I, C._matrix)
            return SpMatrix( node._backend, K, name=name )
        else:
            return node

    def visit_Adjoint(self, node):
        node = self.generic_visit(node)
        C = node._children[0]
        if isinstance(C, SpMatrix) and \
            self.ask_bool('realize <%s>' % node._name):
            name = '{}.H'.format(C._name)
            return SpMatrix( node._backend, C._matrix.getH(), name=name )
        else:
            return node


class StoreMatricesInAdjointOrder(Transform):

    def visit_SpMatrix(self, node):
        """ SpMatrix => Adjoint(SpMatrix.H) """
        M = node._matrix
        if self.ask_bool('adjoint <%s>' % node._name):
            return SpMatrix( node._backend, M.getH(), name=node._name ).H
        return node


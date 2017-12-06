import os
import sys
import time
import logging
import argparse
import numpy as np

import h5py

class dim:
    READ = 0
    PHS1 = 1
    PHS2 = 2
    COIL = 3
    MAPS = 4
    TIME = 10
    NDIM = 20


parser = argparse.ArgumentParser(description='Parallel Imaging and Compressed Sensing.')
parser.add_argument('-i', type=int, default=20, help='number of iterations')
parser.add_argument('--backend', type=str, default='numpy', choices=['mkl', 'numpy', 'cuda', 'customcpu', 'customgpu'])
parser.add_argument('--debug', type=int, default=logging.INFO, help='logging level')
parser.add_argument('--crop', help='crop data before recon: --crop "COIL:2,TIME:4')
parser.add_argument('--lamda', type=float, default=0, help='tikhonov reg parameter')
parser.add_argument('-O', '--recipe', type=int, default=3, choices=range(5), help='optimization level')
parser.add_argument('data', nargs='?', default="scan.h5", help='kspace data in an HDF file')
args = parser.parse_args()

# set up logging
logging.basicConfig(level=args.debug)
log = logging.getLogger("pics")

# instantiate backend
import indigo.backends
B = indigo.backends.get_backend(args.backend)
log.info("using backend: %s", type(B).__name__)

# open input file
hdf = h5py.File(args.data, 'r+')
data = hdf['data'][:]
maps = hdf['maps'][:]
traj = hdf['traj'][:]

# crop input data
crops = [1e6] * dim.NDIM
if args.crop:
    for d, size in eval("{%s}"%args.crop, {}, dim.__dict__).items():
        crops[-(d+1)] = size
        log.info("cropping dim %d to length %d", d, size)

# Read subsets of data to work on
ksp  = data[tuple(slice(0,min(n,c)) for n,c in zip(data.shape, crops))].T
mps  = maps[tuple(slice(0,min(n,c)) for n,c in zip(maps.shape, crops))].T
traj = traj[tuple(slice(0,min(n,c)) for n,c in zip(traj.shape, crops))].T

# determine dimensions
ksp_nc_dims = ksp.shape
ksp_c_dims  = mps.shape[:4] + ksp.shape[4:]
img_dims    = mps.shape[:3] + (1,) + ksp.shape[4:]

log.info('img %s %s', img_dims, ksp.dtype)
log.info('mps %s %s', mps.shape, mps.dtype) 
log.info('ksp %s %s', ksp.shape, ksp.dtype)
log.info('trj %s %s', traj.shape, traj.dtype)

# normalize trajectory coords
traj[0] /= mps.shape[0]
traj[1] /= mps.shape[1]
traj[2] /= mps.shape[2]

for i in range(3):
    log.debug("traj[%d]: %f %f", i, np.amin(traj[i]), np.amax(traj[i]))

# construct operators
C = ksp.shape[dim.COIL]
T = ksp.shape[dim.TIME]
M = ksp.shape[dim.MAPS]
assert M == 1, "No support for multiple maps."
assert T == 1, "No support for multiple timepoints."

slc = [0] * dim.NDIM
slc[dim.READ] = slice(None)
slc[dim.PHS1] = slice(None)
slc[dim.PHS2] = slice(None)

osf = (640/480, 640/480, 640/480) # uniform
#osf = (640/480, 270/208, 432/308) # cpu
#osf = (640/480, 288/208, 400/308) # knl
#osf = (600/480, 270/208, 392/308) # gpu

F1= B.NUFFT(ksp_nc_dims[:3], ksp_c_dims[:3], traj[slc], oversamp=osf, dtype=ksp.dtype)
F = B.KronI(C, F1)
S = B.VStack([B.Diag(mps[:,:,:,c:c+1]) for c in range(C)], name='maps')
A = F * S; A._name = 'SENSE1'

import scipy.sparse as spp

from indigo.transforms import Transform, Visitor
from indigo.operators import (
    Product, UnscaledFFT, SpMatrix, VStack, Eye, Kron
)

class MriGoodAdjoints(Transform):
    def visit_SpMatrix(self, node):
        if 'zpad' in node._name:
            return node.H.realize().H
        else:
            return node

class UseExwriteProperty(Transform):
    def visit_SpMatrix(self, node):
        node._allow_exwrite = True
        return node

class MriRealize(Transform):
    def visit_VStack(self, node):
        return node.realize()

    def visit_Product(self, node):
        l, r = node.children
        if isinstance(r, VStack) and isinstance(l, Kron):
            return node.realize()

        node = self.generic_visit(node)
        l, r = node.children
        if isinstance(r, SpMatrix) and isinstance(l, SpMatrix):
            return node.realize()
        else:
            return node

class DistKroniOverFFT(Transform):
    def visit_Kron(self, node):
        L, R = node.children
        if isinstance(L, Eye) and isinstance(R, Product) and node.has(UnscaledFFT):
            kl = node._backend.Kron( L, R.left_child )
            kr = node._backend.Kron( L, R.right_child )
            return self.visit(kl * kr)
        else:
            return node

class AssocSpMatrices(Transform):
    def visit_Product(self, node):
        l = self.visit(node.left_child)
        r = self.visit(node.right_child)

        try:
            rl = r.left_child
            rr = r.right_child
            if isinstance(l, SpMatrix) and not isinstance(rl, UnscaledFFT):
                return (l*rl) * rr
        except (AttributeError, AssertionError):
            pass

        return l*r
    
class MakeRightLeaning(Transform):
    def visit_Product(self, node):
        l = self.visit(node.left_child)
        r = self.visit(node.right_child)
        if isinstance(l, Product):
            ll = l.left_child
            lr = l.right_child
            return self.visit(ll*(lr*r))
        else:
            return l*r

class MakeLeftLeaning(Transform):
    def visit_Product(self, node):
        l = self.visit(node.left_child)
        r = self.visit(node.right_child)
        if isinstance(r, Product):
            rl = r.left_child
            rr = r.right_child
            return self.visit((l*rl)*rr)
        else:
            return l*r
            
recipe = []
if args.recipe >= 1:
    recipe += [
        MakeRightLeaning,
        AssocSpMatrices,
        DistKroniOverFFT,
        MakeRightLeaning]
if args.recipe >= 2:
    recipe += [MriRealize]
if args.recipe >= 3:
    recipe += [MriGoodAdjoints]
if args.recipe >= 4:
    recipe += [UseExwriteProperty]

A = A.optimize(recipe)

AHA = (A.H * A) + args.lamda * B.Eye(A.shape[1])
AHA._name = 'SENSE'
log.info("tree:\n%s", AHA.dump())

node_mem = AHA.memusage()
alg_mem = 4 * AHA.shape[1] * ksp.dtype.itemsize
log.info('using %d MB of device memory' % ((node_mem+alg_mem)/1e6))

# prep data
AHy = A.H * ksp
AHy /= abs(AHy).max()
x = np.zeros((AHA.shape[1],1), dtype=ksp.dtype, order='F')

# do reconstruction
B.cg(AHA, AHy, x, maxiter=args.i)

# write out full image
img = x.reshape(img_dims, order='F')
if 'rec' in hdf: del hdf['rec']
hdf.create_dataset('rec', data=img.T)
hdf.close()

# write out preview
try:
    from scipy.misc import imsave
    slc = [slice(None)] * img.ndim
    for t in range(T):
        slc[dim.TIME] = t
        im = img[slc].squeeze()
        imf = im.flatten()
        print("%f" % (sum(abs(imf))/len(imf)))
        x2, y2, z2 = (np.array(im.shape) / 2).astype(int)
        imsave("img_t%02d_x.jpg" % t, abs(im[x2,:,:]))
        imsave("img_t%02d_y.jpg" % t, abs(im[:,y2,:]))
        imsave("img_t%02d_z.jpg" % t, abs(im[:,:,z2]))
except ImportError:
    log.warn("install PIL or Pillow to generate preview images.")

log.info("reconstruction complete")

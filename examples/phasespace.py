"""
phasespace.py
----

Implementation of the phase-spacing imaging operator presented in:

3D imaging in volumetric scattering media using phase-space measurements.
Hsiou-Yuan Liu, Eric Jonas, Lei Tian, Jingshan Zhong, Benjamin Recht,
and Laura Waller. Optics Express, 2015

This script expects reconstruction data in an HDF5 file with two datasets:
    'codes' : c, z, py, px
    'imgs'  : z,     y,  x
where C is the number of illumination codes, X*Y*Z is the shape of the region
of interest, and X and Y are padded to PX and PY to avoid aliasing.

"""


import h5py
import logging
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Phase space reconstruction.')
parser.add_argument('-i', '--maxiter', type=int, default=2, help='number of iterations')
parser.add_argument('--backend', type=str, default='numpy', choices=['mkl', 'numpy', 'cuda', 'customcpu', 'customgpu'])
parser.add_argument('--solver', type=str, default='fista', choices=['cg', 'fista'])
parser.add_argument('--debug', type=int, default=logging.INFO, help='logging level')
parser.add_argument('data', nargs='?', default="phasespace.h5", help='acquired data in an HDF file')
args = parser.parse_args()

# set up logging
logging.basicConfig(level=args.debug)
log = logging.getLogger("phasespace")

# instantiate backend
import indigo.backends
B = indigo.backends.get_backend(args.backend)
log.info("using backend: %s", type(B).__name__)

# read problem data
with h5py.File(args.data) as hdf:
    log.info("reconstruction data:")
    imgs    = hdf[ 'imgs'][:]
    codes   = hdf['codes'][:]

    #imgs = imgs[:4,:,:]
    #codes = codes[:4,:3,:,:]

    log.info("imgs  %s %s" % ( imgs.shape,  imgs.dtype))
    log.info("codes %s %s" % (codes.shape, codes.dtype))

c, z, py, px = codes.shape
c,     y,  x = imgs.shape

# construct operator
F1 = B.FFT((py,px), imgs.dtype, name='fft2d')
F = B.KronI(z, F1, name='preffts')
C = B.BlockDiag([
    B.VStack([
        B.Diag(codes[ic,iz].flatten(), name='code_z%02d_c%02d' % (iz,ic)) for ic in range(c)
    ], name='code_z%02d' % iz) for iz in range(z)
], name='codes')
S = B.HStack([ B.Eye(c*py*px,name='sum1') for iz in range(z) ], name='sums')
D = B.Crop((y,x), (py,px), mode='edge', name='crop')
I = B.KronI(c, D*F1.H, name='postffts')
A = I * S * C * F
A._name = 'phasespace'

# optimize tree
from indigo.transforms import *
recipe = [
#    DistributeAdjointOverProd, DistributeKroniOverProd,
    LiftUnscaledFFTs,
    RealizeMatrices,
    MakeRightLeaning,
    GroupRightLeaningProducts,
    RealizeMatrices,
]
A = A.optimize(recipe)
log.info("final tree:\n%s", A.dump())

import matplotlib
matplotlib.use('agg')
from indigo.transforms import SpyOut
SpyOut().visit(A)

# reshape vectors into 2d fortran-ordered arrays
Y = imgs.astype(np.complex64).reshape((1,A.shape[0])).T
X = np.zeros((1,A.shape[1]), dtype=Y.dtype).T

# solver prep
AHy = A.H * Y
AHy_d = B.copy_array(AHy)

AHA = A.H * A
log.info("final tree:\n%s" % AHA.dump())

if args.solver == 'cg':
    B.cg(AHA, AHy, X, maxiter=args.maxiter)

elif args.solver == 'fista':
    def proxg(x_d, alpha):
        B.max(0, x_d)
        
    def gradf(gf, x):
        AHA.eval(gf, x)
        B.axpby(1, gf, -1, AHy_d)

    # do solve
    alpha = .01
    B.apgd(gradf, proxg, alpha, X, maxiter=args.maxiter)

# save output
fluor3d = X.T.reshape((z,py,px))[:,:y,:x]
try:
    from scipy.misc import imsave
    for iz in range(z):
        imsave("slc%02d.jpg" % iz, abs(fluor3d[iz]))
except ImportError:
    log.warn("install PIL or Pillow to generate preview images.")

log.info("reconstruction complete")

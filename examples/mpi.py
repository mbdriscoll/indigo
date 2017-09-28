import logging
import argparse
import numpy as np
from scipy import misc
import scipy.sparse as spp

parser = argparse.ArgumentParser(description='Magnetic Particle Imaging Reconstruction')
parser.add_argument('-i', '--maxiter', type=int, default=20, help='number of iterations')
parser.add_argument('--backend', type=str, default='numpy', choices=['mkl', 'numpy', 'cuda', 'customcpu', 'customgpu'])
parser.add_argument('--debug', type=int, default=logging.INFO, help='logging level')
parser.add_argument('data', nargs='?', default="scan.h5", help='kspace data in an HDF file')
args = parser.parse_args()

img = misc.face()
img = img[:,:,0] # select one color channel
img = img.astype(np.complex64)

x,z = img.shape
p = width = 128
s = overlap = 32
g = gap = p-2*s
snr = 10

pfovs = np.array([img[c:c+p,:] for c in range(0,x-g,g)])

img_dc = np.mean(abs(img))
dc_offsets = 2*np.random.rand(len(pfovs)) * img_dc
pfovs += dc_offsets[:,None,None]

noise = ((np.random.rand(*pfovs.shape)-0.5) + 
      1j*(np.random.rand(*pfovs.shape)-0.5)) / (2*snr)
pfovs += noise

npf, px, pz = pfovs.shape

logging.basicConfig(level=args.debug)
log = logging.getLogger("mpi")

# instantiate backend
from indigo.backends import get_backend
from indigo.transforms import *

B = get_backend(args.backend) # or numpy, cuda, etc.

# construct segmentation operator
indices = np.arange(img.size).reshape(img.shape)
S_shape = (pfovs.size, img.size)
data = np.ones(pfovs.size, dtype=np.complex64)
rows = np.arange(pfovs.size)
cols = np.array([indices[c:c+p,:] for c in range(0,x-g,g)]).flatten()
SegOp = spp.coo_matrix((data, (rows,cols)), shape=S_shape)
S = B.SpMatrix(SegOp.getH(), name='segment').H

# construct DC removal operator
D = B.KronI(npf*px, B.Eye(pz) - (1/pz)*B.One((pz,pz)))

# combine operators into forward model
A = D*S
A = A.optimize()
AHA = A.H * A
log.info("final tree %s", AHA.dump())

# reshape vectors into 2d fortran-ordered arrays
Y = pfovs.copy().reshape((1,-1)).T
X = np.zeros_like(img).reshape((1,-1)).T

AHy = A.H * Y
AHy_d = B.copy_array(AHy)

def proxg(x_d, alpha):
    B.max(0, x_d)
    
def gradf(gf, x):
    # gf = AHA*x - AHy
    AHA.eval(gf, x)
    B.axpby(1, gf, -1, AHy_d)

alpha = 0.05
B.apgd(gradf, proxg, alpha, X, maxiter=args.maxiter)

img_rec = abs(X.T.reshape(img.shape))
try:
    from scipy.misc import imsave
    imsave('mpi.jpg', img_rec)
except ImportError:
    log.warn("install PIL or Pillow to generate preview images.")

log.info("reconstruction complete")

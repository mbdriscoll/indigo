import os
import sys
import time
import logging
import argparse
import pickle
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
parser.add_argument('--trees', help='number of trees to generate')
parser.add_argument('--num_passes', help='number of passes: --num_passes 1,2,3,4')
parser.add_argument('--tree_file', help='file in which to store the generated trees')
parser.add_argument('--backend', type=str, default='numpy', choices=['mkl', 'numpy', 'cuda'])
parser.add_argument('--debug', type=int, default=logging.INFO, help='logging level')
parser.add_argument('--crop', help='crop data before recon: --crop "COIL:2,TIME:4')
parser.add_argument('data', nargs='?', default="scan.h5", help='kspace data in an HDF file')
args = parser.parse_args()

numPasses = [int(x) for x in args.num_passes.split(',')]
numTrees = int(args.trees)

count = 0
trees = []

np.random.seed(0)

for t in range(numTrees):
    for passes in numPasses:
        # set up logging
        logging.basicConfig(level=args.debug)
        log = logging.getLogger("pics")

        # instantiate backend
        if args.backend == 'mkl':
            from slo.backends.mkl  import MklBackend as BACKEND
        elif args.backend == 'cuda':
            from slo.backends.cuda import CudaBackend as BACKEND
        elif args.backend == 'numpy':
            from slo.backends.np   import NumpyBackend as BACKEND
        else:
            log.error("unrecognized backend: %s", args.backend)
        B = BACKEND()
        log.info("using backend: %s", type(B).__name__)

        # open input file
        hdf = h5py.File(args.data, 'r+')
        data = hdf['data']
        maps = hdf['maps']
        traj = hdf['traj']
        dcf  = hdf['dcf']

        # crop input data
        crops = [1e6] * dim.NDIM
        if args.crop:
            for d, size in eval("{%s}"%args.crop, {}, dim.__dict__).items():
                crops[-(d+1)] = size
                log.info("cropping dim %d to length %d", d, size)

        # Read subsets of data to work on
        ksp  = data[tuple(slice(0,min(n,c)) for n,c in zip(data.shape, crops))].T
        mps  = maps[tuple(slice(0,min(n,c)) for n,c in zip(maps.shape, crops))].T
        dcf  =  dcf[tuple(slice(0,min(n,c)) for n,c in zip(dcf.shape,  crops))].T
        traj = traj[tuple(slice(0,min(n,c)) for n,c in zip(traj.shape, crops))].T

        # determine dimensions
        ksp_nc_dims = ksp.shape
        ksp_c_dims  = mps.shape[:4] + ksp.shape[4:]
        img_dims    = mps.shape[:3] + (1,) + ksp.shape[4:]

        log.info('img %s %s', img_dims, ksp.dtype)
        log.info('mps %s %s', mps.shape, mps.dtype) 
        log.info('ksp %s %s', ksp.shape, ksp.dtype)
        log.info('dcf %s %s', dcf.shape, dcf.dtype)
        log.info('trj %s %s', traj.shape, traj.dtype)

        # normalize trajectory coords
        traj[0] /= mps.shape[0]
        traj[1] /= mps.shape[1]
        traj[2] /= mps.shape[2]

        # construct operators
        C = ksp.shape[dim.COIL]
        T = ksp.shape[dim.TIME]
        M = ksp.shape[dim.MAPS]
        assert M == 1, "No support for multiple maps."

        Ss = []
        for c in range(C):
            S_c = B.Diag( mps[:,:,:,c:c+1], name='map%02d' % c )
            Ss.append(S_c)

        slc = [0] * dim.NDIM
        slc[dim.READ] = slice(None)
        slc[dim.PHS1] = slice(None)
        slc[dim.PHS2] = slice(None)

        Gs = []
        for t in range(T):
            slc[dim.TIME] = t
            P_t = B.Diag( np.sqrt(dcf[slc]), name='dcf' )
            G_t, Mk, S, F, Mx, Z, R = B.NUFFT(ksp_nc_dims[:3], ksp_c_dims[:3], traj[slc], dtype=ksp.dtype)
            Gs.append( P_t * G_t * Mk * S )

        S = B.KronI(T, B.VStack([Mx * Z * R * Sc for Sc in Ss], name='maps'))
        F = B.KronI(T*C, F, name='batch_fft')
        G = B.BlockDiag( [B.KronI(C, Gt) for Gt in Gs], name='interp')
        A = G * F * S; A._name = 'SENSE1'

        AHA = A.H * A
        AHA._name = 'SENSE'

        AHA,instructions = AHA.optimize(passes)
        x = np.zeros((AHA.shape[1],1), dtype=ksp.dtype, order='F')
        mem = AHA.memusage(x.shape[1])

        log.info('using %d MB of device memory' % (mem/1e6))
        log.info("tree:\n%s", AHA.dump())

        # prep data
        ksp *= np.sqrt(dcf)
        AHy = A.H * ksp
        AHy /= abs(AHy).max()

        from slo.analyses import Memusage

        # do reconstruction
        begin = time.time()
        B.cg(AHA, AHy, x, maxiter=args.i)
        end = time.time()

        elapsed = end - begin

        trees.append([elapsed, mem, AHA.dump(), instructions])

        count += 1

        if count % 100 == 0:
            f = open(args.tree_file, 'wb')
            pickle.dump(trees, f)
            f.close()

f = open(args.tree_file, 'wb')
pickle.dump(trees, f)
f.close()

import time
import argparse
import numpy as np
import scipy.sparse as spp

from slo.util import rand64c, randM, Timer

def benchmark_axpy(backend, args):
    x = rand64c( int(1e9/8) ) # 1 GB
    y = rand64c( x.size  )

    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    timer = Timer()
    for trial in range(args.trials):
        backend.barrier()
        with timer:
            backend.axpy(y_d, 1.1, x_d)
            backend.barrier()
        
    nsec = timer.median
    nbytes = x.nbytes + 2*y.nbytes
    gbps = nbytes / nsec * 1e-9
    frac = gbps / args.stream * 100
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("axpy, %s, %d threads, %2.0f GB, %2.2f GB/s, %2.0f%% STREAM" % \
        (name, nthreads, nbytes/1e9, gbps, frac), flush=True)


def benchmark_fft(backend, args):
    shape = (660, 286, 423)
    x = rand64c( np.prod(shape), args.batch )
    y = np.zeros_like(x)

    F = backend.UnscaledFFT( shape, dtype=x.dtype )
    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    timer = Timer()
    backend.barrier()
    with timer:
        for trial in range(args.trials):
            F.eval(y_d, x_d)
        backend.barrier()

    XYZ = np.prod(x.shape[:3])
    nsec = timer.median / args.trials
    nflops = 5 * XYZ * np.log2(XYZ)
    nbytes = 4 * x.nbytes
    roofline = args.stream*1e9 * (nflops / nbytes)
    frac = (nflops / nsec) / roofline * 100
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("fft,  %s, %d threads, batch %d, %2.2f GFlops/s, %2.0f%% Roofline" % \
        (name, nthreads, args.batch, nflops/nsec/1e9, frac), flush=True)


def benchmark_csrmm(backend, args):
    N = 150                        # problem scale ~ image edge length
    XYZ  = N**3                    # number of columns
    pXYZ = int(8 * N**3 * 1.35**3) # number of rows (8 coils, 1.35 oversampling factor)

    # make one nonzero per row along diagonal
    indptrs = np.arange(pXYZ+1, dtype=np.int32)
    indices = np.arange(pXYZ, dtype=np.int32) % XYZ
    data    = np.ones(pXYZ, dtype=np.complex64)

    A = spp.csr_matrix((data,indices,indptrs), shape=(pXYZ,XYZ), dtype=np.complex64)
    x = rand64c( XYZ,args.batch)
    y = rand64c(pXYZ,args.batch)

    A_d = backend.SpMatrix(A)
    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    timer = Timer()
    for trial in range(args.trials):
        backend.barrier()
        with timer:
            A_d.eval(y_d, x_d)
            backend.barrier()

    nsec = timer.median
    nbytes = x.nbytes + y.nbytes + A_d._matrix_d.nbytes
    gbps = nbytes / nsec * 1e-9
    frac = gbps / args.stream * 100
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__
    print("csrmm %s, %d threads, %d nnz, batch %d, %2.0f GB, %2.2f GB/s, %2.0f%% STREAM" % \
        (name, nthreads, len(A.data), args.batch, nbytes/1e9, gbps, frac), flush=True)


def main():
    parser = argparse.ArgumentParser(description='benchmark slo')
    parser.add_argument('--all',   action='store_true')
    parser.add_argument('--axpy',  action='store_true')
    parser.add_argument('--fft',   action='store_true')
    parser.add_argument('--csrmm', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--stream', type=float, help='Memory bandwidth in GB/sec.')
    args = parser.parse_args()

    from slo.backends.mkl import MklBackend
    from slo.backends.cuda import CudaBackend

    Backends = [MklBackend,CudaBackend]

    for Backend in Backends:
        backend = Backend()
        if args.axpy  or args.all: benchmark_axpy (backend, args)
        if args.fft   or args.all: benchmark_fft  (backend, args)
        if args.csrmm or args.all: benchmark_csrmm(backend, args)


if __name__ == '__main__':
    main()

import time
import argparse
import logging
import numpy as np
import scipy.sparse as spp

from slo.util import rand64c, randM, Timer

log = logging.getLogger(__name__)

def benchmark_axpy(backend, batch=1, ntrials=10):

    x = rand64c( int(1e9/8) ) # 1 GB
    y = rand64c( x.size  )

    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    times = []
    timer = Timer()
    for trial in range(ntrials):
        with timer:
            backend.axpy(y_d, 1.1, x_d)
        
    nsec = timer.median
    nbytes = x.nbytes + 2*y.nbytes
    gbps = nbytes / nsec * 1e-9
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("axpy, %s, %d threads, %2.0f GB, %2.2f GB/s" % (name, nthreads, nbytes/1e9, gbps), flush=True)


def benchmark_fft(backend, batch=8, ntrials=10):

    x = rand64c( 660, 286, 423, batch )
    y = np.zeros_like(x)

    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    times = []
    timer = Timer()
    for trial in range(ntrials):
        with timer:
            backend.fftn(y_d, x_d)

    XYZ = np.prod(x.shape)
    nsec = timer.median
    nflops = batch * 5 * np.prod(XYZ) * np.log2(XYZ)
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("fft,  %s, %d threads, batch %d, %2.2f GFlops/s" % \
        (name, nthreads, batch, nflops/nsec/1e9), flush=True)


def benchmark_csrmm(backend, batch=8, ntrials=10):
    XYZ = 208 * 308 * 480
    ROxPS = 100 * 654
    nnz = 8175000
    density = nnz / ( XYZ * ROxPS * 2 )

    A = randM(ROxPS, XYZ, density)
    x = rand64c(XYZ,batch)
    y = rand64c(ROxPS,batch)

    A_d = backend.SpMatrix(A)
    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    times = []
    timer = Timer()
    for trial in range(ntrials):
        with timer:
            A_d.eval(y_d, x_d)

    nsec = timer.median
    nbytes = x.nbytes + y.nbytes + A_d._matrix_d.nbytes
    gbps = nbytes / nsec * 1e-9
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("csrmm, %s, %d threads, %d nnz, batch %d, %2.0f GB, %2.2f GB/s" % \
        (name, nthreads, len(A.data), batch, nbytes/1e9, gbps), flush=True)


def main():
    parser = argparse.ArgumentParser(description='benchmark slo')
    parser.add_argument('--all',   action='store_true')
    parser.add_argument('--axpy',  action='store_true')
    parser.add_argument('--fft',   action='store_true')
    parser.add_argument('--csrmm', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()

    from slo.backends.mkl import MklBackend

    backends = [MklBackend]

    for backend in backends:
        B = backend()
        if args.axpy  or args.all: benchmark_axpy (B, batch=args.batch)
        if args.fft   or args.all: benchmark_fft  (B, batch=args.batch)
        if args.csrmm or args.all: benchmark_csrmm(B, batch=args.batch)


if __name__ == '__main__':
    main()

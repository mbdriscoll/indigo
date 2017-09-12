import time
import argparse
import numpy as np
import scipy.sparse as spp

from indigo.util import rand64c, randM, Timer

def benchmark_axpby(backend, args):
    n = 3e9//8
    n = 2.3e6
    x = rand64c(int(n))
    y = rand64c(int(n))
    x_d = backend.copy_array(x)
    y_d = backend.copy_array(y)

    ntrials = args.trials
    backend.axpby(1, y_d, 1.1, x_d)

    backend.barrier()
    timer = Timer()
    for t in range(ntrials):
        with timer:
            backend.axpby(1, y_d, 1.1, x_d)
            backend.barrier()
        
    nsec = timer.min
    nbytes = x_d.nbytes + 2*y_d.nbytes
    gbps = nbytes / nsec * 1e-9
    nthreads = backend.get_max_threads()
    name = backend.__class__.__name__

    print("axpby, %s, %d threads, %d trials, %.0f MB, best %2.2f ms %2.2f GB/s, worst %2.2f ms %2.2f GB/s" % \
        (name, nthreads, ntrials, (x_d.nbytes+y_d.nbytes)/1e6, timer.min * 1000, gbps, timer.max * 1000, nbytes/timer.max * 1e-9), flush=True)


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



def factors(n):
    while n > 1:
        for i in range(2, n + 1):
            if n % i == 0:
                n //= i
                yield i
                break

def apf(x, y, z):
    fx = set(factors(x))
    fy = set(factors(y))
    fz = set(factors(z))
    return np.average(list(fx|fy|fz))

def fft_search(args):

    shape = (308,208,480)
    minosf, maxosf = 1.25, 1.5

    from indigo.backends.cuda import CudaBackend

    best_flops = None
    best_flops_s = None
    best_flops_f = 0
    best_sec = None
    best_sec_s = 1e99
    best_sec_f = 1e99

    print("Generating shapes...", flush=True)
    shapes = []
    for     x in range(int(shape[0]*minosf), int(shape[0]*maxosf)):
      for   y in range(int(shape[1]*minosf), int(shape[1]*maxosf)):
        for z in range(int(shape[2]*minosf), int(shape[2]*maxosf)):
            shapes.append( ((x,y,z), apf(x,y,z)) )

    print("Sorting shapes...", flush=True)
    shapes.sort(key=lambda xyzf: xyzf[-1])

    # best known
    shapes.insert(0, ((448,270,640),'?') )
    shapes.insert(0, ((432,270,600),'?') )

    print("Evaluating shapes...", flush=True)
    for (x,y,z), fp in shapes:

        backend = CudaBackend()
        F = backend.UnscaledFFT( (z,y,x), dtype=np.dtype('complex64') )

        u_d = backend.zero_array( (z*y*x, args.batch), dtype=F.dtype )
        v_d = backend.zero_array( (z*y*x, args.batch), dtype=F.dtype )

        # warmup
        F.eval(v_d, u_d)

        timer = Timer()
        backend.barrier()
        with timer:
            for trial in range(args.trials):
                F.eval(v_d, u_d)
            backend.barrier()

        nsec = timer.median / args.trials
        nflops = args.batch * 5 * x*y*z * np.log2(x*y*z)

        floprate = nflops/nsec
        if floprate > best_flops_f:
            best_flops_f = floprate
            best_flops_s = nsec
            best_flops = (x, y, z)

        if nsec < best_sec_s:
            best_sec_s = nsec
            best_sec_f = floprate
            best_sec = (x, y, z)
        
        print("fft,  %03d xx % 3d x % 3d x % 3d :  %2.2f GFlops/s, %2.0f ms\tbest_flops: %s %2.2f GF/s %2.2f ms\tbest_time: %s %2.2f ms %2.2f GF/s" % \
            (args.batch, x, y, z, nflops/nsec/1e9, nsec*1000, best_flops, best_flops_f/1e9, best_flops_s*1000, best_sec, best_sec_s*1000, best_sec_f/1e9), flush=True)

def main():
    parser = argparse.ArgumentParser(description='benchmark indigo')
    parser.add_argument('--all',   action='store_true')
    parser.add_argument('--axpby', action='store_true')
    parser.add_argument('--fft',   action='store_true')
    parser.add_argument('--csrmm', action='store_true')
    parser.add_argument('--fftsearch', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--stream', type=float, help='Memory bandwidth in GB/sec.')
    args = parser.parse_args()

    from indigo.backends.mkl import MklBackend

    Backends = [MklBackend,]

    for Backend in Backends:
        backend = Backend()
        if args.axpby or args.all: benchmark_axpby(backend, args)
        if args.fft   or args.all: benchmark_fft  (backend, args)
        if args.csrmm or args.all: benchmark_csrmm(backend, args)

        if args.fftsearch: fft_search(args)


if __name__ == '__main__':
    main()

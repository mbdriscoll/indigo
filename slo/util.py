import numpy as np
import scipy.sparse as spp

def rand64c(*shape):
    r = np.random.rand(*shape).astype(np.float32)
    i = np.random.rand(*shape).astype(np.float32)
    arr = (r + 1j*i).astype(np.complex64)
    return arr


def randM(M, N, density):
    A_r = spp.random(M, N, density=density, format='csr', dtype=np.float32)
    A_i = spp.random(M, N, density=density, format='csr', dtype=np.float32)
    A = (A_r + 1j * A_i).astype(np.dtype('complex64'))
    return A

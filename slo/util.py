import numpy as np
import scipy.sparse as spp

def rand64c(*shape):
    """
    Constructs a `np.ndarray` of the requested shape and
    populates it with random np.complex64 values.
    """
    r = np.random.rand(*shape).astype(np.float32)
    i = np.random.rand(*shape).astype(np.float32)
    arr = (r + 1j*i).astype(np.complex64)
    arr = np.asfortranarray(arr)
    return arr


def randM(M, N, density):
    """
    Constructs a `scipy.sparse.spmatrix'  of the requested shape and
    density and populates it with random np.complex64 values.
    """
    A_r = spp.random(M, N, density=density, format='csr', dtype=np.float32)
    A_i = spp.random(M, N, density=density, format='csr', dtype=np.float32)
    A = (A_r + 1j * A_i).astype(np.dtype('complex64'))
    return A

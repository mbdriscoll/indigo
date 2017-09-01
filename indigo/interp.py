import numba as nb
import math
import numpy as np
import scipy.sparse as sparse
__all__ = ['interp_mat', 'interp_funs']


@nb.jit(nopython=True, cache=True)
def lin_interp(table, x):
    if x >= 1:
        return 0.0
    n = len(table)
    idx = int(x * (n - 1))
    frac = x * (n - 1) - idx
    return (1.0 - frac) * table[idx] + frac * table[idx + 1]


@nb.jit(nopython=True, cache=True)
def _interp3_mat(m, N, width, table, coord):

    row = np.empty(m * int(2 * width + 1)**3, dtype=np.int64)
    col = np.empty(m * int(2 * width + 1)**3, dtype=np.int64)
    ker = np.empty(m * int(2 * width + 1)**3, dtype=np.float64)

    c = 0
    for i in range(m):

        pos = (N[0] * coord[0, i] + (N[0] // 2),
               N[1] * coord[1, i] + (N[1] // 2),
               N[2] * coord[2, i] + (N[2] // 2))

        start = (math.ceil(pos[0] - width),
                 math.ceil(pos[1] - width),
                 math.ceil(pos[2] - width))
        
        end = (math.floor(pos[0] + width),
               math.floor(pos[1] + width),
               math.floor(pos[2] + width))

        for z in range(start[2], end[2]):

            wz = lin_interp(table, abs(z - pos[2]) / width)
            jz = (z % N[2]) * N[1] * N[0]

            for y in range(start[1], end[1]):

                wy = wz * lin_interp(table, abs(y - pos[1]) / width)
                jy = (y % N[1]) * N[0] + jz

                for x in range(start[0], end[0]):

                    w = wy * lin_interp(table, abs(x - pos[0]) / width)
                    j = (x % N[0]) + jy

                    row[c] = i
                    col[c] = j
                    ker[c] = w

                    c += 1
    return row[:c], col[:c], ker[:c]


def interp_mat(m, N, width, table, coord, backend):

    ndim = coord.shape[0]

    if ndim == 1:
        _interp_mat = _interp1_mat
    elif ndim == 2:
        _interp_mat = _interp2_mat
    elif ndim == 3:
        _interp_mat = _interp3_mat
    else:
        raise ValueError('Number of dimensions can only be 1, 2 or 3, got %r',
                         ndim)

    row, col, ker = _interp_mat(m, N, width, table, coord)
    
    return sparse.coo_matrix((ker, (row, col)),
                             shape=(m, np.prod(N, dtype=np.int)))

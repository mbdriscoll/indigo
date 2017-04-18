import os
import numba
import math
import hashlib
import logging
import numpy as np
import scipy.sparse as sparse

__all__ = ['interp_mat', 'interp_funs']

log = logging.getLogger(__name__)

@numba.njit
def lin_interp(table, x):
    n = len(table)
    idx = int(x * (n - 2))
    frac = x * (n - 2) - idx
    return (1.0 - frac) * table[idx] + frac * table[idx + 1]


@numba.njit
def _interp2_mat(M, N, width, table, coord):
    row = np.empty(M * int(2 * width + 1)**2, dtype=np.int64)
    col = np.empty(M * int(2 * width + 1)**2, dtype=np.int64)
    ker = np.empty(M * int(2 * width + 1)**2, dtype=np.float64)

    c = 0
    for i in range(M):

        start = np.empty(2, np.int64)
        end = np.empty(2, np.int64)
        pos = np.empty(2, np.float64)
        for d in range(2):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

        for y in range(start[1], end[1]):

            wy = lin_interp(table, abs(y - pos[1]) / width)
            jy = (y % N[1]) * N[0]

            for x in range(start[0], end[0]):

                w = wy * lin_interp(table, abs(x - pos[0]) / width)
                j = (x % N[0]) + jy

                row[c] = i
                col[c] = j
                ker[c] = w

                c += 1
    return row[:c], col[:c], ker[:c]


@numba.njit
def _interp3_mat(M, N, width, table, coord):

    row = np.empty(M * int(2 * width + 1)**3, dtype=np.int64)
    col = np.empty(M * int(2 * width + 1)**3, dtype=np.int64)
    ker = np.empty(M * int(2 * width + 1)**3, dtype=np.float64)

    c = 0
    for i in range(M):

        start = np.empty(3, np.int64)
        end = np.empty(3, np.int64)
        pos = np.empty(3, np.float64)

        for d in range(3):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

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


def _mat_identifier(M, N, width, table, coord):
    m = hashlib.md5()
    m.update(str((M,N,width)).encode('ascii'))
    m.update(table.tobytes())
    m.update(coord.tobytes())
    return m.hexdigest()


def interp_mat(M, N, width, table, coord, num_threads, cache=True, force_cache=False):

    ndim = coord.shape[0]

    if ndim == 2:
        _interp_mat = _interp2_mat
    elif ndim == 3:
        _interp_mat = _interp3_mat
    else:
        raise ValueError('Number of dimensions can only be 2 or 3, got %r',
                         ndim)

    mat = None
    mat_shape = (M, np.prod(N))

    if cache:
        mat_id = _mat_identifier(M, N, width, table, coord)

        if not os.path.exists("cache"):
            os.makedirs("cache")
        mat_file = os.path.join("cache", "interp_%s.npz" % mat_id)

        try:
            f = np.load(mat_file)
            values, indices, indptr = f['values'], f['indices'], f['indptr']
            mat = sparse.csr_matrix((values, indices, indptr), shape=mat_shape)
            log.debug("using cached interpolation matrix.")
        except IOError as e:
            if force_cache:
                raise ValueError("force_cache requested, but no cached matrix exists.")

    if mat is None:
        log.debug("interpolation matrix not cached. recomputing now.")
        row, col, ker = _interp_mat(M, N, width, table, coord)
        mat = sparse.csr_matrix((ker, (row, col)), shape=mat_shape)

        if cache:
            np.savez_compressed( mat_file, values=mat.data, indices=mat.indices, indptr=mat.indptr )

    return mat


@numba.njit
def _interp2(M, N, batch, width, table, coord, src):
    dst = np.zeros((M, batch), src.dtype)

    for i in range(M):
        start = np.empty(2, np.int64)
        end = np.empty(2, np.int64)
        pos = np.empty(2, np.float64)
        for d in range(2):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

        for y in range(start[1], end[1]):

            wy = lin_interp(table, abs(y - pos[1]) / width)
            jy = y * N[0]

            for x in range(start[0], end[0]):

                w = wy * lin_interp(table, abs(x - pos[0]) / width)
                j = x + jy

                for k in range(batch):
                    dst[i, k] += w * src[j, k]

    return dst


@numba.njit
def _interp2H(M, N, batch, width, table, coord, src):
    dst = np.zeros((N[0] * N[1], batch), src.dtype)

    for i in range(M):
        start = np.empty(2, np.int64)
        end = np.empty(2, np.int64)
        pos = np.empty(2, np.float64)
        for d in range(2):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

        for y in range(start[1], end[1]):

            wy = lin_interp(table, abs(y - pos[1]) / width)
            jy = (y % N[1]) * N[0]

            for x in range(start[0], end[0]):

                w = wy * lin_interp(table, abs(x - pos[0]) / width)
                j = (x % N[0]) + jy

                for k in range(batch):
                    dst[j, k] += w * src[i, k]

    return dst


@numba.njit
def _interp3(M, N, batch, width, table, coord, src):
    dst = np.zeros((M, batch), src.dtype)

    for i in range(M):
        start = np.empty(3, np.int64)
        end = np.empty(3, np.int64)
        pos = np.empty(3, np.float64)
        for d in range(3):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

        for z in range(start[2], end[2]):

            wz = lin_interp(table, abs(z - pos[2]) / width)
            jz = (z % N[2]) * N[1] * N[0]

            for y in range(start[1], end[1]):

                wy = wz * lin_interp(table, abs(y - pos[1]) / width)
                jy = (y % N[1]) * N[0] + jz

                for x in range(start[0], end[0]):
                    w = wy * lin_interp(table, abs(x - pos[0]) / width)
                    j = (x % N[0]) + jy

                    for k in range(batch):
                        dst[i, k] += w * src[j, k]

    return dst


@numba.njit
def _interp3H(M, N, batch, width, table, coord, src):
    dst = np.zeros((N[0] * N[1] * N[2], batch), src.dtype)

    for i in range(M):
        start = np.empty(3, np.int64)
        end = np.empty(3, np.int64)
        pos = np.empty(3, np.float64)
        for d in range(3):
            pos[d] = N[d] * (coord[d, i] + 0.5)

            start[d] = math.ceil(pos[d] - width)
            end[d] = math.floor(pos[d] + width)

        for z in range(start[2], end[2]):

            wz = lin_interp(table, abs(z - pos[2]) / width)
            jz = (z % N[2]) * N[1] * N[0]

            for y in range(start[1], end[1]):

                wy = wz * lin_interp(table, abs(y - pos[1]) / width)
                jy = (y % N[1]) * N[0] + jz

                for x in range(start[0], end[0]):
                    w = wy * lin_interp(table, abs(x - pos[0]) / width)
                    j = (x % N[0]) + jy

                    for k in range(batch):
                        dst[j, k] += w * src[i, k]

    return dst


def interp(M, N, batch, width, table, coord, num_threads, x):

    ndim = coord.shape[0]
    if ndim == 2:
        _interp = _interp2
    else:
        _interp = _interp3

    out = _interp(M, N, batch, width, table, coord, x)

    return out


def interpH(M, N, batch, width, table, coord, num_threads, x):

    ndim = coord.shape[0]
    if ndim == 2:
        _interpH = _interp2H
    else:
        _interpH = _interp3H

    out = _interpH(M, N, batch, width, table, coord, x)
    return out


def interp_funs(M, N, batch, width, table, coord, num_threads):
    ''' Interpolation funcations
    Parameters
    ----------
    M : int, number of points to interpolate to
    N : tuple of ints, grid size, has length ndim (either 2 or 3)
    batch : int, number of batch interpolation
    width : int, width
    table : array of floats
    coord : (ndim, M) float array
    '''
    num_threads = min(M, num_threads)

    return (lambda x: interp(M, N, batch, width, table, coord, num_threads, x),
            lambda x: interpH(M, N, batch, width, table, coord, num_threads, x))

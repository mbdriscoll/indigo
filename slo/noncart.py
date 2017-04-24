import numpy as np
import math
import numba
import numexpr as ne


def ftkb(beta, x):
    pi = np.pi
    a = ne.evaluate('sqrt(beta ** 2 - (pi * x) ** 2)')

    ret = np.empty(a.shape, dtype=a.dtype)
    ret[a == 0.0] = 1.0
    b = a[a != 0.0]
    ret[a != 0.0] = ne.evaluate('sinh(b) / b')

    return ret


def rolloff2(oversamp, width, beta, N):

    x, y = np.mgrid[:N[0], :N[1]]

    return ftkb(beta, 0.0)**2 / (ftkb(beta, (x / N[0] - 0.5) * width * 2.0 / oversamp) *
                                 ftkb(beta, (y / N[1] - 0.5) * width * 2.0 / oversamp))


def rolloff3(oversamp, width, beta, N):

    x, y, z = np.mgrid[:N[0], :N[1], :N[2]]

    return ftkb(beta, 0.0)**3 / (ftkb(beta, (x / N[0] - 0.5) * width * 2.0 / oversamp) *
                                 ftkb(beta, (y / N[1] - 0.5) * width * 2.0 / oversamp) *
                                 ftkb(beta, (z / N[2] - 0.5) * width * 2.0 / oversamp))

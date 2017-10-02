import numpy as np
import scipy.sparse as spp
import math

from scipy.special import erf
from scipy.ndimage.morphology import distance_transform_edt

# Generate mapidx, which describes the frames
# to extract from an image
def gen_mapidx(nx, ny, nframes):
    # Perform translation.
    nnx = nny = np.sqrt(nframes)
    x = np.linspace(0, 95, nnx)
    y = np.linspace(0, 95, nny)

    # TODO: Check that this works in general.
    # (Different sizes)
    ttx = np.tile(x, (len(y), 1))
    tty = ttx.T

    tt = np.array(list(zip(ttx.flatten(), tty.flatten()))).T
    cornerx = tt[0,:]
    cornery = tt[1,:]

    cornerx = tt[0,:]
    cornery = tt[1,:]

    Nx = math.ceil(max(cornerx) - min(cornerx)) + nx
    Ny = math.ceil(max(cornery) - min(cornery)) + ny

    frame_corner,frac_shift = idx2corner(cornerx, max(cornery) - cornery, Nx, Ny)

    xi, yi = corner2pos(frame_corner, Nx)
    wrap = lambda x, bound: ((x - 1) % bound) + 1
    rrx, rry = meshgrid(np.arange(1, nx+1), np.arange(1, ny+1))
    mapidx = (wrap(spv(rrx, xi), Nx) - 1) * Nx + wrap(spv(rry, yi), Ny)

    # mapidx-1 to switch to 0-based indexing.
    return (mapidx.flatten()-1, Nx, Ny)

# Generate illumination
def gen_omega(nx, ny, xc, yc, fProbe):
    ph = phaseramps(nx / 2, nx / 2, nx, ny)
    tmp = fProbe * ph
    Fprobe = np.fft.fft2(tmp) * (1 / np.sqrt(tmp.shape[-1]*tmp.shape[-2]))
    Fprobec = np.roll(Fprobe, [-int(round(xc)), -int(round(yc))], axis=[0, 1])
    rrf = np.sqrt(np.tile((np.array(range(1,nx+1)) - nx/2)**2, (len(range(1, nx+1)), 1)) + ((np.array(range(1,nx+1)) - nx/2)**2)[:, np.newaxis])
    rmsk = (rrf < nx*3/8) * erf((nx*3/8-rrf)/5)**2

    probe = rmsk*(np.fft.ifft2(Fprobec) * np.sqrt(Fprobec.shape[-1]*Fprobec.shape[-2]))

    return probe.flatten()

# Generate psi0
def gen_psi0(Nx, Ny, image):
    psi0 = padmat(image, [Ny, Nx], np.mean(image))
    msk = padmat(image*0 + 1, [Ny, Nx], 0.0)
    msk = distance_transform_edt(msk)
    msk = erf(msk/10)
    psi0 = psi0 * msk + psi0[0] * (1 - msk)
    return psi0.flatten()

def gen_Q(nx, ny, nframes, omega, mapidx):
    eps = (np.max(np.abs(omega))**2)*5e-2
    Q = spp.csc_matrix((np.tile(omega+eps, nframes), (np.arange(0, nx*ny*nframes), mapidx.flatten())))
    QHQ = Q.H.dot(Q)
    QHQinv = spp.diags(np.ones(QHQ.diagonal().shape) / QHQ.diagonal())
    return Q,QHQinv

# Helper functions
def corner2pos(frame_corner, Nx):
    xi = np.floor(frame_corner / Nx)
    yi = frame_corner - xi*Nx
    return xi, yi

def meshgrid(x, y):
    tx = np.tile(y[:, np.newaxis], (1, len(x)))
    ty = np.tile(x, (len(y), 1))

    return tx, ty

def spv(xx, ixx):
    return ixx.reshape(ixx.shape + (1,) * xx.ndim) + xx

def stv(xx, ixx):
    return ixx.reshape(ixx.shape + (1,) * xx.ndim) * xx

def idx2corner(ix, iy, Nx, Ny):
    frac_shiftx = (ix - np.floor(ix))
    ix = np.floor(ix)

    frac_shifty = iy - np.floor(iy)
    iy = np.floor(iy)

    # TODO: Check that this works in general.
    # (Different sizes)
    frac_shift = np.array([frac_shiftx, frac_shifty])

    wrap = np.vectorize(lambda x, bound: ((x-1) % bound) + 1)
    frame_corner = np.squeeze(wrap(ix, Nx) - 1 + (wrap(iy, Ny) - 1)*Nx)
    return (frame_corner,frac_shift)

def phaseramps(dx, dy, nx, ny):
    xramp = np.array(range(nx), dtype=np.float64) / nx*2*np.pi
    dx = np.array(dx)
    dy = np.array(dy)
    return np.outer(np.exp(1j * (xramp * dx.flatten())), (np.exp(1j * (xramp * dy.flatten())).T))

def padmat(x, size, value):
    assert(len(size) == len(x.shape) == 2)
    m = size[0]
    n = size[1]
    M, N = x.shape

    y = np.full(size, value, dtype=x.dtype)
    y[:M, :N] = x
    y = np.roll(y, [(m-M)//2, (n-N)//2], axis=[0, 1])
    return y


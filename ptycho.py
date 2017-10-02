import h5py
import numpy as np
import scipy.sparse as spp
import ops

import logging

fname = 'CoPd77827_20x20pts_50nmstps_400x10ms_R_RUN3.cxi'

hdf = h5py.File(fname, 'r+')

data = np.array(hdf['entry_1']['instrument_1']['detector_1']['data'])
background = np.array(hdf['entry_1']['image_1']['background'])
image = np.array(hdf['entry_1']['image_1']['data'])

energy = np.float64(hdf['entry_1']['instrument_1']['source_1']['energy']) * 6.24150934e18
ccdDist = np.float64(hdf['entry_1']['instrument_1']['detector_1']['distance']) * 1e3
ccdPx = np.float64(hdf['entry_1']['instrument_1']['detector_1']['x_pixel_size']) * 1e3

fProbe = np.array(hdf['entry_1']['instrument_1']['detector_1']['Probe'])
fprobe = np.array(hdf['entry_1']['image_1']['probe'])

c=299792458*1e9
h=4.135667516e-15

nframes,ny,nx = data.shape

# Subtract background, remove bad frames,
# and preprocess data.
background = np.fft.fftshift(background)
data -= background
data[0,:,:] = 0
data[18*20+10,:,:] = 0

# Compute center of mass
datam = np.mean(np.abs(data), axis=0)*nframes / (nframes-1)
xc = np.sum(np.sum(datam, axis=1)*range(nx)) / np.sum(datam)
yc = np.sum(np.sum(datam, axis=0)*range(nx)) / np.sum(datam)

datac = np.roll(data, [0, -int(round(xc)), -int(round(yc))], axis=[0, 1, 2])
a_data = np.real(np.sqrt(datac.astype(np.complex64)))

a_data[18*20 + 10, :, :] = 0

# The resolution? If you're interested.
lambd = h * c / energy
res = lambd * (ccdDist / (ccdPx*nx))

# Construct illumination and frame splitting
# matrix, the construct Q operator matrix.
omega = ops.gen_omega(nx, ny, xc, yc, fProbe)
mapidx, Nx, Ny = ops.gen_mapidx(nx, ny, nframes)
Q,QHQinv = ops.gen_Q(nx, ny, nframes, omega, mapidx)

# Initial guess
psi0 = ops.gen_psi0(Nx, Ny, image)
z = Q.dot(psi0)

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('ptycho')

# Construct F and Q operators with Indigo.
from indigo.backends import get_backend
b = get_backend('mkl')
F = b.KronI(nframes, b.FFT((nx, ny), dtype=np.complex64), name='F')

Qop = b.SpMatrix(Q, name='Q')
PQ = Qop * b.SpMatrix(QHQinv, name='QHQinv') * b.SpMatrix(Q.H, name='QH')

print(Q.shape)
print(QHQinv.shape)

FHA = F.H * b.SpMatrix(spp.diags(a_data.flatten()), name='A')
PQFHA = PQ * FHA

# import matplotlib.pyplot as plt
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

print('F (2st part): %s' % (F.dump(),))
print('PQFHA (2nd part): %s' % (PQFHA.dump(),))

import sys

iterations = 10
z_d = b.copy_array(z)
Fz_d = b.zero_array(z.shape, dtype=z.dtype)
Fz = Fz_d.to_host()
for i in range(iterations):
    print('Forward eval:')
    F.eval(Fz_d, z_d)
    print('Done forward eval')

    Fz_d.copy_to(Fz)
    Fz = Fz / np.abs(Fz)
    Fz_d.copy_from(Fz)

    print('Backward eval:')
    PQFHA.eval(z_d, Fz_d)
    print('Done backward eval.')
    '''
    z = tmp_d.to_host()
    psi = Q.H.dot(z)

    samples = z.reshape(nframes, nx, ny)
    sample = 23
    x = samples[sample]
    Fx = np.fft.fft2(x)
    ax1.matshow(np.abs(x))
    ax2.matshow(np.abs(Fx))
    ax3.matshow(np.abs(psi.reshape(Nx, Ny)))
    ax4.matshow(np.abs(np.fft.fft2(psi.reshape(Nx, Ny))))
    '''

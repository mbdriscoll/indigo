import h5py
import numpy as np
import scipy.sparse as spp
import ops

def center(x, isFFT=False):
    if not isFFT:
        x = np.fft.fft2(x)
    x = np.fft.fftshift(x)
    if not isFFT:
        x = np.fft.ifft2(x)
    return x

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
b = get_backend('customgpu')

F = b.KronI(nframes, b.FFT((nx, ny), dtype=np.complex64), name='F')
A = b.SpMatrix(spp.diags(a_data.flatten(), dtype=np.complex64), name='A')

# Operator to project z onto A
PA = F.H * A * F.norm

QOp = b.SpMatrix(Q, name='Q')
QHQinvOp = b.SpMatrix(QHQinv, name='QHQinv')

PQ = QOp * QHQinvOp * QOp.H

PQPA = PQ * PA

from indigo.transforms import Transform, Visitor, LiftUnscaledFFTs, DistributeKroniOverProd
from indigo.operators import Product, Adjoint, SpMatrix, KronI

class DistributeAdjoint(Transform):
    def visit_Adjoint(self, node):
        if isinstance(node.child, KronI):
            return self.visit(b.KronI(node.child._c, node.child.child.H, name=node.child._name))
        elif isinstance(node.child, Product):
            l, r = node.child.children
            return self.visit(r.H) * self.visit(l.H)
        else:
            return node

class RealizeAdjoints(Transform):
    def visit_Adjoint(self, node):
        print('Visiting adjoint... %s' % (node._name))
        if isinstance(node.child, SpMatrix):
            print('Going to try to return SpMatrix')
            return b.SpMatrix(node.child._matrix.H, name=node.child._name + '.H')
        return node

class MakeRightLeaning(Transform):
    def visit_Product(self, node):
        l = self.visit(node.left_child)
        r = self.visit(node.right_child)
        if isinstance(l, Product):
            ll = l.left_child
            lr = l.right_child
            return self.visit(ll*(lr*r))
        else:
            return l*r

class RotateRealizeScaleA(Transform):
    def visit_Product(self, node):
        if node._name == '*A*F.norm':
            return ((node.left * node.right.left).realize() * node.right.right)
        return self.generic_visit(node)

class RealizeQQHQinv(Transform):
    def visit_Product(self, node):
        if node._name == 'Q*QHQinv*Q.H***A*F.norm':
            return (node.left * node.right.left).realize() * node.right.right
        return self.generic_visit(node)

recipe = [DistributeKroniOverProd, LiftUnscaledFFTs, DistributeAdjoint,
          MakeRightLeaning, RotateRealizeScaleA, RealizeAdjoints,
          RealizeQQHQinv]

print('Operator tree: %s' % (PQPA.dump(),))
PQPA = PQPA.optimize(recipe)
print('%s after optimizing.' % (PQPA.dump(),))

import matplotlib.pyplot as plt
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

plt.ion()
plt.show()

iterations = 10

z_d = b.copy_array(z)
new_z_d = b.zero_array(z.shape, dtype=z.dtype)

import time
begin = time.time()
for i in range(iterations):
    print('EVAL\'ing')
    PQPA.eval(new_z_d, z_d)
    print('DONE EVAL\'ing')

    tmp = z_d
    z_d = new_z_d
    new_z_d = tmp

    z = z_d.to_host()
    psi = Q.H.dot(z)

    samples = z.reshape(nframes, nx, ny)
    sample = 23
    x = samples[sample]
    Fx = np.fft.fft2(x)
    ax1.matshow(np.abs(center(x)))
    ax2.matshow(np.abs(center(Fx, isFFT=True)))
    ax3.matshow(np.abs(center(psi.reshape(Nx, Ny))))
    ax4.matshow(np.abs(center(np.fft.fft2(psi.reshape(Nx, Ny)), isFFT=True)))
    plt.pause(0.01)

end = time.time()
print('%s s for %s iterations (%s/iter)' % (end - begin, iterations, (end - begin) / iterations))

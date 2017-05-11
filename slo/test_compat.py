import pytest
import numpy as np
import scipy.sparse as spp
import numpy.testing as npt
from itertools import product

import slo
from slo.backends import available_backends
BACKENDS = available_backends()


@pytest.mark.parametrize("backend,N,K",
    product( BACKENDS, [23,45], [1,2,3] ) )
def test_compat_Multiply(backend, N, K):
    pymr = pytest.importorskip('pymr')
    b = backend()
    x = slo.util.rand64c(N,K)
    d = slo.util.rand64c(N,1)

    D0 = pymr.linop.Multiply( (N,K), (N,K), d, dtype=d.dtype )
    D1 = b.Diag(d)

    y_exp = pymr.util.reshape(D0 * pymr.util.vec(x), (N,K))
    y_act = D1 * x

    npt.assert_allclose(y_act, y_exp, rtol=1e-5)


@pytest.mark.parametrize("backend,X,Y,Z,P,K",
    product( BACKENDS, [23,32], [23,32], [23,32], [1,2,3], [1,2,3] ) )
def test_compat_Zpad(backend, X,Y,Z, P, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    i_shape = (X, Y, Z, K)
    o_shape = (X+2*P, Y+2*P, Z+2*P, K)

    x = slo.util.rand64c( *i_shape )

    D0 = pymr.linop.Zpad( o_shape, i_shape, dtype=x.dtype )
    D1 = b.Zpad(o_shape[:3], i_shape[:3], dtype=x.dtype)

    x_slo = np.asfortranarray(x.reshape((-1,K), order='F'))
    x_pmr = pymr.util.vec(x)
    y_exp = D0 * x_pmr
    y_act = D1 * x_slo

    y_act = y_act.flatten(order='F')
    npt.assert_equal(y_act, y_exp)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_UnscaledFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), normalize=False, center=False, dtype=np.dtype('complex64'))
    D1 = b.UnscaledFFT(shape[:3], dtype=np.dtype('complex64'))

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(y_act, y_exp, rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_UnitaryFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), center=False, dtype=np.dtype('complex64'))
    S, F = b.FFT(shape[:3], dtype=np.dtype('complex64'))
    D1 = S * F

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(y_act, y_exp, rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,K",
    product( BACKENDS, [23,44], [23,44], [23,44], [1,2,3] ) )
def test_compat_CenteredFFT(backend, X,Y,Z, K):
    pymr = pytest.importorskip('pymr')
    b = backend()

    shape = (X, Y, Z, K)
    x = slo.util.rand64c( *shape )

    D0 = pymr.linop.FFT(shape, axes=(0,1,2), dtype=np.dtype('complex64'))
    Mk, S, F, Mx = b.FFTc(shape[:3], dtype=np.dtype('complex64'))
    D1 = Mk * S * F * Mx

    y_exp = D0 * pymr.util.vec(x)
    y_act = D1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose(abs(y_act), abs(y_exp), rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,RO,PS,K,oversamp,n,width",
    product( BACKENDS, [23,44], [23,44], [23,44], [33,34], [35,36], [1,2], [1.375, 1.43], [64], [3] ) )
def test_compat_NUFFT(backend, X, Y, Z, RO, PS, K, oversamp, n, width):
    pymr = pytest.importorskip('pymr')
    b = backend()

    c_dims  = (X, Y, Z, K)
    nc_dims = (1, RO, PS, K)
    t_dims  = (3, RO, PS)

    x = slo.util.rand64c(*c_dims)
    traj = slo.util.rand64c( *t_dims ).real - 0.5
    kwargs = dict(oversamp=oversamp, width=width, n=n, dtype=x.dtype)

    print(nc_dims, c_dims, traj.shape)
    G0 = pymr.linop.NUFFT(nc_dims, c_dims, traj, **kwargs)

    G_t, Mk, S, F, Mx, Z, R = b.NUFFT(nc_dims[:3], c_dims[:3], traj, **kwargs)
    G1 = G_t * Mk * S * F * Mx * Z * R

    x_slo = np.asfortranarray(x.reshape((-1,K), order='F'))
    x_pmr = pymr.util.vec(x)
    y_exp = G0 * x_pmr
    y_act = G1 * x_slo

    y_act = y_act.reshape(-1, order='F')
    npt.assert_allclose(abs(y_act), abs(y_exp), rtol=1e-2)


@pytest.mark.parametrize("backend,X,Y,Z,RO,PS,K,n,width",
    product( BACKENDS, [23,44], [23,44], [23,44], [33,34], [35,36], [1,2], [64], [3] ) )
def test_compat_Interp(backend, X, Y, Z, RO, PS, K, n, width):
    pymr = pytest.importorskip('pymr')
    b = backend()

    N = (X, Y, Z, K)
    M = (1, RO, PS, K)
    T = (3, RO, PS)

    import scipy.signal as signal
    beta = 0.1234
    kb = signal.kaiser(2 * n + 1, beta)[n:]

    x = slo.util.rand64c(*N)
    traj = slo.util.rand64c(*T).real - 0.5

    G0 = pymr.linop.Interp(M, N, traj, width, kb, dtype=x.dtype)
    y_exp = G0 * pymr.util.vec(x)

    G1 = b.Interp(N[:3], traj, width, kb, dtype=x.dtype)
    y_act = G1 * x.reshape((-1,K), order='F')

    y_act = y_act.flatten(order='F')
    npt.assert_allclose( y_act, y_exp, rtol=1e-2)


@pytest.mark.parametrize("backend,forward,os",
    product( BACKENDS, [True,False], [1.375,1.43] ))
def test_compat_SENSE(backend, forward, os):
    pymr = pytest.importorskip('pymr')
    b = backend()

    X, Y, Z = 12, 13, 14
    RO, PS, C, T = 15, 16, 3, 1

    img = slo.util.rand64c(X,  Y,  Z, 1, T)
    mps = slo.util.rand64c(X,  Y,  Z, C, 1)
    ksp = slo.util.rand64c(1, RO, PS, C, T)
    dcf = slo.util.rand64c(1, RO, PS, 1, T).real
    coord=slo.util.rand64c(3, RO, PS, 1, T).real - 0.5

    # pymr
    P = pymr.linop.Multiply(ksp.shape, ksp.shape, dcf, dtype=img.dtype)
    F = pymr.linop.NUFFT(ksp.shape, mps.shape,  coord, dtype=img.dtype)
    S = pymr.linop.Multiply(mps.shape, img.shape, mps, dtype=img.dtype)
    A_pmr = P * F * S

    # slo
    G, Mk, U, F1, Mx, Z, R = b.NUFFT(ksp.shape[:3], mps.shape[:3], coord[...,0,0], dtype=img.dtype)
    P1 = b.Diag( dcf, name='dcf' )

    S = b.VStack([ Mx * Z * R * b.Diag(mps[:,:,:,c]) for c in range(C) ])
    F = b.KronI(C, F1, name='fft')
    P = b.KronI(C, P1 * G * Mk * U)
    A_slo = P * F * S

    # check
    if forward:
        exp = A_pmr * pymr.util.vec(img)
        act = A_slo * img.reshape( (-1,1), order='F' )
    else:
        exp = A_pmr.H * pymr.util.vec(ksp)
        act = A_slo.H * ksp.reshape( (-1,1), order='F' )

    act = act.flatten(order='F')
    npt.assert_allclose(abs(act), abs(exp), rtol=1e-2)

@pytest.mark.parametrize("backend,N",
    product( BACKENDS, [120,130,140] ))
def test_compat_conjgrad(backend, N):
    pymr = pytest.importorskip('pymr')
    b = backend()

    A = slo.util.randM( N, N, 0.5 )
    A = A.H @ A # make positive definite
    y = slo.util.rand64c( N )
    x0 = np.zeros( N, dtype=np.complex64 )

    A_pmr = pymr.linop.Matrix( A.toarray(), dtype=A.dtype )
    x_exp = pymr.alg.cg(A_pmr, A.H * y, x0, maxiter=40)

    A_slo = b.SpMatrix(A)
    b.cg(A_slo, y, x0, maxiter=40)
    x_act = x0.copy()

    npt.assert_allclose(x_act, x_exp, rtol=1e-6)


@pytest.mark.parametrize("oversamp,width,beta,X,Y,Z",
    product( [1.2,3.4], [3,4], [.1,1.1], [12,13], [14,15], [16,17] ))
def test_compat_rolloff(oversamp, width, beta, X, Y, Z):
    pymr = pytest.importorskip('pymr')
    N = (X,Y,Z)

    from pymr.noncart import rolloff3 as pymr_rolloff3
    from slo.noncart import rolloff3 as slo_rolloff3

    r_exp = pymr_rolloff3(oversamp, width, beta, N)
    r_act =  slo_rolloff3(oversamp, width, beta, N)
    npt.assert_allclose(r_exp, r_act)


@pytest.mark.parametrize("backend,M,N,alpha",
    product( BACKENDS, [120,130,140], [111,222,333], [1e-2, 1e-3] ))
def test_compat_apgd(backend, M, N, alpha):
    pymr = pytest.importorskip('pymr')
    b = backend()

    A = slo.util.randM( M, N, 0.5 )
    AHA = A.H @ A # make positive definite
    y = slo.util.rand64c( M )
    AHy = A.H * y
    x0 = np.zeros( N, dtype=np.complex64 )

    A_pmr = pymr.linop.Matrix( AHA.toarray(), dtype=A.dtype )

    # check pymr
    def gradf_pmr(x):
        return A_pmr * x - AHy

    def proxg_pmr(alpha, x0, x1):
        pass

    x_exp = pymr.alg.gd(gradf_pmr, alpha, x0, proxg=proxg_pmr, maxiter=40)

    # check slo
    x0[:] = 0
    A_slo = b.SpMatrix(AHA)
    AHy_d = b.copy_array(AHy)

    def gradf_slo(gf, x):
        # gf = AHA*x - AHy
        A_slo.eval(gf, x)
        b.axpy(gf, -1, AHy_d)

    def proxg_slo(alpha, x):
        # x = thresh(x)
        pass

    b.apgd(gradf_slo, proxg_slo, alpha, x0, maxiter=40)
    x_act = x0.copy()

    npt.assert_allclose(x_act, x_exp)

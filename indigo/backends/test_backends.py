import pytest
import logging
import numpy as np
import scipy.sparse as spp
from itertools import product

log = logging.getLogger(__name__)

import indigo.util
from indigo.backends import available_backends, get_backend
BACKENDS = available_backends()

@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_init(backend, n):
    b = backend()
    arr = indigo.util.rand64c(n)
    d_arr = b.copy_array(arr)
    arr2 = d_arr.to_host()
    np.testing.assert_equal(arr, arr2)


@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy_from(backend, n):
    b = backend()
    arr = indigo.util.rand64c(n)
    d_arr = b.zero_array(arr.shape, arr.dtype)
    d_arr.copy_from(arr)
    arr2 = d_arr.to_host()
    np.testing.assert_equal(arr, arr2)


@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy_to(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.copy_array(arr)
    arr2 = np.zeros_like(arr)
    d_arr.copy_to(arr2)
    np.testing.assert_equal(arr, arr2)


@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.copy_array(arr)
    d_arr2 = d_arr.copy()
    d_arr._zero()
    arr2 = d_arr2.to_host()
    np.testing.assert_equal(arr, arr2)

@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy_into(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.copy_array(arr)
    d_arr2 = b.zero_array(arr.shape, arr.dtype)
    d_arr2.copy(d_arr)
    arr2 = d_arr2.to_host()
    np.testing.assert_equal(arr, arr2)

@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_setitem(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.copy_array(arr)
    d_arr2 = b.zero_array(arr.shape, arr.dtype)
    d_arr2[:] = d_arr
    arr2 = d_arr2.to_host()
    np.testing.assert_equal(arr, arr2)


@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy_from_size_mismatch(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.zero_array((n+1,), arr.dtype)
    with pytest.raises(ValueError):
        d_arr.copy_from(arr)


@pytest.mark.parametrize("backend,n", product( BACKENDS, [4,8,129] ))
def test_array_copy_from_dtype_mismatch(backend, n):
    b = backend()
    arr = np.random.rand(n).astype(np.dtype('complex64'))
    d_arr = b.zero_array(arr.shape, np.complex128)
    with pytest.raises(TypeError):
        d_arr.copy_from(arr)


@pytest.mark.parametrize("backend,N,s",
    product( BACKENDS, [23,45], [-2,-1,1,2] )
)
def test_array_slice_1d_front(backend, N, s):
    b = backend()
    arr = np.arange(10)
    d_arr = b.copy_array(arr)
    d_arr_2 = d_arr[:s]
    arr_2 = d_arr_2.to_host()
    np.testing.assert_equal( arr_2, arr[:s] )


@pytest.mark.parametrize("backend,N,s",
    product( BACKENDS, [23,45], [-2,-1,1,2] )
)
def test_array_slice_1d_back(backend, N, s):
    b = backend()
    arr = np.arange(10)
    d_arr = b.copy_array(arr)
    d_arr_2 = d_arr[s:]
    arr_2 = d_arr_2.to_host()
    np.testing.assert_equal( arr_2, arr[s:] )

@pytest.mark.parametrize("backend,M,N,xb,xe,yb,ye",
    product( BACKENDS, [6,7], [8,9],
        [0,1,2],[4,5], [0,1,2], [6,7] )
)
def test_array_slice_2d(backend, M,N,xb,xe,yb,ye):
    b = backend()
    arr = indigo.util.rand64c(M,N)
    d_arr = b.copy_array(arr)
    d_arr_2 = d_arr[xb:xe,yb:ye]
    arr_2 = d_arr_2.to_host()
    np.testing.assert_equal( arr_2, arr[xb:xe,yb:ye] )


@pytest.mark.parametrize("backend,M,N,xb,xe,yb,ye",
    product( BACKENDS, [6,7], [8,9],
        [0,1,2],[4,5], [0,1,2], [6,7] )
)
def test_array_slice_copy_2d(backend, M,N,xb,xe,yb,ye):
    b = backend()
    arr = indigo.util.rand64c(M,N)
    exp = arr[xb:xe,yb:ye]

    d_arr = b.copy_array(arr)
    sub = d_arr[xb:xe,yb:ye]
    sub2 = b.zeros_like(sub)
    sub2.copy(sub)
    act = sub2.to_host()

    np.testing.assert_equal( exp, act )

@pytest.mark.parametrize("backend", BACKENDS )
def test_array_bad_slice0(backend):
    b = backend()
    arr = indigo.util.rand64c(8,4)
    d_arr = b.copy_array(arr)
    slc   = d_arr[2:6,:]
    with pytest.raises(AssertionError):
        slc.reshape((8,2))

@pytest.mark.parametrize("backend,batch,x,y,z",
    product( BACKENDS, [1,2,4,8], [23,24,25], [23,24,25], [23,24,25] )
)
def test_fft(backend, batch, x, y, z):
    b = backend()
    N = (z, y, x, batch)
    v = np.random.rand(*N) + 1j*np.random.rand(*N)
    v = np.require(v, dtype=np.dtype('complex64'), requirements='F')
    ax = (0,1,2)
    
    # check forward
    w_exp = np.fft.fftn(v, axes=ax)
    v_d = b.copy_array(v)
    u_d = b.copy_array(v)
    b.fftn(u_d, v_d)
    w_act = u_d.to_host()
    np.testing.assert_allclose(w_act, w_exp, atol=1e-2)

    # check adjoint
    v_exp = np.fft.ifftn(w_act, axes=ax) * (x*y*z)
    v_d = b.copy_array(w_act)
    u_d = b.copy_array(w_act)
    b.ifftn(u_d, v_d)
    v_act = u_d.to_host()
    np.testing.assert_allclose(v_act, v_exp, atol=1e-2)

    # check unitary
    np.testing.assert_allclose(v, v_act / (x*y*z), atol=1e-6)


@pytest.mark.parametrize("backend,M,N,K,density",
    product( BACKENDS, [23,45], [45,23], [1,8,9,17], [0.01,0.1,0.5] )
)
def test_csr_matrix(backend, M, N, K, density):
    b = backend()
    c = np.dtype('complex64')
    A = indigo.util.randM(M, N, density)
    A_d = b.csr_matrix(b, A)

    # forward
    x = (np.random.rand(N,K) + 1j * np.random.rand(N,K))
    x = np.require(x, dtype=c, requirements='F')
    y_exp = A.astype(c) * x
    x_d = b.copy_array(x)
    y_d = b.zero_array(y_exp.shape, x.dtype)
    A_d.forward(y_d, x_d)
    y_act = y_d.to_host()
    np.testing.assert_allclose(y_exp, y_act, atol=1e-5)

    # adjoint
    x = (np.random.rand(M,K) + 1j * np.random.rand(M,K))
    x = np.require(x, dtype=c, requirements='C')
    y_exp = A.H.astype(c) * x
    x_d = b.copy_array(x)
    y_d = b.zero_array(y_exp.shape, x.dtype)
    A_d.adjoint(y_d, x_d)
    y_act = y_d.to_host()
    np.testing.assert_allclose(y_exp, y_act, atol=1e-5)


@pytest.mark.parametrize("backend,M,N,K,alpha,beta,stack",
    product( BACKENDS, [23,45], [1,8,9,17], [18,19], [0.0,0.5,1.0,1.5], [0.0,0.5,1.0,1.5], [1,2,5] )
)
def test_exw_csr_matrix(backend, M, N, K, alpha, beta, stack):
    b = backend()
    rowCounts = np.random.randint(0,2,K)
    rowPtrs = np.concatenate( [np.array([0]), np.cumsum(rowCounts)] )
    colInds = np.random.randint(0,M,sum(rowCounts))
    values = indigo.util.rand64c(*colInds.shape)
    A = spp.csr_matrix( (values, colInds, rowPtrs), shape=(K,M) ).T
    A_d = b.csr_matrix(b, A)

    # forward
    x = indigo.util.rand64c(K,N)
    y = indigo.util.rand64c(M,N)
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)
    A_d.forward(y_d, x_d, alpha=alpha, beta=beta)
    y_act = y_d.to_host()
    y_exp = beta * y + alpha * (A @ x)
    np.testing.assert_allclose(y_exp, y_act, atol=1e-3)

    # adjoint
    x = indigo.util.rand64c(M,N)
    y = indigo.util.rand64c(K,N)
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)
    A_d.adjoint(y_d, x_d, alpha=alpha, beta=beta)
    y_act = y_d.to_host()
    y_exp = beta * y + alpha * (A.getH() @ x)
    np.testing.assert_allclose(y_exp, y_act, atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_op_dump(backend):
    b = backend()
    M, N, K, density = 22, 33, 44, 0.50
    A0 = indigo.util.randM(M, N, density)
    A = b.KronI( 4, b.SpMatrix(A0, name='Leaf'), name='Branches' )
    B = A.H * A; B._name = 'Trunk'
    tree = B.dump()
    assert 'Leaf' in tree
    assert 'Trunk' in tree
    assert 'Branches' in tree


@pytest.mark.parametrize("backend,alpha,n,alpha_i",
    product(BACKENDS, [-1.1, -1, -0.1, 0, 0.1, 1, 1.1], [10,23,129], [-2,0,3])
)
def test_blas_scale(backend, alpha, n, alpha_i):
    b = backend()
    x = (np.random.rand(n) + 1j * np.random.rand(n))
    x = np.require(x, dtype=np.dtype('complex64'), requirements='F')
    x_d = b.copy_array(x)

    alpha = alpha + 1j*alpha_i

    y_exp = x * alpha
    b.scale(x_d, alpha)
    y_act = x_d.to_host()

    np.testing.assert_allclose(y_exp, y_act, atol=1e-5)


@pytest.mark.parametrize("backend,device,n",
    product(BACKENDS, [0], [10,23,129])
)
def test_blas_nrm2(backend, device, n):
    b = backend(device)

    x = (np.random.rand(n) + 1j * np.random.rand(n))
    x = np.require(x, dtype=np.dtype('complex64'), requirements='F')
    x_d = b.copy_array(x)

    y_exp = np.linalg.norm(x) ** 2
    y_act = b.norm2(x_d)

    np.testing.assert_allclose(y_exp, y_act, atol=1e-4)


@pytest.mark.parametrize("backend,n",
    product(BACKENDS, [10,23,129,144])
)
def test_blas_dot(backend, n):
    b = backend()
    x = (np.random.rand(n) + 1j * np.random.rand(n))
    y = (np.random.rand(n) + 1j * np.random.rand(n))
    x = np.require(x, dtype=np.dtype('complex64'), requirements='F')
    y = np.require(y, dtype=np.dtype('complex64'), requirements='F')
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)

    y_exp = np.vdot(x, y).real
    y_act = b.dot(x_d, y_d)

    np.testing.assert_allclose(y_exp, y_act, atol=1e-5)


@pytest.mark.parametrize("backend,n,alpha,alpha_i,beta",
    product(BACKENDS, [10,23,129,144], [-2.1, -1.0, -0.1, 0.0, 0.1, 1.0, 1.2], [0,3],[0.0,0.5,1.0,1.5])
)
def test_blas_axpby(backend, n, alpha, alpha_i, beta):
    b = backend()
    x = (np.random.rand(n) + 1j * np.random.rand(n))
    y = (np.random.rand(n) + 1j * np.random.rand(n))
    x = np.require(x, dtype=np.dtype('complex64'), requirements='F')
    y = np.require(y, dtype=np.dtype('complex64'), requirements='F')
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)

    alpha = alpha + 1j*alpha_i

    y_exp = beta*y + alpha*x
    b.axpby(beta, y_d, alpha, x_d)

    y_act = y_d.to_host()

    np.testing.assert_allclose(y_exp, y_act, atol=1e-6)


@pytest.mark.parametrize("backend,m,n,k,alpha,beta,forward",
    product(BACKENDS, [10,23,129,144],
                      [10,23,129,144],
                      [10,23,129,144],
                      [1,0.5,0.0],
                      [0,0.5,0.0],
                      [True, False])
)
def test_blas_cgemm(backend, m, n, k, alpha, beta, forward):
    b = backend()

    y = indigo.util.rand64c(m,n)
    M = indigo.util.rand64c(m,k)
    x = indigo.util.rand64c(k,n)

    if not forward:
        x, y = y, x
        M_exp = np.conj(M.T)
    else:
        M_exp = M
    y_exp = alpha * M_exp.dot(x) + beta * y

    y_d = b.copy_array(y)
    M_d = b.copy_array(M)
    x_d = b.copy_array(x)
    b.cgemm(y_d, M_d, x_d, alpha, beta, forward=forward)
    y_act = y_d.to_host()

    np.testing.assert_allclose(y_exp, y_act, atol=1e-3)



@pytest.mark.parametrize("backend,m",
    product(BACKENDS, [10,23,129,144])
)
def test_iter_cg(backend, m):
    b = backend()
    x = indigo.util.rand64c(m,1)
    y = indigo.util.rand64c(m,1)

    M = b.Eye(m)

    b.cg(M, y, x, maxiter=2)


@pytest.mark.parametrize("backend,m",
    product(BACKENDS, [10,23,129,144])
)
def test_iter_apgd(backend, m):
    b = backend()

    M = b.Eye(m)
    x = indigo.util.rand64c(m,1)
    y = indigo.util.rand64c(m,1)

    x_d = b.copy_array(x)
    y_d = b.copy_array(y)

    def gradf(gf, x_d):
        M.eval(gf, x_d)
        b.axpby(1, gf, -1, y_d)

    def proxg(alpha, x):
        return x

    b.apgd(gradf, proxg, 1.0, x, maxiter=2)


@pytest.mark.parametrize("bname", ['numpy', 'mkl', 'customcpu'])
def test_get_backend(bname):
    b = get_backend(bname)

@pytest.mark.xfail
def test_bad_backend():
    b = get_backend('quantumcomputer')

@pytest.mark.parametrize("backend", BACKENDS)
def test_mem_usage(backend):
    b = backend()
    b.mem_usage()


@pytest.mark.parametrize("backend", BACKENDS)
def test_dndarray_on_host(backend):
    b = backend()
    arr = np.zeros(11, dtype=np.complex64)
    arr_d = b.copy_array(arr)

    with arr_d.on_host() as arr_h:
        arr_h += 1

    arr_h2 = arr_d.to_host()
    assert np.sum(arr_h2) == 11

@pytest.mark.parametrize("backend,val,N", 
    product(BACKENDS, [-1.5,-1,-0.5,0,0.5,1,1.5], [4,5,6])
)
def test_max(backend, val, N):
    b = backend()
    if hasattr(b, 'max'):
        arr = indigo.util.rand64c(N)
        arr_d = b.copy_array(arr)
        b.max(val, arr_d)
        arr_act = arr_d.to_host()
        np.testing.assert_allclose( np.maximum(arr.real, val), arr_act.real )
        np.testing.assert_allclose( np.maximum(arr.imag, val), arr_act.imag )


@pytest.mark.parametrize("backend,M,K,N,alpha,beta,maxoffsets",
    product( BACKENDS, [23,45], [45,23], [1,8,9,17], [0,0.5,1.0,1.5], [0,0.5,1.0,1.5], [1,2,3,4] )
)
def test_dia_matrix(backend, M, K, N, alpha, beta, maxoffsets):
    b = backend()
    if getattr(b.cdiamm, '__isabstractmethod__', False):
        pytest.skip("backed <%s> doesn't implement cdiamm" % backend.__name__)
    c = np.dtype('complex64')
    offsets = np.array(list(set(np.random.randint(-K, M+K, size=maxoffsets))))
    data = np.random.rand(offsets.size, K) + 1j * np.random.rand(offsets.size, K)
    data = data.astype(c)
    A = spp.dia_matrix((data, offsets), shape=(M,K))
    A_d = b.dia_matrix(b, A)

    # forward
    x = (np.random.rand(K,N) + 1j * np.random.rand(K,N))
    y = (np.random.rand(M,N) + 1j * np.random.rand(M,N))
    x = np.require(x, dtype=c, requirements='F')
    y = np.require(y, dtype=c, requirements='F')
    y_exp = beta * y + alpha * (A @ x)

    x_d = b.copy_array(x)
    y_d = b.copy_array(y)
    A_d.forward(y_d, x_d, alpha=alpha, beta=beta)
    y_act = y_d.to_host()
    np.testing.assert_allclose(y_act, y_exp, atol=1e-5)

    x = (np.random.rand(K,N) + 1j * np.random.rand(K,N))
    y = (np.random.rand(M,N) + 1j * np.random.rand(M,N))
    x = np.require(x, dtype=c, requirements='F')
    y = np.require(y, dtype=c, requirements='F')
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)
    x_exp = beta * x + alpha * (A.getH() @ y)
    A_d.adjoint(x_d, y_d, alpha=alpha, beta=beta)
    x_act = x_d.to_host()
    np.testing.assert_allclose(x_act, x_exp, atol=1e-5)


@pytest.mark.parametrize("backend,dtype",
    product(BACKENDS, [np.complex128, np.int32, np.float32, np.float64])
)
def test_only_complex64(backend, dtype):
    b = backend()
    M, N, K, density = 22, 33, 44, 0.50
    A0 = indigo.util.randM(M, N, density).astype(dtype)
    x = b.copy_array( indigo.util.rand64c(N,K).astype(np.complex128) )
    y = b.copy_array( indigo.util.rand64c(M,K).astype(np.complex128) )
    with pytest.raises(AssertionError):
        A = b.SpMatrix(A0).eval(y,x)


@pytest.mark.parametrize("backend,m,k,alpha,beta,forward,left",
    product(BACKENDS, [2,4,5,6], [1,2,3],[0.0,0.5,1.0,1.5],[0.0,0.5,1.0,1.5],[True,False],[True,False])
)
def test_csymm(backend, m, k, alpha, beta, forward, left):
    b = backend()

    data = indigo.util.rand64c(m,m)
    data = np.asfortranarray(data + data.T) # make symmetric
    data.imag = 0 # make real
    M = data

    if left:
        x = indigo.util.rand64c(m,k)
        y = indigo.util.rand64c(m,k)
    else:
        x = indigo.util.rand64c(k,m)
        y = indigo.util.rand64c(k,m)

    if left:
        y_exp = alpha * (M @ x) + beta * y
    else:
        y_exp = alpha * (x @ M) + beta * y

    M_d = b.copy_array(M)
    x_d = b.copy_array(x)
    y_d = b.copy_array(y)

    b.csymm(y_d, M_d, x_d, alpha, beta, left)

    y_act = y_d.to_host()
    np.testing.assert_allclose(y_exp, y_act, atol=1e-5)

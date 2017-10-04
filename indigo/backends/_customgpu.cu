#include <stdio.h>
#include <cublas.h>
#include <cuComplex.h>

//#include "_customgpu.h"

__global__
void cu_normalize(unsigned int N, cuFloatComplex *X) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N)
      return;
    X[n] = cuCdivf(X[n], make_cuFloatComplex(cuCabsf(X[n]), 0.0f));
}

__global__
void cu_max(unsigned int N, float val, float *arr) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N)
        return;
    arr[n] = fmaxf(arr[n], val);
}

__global__
void cu_onemm(
    unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *X, unsigned int ldx,
    cuFloatComplex beta, cuFloatComplex *Y, unsigned int ldy
) {
    // matrix is M-K, X is K-N, Y is M-N
    // strategy: accum columns of X, broadcast into columns of Y

    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N)
        return;

    cuFloatComplex acc = make_cuComplex(0,0);
    for (unsigned int k = 0; k < K; k++)
        acc = cuCaddf(acc, X[k+n*ldx]);
    for (unsigned int m = 0; m < M; m++)
        Y[m+n*ldy] = cuCaddf( cuCmulf(beta, Y[m+n*ldy]), cuCmulf(alpha, acc));
}

__global__
void cu_exw_csrmm_H(unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    cuFloatComplex *X, unsigned int ldx, cuFloatComplex beta,
    cuFloatComplex *Y, unsigned int ldy)
{
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M)
        return;

    extern __shared__ cuFloatComplex x[];

    unsigned int ptrb = rowPtrs[m],
                 ptre = rowPtrs[m+1];

    if (ptrb == ptre)
        return;

    #pragma unroll
    for (unsigned int n = 0; n < N; n++)
        x[n] = X[m+n*M];

    for (unsigned int idx = ptrb; idx < ptre; idx++) {
        unsigned int k = colInds[idx];
        cuFloatComplex v = cuCmulf(alpha, cuConjf(values[idx]));

        #pragma unroll
        for (unsigned int n = 0; n < N; n++)
            Y[k+n*K] = cuCmulf(v, x[n]);
    }
}

extern "C"
void c_normalize(unsigned int N, cuFloatComplex *X) {
    int tpb = 128;
    int nb = (N+tpb-1)/tpb;
    cu_normalize<<<nb,tpb>>>(N, X);
}

extern "C"
void c_max(unsigned int N, float val, float *arr) {
    int tpb = 128;
    int nb = (N+tpb-1)/tpb;
    cu_max<<<nb,tpb>>>(N, val, arr);
}

extern "C"
void c_onemm(
    unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *X, unsigned int ldx,
    cuFloatComplex beta, cuFloatComplex *Y, unsigned int ldy
) {
    int tpb = 128;
    int nb = (N+tpb-1)/tpb;
    cu_onemm<<<nb,tpb>>>(M, N, K, alpha, X, ldx, beta, Y, ldy);
}

extern "C"
void c_exw_csrmm_H(unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    cuFloatComplex *X, unsigned int ldx, cuFloatComplex beta,
    cuFloatComplex *Y, unsigned int ldy)
{
    // Y[:] *= beta
    if (cuCrealf(beta) == 0 && cuCimagf(beta) == 0)
        cudaMemset(Y, 0, K*N*sizeof(cuFloatComplex));
    else
        cublasCscal(K*N, beta, Y, 1);

    // Y[:] += alpha * AX
    int tpb = 128;
    int nb = (M+tpb-1)/tpb;
    int ns = N * sizeof(cuFloatComplex);
    cu_exw_csrmm_H<<<nb,tpb,ns>>>(M, N, K, alpha, values, colInds, rowPtrs, X, ldx, beta, Y, ldy);
}

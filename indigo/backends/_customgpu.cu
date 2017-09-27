#include <stdio.h>
#include <cublas.h>
#include <cuComplex.h>

//#include "_customgpu.h"

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
    extern __shared__ cuFloatComplex acc[];
    int tid = threadIdx.x,
        n = blockIdx.x;

    // per-thread accumulate down X
    cuFloatComplex acc_n = make_cuComplex(0,0);
    for (unsigned int k = tid; k < K; k += blockDim.x)
        acc_n = cuCaddf(acc_n, X[k+n*ldx]);
    acc[tid] = acc_n;
    __syncthreads();

    // reduce across block
    for (int i = blockDim.x/2; i > 0; i /= 2) {
        if (tid < i)
            acc[tid] = cuCaddf(acc[tid], acc[tid+i]);
        __syncthreads();
    }

    // broadcast across block
    acc_n = acc[0];

    // broadcast into Y
    for (unsigned int m = tid; m < M; m += blockDim.x)
        Y[m+n*ldy] = cuCaddf( cuCmulf(beta, Y[m+n*ldy]), cuCmulf(alpha, acc_n));
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
    int nb = N;
    int ns = tpb * sizeof(cuFloatComplex);
    cu_onemm<<<nb,tpb,ns>>>(M, N, K, alpha, X, ldx, beta, Y, ldy);
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

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

    unsigned int ptrb = rowPtrs[m],
                 ptre = rowPtrs[m+1];

    if (ptrb == ptre)
        return;

    extern __shared__ cuFloatComplex _x[];
    cuFloatComplex *x = _x + N*threadIdx.x;

    #pragma unroll
    for (int n = 0; n < N; n++)
        x[n] = cuCmulf(alpha, X[m+n*ldx]);

    for (unsigned int idx = ptrb; idx < ptre; idx++) {
        unsigned int k = colInds[idx];
        cuFloatComplex v = cuConjf(values[idx]);

        #pragma unroll
        for (unsigned int n = 0; n < N; n++) {
            Y[k+n*ldy] = cuCaddf( cuCmulf(beta,Y[k+n*ldy]), cuCmulf(v,x[n]) );
        }
    }
}

__global__
void cu_diamm(
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int nOffsets, int *offsets, cuFloatComplex *data,
    cuFloatComplex alpha, cuFloatComplex *X, unsigned int ldx,
    cuFloatComplex beta, cuFloatComplex *Y, unsigned int ldy
) {
    // Y is M-by-N
    // X is K-by-N
    // A is M-by-K
    // data is _-by-K

    int m = blockIdx.x*blockDim.x + threadIdx.x; // row
    if (m >= M)
        return;

    #pragma unroll
    for (int n = 0; n < N; n++)
        Y[m+n*ldy] = cuCmulf(beta, Y[m+n*ldy]);

    for (int d = 0; d < nOffsets; d++) {
        int k = m + offsets[d]; // col
        if (0 <= k && k < K) {
            cuFloatComplex v = cuCmulf(alpha, data[k+d*K]);
            #pragma unroll
            for (int n = 0; n < N; n++)
                Y[m+n*ldy] = cuCaddf(Y[m+n*ldy], cuCmulf(v, X[k+n*ldx]));
        }
    }
}

__global__
void cu_diammH(
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int nOffsets, int *offsets, cuFloatComplex *data,
    cuFloatComplex alpha, cuFloatComplex *X, unsigned int ldx,
    cuFloatComplex beta, cuFloatComplex *Y, unsigned int ldy
) {
    // Y is M-by-N
    // X is K-by-N
    // AH is M-by-K, A is K-by-M
    // data is _-by-M

    int m = blockIdx.x*blockDim.x + threadIdx.x; // row
    if (m >= M)
        return;

    #pragma unroll
    for (int n = 0; n < N; n++)
        Y[m+n*ldy] = cuCmulf(beta, Y[m+n*ldy]);

    for (int d = 0; d < nOffsets; d++) {
        int k = m - offsets[d]; // col
        if (0 <= k && k < K) {
            cuFloatComplex v = cuCmulf(alpha, cuConjf(data[m+d*M]));
            #pragma unroll
            for (int n = 0; n < N; n++)
                Y[m+n*ldy] = cuCaddf(Y[m+n*ldy], cuCmulf(v, X[k+n*ldx]));
        }
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
void c_diamm(
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int nOffsets, int *offsets, cuFloatComplex *data,
    cuFloatComplex alpha, cuFloatComplex *X, unsigned int ldx,
    cuFloatComplex beta, cuFloatComplex *Y, unsigned int ldy, int adjoint
) {
    int tpb = 128;
    int nb = (M + tpb - 1) / tpb;
    if (adjoint) {
        cu_diammH<<<nb,tpb>>>(M, N, K, nOffsets, offsets, data,
            alpha, X, ldx, beta, Y, ldy);
    } else {
        cu_diamm<<<nb,tpb>>>(M, N, K, nOffsets, offsets, data,
            alpha, X, ldx, beta, Y, ldy);
    }
}

extern "C"
void c_exw_csrmm_H(unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    cuFloatComplex *X, unsigned int ldx, cuFloatComplex beta,
    cuFloatComplex *Y, unsigned int ldy)
{
    // Y[:] = beta*Y + alpha*A*X
    int tpb = 128;
    int nb = (M+tpb-1)/tpb;
    int ns = N * tpb * sizeof(cuFloatComplex);
    cu_exw_csrmm_H<<<nb,tpb,ns>>>(M, N, K, alpha, values, colInds, rowPtrs, X, ldx, beta, Y, ldy);
}

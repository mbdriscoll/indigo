#include <stdio.h>
#include <cublas.h>
#include <cuComplex.h>

#include "_customgpu.h"

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

    for (unsigned int idx = rowPtrs[m]; idx < rowPtrs[m+1]; idx++) {
        unsigned int k = colInds[idx];
        cuFloatComplex v = cuCmulf(alpha, cuConjf(values[idx]));

        #pragma unroll
        for (unsigned int n = 0; n < N; n++)
            Y[k+n*K] = cuCmulf(v, X[m+n*M]);
    }
}

extern "C"
void c_exw_csrmm_H(unsigned int M, unsigned int N, unsigned int K,
    cuFloatComplex alpha, cuFloatComplex *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    cuFloatComplex *X, unsigned int ldx, cuFloatComplex beta,
    cuFloatComplex *Y, unsigned int ldy)
{
    // Y[:] *= beta
    cublasCscal(K*N, beta, Y, 1);

    // Y[:] += alpha * AX
    int tpb = 128;
    int nb = (M+tpb-1)/tpb;
    cu_exw_csrmm_H<<<nb,tpb>>>(M, N, K, alpha, values, colInds, rowPtrs, X, ldx, beta, Y, ldy);
}

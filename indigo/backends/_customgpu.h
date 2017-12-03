#include <complex.h>

void c_exw_csrmm_H(void *cublasHandle,
    unsigned int M, unsigned int K, unsigned int N,
    complex float alpha, complex float *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    complex float *X, unsigned int ldx, complex float beta,
    complex float *Y, unsigned int ldy);

void c_onemm(
    unsigned int M, unsigned int N, unsigned int K,
    complex float alpha, complex float *X, unsigned int ldx,
    complex float beta, complex float *Y, unsigned int ldy
);

void c_diamm(
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int nOffsets, int *offsets, complex float *data,
    complex float alpha, complex float *X, unsigned int ldx,
    complex float beta, complex float *Y, unsigned int ldy, int adjoint
);

void c_max(unsigned int N, float val, float *arr);

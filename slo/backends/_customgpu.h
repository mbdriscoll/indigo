#include <complex.h>

void c_exw_csrmm_H(unsigned int M, unsigned int K, unsigned int N,
    complex float alpha, complex float *values,
    unsigned int *colInds, unsigned int *rowPtrs,
    complex float *X, unsigned int ldx, complex float beta,
    complex float *Y, unsigned int ldy);

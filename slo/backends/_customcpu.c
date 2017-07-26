#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>


void custom_ccsrmm(
    int transA, int M, int N, int K, complex float alpha,
    complex float *val, int *col, int *pntrb, int *pntre,
    complex float *B, int ldb, complex float beta,
    complex float *C, int ldc
) { 
    if (transA) {
        #pragma omp parallel
        {
            #pragma omp for
            for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                C[k+n*ldc] *= beta;

            #pragma omp for schedule(guided)
            for (int m = 0; m < M; m++) {
                for (int i = pntrb[m]; i < pntre[m]; i++) {
                    int k = col[i];
                    complex float A_km = conjf(val[i]);
                    for (int n = 0; n < N; n++) {
                        complex float ans = alpha * A_km * B[m+n*ldb];
                        float *acc = (float*) &C[k+n*ldc];

                        #pragma omp atomic
                        acc[0] += crealf(ans);
                        
                        #pragma omp atomic
                        acc[1] += cimagf(ans);
                    }
                }
            }   
        }
    } else {
        #pragma omp parallel
        {
            complex float acc[N];

            #pragma omp for schedule(guided)
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++)
                    acc[n] = 0;
                for (int i = pntrb[m]; i < pntre[m]; i++) {
                    int k = col[i];
                    complex float v = val[i];
                    for (int n = 0; n < N; n++)
                        acc[n] += v * B[k+n*ldb];
                }
                for (int n = 0; n < N; n++)
                    C[m+n*ldc] = alpha * acc[n] + beta * C[m+n*ldc];
            }   
        }
    }
}


// --------------------------------------------------------------------------
// Python interface
// --------------------------------------------------------------------------

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include <numpy/arrayobject.h>

static PyObject*
py_ccsrmm(PyObject *self, PyObject *args)
{
    PyObject *py_alpha, *py_beta;
    int adjoint, ldx, ldy, M, N, K;
    PyArrayObject *py_Y, *py_colind, *py_rowptr, *py_vals, *py_X;
    if (!PyArg_ParseTuple(args, "piiiOOOOOiOOi",
        &adjoint, &M, &N, &K, &py_alpha,
        &py_vals, &py_colind, &py_rowptr,
        &py_X, &ldx, &py_beta, &py_Y, &ldy))
        return NULL;

    int *rowPtrs = PyArray_DATA(py_rowptr);
    int *colInds = PyArray_DATA(py_colind);
    complex float *values = PyArray_DATA(py_vals),
                  *Y = PyArray_DATA(py_Y),
                  *X = PyArray_DATA(py_X);

    float alpha_r = (float) PyComplex_RealAsDouble( py_alpha ),
          alpha_i = (float) PyComplex_ImagAsDouble( py_alpha ),
           beta_r = (float) PyComplex_RealAsDouble( py_beta  ),
           beta_i = (float) PyComplex_ImagAsDouble( py_beta  );
    complex float alpha = alpha_r + I * alpha_i,
                   beta =  beta_r + I *  beta_i;

    custom_ccsrmm(adjoint, M, N, K, alpha, values, colInds, &rowPtrs[0], &rowPtrs[1], X, ldx, beta, Y, ldy);

    Py_RETURN_NONE;
}

static PyMethodDef _customcpuMethods[] = {
    { "ccsrmm", py_ccsrmm, METH_VARARGS, NULL },
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef _customcpu = {
    PyModuleDef_HEAD_INIT, "_customcpu", NULL, -1, _customcpuMethods,
};

PyMODINIT_FUNC
PyInit__customcpu(void)
{
    import_array();
    return PyModule_Create(&_customcpu);
}

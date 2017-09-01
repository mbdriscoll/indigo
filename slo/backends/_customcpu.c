#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

extern void mkl_ccsrmv (const char *transa , const int *m , const int *k , const complex float *alpha , const char *matdescra , const complex float *val , const int *indx , const int *pntrb , const int *pntre , const complex float *x , const complex float *beta , complex float *y );

void custom_ccc_csrmm(
    unsigned int transA, unsigned int M, unsigned int N, unsigned int K, complex float alpha,
    complex float *val, unsigned int *col, unsigned int *pntrb, unsigned int *pntre,
    complex float *B, unsigned int ldb, complex float beta,
    complex float *C, unsigned int ldc, int exwrite
) { 
    if (transA && exwrite) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (unsigned int k = 0; k < K; k++) {
                #pragma unroll
                for (unsigned int n = 0; n < N; n++) {
                    C[k+n*ldc] *= beta;
                }
            }

             #pragma omp for schedule(static)
             for (unsigned int m = 0; m < M; m++) {
                for (unsigned int i = pntrb[m]; i < pntre[m]; i++) {
                    unsigned int k = col[i];
                    complex float v = alpha * conjf(val[i]);

                    #pragma unroll
                    for (unsigned int n = 0; n < N; n++)
                        C[k+n*ldc] += v * B[m+n*ldb];
                }
            }   
        }
    } else if (N==1 && transA==0) {

        char adj = (transA) ? 'C' : 'N';
        char *matdescr = "G NC  ";
        mkl_ccsrmv(&adj, (const int *) &M, (const int *) &K, &alpha, matdescr, val, (const int *) col, (const int *) pntrb, (const int *) pntre, B, &beta, C);

    } else if (transA) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (unsigned int k = 0; k < K; k++) {
                #pragma unroll
                for (unsigned int n = 0; n < N; n++) {
                    C[k+n*ldc] *= beta;
                }
            }

             #pragma omp for schedule(static)
             for (unsigned int m = 0; m < M; m++) {
                for (unsigned int i = pntrb[m]; i < pntre[m]; i++) {
                    unsigned int k = col[i];
                    complex float v = alpha * conjf(val[i]);

                    #pragma unroll
                    for (unsigned int n = 0; n < N; n++) {
                        complex float res = v * B[m+n*ldb];
                        float *out = (float*) &C[k+n*ldc];

                        #pragma omp atomic
                        out[0] += crealf(res);

                        #pragma omp atomic
                        out[1] += cimagf(res);
                    }
                }
            }   
        }
    } else {
        #pragma omp parallel
        {
            complex float * acc = malloc(N * sizeof(complex float));

            #pragma omp for schedule(static)
            for (unsigned int m = 0; m < M; m++) {
                #pragma unroll
                for (unsigned int n = 0; n < N; n++)
                    acc[n] = 0;
                for (unsigned int i = pntrb[m]; i < pntre[m]; i++) {
                    unsigned int k = col[i];
                    complex float v = val[i];
                    #pragma unroll
                    for (unsigned int n = 0; n < N; n++)
                        acc[n] += v * B[k+n*ldb];
                }
                #pragma unroll
                for (unsigned int n = 0; n < N; n++)
                    C[m+n*ldc] = alpha * acc[n] + beta * C[m+n*ldc];
            }   

            free(acc);
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
py_csrmm(PyObject *self, PyObject *args)
{
    PyObject *py_alpha, *py_beta;
    unsigned int adjoint, ldx, ldy, M, N, K, exw;
    PyArrayObject *py_Y, *py_colind, *py_rowptr, *py_vals, *py_X;
    if (!PyArg_ParseTuple(args, "piiiOOOOOiOOip",
        &adjoint, &M, &N, &K, &py_alpha,
        &py_vals, &py_colind, &py_rowptr,
        &py_X, &ldx, &py_beta, &py_Y, &ldy, &exw))
        return NULL;

    unsigned int *rowPtrs = PyArray_DATA(py_rowptr);
    unsigned int *colInds = PyArray_DATA(py_colind);
    void *values = PyArray_DATA(py_vals),
                  *Y = PyArray_DATA(py_Y),
                  *X = PyArray_DATA(py_X);

    float alpha_r = (float) PyComplex_RealAsDouble( py_alpha ),
          alpha_i = (float) PyComplex_ImagAsDouble( py_alpha ),
           beta_r = (float) PyComplex_RealAsDouble( py_beta  ),
           beta_i = (float) PyComplex_ImagAsDouble( py_beta  );
    complex float alpha = alpha_r + I * alpha_i,
                   beta =  beta_r + I *  beta_i;

    PyArray_Descr *descr = PyArray_DTYPE(py_vals);
    if ( PyDataType_ISCOMPLEX(descr) )
        custom_ccc_csrmm(adjoint, M, N, K, alpha, values, colInds, &rowPtrs[0], &rowPtrs[1], X, ldx, beta, Y, ldy, exw);
    else
        assert(0 && "float times complex not implemented");

    Py_RETURN_NONE;
}

static PyObject*
py_inspect(PyObject *self, PyObject *args)
{
    unsigned int M, K;
    PyArrayObject *py_colInds, *py_rowPtrs;

    if (!PyArg_ParseTuple(args, "iiOO",
        &M, &K, &py_colInds, &py_rowPtrs))
        return NULL;

    unsigned int *rowPtrs = PyArray_DATA(py_rowPtrs);
    unsigned int *colInds = PyArray_DATA(py_colInds);

    int nzrows = 0,
        nzcols = 0,
        exwrite = 1;
    int *nz = malloc(K * sizeof(int)); // nonzeros in each col
    memset(nz, 0, K * sizeof(int));

    for (unsigned int m = 0; m < M; m++) {
        unsigned int b = rowPtrs[m+0],
                     e = rowPtrs[m+1];
        if (e>b)
            nzrows += 1;
        for (unsigned int idx = b; idx < e; idx++)
            nz[ colInds[idx] ] += 1;
    }   

    for (unsigned int k = 0; k < K; k++) {
        if (nz[k] > 0)
            nzcols += 1;
        if (nz[k] > 1)
            exwrite = 0;
    }
    free(nz);
    return Py_BuildValue("iii", nzrows, nzcols, exwrite);
}


static PyMethodDef _customcpuMethods[] = {
    { "csrmm", py_csrmm, METH_VARARGS, NULL },
    { "inspect", py_inspect, METH_VARARGS, NULL },
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

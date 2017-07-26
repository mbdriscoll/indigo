#include <complex.h>
#include <assert.h>

#include <omp.h>


int custom_csrmm(
    int transA, int M, int N, int K, complex float alpha,
    complex float *val, int *col, int *pntrb, int *pntre,
    complex float *B, int ldb, complex float beta,
    complex float *C, int ldc
) { 
    assert(transA == 0);

    //#pragma omp parallel
    {
        complex float acc[N];

        //#pragma omp for schedule(guided)
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++)
                acc[n] = 0;
            for (int i = pntrb[m]; i < pntre[m]; i++) {
                int k = col[i];
                for (int n = 0; n < N; n++)
                    acc[n] += val[i] * B[k+n*ldb];
            }
            for (int n = 0; n < N; n++)
                C[m+n*ldc] = alpha * acc[n] + beta * C[m+n*ldc];
        }   
    }

    return 0;
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
    int adjoint = 0;
    PyObject *py_Ashape, *py_alpha, *py_beta;
    PyArrayObject *py_Y, *py_colind, *py_rowptr, *py_vals, *py_X;
    if (!PyArg_ParseTuple(args, "OOOOOOOOp",
        &py_Y, &py_Ashape, &py_colind, &py_rowptr, &py_vals, &py_X, &py_alpha, &py_beta, &adjoint))
        return NULL;

    int *rowPtrs = PyArray_DATA(py_rowptr);
    int *colInds = PyArray_DATA(py_colind);
    complex float *values = PyArray_DATA(py_vals);

    complex float *Y = PyArray_DATA(py_Y),
                  *X = PyArray_DATA(py_X);

    int M = PyArray_DIM(py_Y, 0),
        N = PyArray_DIM(py_Y, 1),
        K = PyArray_DIM(py_X, 0);

    float alpha_r = (float) PyComplex_RealAsDouble( py_alpha ),
          alpha_i = (float) PyComplex_ImagAsDouble( py_alpha ),
           beta_r = (float) PyComplex_RealAsDouble( py_beta  ),
           beta_i = (float) PyComplex_ImagAsDouble( py_beta  );

    complex float alpha = alpha_r + I * alpha_i,
                   beta =  beta_r + I *  beta_i;

    int ldx = PyLong_AsLong(PyObject_GetItem( PyObject_GetAttrString((PyObject*) py_X, "_leading_dims"), PyLong_FromLong(0) ));
    int ldy = PyLong_AsLong(PyObject_GetItem( PyObject_GetAttrString((PyObject*) py_Y, "_leading_dims"), PyLong_FromLong(0) ));

    custom_csrmm(adjoint, M, N, K, alpha, values, colInds, rowPtrs, rowPtrs+1, X, ldx, beta, Y, ldy);

    Py_RETURN_NONE;
}

static PyMethodDef _customcpuMethods[] = {
    { "csrmm", py_csrmm, METH_VARARGS, NULL },
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

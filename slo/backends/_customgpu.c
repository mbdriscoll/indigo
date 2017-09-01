#include <complex.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include <numpy/arrayobject.h>

#include "_customgpu.h"

static PyObject*
py_exw_csrmm_H(PyObject *self, PyObject *args)
{
    PyObject *py_alpha, *py_beta;
    unsigned int ldx, ldy, M, N, K;
    unsigned long rowPtrs, colInds, values, X, Y;
    if (!PyArg_ParseTuple(args, "iiiOkkkkiOki",
        &M, &N, &K, &py_alpha,
        &values, &colInds, &rowPtrs,
        &X, &ldx, &py_beta, &Y, &ldy))
        return NULL;

    float alpha_r = (float) PyComplex_RealAsDouble( py_alpha ),
          alpha_i = (float) PyComplex_ImagAsDouble( py_alpha ),
           beta_r = (float) PyComplex_RealAsDouble( py_beta  ),
           beta_i = (float) PyComplex_ImagAsDouble( py_beta  );
    complex float alpha = alpha_r + I * alpha_i,
                   beta =  beta_r + I *  beta_i;

    c_exw_csrmm_H(M, N, K, alpha, (complex float*) values, (unsigned int *) colInds, (unsigned int *) rowPtrs, (complex float *) X, ldx, beta, (complex float *) Y, ldy);

    Py_RETURN_NONE;
}

static PyMethodDef _customgpuMethods[] = {
    { "exw_csrmm", py_exw_csrmm_H, METH_VARARGS, NULL },
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef _customgpu = {
    PyModuleDef_HEAD_INIT, "_customgpu", NULL, -1, _customgpuMethods,
};

PyMODINIT_FUNC
PyInit__customgpu(void)
{
    import_array();
    return PyModule_Create(&_customgpu);
}

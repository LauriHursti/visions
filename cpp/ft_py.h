#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* getKpBoxes(PyObject* dummy, PyObject* args);
static PyObject* getFTKeypoints(PyObject* obj, PyObject* args);
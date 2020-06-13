#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <tuple>
#include <numpy/arrayobject.h>

#include "constants.h"
#include "ft_api.h"
#include "ft_py.h"

using std::tie;

static PyObject* getKpBoxes(PyObject* dummy, PyObject* args)
{
    PyArrayObject* imgArg;
    int padding = 0;
    int count = 4000;
    int scales = 4;
    int threshold = 12;
    bool positives = true;
    bool negatives = true;
    int wLimit = DIM_SIZE;
    int hLimit = DIM_SIZE;

    // For some reason the first int was always set to zero by PyArg_ParseTuple
    // For that reason, there's an extra argument called "padding"
    if (!PyArg_ParseTuple(args, "O!iiiippii", &PyArray_Type, &imgArg, &padding, &count, &scales, &threshold, &positives, &negatives, &wLimit, &hLimit))
    {
        return NULL;
    }

    // The first argument is converted to a uint8_t pointer
    // The size of the content would be DIM_SIZE * DIM_SIZE
	uint8_t* imgUint = (uint8_t*) PyArray_DATA(imgArg);
    vector<array<int, 6>> kps = getFASTextKeypoints(imgUint, count, scales, threshold, positives, negatives);

    int* boxes;
    int boxCount;
    tie(boxes, boxCount) = py_getFASTextBoxes(imgUint, kps, wLimit, hLimit);
    
    long dims[2] = {boxCount, 5};

    PyObject* retValue = PyArray_SimpleNewFromData(2, dims, NPY_INT32, &boxes[0]);
    Py_INCREF(retValue);

    return retValue;
}

static PyObject* getFTKeypoints(PyObject* obj, PyObject* args)
{
    // Function parameters
    PyArrayObject* imgArg = NULL;
    int padding = 0;
    int count = 4000;
    int scales = 4;
    int threshold = 12;
    bool positives = true;
    bool negatives = true;

    // String "O|iiippii" means that after the image there's optional arguments of int, int, int, bool, bool
	if (!PyArg_ParseTuple(args, "O!iiiipp", &PyArray_Type, &imgArg, &padding, &count, &scales, &threshold, &positives, &negatives))
	{
        return NULL;
    }

	uint8_t* imgUint = (uint8_t*) PyArray_DATA(imgArg);
    int* kps;
    int kpCount;
	tie(kps, kpCount) = py_getFASTextKeypoints(imgUint, count, scales, threshold, positives, negatives);

    long dims[2] = {kpCount, 5};

    PyObject* retValue = PyArray_SimpleNewFromData(2, dims, NPY_INT32, &kps[0]);
    Py_INCREF(retValue);

    return retValue;
}

static PyObject* kpBoxDBSCAN(PyObject* obj, PyObject* args)
{
    // Function parameters
    PyArrayObject* boxesArg = NULL;
    int padding = 0;
    int boxCount = 0;
    float eps = 10;
    int min_samples = 10;
  
	if (!PyArg_ParseTuple(args, "O!iifi", &PyArray_Type, &boxesArg, &padding, &boxCount, &eps, &min_samples))
	{
        return NULL;
    }

    // boxesArg is a numpy array of shape (n, 4) that mark the bounding points of a keypoint CC
    // in order "left, right, top, bottom"
	int32_t* boxes = (int32_t*) PyArray_DATA(boxesArg);
    int16_t* labels;
    int labelCount;
	tie(labels, labelCount) = py_getCompClusters(boxes, boxCount, eps, min_samples);

    long dims[1] = {labelCount};

    PyObject* retValue = PyArray_SimpleNewFromData(1, dims, NPY_INT16, &labels[0]);
    Py_INCREF(retValue);

    return retValue;
}

static PyMethodDef ftMethods[] = {
    {"getFTKeypoints", getFTKeypoints, METH_VARARGS, "Get FASText keypoints"},
    {"getKpBoxes", getKpBoxes, METH_VARARGS, "Get bounding boxes for FASText connected components"},
    {"kpBoxDBSCAN", kpBoxDBSCAN, METH_VARARGS, "Get clusters of connected component bounding boxes"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ftpy = {
    PyModuleDef_HEAD_INIT,
    "ftpy",
    "Placeholder documentation",
    -1,
    ftMethods
};

PyMODINIT_FUNC PyInit_libftpy(void)
{
    import_array();
    return PyModule_Create(&ftpy);
}
#pragma once

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace tensorrt_inference {
namespace py = pybind11;

void pybind_detection(py::module& m);
void pybind_model(py::module& m);

}  // namespace tensorrt_inference
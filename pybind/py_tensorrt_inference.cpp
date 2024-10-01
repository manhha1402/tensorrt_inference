#include "py_tensorrt_inference.h"

#include "ndarray_converter.h"
namespace tensorrt_inference {
namespace py = pybind11;
PYBIND11_MODULE(tensorrt_inference_py, m) {
  NDArrayConverter::init_numpy();
  py::module m_submodule_detection = m.def_submodule("detection");
  pybind_detection(m_submodule_detection);

  py::module m_submodule_model = m.def_submodule("model");
  pybind_model(m_submodule_model);
}
}  // namespace tensorrt_inference
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ndarray_converter.h"
#include "py_tensorrt_inference.h"
#include "tensorrt_inference/tensorrt_inference.h"
namespace py = pybind11;
using namespace py::literals;
namespace tensorrt_inference {
class PyDetection {
 public:
  PyDetection(
      const std::string &model_name,
      tensorrt_inference::Options options = tensorrt_inference::Options(),
      const std::string &model_dir = "") {
    std::filesystem::path model_path;
    if (model_dir.empty()) {
      model_path =
          std::filesystem::path(std::getenv("HOME")) / "data" / "weights";
    } else {
      model_path = std::filesystem::path(model_dir);
    }
    if (model_name.find("yolov8") != std::string::npos) {
      detector_ = std::make_unique<tensorrt_inference::YoloV8>(
          model_name, options, model_path);
    } else if (model_name.find("yolov9") != std::string::npos) {
      detector_ = std::make_unique<tensorrt_inference::YoloV9>(
          model_name, options, model_path);
    } else if (model_name.find("facedetector") != std::string::npos) {
      detector_ = std::make_unique<tensorrt_inference::RetinaFace>(
          model_name, options, model_path);
    } else if (model_name.find("retinaface") != std::string::npos) {
      detector_ = std::make_unique<tensorrt_inference::RetinaFace>(
          model_name, options, model_path);
    } else {
      throw std::runtime_error("unkown model");
    }
  }

  std::vector<Object> detect(const cv::Mat &image,
                             const DetectionParams &params,
                             const std::vector<std::string> &detected_class) {
    return detector_->detect(image, params, detected_class);
  }
  cv::Mat drawObjectLabels(const cv::Mat &image,
                           const std::vector<Object> &objects,
                           const DetectionParams &params, unsigned int scale) {
    return detector_->drawObjectLabels(image, objects, params, scale);
  }

 private:
  std::unique_ptr<tensorrt_inference::Detection> detector_;
};

void pybind_detection(py::module &m) {
  // Options

  // Enum class binding for Precision
  py::enum_<Precision>(m, "Precision")
      .value("FP32", Precision::FP32)
      .value("FP16", Precision::FP16)
      .value("INT8", Precision::INT8)
      .export_values();  // Exposes the enum values to Python

  // Struct binding for Options
  py::class_<Options>(m, "Options")
      .def(py::init<>())  // Default constructor
      .def_readwrite("precision", &Options::precision)
      .def_readwrite("calibrationDataDirectoryPath",
                     &Options::calibrationDataDirectoryPath)
      .def_readwrite("calibrationBatchSize", &Options::calibrationBatchSize)
      .def_readwrite("optBatchSize", &Options::optBatchSize)
      .def_readwrite("maxBatchSize", &Options::maxBatchSize)
      .def_readwrite("deviceIndex", &Options::deviceIndex)
      .def_readwrite("engine_file_dir", &Options::engine_file_dir)
      .def_readwrite("maxInputWidth", &Options::maxInputWidth)
      .def_readwrite("minInputWidth", &Options::minInputWidth)
      .def_readwrite("optInputWidth", &Options::optInputWidth)
      .def_readwrite("MIN_DIMS_", &Options::MIN_DIMS_)
      .def_readwrite("OPT_DIMS_", &Options::OPT_DIMS_)
      .def_readwrite("MAX_DIMS_", &Options::MAX_DIMS_);

  m.doc() = "Tensorrt Inference";
  py::class_<PyDetection>(m, "Detection")
      .def(py::init<const std::string, const tensorrt_inference::Options &,
                    const std::string &>(),
           py::arg("model_name"),
           py::arg("options") =
               tensorrt_inference::Options(),  // Default argument
           py::arg("model_dir") = "")
      .def("detect", &PyDetection::detect, "image"_a, "params"_a, "params"_a)
      .def("draw", &PyDetection::drawObjectLabels, "image"_a, "objects"_a,
           "params"_a, "scale"_a);

  py::class_<tensorrt_inference::DetectionParams>(m, "DetectionParams")
      .def(py::init<>())
      .def(py::init<float, float, float, float, int>(),
           py::arg("obj_threshold") = 0.5, py::arg("nms_threshold") = 0.65,
           py::arg("seg_threshold") = 0.5, py::arg("kps_threshold") = 0.5,
           py::arg("num_detect") = 20)
      .def_readwrite("obj_threshold",
                     &tensorrt_inference::DetectionParams::obj_threshold)
      .def_readwrite("nms_threshold",
                     &tensorrt_inference::DetectionParams::nms_threshold)
      .def_readwrite("seg_threshold",
                     &tensorrt_inference::DetectionParams::seg_threshold)
      .def_readwrite("kps_threshold",
                     &tensorrt_inference::DetectionParams::kps_threshold)
      .def_readwrite("num_detect",
                     &tensorrt_inference::DetectionParams::num_detect);
  py::class_<cv::Rect>(m, "Rect")
      .def(py::init<>())                    // Default constructor
      .def(py::init<int, int, int, int>())  // (x, y, width, height)
      .def_readwrite("x", &cv::Rect::x)     // Expose members
      .def_readwrite("y", &cv::Rect::y)
      .def_readwrite("width", &cv::Rect::width)
      .def_readwrite("height", &cv::Rect::height)
      .def("__repr__", [](const cv::Rect &r) {
        return "<Rect(x=" + std::to_string(r.x) + ", y=" + std::to_string(r.y) +
               ", width=" + std::to_string(r.width) +
               ", height=" + std::to_string(r.height) + ")>";
      });
  py::class_<tensorrt_inference::Object> object(m, "Object");
  object.def(py::init<>());
  object.def_readwrite("label", &tensorrt_inference::Object::label);
  object.def_readwrite("probability", &tensorrt_inference::Object::probability);
  object.def_readwrite("rect", &tensorrt_inference::Object::rect);
  object.def_readwrite("box_mask", &tensorrt_inference::Object::box_mask);
  object.def_readwrite("kps", &tensorrt_inference::Object::kps);
  object.def_readwrite("landmarks", &tensorrt_inference::Object::landmarks);
}

}  // namespace tensorrt_inference

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ndarray_converter.h"
#include "py_tensorrt_inference.h"
#include "tensorrt_inference/tensorrt_inference.h"
namespace py = pybind11;
using namespace py::literals;
namespace tensorrt_inference
{
  class PyModel
  {
  public:
    PyModel(const std::string &model_name,
            tensorrt_inference::Options options = tensorrt_inference::Options(),
            const std::string &model_dir = "")
    {
      std::filesystem::path model_path;
      if (model_dir.empty())
      {
        model_path =
            std::filesystem::path(std::getenv("HOME")) / "data" / "weights";
      }
      else
      {
        model_path = std::filesystem::path(model_dir);
      }
      if (model_name.find("FaceNet") != std::string::npos)
      {
        model_ = std::make_unique<tensorrt_inference::FaceRecognition>(
            model_name, options, model_path);
      }
      else
      {
        throw std::runtime_error("unkown model");
      }
    }
    cv::Mat getEmbedding(const cv::Mat &cropped_face)
    {
      std::unordered_map<std::string, std::vector<float>> feature_vectors;
      bool res = model_->doInference(cropped_face, feature_vectors);
      cv::Mat out = cv::Mat(feature_vectors.begin()->second.size(), 1, CV_32F,
                            feature_vectors.begin()->second.data());
      cv::Mat out_norm;
      cv::normalize(out, out_norm);
      return out_norm;
    }

  private:
    std::unique_ptr<tensorrt_inference::Model> model_;
  };
  void pybind_model(py::module &m)
  {
    py::class_<PyModel>(m, "Model")
        .def(py::init<const std::string, const tensorrt_inference::Options &,
                      const std::string &>(),
             py::arg("model_name"),
             py::arg("options") =
                 tensorrt_inference::Options(), // Default argument
             py::arg("model_dir") = "")
        .def("get_embedding", &PyModel::getEmbedding, "image"_a);
  }
} // namespace tensorrt_inference
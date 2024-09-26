#pragma once
#include <fstream>

#include "tensorrt_inference/detection.h"
namespace tensorrt_inference {
struct CroppedFace {
  cv::Mat face;
  cv::Mat face_mat;
  cv::Rect rect;
};
class ArcFace : public Detection {
 public:
  // Builds the onnx model into a TensorRT engine, and loads the engine into
  // memory
  ArcFace(const std::string &model_name,
          tensorrt_inference::Options options = tensorrt_inference::Options(),
          const std::filesystem::path &model_dir =
              std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
  std::vector<CroppedFace> getCroppedFaces(const cv::Mat &frame,
                                           const std::vector<Object> &faces);
  void preprocessFace(cv::Mat &face, cv::Mat &output);
  void preprocessFaces();

 private:
  // Postprocess the output
  std::vector<Object> postprocess(
      std::unordered_map<std::string, std::vector<float>> &feature_vectors,
      const DetectionParams &params = DetectionParams(),
      const std::vector<std::string> &detected_class = {}) override;
};
}  // namespace tensorrt_inference
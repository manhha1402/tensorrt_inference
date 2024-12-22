#pragma once
#include <fstream>

#include "tensorrt_inference/detection.h"
namespace tensorrt_inference
{
  class RetinaFace : public Detection
  {
    struct anchorBox
    {
      float cx;
      float cy;
      float sx;
      float sy;
    };

  public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into
    // memory
    RetinaFace(
        const std::string &model_name,
        tensorrt_inference::Options options = tensorrt_inference::Options(),
        const std::filesystem::path &model_dir =
            std::filesystem::path(std::getenv("HOME")) / "data" / "weights");

  private:
    // Postprocess the output
    std::vector<Object> postprocess(
        std::unordered_map<std::string, std::vector<float>> &feature_vectors,
        const DetectionParams &params = DetectionParams(),
        const std::vector<std::string> &detected_class = {}) override;

    void create_anchor_retinaface(std::vector<anchorBox> &anchor, int w, int h);
  };
} // namespace tensorrt_inference
#pragma once
#include <fstream>

#include "tensorrt_inference/model.h"
namespace tensorrt_inference {

class FaceRecognition : public Model {
 public:
  // Builds the onnx model into a TensorRT engine, and loads the engine into
  // memory
  FaceRecognition(
      const std::string &model_name,
      tensorrt_inference::Options options = tensorrt_inference::Options(),
      const std::filesystem::path &model_dir =
          std::filesystem::path(std::getenv("HOME")) / "data" / "weights");
};
}  // namespace tensorrt_inference
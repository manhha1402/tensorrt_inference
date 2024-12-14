#pragma once

#include <array>
#include <variant>
#include <opencv2/core/cuda.hpp>
#include <vector>

#include "NvInfer.h"  // Include for nvinfer1::Dims and nvinfer1::Dims3
#include "NvInferPlugin.h"
namespace tensorrt_inference {
struct NetInfo {
  void *buffer;
  nvinfer1::Dims dims;
  size_t tensor_length = 0;
  nvinfer1::DataType data_type;
};

class IEngine {
 public:
  virtual ~IEngine() = default;
  virtual bool buildLoadNetwork(const std::string &onnx_file) = 0;
  virtual bool loadNetwork(std::string trtModelPath) = 0;
//   virtual bool runInference(
//       cv::cuda::GpuMat &input,
//       std::unordered_map<std::string, std::vector<FeatureType>> &featureVectors) = 0;
  virtual const std::unordered_map<std::string, NetInfo> &getInputInfo()
      const = 0;
  virtual const std::unordered_map<std::string, NetInfo> &getOutputInfo()
      const = 0;
};
}  // namespace tensorrt_inference
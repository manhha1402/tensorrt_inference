#pragma once

#include "NvInfer.h" // Include for nvinfer1::Dims and nvinfer1::Dims3
#include <array>
#include <opencv2/core/cuda.hpp>
#include <vector>
namespace tensorrt_inference {
struct NetInfo {
  void *buffer;
  nvinfer1::Dims dims;
  size_t tensor_length = 0;
};

template <typename T> class IEngine {
public:
  virtual ~IEngine() = default;
  virtual bool buildLoadNetwork(const std::string &onnx_file) = 0;
  virtual bool loadNetwork(std::string trtModelPath) = 0;
  virtual bool runInference(
      const cv::cuda::GpuMat &input,
      std::unordered_map<std::string, std::vector<T>> &featureVectors) = 0;
  virtual const std::unordered_map<std::string, NetInfo> &
  getInputInfo() const = 0;
  virtual const std::unordered_map<std::string, NetInfo> &
  getOutputInfo() const = 0;
};
} // namespace tensorrt_inference
#pragma once
#include <spdlog/spdlog.h>

#include <filesystem>

#include "tensorrt_inference/tensorrt_api/util/Util.h"
namespace tensorrt_inference {
template <typename T>
bool Engine<T>::runInference(
    cv::cuda::GpuMat &inputs,
    std::unordered_map<std::string, std::vector<T>> &feature_vectors) {
  // First we do some error checking
  if (inputs.empty()) {
    spdlog::error("Provided input vector is empty!");
    return false;
  }
  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
  input_map_[input_map_.begin()->first].tensor_length =
      inputs.cols * inputs.rows * inputs.channels();

  // Set the address of the input buffers
  bool status = m_context->setTensorAddress(input_map_.begin()->first.c_str(),
                                            inputs.ptr<void>());
  if (!status) {
    return false;
  }
  for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
    bool status =
        m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
    if (!status) {
      return false;
    }
  }
  if (!m_context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  // Run inference.
  status = m_context->enqueueV3(inferenceCudaStream);
  if (!status) {
    return false;
  }
  // Copy the outputs back to CPU
  // Batch
  for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
    feature_vectors[it->first].resize(it->second.tensor_length);
    // Copy the output
    Util::checkCudaErrorCode(
        cudaMemcpyAsync(feature_vectors[it->first].data(), it->second.buffer,
                        it->second.tensor_length * sizeof(T),
                        cudaMemcpyDeviceToHost, inferenceCudaStream));
  }

  // Synchronize the cuda stream
  Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
  Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
  return true;
}
}  // namespace tensorrt_inference
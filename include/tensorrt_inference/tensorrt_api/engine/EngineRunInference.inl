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
  if (!m_context->allInputDimensionsSpecified()) {
    auto msg = "Error, not all required dimensions specified.";
    spdlog::error(msg);
    throw std::runtime_error(msg);
  }
  //   std::cout << inputs.cols * inputs.rows * inputs.channels() << std::endl;
  //   std::cout << input_map_.at("input0").tensor_length << std::endl;
  const uint32_t tensor_length = inputs.cols * inputs.rows * inputs.channels();
  const std::string &input_tensor_name = input_map_.begin()->first;
  //   input_map_[tensor_name].tensor_length = tensor_length;
  //   Util::checkCudaErrorCode(cudaMallocAsync(&input_map_[tensor_name].buffer,
  //                                            tensor_length * sizeof(T),
  //                                            inferenceCudaStream));
  //   // DMA input batch data to device, infer on the batch asynchronously, and
  //   DMA
  //   // output back to host
  //   for (auto it = input_map_.begin(); it != input_map_.end(); ++it) {
  //     Util::checkCudaErrorCode(
  //         cudaMemcpyAsync(it->second.buffer, (T *)inputs.ptr<T>(0),
  //                         it->second.tensor_length * sizeof(T),
  //                         cudaMemcpyDeviceToDevice, inferenceCudaStream));
  //   }
  // Ensure all dynamic bindings have been defined.

  // Set the address of the input  buffers
  //   for (auto it = input_map_.begin(); it != input_map_.end(); ++it) {
  //     bool status =
  //         m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
  //     if (!status) {
  //       return false;
  //     }
  //   }
  bool status = m_context->setTensorAddress(input_tensor_name.c_str(),
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
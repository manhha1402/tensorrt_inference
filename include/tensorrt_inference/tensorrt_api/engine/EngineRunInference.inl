#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>
#include "tensorrt_inference/tensorrt_api/util/Util.h"
namespace tensorrt_inference
{
template <typename T>
bool Engine<T>::runInference(const cv::Mat &inputs, std::unordered_map<std::string,std::vector<T>> &feature_vectors) {
    // First we do some error checking
    if (inputs.empty()) {
        spdlog::error("Provided input vector is empty!");
        return false;
    }
    // Create the cuda stream that will be used for inference
    cudaStream_t inferenceCudaStream;
    Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));
    
      // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    for (auto it = input_map_.begin(); it !=input_map_.end(); ++ it) {
        Util::checkCudaErrorCode(cudaMemcpyAsync(it->second.buffer, (T *)inputs.ptr<T>(0),
                                    it->second.tensor_length * sizeof(T),
                                    cudaMemcpyHostToDevice, inferenceCudaStream));
    }
    // Ensure all dynamic bindings have been defined.
    if (!m_context->allInputDimensionsSpecified()) {
        auto msg = "Error, not all required dimensions specified.";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    // Set the address of the input  buffers
    for (auto it = input_map_.begin(); it !=input_map_.end(); ++ it) {
        bool status = m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
        if (!status) {
            return false;
        }
    }
    // Set the address of the output buffers
    for (auto it = output_map_.begin(); it !=output_map_.end(); ++ it) {
        bool status = m_context->setTensorAddress(it->first.c_str(), it->second.buffer);
        if (!status) {
            return false;
        }
    }
    // Run inference.
    bool status = m_context->enqueueV3(inferenceCudaStream);
    if (!status) {
        return false;
    }
    // Copy the outputs back to CPU
        // Batch
    for (auto it = output_map_.begin(); it !=output_map_.end(); ++ it) {
        feature_vectors[it->first].resize(it->second.tensor_length);
        // Copy the output
        Util::checkCudaErrorCode(cudaMemcpyAsync(feature_vectors[it->first].data(),
                                                it->second.buffer,
                                                it->second.tensor_length * sizeof(T),
                                                cudaMemcpyDeviceToHost,
                                                inferenceCudaStream));
        
    }
 

    // Synchronize the cuda stream
    Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
    Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));
    return true;
}
}
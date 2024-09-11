#pragma once
#include <filesystem>
#include <spdlog/spdlog.h>
namespace tensorrt_inference
{
template <typename T>
void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<T>> &output) {
    if (input.size() != 1) {
        auto msg = "The feature vector has incorrect dimensions!";
        spdlog::error(msg);
        throw std::logic_error(msg);
    }

    output = std::move(input[0]);
}

template <typename T> void Engine<T>::transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output) {
    // if (input.size() != 1 || input[0].size() != 1) {
    //     auto msg = "The feature vector has incorrect dimensions!";
    //     spdlog::error(msg);
    //     throw std::logic_error(msg);
    // }
    output = std::move(input[0][0]);
}

template <typename T>
cv::cuda::GpuMat Engine<T>::resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                const cv::Scalar &bgcolor) {
    float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
    int unpad_w = r * input.cols;
    int unpad_h = r * input.rows;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

template <typename T> void Engine<T>::getDeviceNames(std::vector<std::string> &deviceNames) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    for (int device = 0; device < numGPUs; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        deviceNames.push_back(std::string(prop.name));
    }
}

template <typename T> std::string Engine<T>::serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
    const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
    std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

    // Add the GPU device name to the file to ensure that the model is only used
    // on devices with the exact same GPU
    std::vector<std::string> deviceNames;
    getDeviceNames(deviceNames);

    if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
        auto msg = "Error, provided device index is out of range!";
        spdlog::error(msg);
        throw std::runtime_error(msg);
    }

    auto deviceName = deviceNames[options.deviceIndex];
    // Remove spaces from the device name
    deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());

    engineName += "." + deviceName;

    // Serialize the specified options into the filename
    if (options.precision == Precision::FP16) {
        engineName += ".fp16";
    } else if (options.precision == Precision::FP32) {
        engineName += ".fp32";
    } else {
        engineName += ".int8";
    }

    engineName += "." + std::to_string(options.maxBatchSize);
    engineName += "." + std::to_string(options.optBatchSize);
    engineName += "." + std::to_string(options.minInputWidth);
    engineName += "." + std::to_string(options.optInputWidth);
    engineName += "." + std::to_string(options.maxInputWidth);
    spdlog::info("Engine name: {}", engineName);
    return engineName;
}


template <typename T> void Engine<T>::clearGpuBuffers() {
    if (!input_map_.empty()) {
        // Free GPU memory of inputs
        for (auto it = input_map_.begin(); it != input_map_.end(); ++it) {
            Util::checkCudaErrorCode(cudaFree(it->second.buffer));
        }
        input_map_.clear();
    }
    if (!output_map_.empty()) {
        // Free GPU memory of inputs
        for (auto it = output_map_.begin(); it != output_map_.end(); ++it) {
            Util::checkCudaErrorCode(cudaFree(it->second.buffer));
        }
        output_map_.clear();
    }
}
}
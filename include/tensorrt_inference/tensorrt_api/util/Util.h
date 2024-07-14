#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
namespace tensorrt_inference
{
namespace Util {
    // Checks if a file exists at the given file path
    bool doesFileExist(const std::string &filepath);

    // Checks and logs CUDA error codes
    void checkCudaErrorCode(cudaError_t code);

    // Retrieves a list of file names in the specified directory
    std::vector<std::string> getFilesInDirectory(const std::string &dirPath);
}
}
#include "tensorrt_inference/tensorrt_api/util/Util.inl"

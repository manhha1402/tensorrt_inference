#pragma once

#include <cuda_runtime.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
namespace tensorrt_inference {
namespace Util {
// Checks if a file exists at the given file path
bool doesFileExist(const std::string &filepath);

// Checks and logs CUDA error codes
void checkCudaErrorCode(cudaError_t code);

// Retrieves a list of file names in the specified directory
std::vector<std::string> getFilesInDirectory(const std::string &dirPath);

// Get folder of given filepath
std::filesystem::path getFolderOfFile(const std::string &filepath);
} // namespace Util
} // namespace tensorrt_inference
#include "tensorrt_inference/tensorrt_api/util/Util.inl"
